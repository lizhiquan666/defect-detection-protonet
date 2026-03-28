#include "utils.hpp"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace fs = std::filesystem;

namespace {

float clamp_float(float v, float lo, float hi) {
    return std::max(lo, std::min(v, hi));
}

float rotated_iou(const RBox& a, const RBox& b) {
    cv::RotatedRect ra(cv::Point2f(a.cx, a.cy), cv::Size2f(std::max(a.w, 1e-3f), std::max(a.h, 1e-3f)), a.angle_deg);
    cv::RotatedRect rb(cv::Point2f(b.cx, b.cy), cv::Size2f(std::max(b.w, 1e-3f), std::max(b.h, 1e-3f)), b.angle_deg);

    std::vector<cv::Point2f> inter_pts;
    const int inter_type = cv::rotatedRectangleIntersection(ra, rb, inter_pts);
    if (inter_type == cv::INTERSECT_NONE || inter_pts.empty()) {
        return 0.0f;
    }

    float inter_area = 0.0f;
    if (inter_type == cv::INTERSECT_FULL) {
        inter_area = std::min(a.w * a.h, b.w * b.h);
    } else {
        std::vector<cv::Point2f> hull;
        cv::convexHull(inter_pts, hull);
        inter_area = std::fabs(cv::contourArea(hull));
    }

    const float area_a = std::max(a.w * a.h, 1e-6f);
    const float area_b = std::max(b.w * b.h, 1e-6f);
    const float uni = area_a + area_b - inter_area;
    if (uni <= 1e-6f) return 0.0f;
    return inter_area / uni;
}

std::vector<RBox> rotated_nms(const std::vector<RBox>& boxes, float iou_thresh) {
    if (boxes.empty()) return {};
    if (iou_thresh <= 0.0f) return boxes;

    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return boxes[i].score > boxes[j].score;
    });

    std::vector<RBox> keep;
    for (int idx : order) {
        const auto& cand = boxes[idx];
        bool suppressed = false;
        for (const auto& kept : keep) {
            if (rotated_iou(cand, kept) > iou_thresh) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(cand);
    }
    return keep;
}

} // namespace

// ====================== 原型加载 ======================
std::vector<torch::Tensor> load_prototype_features(const std::string& proto_dir,
                                                   const std::string& class_name,
                                                   torch::Device device) {
    std::vector<torch::Tensor> protos(3);
    const fs::path base = fs::path(proto_dir) / class_name;

    torch::load(protos[0], (base / "layer1.pt").string());
    torch::load(protos[1], (base / "layer2.pt").string());
    torch::load(protos[2], (base / "layer3.pt").string());

    for (auto& p : protos) p = p.to(device);
    return protos;
}

// ====================== 原型匹配 ======================
std::vector<torch::Tensor> get_prototype_features(const std::vector<torch::Tensor>& features,
                                                  const std::vector<torch::Tensor>& proto_features) {
    std::vector<torch::Tensor> closest(3);
    for (int i = 0; i < 3; ++i) {
        auto fi = features[i].view({features[i].size(0), -1});
        auto pi = proto_features[i].view({proto_features[i].size(0), -1});
        auto dist = torch::cdist(fi.unsqueeze(0), pi.unsqueeze(0)).squeeze(0);
        auto inds = std::get<1>(dist.min(1));
        closest[i] = proto_features[i].index_select(0, inds);
    }
    return closest;
}

// ====================== 残差 & 拼接 ======================
std::vector<torch::Tensor> get_residual_features(const std::vector<torch::Tensor>& features,
                                                 const std::vector<torch::Tensor>& closest_protos) {
    std::vector<torch::Tensor> res(3);
    for (int i = 0; i < 3; ++i) {
        res[i] = torch::mse_loss(features[i], closest_protos[i], torch::Reduction::None);
    }
    return res;
}

std::vector<torch::Tensor> get_concatenated_features(const std::vector<torch::Tensor>& features1,
                                                     const std::vector<torch::Tensor>& features2) {
    std::vector<torch::Tensor> concat(3);
    for (int i = 0; i < 3; ++i) {
        concat[i] = torch::cat({features1[i], features2[i]}, 1);
    }
    return concat;
}

// ====================== 热图后处理 ======================
torch::Tensor postprocess_anomaly_map(const torch::Tensor& logits) {
    if (logits.dim() == 4 && logits.size(1) == 2) {
        return torch::softmax(logits, 1).slice(1, 1, 2);
    }
    if (logits.dim() == 4 && logits.size(1) == 1) {
        auto min_v = logits.min().item<float>();
        auto max_v = logits.max().item<float>();
        if (min_v >= 0.0f && max_v <= 1.0f) {
            return logits;
        }
    }
    return torch::sigmoid(logits);
}

float compute_image_level_score(const torch::Tensor& score_map) {
    auto flat = score_map.view({score_map.size(0), -1});
    auto [values, _] = torch::topk(flat, 100, 1);
    return values.mean(1).item<float>();
}

// ====================== RBox 提取 ======================
std::vector<RBox> extract_rboxes_from_heatmap(const torch::Tensor& score_map,
                                              const RBoxPostprocessOptions& opt) {
    torch::Tensor map = score_map;
    if (map.dim() == 4) map = map.squeeze(0).squeeze(0);
    else if (map.dim() == 3) map = map.squeeze(0);

    torch::Tensor cpu_map = map.cpu().to(torch::kFloat32).contiguous();
    int H = cpu_map.size(0), W = cpu_map.size(1);

    cv::Mat gray(H, W, CV_32F, cpu_map.data_ptr<float>());
    cv::Mat binary;
    cv::threshold(gray, binary, opt.thresh, 1.0, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8UC1, 255.0);

    const int k = std::max(1, opt.morph_kernel);
    if (k > 1) {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        if (opt.morph_close_iter > 0) {
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), opt.morph_close_iter);
        }
        if (opt.morph_open_iter > 0) {
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), opt.morph_open_iter);
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<RBox> rboxes;
    rboxes.reserve(contours.size());

    const float min_area = std::max(opt.min_area, 0.0f);
    const float max_area = std::max(opt.max_area, min_area);
    const float min_aspect = std::max(opt.min_aspect, 1.0f);
    const float max_aspect = std::max(opt.max_aspect, min_aspect);
    const float min_side = std::max(opt.min_side, 0.0f);

    const float edge_margin = clamp_float(opt.edge_margin_ratio, 0.0f, 0.45f);
    const float edge_x = edge_margin * static_cast<float>(W);
    const float edge_y = edge_margin * static_cast<float>(H);

    const float ignore_tl_w = clamp_float(opt.ignore_tl_width_ratio, 0.0f, 1.0f) * static_cast<float>(W);
    const float ignore_tl_h = clamp_float(opt.ignore_tl_height_ratio, 0.0f, 1.0f) * static_cast<float>(H);

    const float ign_x0 = clamp_float(opt.ignore_rect_x0_ratio, 0.0f, 1.0f) * static_cast<float>(W);
    const float ign_y0 = clamp_float(opt.ignore_rect_y0_ratio, 0.0f, 1.0f) * static_cast<float>(H);
    const float ign_x1 = clamp_float(opt.ignore_rect_x1_ratio, 0.0f, 1.0f) * static_cast<float>(W);
    const float ign_y1 = clamp_float(opt.ignore_rect_y1_ratio, 0.0f, 1.0f) * static_cast<float>(H);
    const bool use_ignore_rect = (ign_x1 > ign_x0) && (ign_y1 > ign_y0);

    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_area) continue;
        if (area > max_area) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        const float rw = std::max(rr.size.width, 1e-3f);
        const float rh = std::max(rr.size.height, 1e-3f);
        const float long_side = std::max(rw, rh);
        const float short_side = std::min(rw, rh);
        if (short_side < min_side) continue;

        const float aspect = long_side / std::max(short_side, 1e-3f);
        if (aspect < min_aspect || aspect > max_aspect) continue;

        if (rr.center.x < edge_x || rr.center.x > static_cast<float>(W) - edge_x ||
            rr.center.y < edge_y || rr.center.y > static_cast<float>(H) - edge_y) {
            continue;
        }

        // Optional hard ignore zone for top-left text/print area.
        // Only affects detections whose centers fall inside this corner region.
        if (ignore_tl_w > 0.0f && ignore_tl_h > 0.0f &&
            rr.center.x <= ignore_tl_w && rr.center.y <= ignore_tl_h) {
            continue;
        }

        if (use_ignore_rect &&
            rr.center.x >= ign_x0 && rr.center.x <= ign_x1 &&
            rr.center.y >= ign_y0 && rr.center.y <= ign_y1) {
            continue;
        }

        RBox box{rr.center.x, rr.center.y, rw, rh, rr.angle};
        if (box.w < box.h) {
            std::swap(box.w, box.h);
            box.angle_deg += 90.0f;
        }

        cv::Rect br = rr.boundingRect();
        br &= cv::Rect(0, 0, W, H);
        float score = 0.0f;
        if (br.area() > 0) {
            cv::Mat roi = gray(br);
            cv::Mat mask(br.height, br.width, CV_8UC1, cv::Scalar(0));
            std::vector<std::vector<cv::Point>> one{c};
            cv::drawContours(mask, one, 0, cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -br.tl());
            score = static_cast<float>(cv::mean(roi, mask)[0]);
        }
        if (score < opt.min_score) {
            continue;
        }
        box.score = score;

        rboxes.push_back(box);
    }
    return rotated_nms(rboxes, opt.nms_iou_thresh);
}

// ====================== 单图输出（优化版） ======================
void process_and_save_results(const std::string& input_bmp_path,
                              const torch::Tensor& anomaly_logits,
                              const std::string& output_dir,
                              const RBoxPostprocessOptions& opt) {
    if (anomaly_logits.sizes().size() != 4 || anomaly_logits.size(1) != 1) {
        std::cerr << "错误: anomaly_logits 必须是 (B,1,H,W) 形状\n";
        return;
    }

    cv::Mat orig = cv::imread(input_bmp_path);
    if (orig.empty()) {
        std::cerr << "无法读取图像: " << input_bmp_path << std::endl;
        return;
    }

    torch::Tensor score_map = postprocess_anomaly_map(anomaly_logits);

    torch::Tensor map_for_scale = score_map;
    if (map_for_scale.dim() == 4) {
        map_for_scale = map_for_scale.squeeze(0).squeeze(0);
    } else if (map_for_scale.dim() == 3) {
        map_for_scale = map_for_scale.squeeze(0);
    }

    const float sx = static_cast<float>(orig.cols) / static_cast<float>(map_for_scale.size(1));
    const float sy = static_cast<float>(orig.rows) / static_cast<float>(map_for_scale.size(0));

    // 保存热图
    torch::Tensor heat_t = score_map.squeeze(0).squeeze(0).cpu().to(torch::kFloat32).contiguous();
    cv::Mat heat_gray(heat_t.size(0), heat_t.size(1), CV_32F, heat_t.data_ptr<float>());
    cv::Mat heat_resized, heat_colored;
    cv::resize(heat_gray, heat_resized, orig.size());
    cv::Mat heat_norm;
    cv::normalize(heat_resized, heat_norm, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    heat_resized = heat_norm;
    cv::applyColorMap(heat_resized, heat_colored, cv::COLORMAP_JET);

    std::string base = fs::path(input_bmp_path).stem().string();
    std::string out_dir = output_dir.empty() ? fs::path(input_bmp_path).parent_path().string() : output_dir;

    cv::imwrite(out_dir + "/" + base + "_heat.png", heat_colored);

    // 提取并绘制 RBox
    const float area_scale = std::max(sx * sy, 1e-6f);
    const float scale_len = std::sqrt(area_scale);

    RBoxPostprocessOptions map_opt = opt;
    map_opt.min_area = opt.min_area / area_scale;
    map_opt.max_area = opt.max_area / area_scale;
    map_opt.min_side = opt.min_side / std::max(scale_len, 1e-6f);

    std::vector<RBox> rboxes = extract_rboxes_from_heatmap(score_map, map_opt);
    std::vector<RBox> scaled_rboxes;
    scaled_rboxes.reserve(rboxes.size());

    cv::Mat boxed = orig.clone();
    for (auto box : rboxes) {
        box.cx *= sx;
        box.cy *= sy;
        box.w *= sx;
        box.h *= sy;

        // Second-stage area gate in original image coordinates.
        // This makes --max_area/--min_area behavior robust to map/image scale conversion.
        const float box_area = std::max(box.w * box.h, 0.0f);
        if (box_area < opt.min_area || box_area > opt.max_area) {
            continue;
        }

        scaled_rboxes.push_back(box);

        cv::Point2f pts[4];
        cv::RotatedRect(cv::Point2f(box.cx, box.cy), cv::Size2f(box.w, box.h), box.angle_deg).points(pts);
        for (int i = 0; i < 4; ++i) {
            cv::line(boxed, pts[i], pts[(i + 1) % 4], cv::Scalar(0, 255, 0), 3);
        }
    }
    cv::imwrite(out_dir + "/result_" + base + ".bmp", boxed);

    // 保存 TXT
    std::ofstream ofs(out_dir + "/result_" + base + ".txt");
    ofs << "defect_count: " << scaled_rboxes.size() << "\n";
    for (const auto& b : scaled_rboxes) {
        ofs << std::fixed << std::setprecision(1)
            << "cx=" << b.cx << " cy=" << b.cy
            << " w=" << b.w << " h=" << b.h << " angle=" << b.angle_deg << "\n";
    }

    // 控制台输出（优化版）
    std::cout << "  ✓ 处理完成: " << base
              << " | 缺陷数量: " << rboxes.size() << "\n";
}

// ====================== 批量处理（赛题专用优化版） ======================
void process_folder_batch(const std::string& root_dir,
                          const std::vector<torch::Tensor>& anomaly_logits_list,
                          const RBoxPostprocessOptions& opt) {
    std::cout << "\n=== 开始批量处理验证集 ===\n";
    std::cout << "根目录: " << root_dir << "\n";
    std::cout << "待处理预测结果数量: " << anomaly_logits_list.size() << "\n\n";

    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        std::cerr << "错误：根目录不存在或不是文件夹: " << root_dir << "\n";
        return;
    }

    size_t idx = 0;

    const std::vector<int> val_folders = {2, 4, 6, 8, 10};

for (int f : val_folders) {
    fs::path folder = fs::path(root_dir) / std::to_string(f);
    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        std::cout << "跳过（文件夹不存在）: " << folder.string() << "\n";
        continue;
    }

        std::cout << "正在处理子文件夹 [" << f << "/10]: " << folder.string() << "\n";

        std::vector<std::string> bmps;
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".bmp") {
                bmps.push_back(entry.path().string());
            }
        }
        std::sort(bmps.begin(), bmps.end());

        std::cout << "  找到 " << bmps.size() << " 张 BMP 图片\n";

        for (const auto& bmp_path : bmps) {
            if (idx >= anomaly_logits_list.size()) {
                std::cout << "  已处理完所有预测结果，提前结束\n";
                return;
            }
            process_and_save_results(bmp_path, anomaly_logits_list[idx], root_dir, opt);
            ++idx;
        }
    }

    std::cout << "\n=== 批量处理完成！共处理 " << idx << " 张图片 ===\n";
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif