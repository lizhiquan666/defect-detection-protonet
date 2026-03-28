#include "dataset.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

static constexpr double kImagenetMean[3] = {0.485, 0.456, 0.406};
static constexpr double kImagenetStd[3] = {0.229, 0.224, 0.225};

static cv::Mat read_image_rgb_u8(const std::filesystem::path& path) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static cv::Mat safe_read_image_rgb_u8(const std::filesystem::path& path) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        return {};
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static cv::Mat ensure_rgb_3ch(const cv::Mat& rgb_or_gray) {
    if (rgb_or_gray.channels() == 3) {
        return rgb_or_gray;
    }
    if (rgb_or_gray.channels() == 1) {
        cv::Mat rgb;
        cv::cvtColor(rgb_or_gray, rgb, cv::COLOR_GRAY2RGB);
        return rgb;
    }
    throw std::runtime_error("Unsupported channel count");
}

static cv::Mat resize_square(const cv::Mat& img, int size, int interp) {
    cv::Mat out;
    cv::resize(img, out, cv::Size(size, size), 0, 0, interp);
    return out;
}

// torchvision.transforms.Resize(int): resize shorter side to `size` while preserving aspect ratio.
static cv::Mat resize_shorter_side(const cv::Mat& img, int size, int interp) {
    if (img.empty()) {
        return img;
    }
    const int h = img.rows;
    const int w = img.cols;
    if (h <= 0 || w <= 0) {
        return img;
    }
    if (std::min(h, w) == size) {
        return img.clone();
    }

    const float scale = (h < w) ? (static_cast<float>(size) / static_cast<float>(h))
                                : (static_cast<float>(size) / static_cast<float>(w));

    const int new_h = std::max(1, static_cast<int>(std::lround(h * scale)));
    const int new_w = std::max(1, static_cast<int>(std::lround(w * scale)));

    cv::Mat out;
    cv::resize(img, out, cv::Size(new_w, new_h), 0, 0, interp);
    return out;
}

static cv::Mat center_crop_square(const cv::Mat& img, int crop_size) {
    if (img.rows < crop_size || img.cols < crop_size) {
        throw std::runtime_error("center_crop_square: image smaller than crop");
    }
    int y0 = (img.rows - crop_size) / 2;
    int x0 = (img.cols - crop_size) / 2;
    cv::Rect roi(x0, y0, crop_size, crop_size);
    return img(roi).clone();
}

static torch::Tensor rgb_u8_to_normalized_tensor(const cv::Mat& rgb_u8) {
    cv::Mat rgb;
    if (rgb_u8.type() != CV_8UC3) {
        rgb_u8.convertTo(rgb, CV_8UC3);
    } else {
        rgb = rgb_u8;
    }

    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(rgb_f.data, {rgb_f.rows, rgb_f.cols, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    tensor = tensor.permute({2, 0, 1}).contiguous();

    // Normalize
    torch::Tensor mean = torch::tensor({kImagenetMean[0], kImagenetMean[1], kImagenetMean[2]}, tensor.options()).view({3, 1, 1});
    torch::Tensor std = torch::tensor({kImagenetStd[0], kImagenetStd[1], kImagenetStd[2]}, tensor.options()).view({3, 1, 1});
    tensor = (tensor - mean) / std;

    return tensor.clone();
}

static torch::Tensor rgb_u8_to_tensor_normalized(const cv::Mat& rgb_u8, const std::array<double, 3>& mean, const std::array<double, 3>& stdev) {
    cv::Mat rgb;
    if (rgb_u8.type() != CV_8UC3) {
        rgb = ensure_rgb_3ch(rgb_u8);
    } else {
        rgb = rgb_u8;
    }

    cv::Mat rgb_f;
    rgb.convertTo(rgb_f, CV_32FC3, 1.0 / 255.0);

    auto tensor = torch::from_blob(rgb_f.data, {rgb_f.rows, rgb_f.cols, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    tensor = tensor.permute({2, 0, 1}).contiguous();

    torch::Tensor mean_t = torch::tensor({mean[0], mean[1], mean[2]}, tensor.options()).view({3, 1, 1});
    torch::Tensor std_t = torch::tensor({stdev[0], stdev[1], stdev[2]}, tensor.options()).view({3, 1, 1});
    tensor = (tensor - mean_t) / std_t;

    return tensor.clone();
}

static torch::Tensor mask_u8_to_tensor01(const cv::Mat& mask_u8) {
    cv::Mat m;
    if (mask_u8.channels() == 3) {
        // OpenCV imread returns BGR; treat 3ch mask as BGR for consistency.
        cv::cvtColor(mask_u8, m, cv::COLOR_BGR2GRAY);
    } else {
        m = mask_u8;
    }
    if (m.type() != CV_8UC1) {
        m.convertTo(m, CV_8UC1);
    }

    cv::Mat mf;
    m.convertTo(mf, CV_32FC1, 1.0 / 255.0);
    // Binarize like typical gt masks
    cv::threshold(mf, mf, 0.5, 1.0, cv::THRESH_BINARY);

    auto t = torch::from_blob(mf.data, {mf.rows, mf.cols}, torch::TensorOptions().dtype(torch::kFloat32));
    t = t.unsqueeze(0).contiguous();
    return t.clone();
}

c10::optional<RBox> mask_to_rbox(const cv::Mat& mask_u8) {
    cv::Mat m;
    if (mask_u8.empty()) {
        return c10::nullopt;
    }
    if (mask_u8.channels() == 3) {
        // Treat 3ch input as BGR (OpenCV default).
        cv::cvtColor(mask_u8, m, cv::COLOR_BGR2GRAY);
    } else {
        m = mask_u8;
    }
    if (m.type() != CV_8UC1) {
        m.convertTo(m, CV_8UC1);
    }

    cv::Mat bin;
    cv::threshold(m, bin, 0, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return c10::nullopt;
    }

    size_t best_i = 0;
    double best_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > best_area) {
            best_area = a;
            best_i = i;
        }
    }

    if (best_area <= 0.0) {
        return c10::nullopt;
    }

    cv::RotatedRect rr = cv::minAreaRect(contours[best_i]);

    // Normalize OpenCV's rotated-rect representation to a stable convention:
    // - keep w >= h
    // - angle in [-90, 90)
    // This prevents downstream consumers from seeing discontinuous angle flips.
    float w = rr.size.width;
    float h = rr.size.height;
    float angle = rr.angle;
    if (w < h) {
        std::swap(w, h);
        angle += 90.0F;
    }
    if (w <= 0.0F || h <= 0.0F) {
        return c10::nullopt;
    }
    // Ensure angle in [-90, 90)
    while (angle >= 90.0F) angle -= 180.0F;
    while (angle < -90.0F) angle += 180.0F;

    RBox r;
    r.cx = rr.center.x;
    r.cy = rr.center.y;
    r.w = w;
    r.h = h;
    r.angle_deg = angle;
    return r;
}

torch::Tensor rbox_to_tensor(const RBox& rbox, const torch::TensorOptions& options) {
    return torch::tensor({rbox.cx, rbox.cy, rbox.w, rbox.h, rbox.angle_deg}, options);
}

// =====================
// MVTEC
// =====================

const std::vector<std::string> MVTEC::CLASS_NAMES = {
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"};

MVTEC::MVTEC(std::filesystem::path root,
             std::optional<std::string> class_name,
             bool train,
             DatasetConfig cfg)
    : root_(std::move(root)), class_name_(std::move(class_name)), train_(train), cfg_(cfg) {

    class_to_idx_ = {
        {"bottle", 0}, {"cable", 1}, {"capsule", 2}, {"carpet", 3}, {"grid", 4},
        {"hazelnut", 5}, {"leather", 6}, {"metal_nut", 7}, {"pill", 8}, {"screw", 9},
        {"tile", 10}, {"toothbrush", 11}, {"transistor", 12}, {"wood", 13}, {"zipper", 14},
    };
    for (const auto& kv : class_to_idx_) {
        idx_to_class_[kv.second] = kv.first;
    }

    if (!class_name_.has_value()) {
        load_all_data_();
    } else {
        auto [imgs, labels, masks, types] = load_data_(class_name_.value());
        image_paths_ = std::move(imgs);
        labels_ = std::move(labels);
        mask_paths_ = std::move(masks);
        img_types_ = std::move(types);
    }
}

size_t MVTEC::size() const { return image_paths_.size(); }

static std::vector<std::filesystem::path> list_files_with_ext(const std::filesystem::path& dir, const std::string& ext) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::exists(dir)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto p = entry.path();
        if (p.extension().string() == ext) {
            out.push_back(p);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

static bool is_image_file_ext(const std::filesystem::path& p) {
    std::string ext = to_lower_copy(p.extension().string());
    return ext == ".bmp" || ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

static bool is_mask_filename(const std::filesystem::path& p) {
    std::string stem = to_lower_copy(p.stem().string());
    if (stem.size() >= 2 && stem.substr(stem.size() - 2) == "_t") {
        return true;
    }
    if (stem.size() >= 5 && stem.substr(stem.size() - 5) == "_mask") {
        return true;
    }
    return false;
}

static std::vector<std::filesystem::path> list_image_files_excluding_masks(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> out;
    if (!std::filesystem::exists(dir)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto p = entry.path();
        if (!is_image_file_ext(p)) {
            continue;
        }
        if (is_mask_filename(p)) {
            continue;
        }
        out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    return out;
}

static std::optional<std::filesystem::path> find_mask_for_image_same_dir(const std::filesystem::path& img_path) {
    const auto parent = img_path.parent_path();
    const std::string stem = img_path.stem().string();

    std::vector<std::filesystem::path> candidates = {
        parent / (stem + "_t.bmp"),
        parent / (stem + "_t.png"),
        parent / (stem + "_mask.png"),
        parent / (stem + "_mask.bmp")};

    for (const auto& p : candidates) {
        if (std::filesystem::exists(p) && std::filesystem::is_regular_file(p)) {
            return p;
        }
    }
    return std::nullopt;
}

static std::optional<std::pair<std::filesystem::path, std::filesystem::path>>
find_ok_ng_dirs_under_class(const std::filesystem::path& class_dir) {
    std::optional<std::filesystem::path> ok_dir;
    std::optional<std::filesystem::path> ng_dir;

    if (!std::filesystem::exists(class_dir)) {
        return std::nullopt;
    }

    for (const auto& entry : std::filesystem::directory_iterator(class_dir)) {
        if (!entry.is_directory()) {
            continue;
        }

        const auto dpath = entry.path();
        std::string name = to_lower_copy(dpath.filename().string());

        if (!ok_dir.has_value()) {
            if ((name.find("ok") != std::string::npos) && (name.find("ng") == std::string::npos)) {
                ok_dir = dpath;
            }
        }
        if (!ng_dir.has_value()) {
            if (name.find("ng") != std::string::npos) {
                ng_dir = dpath;
            }
        }
    }

    if (ok_dir.has_value() && ng_dir.has_value()) {
        return std::make_pair(ok_dir.value(), ng_dir.value());
    }
    return std::nullopt;
}

static std::vector<std::string> list_subdirs(const std::filesystem::path& dir) {
    std::vector<std::string> out;
    if (!std::filesystem::exists(dir)) {
        return out;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_directory()) {
            out.push_back(entry.path().filename().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::tuple<std::vector<std::filesystem::path>, std::vector<int64_t>, std::vector<std::optional<std::filesystem::path>>, std::vector<std::string>>
MVTEC::load_data_(const std::string& class_name) const {
    const std::string phase = train_ ? "train" : "test";

    std::vector<std::filesystem::path> image_paths;
    std::vector<int64_t> labels;
    std::vector<std::optional<std::filesystem::path>> mask_paths;
    std::vector<std::string> types;

    const auto image_dir = root_ / class_name / phase;
    const auto mask_dir = root_ / class_name / "ground_truth";

    auto img_types = list_subdirs(image_dir);
    for (const auto& img_type : img_types) {
        const auto img_type_dir = image_dir / img_type;
        auto img_files = list_files_with_ext(img_type_dir, ".png");
        for (const auto& f : img_files) {
            image_paths.push_back(f);
        }

        if (img_type == "good") {
            labels.insert(labels.end(), img_files.size(), 0);
            mask_paths.insert(mask_paths.end(), img_files.size(), std::nullopt);
            types.insert(types.end(), img_files.size(), "good");
        } else {
            labels.insert(labels.end(), img_files.size(), 1);
            const auto gt_type_dir = mask_dir / img_type;
            for (const auto& f : img_files) {
                const std::string stem = f.stem().string();
                mask_paths.push_back(gt_type_dir / (stem + "_mask.png"));
                types.push_back(img_type);
            }
        }
    }

    return {image_paths, labels, mask_paths, types};
}

void MVTEC::load_all_data_() {
    image_paths_.clear();
    labels_.clear();
    mask_paths_.clear();
    img_types_.clear();

    for (const auto& cn : CLASS_NAMES) {
        auto [imgs, labels, masks, types] = load_data_(cn);
        image_paths_.insert(image_paths_.end(), imgs.begin(), imgs.end());
        labels_.insert(labels_.end(), labels.begin(), labels.end());
        mask_paths_.insert(mask_paths_.end(), masks.begin(), masks.end());
        img_types_.insert(img_types_.end(), types.begin(), types.end());
    }
}

static std::string basename_no_ext(const std::filesystem::path& p) {
    return p.stem().string();
}

Sample MVTEC::load_sample_(const std::filesystem::path& image_path,
                           int64_t label,
                           const std::optional<std::filesystem::path>& mask_path,
                           const std::string& img_type,
                           const std::optional<std::string>& class_name_override) const {

    std::string class_name = class_name_override.has_value() ? class_name_override.value() : (class_name_.has_value() ? class_name_.value() : "");
    if (class_name.empty()) {
        // root/class/phase/type/file.png
        // take -4 component
        auto parts = image_path;
        class_name = image_path.parent_path().parent_path().parent_path().filename().string();
    }

    cv::Mat rgb = read_image_rgb_u8(image_path);

    // Handle grayscale classes
    rgb = resize_shorter_side(rgb, cfg_.img_size, cv::INTER_LANCZOS4);
    rgb = center_crop_square(rgb, cfg_.crp_size);
    torch::Tensor image_t = rgb_u8_to_normalized_tensor(rgb);

    torch::Tensor mask_t;
    c10::optional<RBox> rbox;
    if (label == 0 || !mask_path.has_value()) {
        mask_t = torch::zeros({1, cfg_.msk_crp_size, cfg_.msk_crp_size}, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        cv::Mat m = cv::imread(mask_path->string(), cv::IMREAD_GRAYSCALE);
        if (m.empty()) {
            throw std::runtime_error("Failed to read mask: " + mask_path->string());
        }
        m = resize_shorter_side(m, cfg_.msk_size, cv::INTER_NEAREST);
        m = center_crop_square(m, cfg_.msk_crp_size);
        rbox = mask_to_rbox(m);
        mask_t = mask_u8_to_tensor01(m);
    }

    int64_t out_label = label;
    if (train_) {
        auto it = class_to_idx_.find(class_name);
        if (it == class_to_idx_.end()) {
            throw std::runtime_error("Unknown class_name: " + class_name);
        }
        out_label = it->second;
    }

    Sample s;
    s.image = image_t;
    s.label = out_label;
    s.mask = mask_t;
    s.basename = basename_no_ext(image_path);
    s.img_type = img_type;
    s.rbox = rbox;
    return s;
}

Sample MVTEC::get(size_t idx) const {
    const auto& image_path = image_paths_.at(idx);
    const int64_t label = labels_.at(idx);
    const auto& mask_path = mask_paths_.at(idx);
    const auto& img_type = img_types_.at(idx);

    std::optional<std::string> class_name_override;
    if (!class_name_.has_value()) {
        class_name_override = image_path.parent_path().parent_path().parent_path().filename().string();
    }

    return load_sample_(image_path, label, mask_path, img_type, class_name_override);
}

void MVTEC::update_class_to_idx(const std::unordered_map<std::string, int64_t>& class_to_idx) {
    for (const auto& kv : class_to_idx_) {
        auto it = class_to_idx.find(kv.first);
        if (it != class_to_idx.end()) {
            class_to_idx_[kv.first] = it->second;
        }
    }
    idx_to_class_.clear();
    for (const auto& kv : class_to_idx_) {
        idx_to_class_[kv.second] = kv.first;
    }
}

// =====================
// TRAINMVTEC
// =====================

static uint32_t seed_from_rd() {
    std::random_device rd;
    return (static_cast<uint32_t>(rd()) << 16) ^ static_cast<uint32_t>(rd());
}

static cv::Mat generate_perlin_like_mask_u8(int h,
                                            int w,
                                            int grid,
                                            float thresh,
                                            std::mt19937& rng) {
    const int gh = std::max(2, grid);
    const int gw = std::max(2, grid);
    std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

    cv::Mat coarse(gh, gw, CV_32F);
    for (int y = 0; y < gh; ++y) {
        float* row = coarse.ptr<float>(y);
        for (int x = 0; x < gw; ++x) {
            row[x] = uni01(rng);
        }
    }

    cv::Mat noise;
    cv::resize(coarse, noise, cv::Size(w, h), 0.0, 0.0, cv::INTER_CUBIC);
    cv::GaussianBlur(noise, noise, cv::Size(0, 0), 2.0, 2.0);

    cv::Mat mask;
    cv::threshold(noise, mask, static_cast<double>(thresh), 1.0, cv::THRESH_BINARY);
    mask.convertTo(mask, CV_8UC1, 255.0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
    return mask;
}

static void maybe_apply_perlin_augmentation(torch::Tensor& img_t,
                                            torch::Tensor& mask_t,
                                            int64_t& label,
                                            const DatasetConfig& cfg,
                                            std::mt19937& rng) {
    if (!cfg.perlin_enable || cfg.perlin_prob <= 0.0f) {
        return;
    }
    if (img_t.dim() != 3 || img_t.size(0) != 3 || mask_t.dim() != 3 || mask_t.size(0) != 1) {
        return;
    }

    std::uniform_real_distribution<float> prob01(0.0f, 1.0f);
    if (prob01(rng) >= cfg.perlin_prob) {
        return;
    }

    const int h = static_cast<int>(img_t.size(1));
    const int w = static_cast<int>(img_t.size(2));
    if (h <= 1 || w <= 1) {
        return;
    }

    cv::Mat perlin_mask_u8 = generate_perlin_like_mask_u8(
        h,
        w,
        std::max(2, cfg.perlin_grid),
        std::clamp(cfg.perlin_thresh, 0.05f, 0.95f),
        rng);

    const float area_ratio = static_cast<float>(cv::countNonZero(perlin_mask_u8)) / static_cast<float>(h * w);
    if (area_ratio < 0.01f || area_ratio > 0.45f) {
        return;
    }

    cv::Mat mf;
    perlin_mask_u8.convertTo(mf, CV_32FC1, 1.0 / 255.0);
    auto perlin_mask = torch::from_blob(mf.data, {h, w}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    perlin_mask = perlin_mask.unsqueeze(0).to(img_t.device());

    auto m3 = perlin_mask.repeat({3, 1, 1});
    auto noise = torch::randn_like(img_t) * std::max(0.0f, cfg.perlin_noise_std);
    // Blend random texture into selected regions while keeping the rest unchanged.
    img_t = img_t + m3 * noise;
    img_t = torch::clamp(img_t, -3.5, 3.5);

    mask_t = torch::clamp(mask_t + perlin_mask, 0.0, 1.0);
    label = 1;
}

static torch::Tensor mask_u8_to_tensor_chw01(const cv::Mat& mask_u8,
                                             const torch::Device& device) {
    cv::Mat f;
    mask_u8.convertTo(f, CV_32FC1, 1.0 / 255.0);
    auto t = torch::from_blob(f.data, {f.rows, f.cols}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return t.unsqueeze(0).to(device);
}

static cv::Mat make_scratch_mask_u8(int h, int w, std::mt19937& rng) {
    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
    std::uniform_int_distribution<int> n_lines(1, 2);
    std::uniform_int_distribution<int> px(0, std::max(0, w - 1));
    std::uniform_int_distribution<int> py(0, std::max(0, h - 1));
    std::uniform_real_distribution<float> angle(-35.0f, 35.0f);
    std::uniform_real_distribution<float> len_ratio(0.10f, 0.35f);
    std::uniform_int_distribution<int> thick(1, 3);

    for (int i = 0; i < n_lines(rng); ++i) {
        const float a = angle(rng) * static_cast<float>(CV_PI / 180.0);
        const float L = len_ratio(rng) * static_cast<float>(w);
        const int x0 = px(rng);
        const int y0 = py(rng);
        const int x1 = static_cast<int>(std::round(static_cast<float>(x0) + L * std::cos(a)));
        const int y1 = static_cast<int>(std::round(static_cast<float>(y0) + L * std::sin(a)));
        cv::line(mask, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255), thick(rng), cv::LINE_AA);
    }
    return mask;
}

static cv::Mat make_missing_print_mask_u8(int h, int w, std::mt19937& rng) {
    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
    std::uniform_int_distribution<int> n_blocks(1, 2);
    std::uniform_int_distribution<int> orient(0, 9); // horizontal-biased
    std::uniform_real_distribution<float> rw(0.03f, 0.12f);
    std::uniform_real_distribution<float> rh(0.01f, 0.05f);

    for (int i = 0; i < n_blocks(rng); ++i) {
        const bool horizontal = orient(rng) < 7;
        int bw = std::max(2, static_cast<int>(std::round(rw(rng) * w)));
        int bh = std::max(2, static_cast<int>(std::round(rh(rng) * h)));
        if (horizontal) bh = std::max(2, bh / 2);
        else bw = std::max(2, bw / 2);

        std::uniform_int_distribution<int> px(0, std::max(0, w - bw));
        std::uniform_int_distribution<int> py(0, std::max(0, h - bh));
        cv::Rect r(px(rng), py(rng), bw, bh);
        cv::rectangle(mask, r, cv::Scalar(255), cv::FILLED);
    }
    return mask;
}

static cv::Mat make_blob_mask_u8(int h, int w, std::mt19937& rng) {
    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));
    std::uniform_int_distribution<int> n_shapes(1, 3);
    std::uniform_int_distribution<int> px(0, std::max(0, w - 1));
    std::uniform_int_distribution<int> py(0, std::max(0, h - 1));
    std::uniform_real_distribution<float> rs(0.02f, 0.10f);
    std::uniform_int_distribution<int> shape_pick(0, 1);

    for (int i = 0; i < n_shapes(rng); ++i) {
        const int cx = px(rng);
        const int cy = py(rng);
        const int r = std::max(2, static_cast<int>(std::round(rs(rng) * std::min(h, w))));
        if (shape_pick(rng) == 0) {
            cv::ellipse(mask, cv::Point(cx, cy), cv::Size(r, std::max(2, r / 2)),
                        static_cast<double>(px(rng) % 180), 0, 360, cv::Scalar(255), cv::FILLED);
        } else {
            std::vector<cv::Point> tri;
            tri.emplace_back(cx, cy - r);
            tri.emplace_back(cx - r, cy + r);
            tri.emplace_back(cx + r, cy + r);
            cv::fillConvexPoly(mask, tri, cv::Scalar(255));
        }
    }

    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 1.0, 1.0);
    cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
    return mask;
}

static bool maybe_apply_type_directed_augmentation(torch::Tensor& img_t,
                                                    torch::Tensor& mask_t,
                                                    int64_t& label,
                                                    const DatasetConfig& cfg,
                                                    std::mt19937& rng) {
    if (!cfg.typed_aug_enable) {
        return false;
    }
    if (img_t.dim() != 3 || img_t.size(0) != 3 || mask_t.dim() != 3 || mask_t.size(0) != 1) {
        return false;
    }

    const float p_s = std::max(0.0f, cfg.scratch_prob);
    const float p_m = std::max(0.0f, cfg.missing_print_prob);
    const float p_b = std::max(0.0f, cfg.blob_prob);
    const float p_sum = p_s + p_m + p_b;
    if (p_sum <= 0.0f) {
        return false;
    }

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const float pick = u01(rng) * p_sum;

    enum class AugType { Scratch, Missing, Blob };
    AugType aug = AugType::Scratch;
    if (pick < p_s) {
        aug = AugType::Scratch;
    } else if (pick < p_s + p_m) {
        aug = AugType::Missing;
    } else {
        aug = AugType::Blob;
    }

    const int h = static_cast<int>(img_t.size(1));
    const int w = static_cast<int>(img_t.size(2));
    cv::Mat mask_u8;
    if (aug == AugType::Scratch) {
        mask_u8 = make_scratch_mask_u8(h, w, rng);
    } else if (aug == AugType::Missing) {
        mask_u8 = make_missing_print_mask_u8(h, w, rng);
    } else {
        mask_u8 = make_blob_mask_u8(h, w, rng);
    }

    const float area_ratio = static_cast<float>(cv::countNonZero(mask_u8)) / static_cast<float>(h * w);
    if (area_ratio < 0.002f || area_ratio > 0.18f) {
        return false;
    }

    auto region = mask_u8_to_tensor_chw01(mask_u8, img_t.device());
    auto region3 = region.repeat({3, 1, 1});

    if (aug == AugType::Scratch) {
        const float mag = std::max(0.05f, cfg.scratch_strength);
        auto noise = torch::randn_like(img_t) * mag;
        img_t = torch::clamp(img_t + region3 * noise, -3.5, 3.5);
    } else if (aug == AugType::Missing) {
        const float alpha = std::clamp(cfg.missing_print_alpha, 0.0f, 0.98f);
        img_t = torch::clamp(img_t * (1.0f - alpha * region3), -3.5, 3.5);
    } else {
        const float stdv = std::max(0.05f, cfg.blob_noise_std);
        auto noise = torch::randn_like(img_t) * stdv;
        std::uniform_real_distribution<float> bias(-0.15f, 0.15f);
        auto bias_t = torch::full_like(img_t, bias(rng));
        img_t = torch::clamp(img_t + region3 * (noise + bias_t), -3.5, 3.5);
    }

    mask_t = torch::clamp(mask_t + region, 0.0, 1.0);
    label = 1;
    return true;
}

TRAINMVTEC::TRAINMVTEC(std::filesystem::path data_path,
                       std::string class_name,
                       bool train,
                       DatasetConfig cfg,
                       int num_anomalies)
    : dataset_path_(std::move(data_path)),
      class_name_(std::move(class_name)),
      train_(train),
      cfg_(cfg),
    num_load_anomalies_(num_anomalies),
    rng_(seed_from_rd()) {

    PathsPack pack = load_dataset_folder_();
    n_imgs_ = std::move(pack.n_imgs);
    n_labels_ = std::move(pack.n_labels);
    n_masks_ = std::move(pack.n_masks);
    a_imgs_ = std::move(pack.a_imgs);
    a_labels_ = std::move(pack.a_labels);
    a_masks_ = std::move(pack.a_masks);
}

size_t TRAINMVTEC::size() const {
    return n_imgs_.size() + (a_imgs_.empty() ? 0 : a_imgs_.size());
}

size_t TRAINMVTEC::normal_size() const {
    return n_imgs_.size();
}

size_t TRAINMVTEC::anomaly_size() const {
    return a_imgs_.size();
}

std::tuple<torch::Tensor, int64_t, torch::Tensor> TRAINMVTEC::load_image_and_mask_(
    const std::filesystem::path& img_path,
    int64_t label,
    const std::optional<std::filesystem::path>& mask_path) const {

    cv::Mat rgb = read_image_rgb_u8(img_path);
    rgb = resize_shorter_side(rgb, cfg_.img_size, cv::INTER_LANCZOS4);
    rgb = center_crop_square(rgb, cfg_.crp_size);

    torch::Tensor img_t = rgb_u8_to_normalized_tensor(rgb);

    torch::Tensor mask_t;
    if (label == 0 || !mask_path.has_value()) {
        mask_t = torch::zeros({1, cfg_.msk_crp_size, cfg_.msk_crp_size}, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        cv::Mat m = cv::imread(mask_path->string(), cv::IMREAD_GRAYSCALE);
        if (m.empty()) {
            throw std::runtime_error("Failed to read mask: " + mask_path->string());
        }
        m = resize_shorter_side(m, cfg_.msk_size, cv::INTER_NEAREST);
        m = center_crop_square(m, cfg_.msk_crp_size);
        mask_t = mask_u8_to_tensor01(m);
    }

    return {img_t, label, mask_t};
}

Sample TRAINMVTEC::get(size_t idx) {
    const size_t total = size();
    if (idx >= total) {
        throw std::out_of_range(
            "TRAINMVTEC::get: idx out of range (idx=" + std::to_string(idx) +
            ", size=" + std::to_string(total) + ")");
    }

    if (idx < n_imgs_.size()) {
        auto [img_t, label, mask_t] = load_image_and_mask_(n_imgs_.at(idx), 0, n_masks_.at(idx));
        if (train_) {
            const bool typed_aug_applied = maybe_apply_type_directed_augmentation(img_t, mask_t, label, cfg_, rng_);
            if (!typed_aug_applied && cfg_.perlin_enable) {
                maybe_apply_perlin_augmentation(img_t, mask_t, label, cfg_, rng_);
            }
        }
        Sample s;
        s.image = img_t;
        s.label = label;
        s.mask = mask_t;
        s.basename = basename_no_ext(n_imgs_.at(idx));
        s.img_type = "good";
        s.rbox = c10::nullopt;
        return s;
    }

    if (a_imgs_.empty()) {
        throw std::runtime_error("No anomaly images available");
    }

    const size_t a_idx = (idx - n_imgs_.size()) % a_imgs_.size();
    auto img_path = a_imgs_.at(a_idx);
    auto mask_path = a_masks_.at(a_idx);

    auto [img_t, label, mask_t] = load_image_and_mask_(img_path, 1, mask_path);

    Sample s;
    s.image = img_t;
    s.label = 1;
    s.mask = mask_t;
    s.basename = basename_no_ext(img_path);
    s.img_type = "real";
    // rbox from mask
    {
        torch::Tensor m_u8 = (mask_t.squeeze(0) * 255.0F).to(torch::kU8).contiguous();
        cv::Mat mcv(m_u8.size(0), m_u8.size(1), CV_8UC1, m_u8.data_ptr());
        s.rbox = mask_to_rbox(mcv);
    }
    return s;
}

TRAINMVTEC::PathsPack TRAINMVTEC::load_dataset_folder_() const {
    PathsPack out;

    std::filesystem::path class_dir = dataset_path_ / class_name_;

    // Support both inputs:
    // 1) dataset_path is dataset root and class_name is class folder (root/class)
    // 2) dataset_path is already class folder (e.g. .../1), with --class 1
    std::error_code ec;
    if ((!std::filesystem::exists(class_dir, ec) || !std::filesystem::is_directory(class_dir, ec)) &&
        std::filesystem::exists(dataset_path_, ec) && std::filesystem::is_directory(dataset_path_, ec)) {
        const bool dataset_has_ok_ng = find_ok_ng_dirs_under_class(dataset_path_).has_value();
        const bool dataset_has_mvtec_layout =
            std::filesystem::exists(dataset_path_ / "train", ec) ||
            std::filesystem::exists(dataset_path_ / "test", ec);

        if (dataset_has_ok_ng || dataset_has_mvtec_layout) {
            class_dir = dataset_path_;
        }
    }

    // 1) 优先兼容你们当前结构：class_dir 下存在 *ok* 与 *ng* 两个目录，且 ng 掩膜与图像同目录（*_t.bmp）。
    if (auto ok_ng_dirs = find_ok_ng_dirs_under_class(class_dir); ok_ng_dirs.has_value()) {
        const auto& ok_dir = ok_ng_dirs->first;
        const auto& ng_dir = ok_ng_dirs->second;

        // normal
        auto n_files = list_image_files_excluding_masks(ok_dir);
        for (const auto& f : n_files) {
            out.n_imgs.push_back(f);
            out.n_labels.push_back(0);
            out.n_masks.push_back(std::nullopt);
        }

        // anomaly + mask
        auto a_files = list_image_files_excluding_masks(ng_dir);
        std::mt19937 rng_local(123);
        std::shuffle(a_files.begin(), a_files.end(), rng_local);

        size_t take_n = a_files.size();
        if (num_load_anomalies_ > 0) {
            take_n = std::min<size_t>(take_n, static_cast<size_t>(num_load_anomalies_));
        }

        for (size_t i = 0; i < take_n; ++i) {
            const auto& img = a_files[i];
            auto mask = find_mask_for_image_same_dir(img);
            if (!mask.has_value()) {
                throw std::runtime_error("Missing mask for anomaly image: " + img.string() + " (expected *_t.bmp in same folder)");
            }

            out.a_imgs.push_back(img);
            out.a_labels.push_back(1);
            out.a_masks.push_back(mask.value());
        }

        return out;
    }

    // 2) 回退到原始 MVTEC 目录结构。
    {
        auto img_dir = class_dir / "test";
        auto gt_dir = class_dir / "ground_truth";

        auto ano_types = list_subdirs(img_dir);
        const int num_ano_types = static_cast<int>(ano_types.size()) - 1;
        const int anomaly_nums_per_type = (num_ano_types > 0) ? (num_load_anomalies_ / num_ano_types) : 0;
        const int extra_nums = (num_ano_types > 0) ? (num_load_anomalies_ % num_ano_types) : 0;

        std::vector<std::filesystem::path> extra_ano_imgs;
        std::vector<std::filesystem::path> extra_ano_gts;

        std::mt19937 rng_local(123);

        for (const auto& t : ano_types) {
            auto t_dir = img_dir / t;
            auto files = list_files_with_ext(t_dir, ".png");
            if (t == "good") {
                continue;
            }
            std::shuffle(files.begin(), files.end(), rng_local);

            const int take = std::min(anomaly_nums_per_type, static_cast<int>(files.size()));
            for (int i = 0; i < take; ++i) {
                out.a_imgs.push_back(files[i]);
                out.a_labels.push_back(1);
                auto stem = files[i].stem().string();
                out.a_masks.push_back(gt_dir / t / (stem + "_mask.png"));
            }
            for (size_t i = static_cast<size_t>(take); i < files.size(); ++i) {
                extra_ano_imgs.push_back(files[i]);
                auto stem = files[i].stem().string();
                extra_ano_gts.push_back(gt_dir / t / (stem + "_mask.png"));
            }
        }

        if (extra_nums > 0 && extra_ano_imgs.size() == extra_ano_gts.size() && !extra_ano_imgs.empty()) {
            std::vector<size_t> inds(extra_ano_imgs.size());
            std::iota(inds.begin(), inds.end(), 0);
            std::shuffle(inds.begin(), inds.end(), rng_local);
            const int take = std::min<int>(extra_nums, static_cast<int>(inds.size()));
            for (int i = 0; i < take; ++i) {
                size_t ind = inds[i];
                out.a_imgs.push_back(extra_ano_imgs[ind]);
                out.a_labels.push_back(1);
                out.a_masks.push_back(extra_ano_gts[ind]);
            }
        }

        auto n_dir = class_dir / "train" / "good";
        auto n_files = list_files_with_ext(n_dir, ".png");
        for (const auto& f : n_files) {
            out.n_imgs.push_back(f);
            out.n_labels.push_back(0);
            out.n_masks.push_back(std::nullopt);
        }
    }

    if (out.n_imgs.empty() && out.a_imgs.empty()) {
        throw std::runtime_error(
            "No training samples found. Checked class_dir: " + class_dir.string() +
            "\nHint: pass --train_root as dataset root (containing class subfolders) "
            "or as class folder directly (containing *ok*/*ng* or train/test)."
        );
    }

    return out;
}

// =====================
// BTAD
// =====================

const std::vector<std::string> BTAD::CLASS_NAMES = {"01", "02", "03"};

BTAD::BTAD(std::filesystem::path root,
           std::optional<std::string> class_name,
           bool train,
           DatasetConfig cfg)
    : root_(std::move(root)), class_name_(std::move(class_name)), train_(train), cfg_(cfg), rng_(seed_from_rd()) {

    class_to_idx_ = {{"01", 0}, {"02", 1}, {"03", 2}};
    for (const auto& kv : class_to_idx_) {
        idx_to_class_[kv.second] = kv.first;
    }

    if (!class_name_.has_value()) {
        load_all_data_();
    } else {
        auto [imgs, labels, masks, types] = load_data_(class_name_.value());
        image_paths_ = std::move(imgs);
        labels_ = std::move(labels);
        mask_paths_ = std::move(masks);
        img_types_ = std::move(types);
    }
}

size_t BTAD::size() const { return image_paths_.size(); }

std::tuple<std::vector<std::filesystem::path>, std::vector<int64_t>, std::vector<std::optional<std::filesystem::path>>, std::vector<std::string>>
BTAD::load_data_(const std::string& class_name) const {
    const std::string phase = train_ ? "train" : "test";

    std::vector<std::filesystem::path> image_paths;
    std::vector<int64_t> labels;
    std::vector<std::optional<std::filesystem::path>> mask_paths;
    std::vector<std::string> types;

    auto img_dir = root_ / class_name / phase;
    auto gt_dir = root_ / class_name / "ground_truth";

    auto img_types = list_subdirs(img_dir);
    for (const auto& t : img_types) {
        auto t_dir = img_dir / t;
        // btad has bmp or png
        std::vector<std::filesystem::path> files;
        if (std::filesystem::exists(t_dir)) {
            for (const auto& e : std::filesystem::directory_iterator(t_dir)) {
                if (!e.is_regular_file()) {
                    continue;
                }
                auto ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".bmp" || ext == ".png") {
                    files.push_back(e.path());
                }
            }
        }
        std::sort(files.begin(), files.end());

        image_paths.insert(image_paths.end(), files.begin(), files.end());

        if (t == "ok") {
            labels.insert(labels.end(), files.size(), 0);
            mask_paths.insert(mask_paths.end(), files.size(), std::nullopt);
            types.insert(types.end(), files.size(), "ok");
        } else {
            labels.insert(labels.end(), files.size(), 1);
            auto gt_type_dir = gt_dir / t;
            for (const auto& f : files) {
                auto stem = f.stem().string();
                if (class_name == "03") {
                    mask_paths.push_back(gt_type_dir / (stem + ".bmp"));
                } else {
                    mask_paths.push_back(gt_type_dir / (stem + ".png"));
                }
                types.push_back(t);
            }
        }
    }

    return {image_paths, labels, mask_paths, types};
}

void BTAD::load_all_data_() {
    image_paths_.clear();
    labels_.clear();
    mask_paths_.clear();
    img_types_.clear();

    for (const auto& cn : CLASS_NAMES) {
        auto [imgs, labels, masks, types] = load_data_(cn);
        image_paths_.insert(image_paths_.end(), imgs.begin(), imgs.end());
        labels_.insert(labels_.end(), labels.begin(), labels.end());
        mask_paths_.insert(mask_paths_.end(), masks.begin(), masks.end());
        img_types_.insert(img_types_.end(), types.begin(), types.end());
    }
}

Sample BTAD::get(size_t idx) const {
    const auto& image_path = image_paths_.at(idx);
    const int64_t label = labels_.at(idx);
    const auto& mask_path = mask_paths_.at(idx);
    const auto& img_type = img_types_.at(idx);

    cv::Mat rgb = read_image_rgb_u8(image_path);
    rgb = resize_shorter_side(rgb, cfg_.img_size, cv::INTER_LANCZOS4);
    rgb = center_crop_square(rgb, cfg_.crp_size);

    if (train_) {
        std::uniform_real_distribution<float> u01(0.0F, 1.0F);
        if (u01(rng_) < 0.5F) {
            cv::flip(rgb, rgb, 1);
        }
    }

    const std::array<double, 3> mean = {0.5, 0.5, 0.5};
    const std::array<double, 3> stdev = {0.5, 0.5, 0.5};
    torch::Tensor img_t = rgb_u8_to_tensor_normalized(rgb, mean, stdev);

    torch::Tensor mask_t;
    c10::optional<RBox> rbox;
    if (label == 0 || !mask_path.has_value()) {
        mask_t = torch::zeros({1, cfg_.msk_crp_size, cfg_.msk_crp_size}, torch::TensorOptions().dtype(torch::kFloat32));
    } else {
        cv::Mat m = cv::imread(mask_path->string(), cv::IMREAD_GRAYSCALE);
        if (m.empty()) {
            throw std::runtime_error("Failed to read mask: " + mask_path->string());
        }
        m = resize_shorter_side(m, cfg_.msk_size, cv::INTER_NEAREST);
        m = center_crop_square(m, cfg_.msk_crp_size);
        rbox = mask_to_rbox(m);
        mask_t = mask_u8_to_tensor01(m);
    }

    int64_t out_label = label;
    if (train_) {
        std::string cn = class_name_.has_value() ? class_name_.value() : image_path.parent_path().parent_path().parent_path().filename().string();
        auto it = class_to_idx_.find(cn);
        if (it == class_to_idx_.end()) {
            throw std::runtime_error("Unknown class_name: " + cn);
        }
        out_label = it->second;
    }

    Sample s;
    s.image = img_t;
    s.label = out_label;
    s.mask = mask_t;
    s.basename = basename_no_ext(image_path);
    s.img_type = img_type;
    s.rbox = rbox;
    return s;
}

void BTAD::update_class_to_idx(const std::unordered_map<std::string, int64_t>& class_to_idx) {
    for (const auto& kv : class_to_idx_) {
        auto it = class_to_idx.find(kv.first);
        if (it != class_to_idx.end()) {
            class_to_idx_[kv.first] = it->second;
        }
    }
    idx_to_class_.clear();
    for (const auto& kv : class_to_idx_) {
        idx_to_class_[kv.second] = kv.first;
    }
}

// =====================
// BalancedBatchSampler
// =====================

BalancedBatchSampler::BalancedBatchSampler(Config cfg, std::vector<int64_t> normal_idx, std::vector<int64_t> anomaly_idx)
    : cfg_(cfg), normal_idx_(std::move(normal_idx)), anomaly_idx_(std::move(anomaly_idx)), rng_(seed_from_rd()) {

    if (cfg_.num_anomalies != 0) {
        n_normal_ = cfg_.batch_size / 2;
        n_anomaly_ = cfg_.batch_size - n_normal_;
    } else {
        n_normal_ = cfg_.batch_size;
        n_anomaly_ = 0;
    }
}

int64_t BalancedBatchSampler::length() const { return cfg_.steps_per_epoch; }

std::vector<int64_t> BalancedBatchSampler::next() {
    if (step_ >= cfg_.steps_per_epoch) {
        step_ = 0;
    }
    ++step_;

    std::vector<int64_t> batch;
    batch.reserve(static_cast<size_t>(cfg_.batch_size));

    // Match Python generator semantics more closely: iterate through a random permutation without replacement.
    auto refill_perm = [&](std::vector<int64_t>& perm, const std::vector<int64_t>& src) {
        perm = src;
        std::shuffle(perm.begin(), perm.end(), rng_);
    };

    if (normal_perm_.empty() && !normal_idx_.empty()) {
        refill_perm(normal_perm_, normal_idx_);
        normal_pos_ = 0;
    }
    if (anomaly_perm_.empty() && !anomaly_idx_.empty()) {
        refill_perm(anomaly_perm_, anomaly_idx_);
        anomaly_pos_ = 0;
    }

    for (int64_t i = 0; i < n_normal_; ++i) {
        if (normal_idx_.empty()) {
            break;
        }
        if (normal_pos_ >= normal_perm_.size()) {
            refill_perm(normal_perm_, normal_idx_);
            normal_pos_ = 0;
        }
        batch.push_back(normal_perm_[normal_pos_++]);
    }

    for (int64_t i = 0; i < n_anomaly_; ++i) {
        if (anomaly_idx_.empty()) {
            break;
        }
        if (anomaly_pos_ >= anomaly_perm_.size()) {
            refill_perm(anomaly_perm_, anomaly_idx_);
            anomaly_pos_ = 0;
        }
        batch.push_back(anomaly_perm_[anomaly_pos_++]);
    }

    return batch;
}

} 
#ifdef _MSC_VER
#pragma warning(pop)
#endif