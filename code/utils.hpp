#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

struct RBox {
    float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f, angle_deg = 0.0f;
    float score = 1.0f;
};

struct RBoxPostprocessOptions {
    float thresh = 0.5f;
    float min_score = 0.0f;
    float min_area = 100.0f;
    float max_area = 1e9f;
    float min_aspect = 1.0f;
    float max_aspect = 8.0f;
    float min_side = 8.0f;
    float edge_margin_ratio = 0.0f;
    float ignore_tl_width_ratio = 0.0f;
    float ignore_tl_height_ratio = 0.0f;
    // Optional custom ignore rectangle in ratio coordinates [0,1].
    // Active only when ignore_rect_x1_ratio > ignore_rect_x0_ratio and
    // ignore_rect_y1_ratio > ignore_rect_y0_ratio.
    float ignore_rect_x0_ratio = 0.0f;
    float ignore_rect_y0_ratio = 0.0f;
    float ignore_rect_x1_ratio = 0.0f;
    float ignore_rect_y1_ratio = 0.0f;
    float nms_iou_thresh = 0.2f;
    int morph_kernel = 3;
    int morph_close_iter = 1;
    int morph_open_iter = 1;
};

// ====================== 原型相关 ======================
std::vector<torch::Tensor> load_prototype_features(const std::string& proto_dir,
                                                   const std::string& class_name,
                                                   torch::Device device);

std::vector<torch::Tensor> get_prototype_features(const std::vector<torch::Tensor>& features,
                                                  const std::vector<torch::Tensor>& proto_features);

std::vector<torch::Tensor> get_residual_features(const std::vector<torch::Tensor>& features,
                                                 const std::vector<torch::Tensor>& closest_protos);

std::vector<torch::Tensor> get_concatenated_features(const std::vector<torch::Tensor>& features1,
                                                     const std::vector<torch::Tensor>& features2);

// ====================== 热图后处理 & RBox ======================
torch::Tensor postprocess_anomaly_map(const torch::Tensor& logits);

float compute_image_level_score(const torch::Tensor& score_map);

std::vector<RBox> extract_rboxes_from_heatmap(const torch::Tensor& score_map,
                                              const RBoxPostprocessOptions& opt = {});

// ====================== 赛题输出 ======================
void process_and_save_results(const std::string& input_bmp_path,
                              const torch::Tensor& anomaly_logits,
                              const std::string& output_dir = "",
                              const RBoxPostprocessOptions& opt = {});

void process_folder_batch(const std::string& root_dir,
                          const std::vector<torch::Tensor>& anomaly_logits_list,
                          const RBoxPostprocessOptions& opt = {});

#ifdef _MSC_VER
#pragma warning(pop)
#endif