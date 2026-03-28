#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <c10/util/Optional.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

#if defined(PRNET_ENABLE_COPY_PASTE) || defined(PRNET_USE_COPY_PASTE) || defined(PRNET_ENABLE_PERLIN) || defined(PRNET_USE_PERLIN)
#error "Legacy augmentations (copy_paste/perlin) were removed and must not be enabled in dataset.hpp/cpp."
#endif

struct RBox {
    float cx = 0.0F;
    float cy = 0.0F;
    float w = 0.0F;
    float h = 0.0F;
    float angle_deg = 0.0F; // OpenCV minAreaRect angle (degrees)
};

c10::optional<RBox> mask_to_rbox(const cv::Mat& mask_u8);
torch::Tensor rbox_to_tensor(const RBox& rbox, const torch::TensorOptions& options = torch::TensorOptions().dtype(torch::kFloat));

struct Sample {
    torch::Tensor image;  // (3,H,W) float, normalized
    int64_t label = 0;    // train: class idx; test: 0 normal / 1 anomaly
    torch::Tensor mask;   // (1,H,W) float in {0,1}
    std::string basename;
    std::string img_type;
    c10::optional<RBox> rbox; // from mask (if any)
};

struct DatasetConfig {
    int img_size = 256;
    int crp_size = 256;
    int msk_size = 256;
    int msk_crp_size = 256;

    // Optional synthetic anomaly augmentation (disabled by default).
    bool perlin_enable = false;
    float perlin_prob = 0.0f;
    int perlin_grid = 8;
    float perlin_thresh = 0.58f;
    float perlin_noise_std = 0.35f;

    // Type-directed synthetic anomaly augmentations (all disabled by default).
    bool typed_aug_enable = false;
    float scratch_prob = 0.0f;        // slender scratch-like perturbation
    float missing_print_prob = 0.0f;  // local erase for missing-print defects
    float blob_prob = 0.0f;           // block/blob dirt-like perturbation
    float scratch_strength = 0.20f;
    float missing_print_alpha = 0.70f;
    float blob_noise_std = 0.18f;
};

class MVTEC {
public:
    static const std::vector<std::string> CLASS_NAMES;

    MVTEC(std::filesystem::path root,
          std::optional<std::string> class_name,
          bool train,
          DatasetConfig cfg = {});

    size_t size() const;
    Sample get(size_t idx) const;

    void update_class_to_idx(const std::unordered_map<std::string, int64_t>& class_to_idx);

private:
    std::tuple<std::vector<std::filesystem::path>, std::vector<int64_t>, std::vector<std::optional<std::filesystem::path>>, std::vector<std::string>>
    load_data_(const std::string& class_name) const;

    void load_all_data_();

    Sample load_sample_(const std::filesystem::path& image_path,
                        int64_t label,
                        const std::optional<std::filesystem::path>& mask_path,
                        const std::string& img_type,
                        const std::optional<std::string>& class_name_override) const;

private:
    std::filesystem::path root_;
    std::optional<std::string> class_name_;
    bool train_;
    DatasetConfig cfg_;

    std::vector<std::filesystem::path> image_paths_;
    std::vector<int64_t> labels_;
    std::vector<std::optional<std::filesystem::path>> mask_paths_;
    std::vector<std::string> img_types_;

    std::unordered_map<std::string, int64_t> class_to_idx_;
    std::unordered_map<int64_t, std::string> idx_to_class_;
};

class TRAINMVTEC {
public:
    TRAINMVTEC(std::filesystem::path data_path,
               std::string class_name,
               bool train,
               DatasetConfig cfg = {},
               int num_anomalies = 10);

    size_t size() const;
    size_t normal_size() const;
    size_t anomaly_size() const;
    Sample get(size_t idx);

private:
    struct PathsPack {
        std::vector<std::filesystem::path> n_imgs;
        std::vector<int64_t> n_labels;
        std::vector<std::optional<std::filesystem::path>> n_masks;
        std::vector<std::filesystem::path> a_imgs;
        std::vector<int64_t> a_labels;
        std::vector<std::filesystem::path> a_masks;
    };

    // Supports two layouts:
    // 1) MVTEC-style: <root>/<class>/train|test|ground_truth
    // 2) Custom semi-supervised: <root>/<class>/*ok* and *ng* folders,
    //    where anomaly masks are side-by-side files named *_t.bmp (or *_t.png).
    PathsPack load_dataset_folder_() const;

    std::tuple<torch::Tensor, int64_t, torch::Tensor> load_image_and_mask_(
        const std::filesystem::path& img_path,
        int64_t label,
        const std::optional<std::filesystem::path>& mask_path) const;

private:
    std::filesystem::path dataset_path_;
    std::string class_name_;
    bool train_;

    DatasetConfig cfg_;
    int num_load_anomalies_;

    std::vector<std::filesystem::path> n_imgs_;
    std::vector<int64_t> n_labels_;
    std::vector<std::optional<std::filesystem::path>> n_masks_;

    std::vector<std::filesystem::path> a_imgs_;
    std::vector<int64_t> a_labels_;
    std::vector<std::filesystem::path> a_masks_;

    mutable std::mt19937 rng_;

};

class BTAD {
public:
    static const std::vector<std::string> CLASS_NAMES;

    BTAD(std::filesystem::path root,
         std::optional<std::string> class_name,
         bool train,
         DatasetConfig cfg = {});

    size_t size() const;
    Sample get(size_t idx) const;

    void update_class_to_idx(const std::unordered_map<std::string, int64_t>& class_to_idx);

private:
    std::tuple<std::vector<std::filesystem::path>, std::vector<int64_t>, std::vector<std::optional<std::filesystem::path>>, std::vector<std::string>>
    load_data_(const std::string& class_name) const;

    void load_all_data_();

private:
    std::filesystem::path root_;
    std::optional<std::string> class_name_;
    bool train_;
    DatasetConfig cfg_;

    std::vector<std::filesystem::path> image_paths_;
    std::vector<int64_t> labels_;
    std::vector<std::optional<std::filesystem::path>> mask_paths_;
    std::vector<std::string> img_types_;

    std::unordered_map<std::string, int64_t> class_to_idx_;
    std::unordered_map<int64_t, std::string> idx_to_class_;

    // For train-time random flip (to match Python BTAD)
    mutable std::mt19937 rng_;
};

class BalancedBatchSampler {
public:
    struct Config {
        int64_t batch_size = 32;
        int64_t steps_per_epoch = 100;
        int64_t num_anomalies = 10;
    };

    BalancedBatchSampler(Config cfg, std::vector<int64_t> normal_idx, std::vector<int64_t> anomaly_idx);

    int64_t length() const;
    std::vector<int64_t> next();

private:
    Config cfg_;
    std::vector<int64_t> normal_idx_;
    std::vector<int64_t> anomaly_idx_;

    std::mt19937 rng_;
    int64_t n_normal_ = 0;
    int64_t n_anomaly_ = 0;
    int64_t step_ = 0;

    std::vector<int64_t> normal_perm_;
    std::vector<int64_t> anomaly_perm_;
    size_t normal_pos_ = 0;
    size_t anomaly_pos_ = 0;
};

} 
#ifdef _MSC_VER
#pragma warning(pop)
#endif