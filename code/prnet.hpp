#pragma once

#include <torch/torch.h>
#include <torch/script.h>

#include "attention.hpp"

#include <cstdint>
#include <string>
#include <vector>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

struct BasicBlockImpl : torch::nn::Module {
  static constexpr int64_t kExpansion = 1;

  BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride = 1);

  torch::Tensor forward(const torch::Tensor& x);

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::Sequential downsample;
  int64_t stride_ = 1;
};
TORCH_MODULE(BasicBlock);

struct ResNet18FeaturesImpl : torch::nn::Module {
  ResNet18FeaturesImpl();

  std::vector<torch::Tensor> forward_features(const torch::Tensor& x);

  torch::Tensor forward(const torch::Tensor& x);

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::ReLU relu;
  torch::nn::MaxPool2d maxpool{nullptr};

  torch::nn::Sequential layer1;
  torch::nn::Sequential layer2;
  torch::nn::Sequential layer3;
  torch::nn::Sequential layer4;

private:
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride);
  int64_t in_planes_ = 64;
};
TORCH_MODULE(ResNet18Features);

struct UpsampleConvImpl : torch::nn::Module {
  UpsampleConvImpl(int64_t in_channels, int64_t out_channels, int64_t scale_factor = 2);
  torch::Tensor forward(const torch::Tensor& x);

  int64_t scale_factor_ = 2;
  torch::nn::Conv2d conv{nullptr};
};
TORCH_MODULE(UpsampleConv);

struct MultiScaleFusionImpl : torch::nn::Module {
  explicit MultiScaleFusionImpl(const std::vector<int64_t>& channels);

  // Expects 3 tensors: layer1, layer2, layer3
  std::vector<torch::Tensor> forward(const torch::Tensor& layer1_x,
                                     const torch::Tensor& layer2_x,
                                     const torch::Tensor& layer3_x);

  UpsampleConv l2_to_l1{nullptr};
  UpsampleConv l3_to_l1{nullptr};

  torch::nn::Conv2d l1_to_l2{nullptr};
  UpsampleConv l3_to_l2{nullptr};

  torch::nn::Conv2d l1_to_l3{nullptr};
  torch::nn::Conv2d l2_to_l3{nullptr};
};
TORCH_MODULE(MultiScaleFusion);

using MultiSizeAttention = ::MultiSizeAttention;
using MultiSizeAttentionModule = ::MultiSizeAttentionModule;

struct PRNetImpl : torch::nn::Module {
  enum class HeatmapMode {
    Residual = 0,
    Supervised = 1,
    ResidualPlusSupervised = 2,
  };

  struct Options {
    int64_t num_classes = 2;
    int64_t input_h = 256;
    int64_t input_w = 256;
    // Optional explicit path. If empty, PRNet will auto-discover from common relative locations.
    std::string encoder_ts_path;
  };

  explicit PRNetImpl(const Options& options = Options());

  // proto_features should contain 3 tensors (layer1/2/3 prototypes):
  //   proto_features[i] shape: (K, C_i, H_i, W_i)
  // where (C_i, H_i, W_i) must match the corresponding encoder feature map.
  torch::Tensor forward(const torch::Tensor& images,
                        const std::vector<torch::Tensor>& proto_features);

  // training_branch=false: inference behavior (no grad in encoder-heavy path)
  // training_branch=true: enables full autograd path for supervised training.
  torch::Tensor forward(const torch::Tensor& images,
                        const std::vector<torch::Tensor>& proto_features,
                        bool training_branch);

  // Output heatmap in selected mode:
  // - Residual: residual-based map from prototype distances
  // - Supervised: map decoded from segmentation logits
  // - ResidualPlusSupervised: average of both
  torch::Tensor forward_heatmap(const torch::Tensor& images,
                                const std::vector<torch::Tensor>& proto_features,
                                HeatmapMode mode = HeatmapMode::ResidualPlusSupervised,
                                bool training_branch = false);

  // Expose encoder multi-scale features for training utilities (e.g. dynamic prototype updates).
  // Returns 4 tensors: layer1, layer2, layer3, layer4.
  std::vector<torch::Tensor> extract_encoder_features(const torch::Tensor& images,
                                                      bool no_grad = true);

  // Helper: load prototype tensors saved by torch.save(list_of_tensors, path)
  // Expected order: [layer1, layer2, layer3]
  void load_prototypes(const std::string& pt_path);

  // Access loaded prototypes
  const std::vector<torch::Tensor>& prototypes() const { return prototypes_; }

private:
  Options options_;

  // Backbone (TorchScript)
  torch::jit::Module encoder_ts_;
  c10::Device encoder_device_{torch::kCPU};
  bool encoder_device_initialized_ = false;

  // Inferred dimensions for layer1/2/3 (excluding deepest layer4)
  std::vector<int64_t> heights_;
  std::vector<int64_t> widths_;
  std::vector<int64_t> feature_dims_;  // [C1, C2, C3, C4]

  // Fusion + attention
  MultiScaleFusion ms_fuser1{nullptr};
  MultiScaleFusion ms_fuser2{nullptr};
  MultiSizeAttentionModule attn_module{nullptr};

  // Decoder
  torch::nn::Sequential up4_to_3;
  torch::nn::Sequential conv_block3;
  torch::nn::Sequential up3_to_2;
  torch::nn::Sequential conv_block2;
  torch::nn::Sequential up2_to_1;
  torch::nn::Sequential conv_block1;
  torch::nn::Sequential up1_to_0;
  torch::nn::Conv2d conv_out{nullptr};

  std::vector<torch::Tensor> prototypes_;

private:
  struct ForwardOutputs {
    torch::Tensor logits;
    torch::Tensor residual_heatmap;
    torch::Tensor supervised_heatmap;
  };

  void ensure_encoder_device_(const c10::Device& device);
  std::vector<torch::Tensor> forward_encoder_(const torch::Tensor& images);

  ForwardOutputs forward_impl_(const torch::Tensor& images,
                               const std::vector<torch::Tensor>& proto_features);
  static torch::Tensor logits_to_supervised_heatmap_(const torch::Tensor& logits);
  static torch::Tensor normalize_heatmap_minmax_(const torch::Tensor& heatmap);

  // Prototype/residual utils
  static std::vector<torch::Tensor> get_prototype_features(const std::vector<torch::Tensor>& features,
                                                           const std::vector<torch::Tensor>& proto_features);
  static std::vector<torch::Tensor> get_residual_features(const std::vector<torch::Tensor>& features,
                                                          const std::vector<torch::Tensor>& proto_features);
  static std::vector<torch::Tensor> get_concatenated_features(const std::vector<torch::Tensor>& features1,
                                                              const std::vector<torch::Tensor>& features2);

  void infer_featuremap_dims_();
};
TORCH_MODULE(PRNet);

} 
#ifdef _MSC_VER
#pragma warning(pop)
#endif