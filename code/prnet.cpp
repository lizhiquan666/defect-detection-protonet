#include "prnet.hpp"
#include <filesystem>
#include <stdexcept>
#include <utility>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

namespace {

std::filesystem::path normalize_existing_path(const std::filesystem::path& path) {
  return std::filesystem::weakly_canonical(path);
}

std::filesystem::path find_in_cwd_ancestors(const std::filesystem::path& relative_path,
                                            int max_levels = 6) {
  auto base = std::filesystem::current_path();
  for (int i = 0; i <= max_levels; ++i) {
    auto candidate = base / relative_path;
    if (std::filesystem::exists(candidate)) {
      return normalize_existing_path(candidate);
    }
    if (!base.has_parent_path()) {
      break;
    }
    base = base.parent_path();
  }
  return {};
}

std::filesystem::path resolve_readable_path(const std::string& input_path,
                                            const std::vector<std::filesystem::path>& fallbacks = {}) {
  if (!input_path.empty()) {
    const std::filesystem::path raw(input_path);

    if (raw.is_absolute()) {
      if (std::filesystem::exists(raw)) {
        return normalize_existing_path(raw);
      }
      throw std::runtime_error("Path does not exist: " + raw.string());
    }

    if (std::filesystem::exists(raw)) {
      return normalize_existing_path(raw);
    }

    auto found = find_in_cwd_ancestors(raw);
    if (!found.empty()) {
      return found;
    }
  }

  for (const auto& fallback : fallbacks) {
    if (std::filesystem::exists(fallback)) {
      return normalize_existing_path(fallback);
    }

    if (fallback.is_relative()) {
      auto found = find_in_cwd_ancestors(fallback);
      if (!found.empty()) {
        return found;
      }
    }
  }

  std::string error = "Unable to resolve path";
  if (!input_path.empty()) {
    error += " from input: " + input_path;
  }
  if (!fallbacks.empty()) {
    error += " (fallbacks:";
    for (const auto& fallback : fallbacks) {
      error += " " + fallback.string();
    }
    error += " )";
  }
  throw std::runtime_error(error);
}

}  // namespace

// -----------------------------
// BasicBlock
// -----------------------------

BasicBlockImpl::BasicBlockImpl(int64_t in_planes, int64_t planes, int64_t stride)
    : stride_(stride) {
  conv1 = register_module(
      "conv1",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 3)
                            .stride(stride)
                            .padding(1)
                            .bias(false)));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));

  conv2 = register_module(
      "conv2",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3)
                            .stride(1)
                            .padding(1)
                            .bias(false)));
  bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

  if (stride != 1 || in_planes != planes) {
    downsample = register_module(
        "downsample",
        torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1)
                                  .stride(stride)
                                  .bias(false)),
            torch::nn::BatchNorm2d(planes)));
  } else {
    downsample = register_module("downsample", torch::nn::Sequential());
  }
}

torch::Tensor BasicBlockImpl::forward(const torch::Tensor& x) {
  auto identity = x;

  auto out = conv1->forward(x);
  out = bn1->forward(out);
  out = torch::relu(out);

  out = conv2->forward(out);
  out = bn2->forward(out);

  if (!downsample->is_empty()) {
    identity = downsample->forward(x);
  }

  out += identity;
  out = torch::relu(out);
  return out;
}

ResNet18FeaturesImpl::ResNet18FeaturesImpl() : relu(torch::nn::ReLUOptions(true)) {
  conv1 = register_module(
      "conv1",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7)
                            .stride(2)
                            .padding(3)
                            .bias(false)));
  bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
  maxpool = register_module(
      "maxpool",
      torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

  layer1 = register_module("layer1", _make_layer(64, 2, 1));
  layer2 = register_module("layer2", _make_layer(128, 2, 2));
  layer3 = register_module("layer3", _make_layer(256, 2, 2));
  layer4 = register_module("layer4", _make_layer(512, 2, 2));
}

torch::nn::Sequential ResNet18FeaturesImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
  torch::nn::Sequential layers;

  // First block may downsample
  layers->push_back(BasicBlock(in_planes_, planes, stride));
  in_planes_ = planes * BasicBlockImpl::kExpansion;

  for (int64_t i = 1; i < blocks; ++i) {
    layers->push_back(BasicBlock(in_planes_, planes, 1));
  }

  return layers;
}

std::vector<torch::Tensor> ResNet18FeaturesImpl::forward_features(const torch::Tensor& x) {
  auto out = conv1->forward(x);
  out = bn1->forward(out);
  out = relu->forward(out);
  out = maxpool->forward(out);

  auto f1 = layer1->forward(out);
  auto f2 = layer2->forward(f1);
  auto f3 = layer3->forward(f2);
  auto f4 = layer4->forward(f3);

  return {f1, f2, f3, f4};
}

torch::Tensor ResNet18FeaturesImpl::forward(const torch::Tensor& x) {
  // Default forward returns deepest feature (not used directly in PRNet)
  return forward_features(x).back();
}

UpsampleConvImpl::UpsampleConvImpl(int64_t in_channels, int64_t out_channels, int64_t scale_factor)
    : scale_factor_(scale_factor) {
  conv = register_module(
      "conv",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(1).bias(true)));
}

torch::Tensor UpsampleConvImpl::forward(const torch::Tensor& x) {
  auto y = torch::nn::functional::interpolate(
      x,
      torch::nn::functional::InterpolateFuncOptions()
          .scale_factor(std::vector<double>{static_cast<double>(scale_factor_), static_cast<double>(scale_factor_)})
          .mode(torch::kBilinear)
          .align_corners(true));
  y = conv->forward(y);
  return y;
}

// -----------------------------
// MultiScaleFusion
// -----------------------------

MultiScaleFusionImpl::MultiScaleFusionImpl(const std::vector<int64_t>& channels) {
  TORCH_CHECK(channels.size() == 3, "MultiScaleFusion expects exactly 3 channel entries.");

  l2_to_l1 = register_module("l2_to_l1", UpsampleConv(channels[1], channels[0], 2));
  l3_to_l1 = register_module("l3_to_l1", UpsampleConv(channels[2], channels[0], 4));

  // Depthwise downsample convs (groups == in_channels)
  l1_to_l2 = register_module(
      "l1_to_l2",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[0], channels[1], 3)
                            .stride(2)
                            .padding(1)
                            .groups(channels[0])
                            .bias(true)));
  l3_to_l2 = register_module("l3_to_l2", UpsampleConv(channels[2], channels[1], 2));

  l1_to_l3 = register_module(
      "l1_to_l3",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[0], channels[2], 5)
                            .stride(4)
                            .padding(2)
                            .groups(channels[0])
                            .bias(true)));
  l2_to_l3 = register_module(
      "l2_to_l3",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(channels[1], channels[2], 3)
                            .stride(2)
                            .padding(1)
                            .groups(channels[1])
                            .bias(true)));
}

std::vector<torch::Tensor> MultiScaleFusionImpl::forward(const torch::Tensor& layer1_x,
                                                         const torch::Tensor& layer2_x,
                                                         const torch::Tensor& layer3_x) {
  auto layer2_x_to_1 = l2_to_l1->forward(layer2_x);
  auto layer3_x_to_1 = l3_to_l1->forward(layer3_x);
  auto out1 = layer1_x + layer2_x_to_1 + layer3_x_to_1;

  auto layer1_x_to_2 = l1_to_l2->forward(layer1_x);
  auto layer3_x_to_2 = l3_to_l2->forward(layer3_x);
  auto out2 = layer2_x + layer1_x_to_2 + layer3_x_to_2;

  auto layer1_x_to_3 = l1_to_l3->forward(layer1_x);
  auto layer2_x_to_3 = l2_to_l3->forward(layer2_x);
  auto out3 = layer3_x + layer1_x_to_3 + layer2_x_to_3;

  return {out1, out2, out3};
}

// -----------------------------
// PRNet
// -----------------------------

PRNetImpl::PRNetImpl(const Options& options) : options_(options) {
  std::cout << "[PRNET] Resolving encoder path ...\n";
  options_.encoder_ts_path =
      resolve_readable_path(options_.encoder_ts_path,
                            {"code/resnet18_encoder_ts.pt", "resnet18_encoder_ts.pt"})
          .string();
  std::cout << "[PRNET] Encoder path: " << options_.encoder_ts_path << "\n";

  std::cout << "[PRNET] Loading TorchScript encoder ...\n";
  encoder_ts_ = torch::jit::load(options_.encoder_ts_path);
  encoder_ts_.eval();
  std::cout << "[PRNET] TorchScript encoder loaded\n";
  encoder_device_initialized_ = false;

  std::cout << "[PRNET] Inferring feature map dims ...\n";
  infer_featuremap_dims_();
  std::cout << "[PRNET] Feature map dims inferred\n";

  // feature_dims_ = [C1,C2,C3,C4]
  const std::vector<int64_t> fuse_channels = {feature_dims_[0], feature_dims_[1], feature_dims_[2]};
  std::cout << "[PRNET] Creating ms_fuser1 ...\n";
  ms_fuser1 = register_module("ms_fuser1", MultiScaleFusion(fuse_channels));
  std::cout << "[PRNET] ms_fuser1 ready\n";

  const std::vector<int64_t> in_channels = {2 * feature_dims_[0], 2 * feature_dims_[1], 2 * feature_dims_[2]};
  std::cout << "[PRNET] Creating ms_fuser2/attn_module ...\n";
  ms_fuser2 = register_module("ms_fuser2", MultiScaleFusion(in_channels));
  attn_module = register_module("attn_module", MultiSizeAttentionModule(in_channels, heights_, std::vector<int64_t>{}, 1));
  std::cout << "[PRNET] ms_fuser2/attn_module ready\n";

  // Decoder (mirrors python implementation)
  std::cout << "[PRNET] Building decoder ...\n";
  up4_to_3 = register_module(
      "up4_to_3",
      torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{2.0, 2.0})
                                  .mode(torch::kBilinear)
                                  .align_corners(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[3], feature_dims_[2], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[2]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  conv_block3 = register_module(
      "conv_block3",
      torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[2] * 3, feature_dims_[2], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[2]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[2], feature_dims_[2], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[2]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  up3_to_2 = register_module(
      "up3_to_2",
      torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{2.0, 2.0})
                                  .mode(torch::kBilinear)
                                  .align_corners(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[2], feature_dims_[1], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[1]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  conv_block2 = register_module(
      "conv_block2",
      torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[1] * 3, feature_dims_[1], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[1]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[1], feature_dims_[1], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[1]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  up2_to_1 = register_module(
      "up2_to_1",
      torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{2.0, 2.0})
                                  .mode(torch::kBilinear)
                                  .align_corners(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[1], feature_dims_[0], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[0]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  conv_block1 = register_module(
      "conv_block1",
      torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[0] * 3, feature_dims_[0], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[0]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[0], feature_dims_[0], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[0]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  up1_to_0 = register_module(
      "up1_to_0",
      torch::nn::Sequential(
          torch::nn::Upsample(torch::nn::UpsampleOptions()
                                  .scale_factor(std::vector<double>{4.0, 4.0})
                                  .mode(torch::kBilinear)
                                  .align_corners(true)),
          torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[0], feature_dims_[0], 3).padding(1)),
          torch::nn::BatchNorm2d(feature_dims_[0]),
          torch::nn::ReLU(torch::nn::ReLUOptions(true))));

  conv_out = register_module(
      "conv_out",
      torch::nn::Conv2d(torch::nn::Conv2dOptions(feature_dims_[0], options_.num_classes, 3).padding(1)));
  std::cout << "[PRNET] Decoder ready\n";
}

void PRNetImpl::ensure_encoder_device_(const c10::Device& device) {
  if (!encoder_device_initialized_ || encoder_device_ != device) {
    encoder_ts_ = torch::jit::load(options_.encoder_ts_path, device);
    encoder_ts_.eval();
    encoder_device_ = device;
    encoder_device_initialized_ = true;
  }
}

std::vector<torch::Tensor> PRNetImpl::forward_encoder_(const torch::Tensor& images) {
  ensure_encoder_device_(images.device());

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(images);

  auto out = encoder_ts_.forward(inputs);
  auto tup = out.toTuple();
  TORCH_CHECK(tup, "TorchScript encoder forward did not return a tuple");

  const auto& elems = tup->elements();
  TORCH_CHECK(elems.size() >= 4,
              "TorchScript encoder must return at least 4 tensors (layer1, layer2, layer3, layer4)");

  std::vector<torch::Tensor> feats;
  feats.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    feats.push_back(elems[i].toTensor());
  }
  return feats;
}

void PRNetImpl::infer_featuremap_dims_() {
  std::cout << "[PRNET] infer_featuremap_dims_: create dry tensor\n";
  torch::NoGradGuard no_grad;

  auto dry = torch::empty({1, 3, options_.input_h, options_.input_w}, torch::TensorOptions().device(torch::kCPU));
  std::cout << "[PRNET] infer_featuremap_dims_: run encoder forward\n";
  auto feats = forward_encoder_(dry);
  std::cout << "[PRNET] infer_featuremap_dims_: encoder forward done\n";

  feature_dims_.clear();
  heights_.clear();
  widths_.clear();

  for (const auto& f : feats) {
    feature_dims_.push_back(f.size(1));
  }

  // Heights/widths for layer1/2/3 only (excluding deepest layer4)
  for (int i = 0; i < 3; ++i) {
    heights_.push_back(feats[i].size(2));
    widths_.push_back(feats[i].size(3));
  }
}

void PRNetImpl::load_prototypes(const std::string& pt_path) {
  prototypes_.clear();
  const auto resolved_pt = resolve_readable_path(pt_path);
  torch::load(prototypes_, resolved_pt.string());
  TORCH_CHECK(prototypes_.size() == 3, "Expected 3 prototype tensors in .pt (layer1/2/3)");
}

std::vector<torch::Tensor> PRNetImpl::get_prototype_features(const std::vector<torch::Tensor>& features,
                                                             const std::vector<torch::Tensor>& proto_features) {
  TORCH_CHECK(features.size() == proto_features.size(), "features and proto_features must have same size");

  std::vector<torch::Tensor> matched;
  matched.reserve(features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    const auto& fi = features[i];          // (B, C, H, W)
    const auto& pi = proto_features[i];    // (K, C, H, W)
    TORCH_CHECK(fi.dim() == 4 && pi.dim() == 4, "prototype matching expects 4D tensors");

    const auto B = fi.size(0);
    const auto K = pi.size(0);
    TORCH_CHECK(K > 0, "prototype K must be > 0");

    // Flatten: fi_flat (B, D), pi_flat (K, D)
    auto fi_flat = fi.view({B, -1});
    auto pi_flat = pi.view({K, -1});

    // dist: (B, K)
    auto dist = torch::cdist(fi_flat.unsqueeze(0), pi_flat.unsqueeze(0)).squeeze(0);
    auto inds = std::get<1>(dist.min(1));  // (B)

    auto matched_pi = pi.index_select(0, inds);
    matched.push_back(matched_pi);
  }

  return matched;
}

std::vector<torch::Tensor> PRNetImpl::get_residual_features(const std::vector<torch::Tensor>& features,
                                                            const std::vector<torch::Tensor>& proto_features) {
  TORCH_CHECK(features.size() == proto_features.size(), "features and proto_features must have same size");
  std::vector<torch::Tensor> residual;
  residual.reserve(features.size());

  for (size_t i = 0; i < features.size(); ++i) {
    auto ri = (features[i] - proto_features[i]).pow(2);
    residual.push_back(ri);
  }

  return residual;
}

std::vector<torch::Tensor> PRNetImpl::get_concatenated_features(const std::vector<torch::Tensor>& features1,
                                                                const std::vector<torch::Tensor>& features2) {
  TORCH_CHECK(features1.size() == features2.size(), "features1 and features2 must have same size");
  std::vector<torch::Tensor> out;
  out.reserve(features1.size());

  for (size_t i = 0; i < features1.size(); ++i) {
    out.push_back(torch::cat({features1[i], features2[i]}, 1));
  }

  return out;
}

torch::Tensor PRNetImpl::logits_to_supervised_heatmap_(const torch::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 4, "logits_to_supervised_heatmap_ expects logits [N,C,H,W]");

  if (logits.size(1) == 1) {
    return torch::sigmoid(logits);
  }
  TORCH_CHECK(logits.size(1) >= 2, "logits channel must be >=1");
  return torch::softmax(logits, 1).slice(1, 1, 2);
}

torch::Tensor PRNetImpl::normalize_heatmap_minmax_(const torch::Tensor& heatmap) {
  TORCH_CHECK(heatmap.dim() == 4 && heatmap.size(1) == 1,
              "normalize_heatmap_minmax_ expects [N,1,H,W]");

  auto x = heatmap;
  auto x_flat = x.view({x.size(0), -1});
  auto min_v = std::get<0>(x_flat.min(1, true));
  auto max_v = std::get<0>(x_flat.max(1, true));
  auto denom = (max_v - min_v).clamp_min(1e-6);

  auto out = (x_flat - min_v) / denom;
  return out.view_as(x);
}

PRNetImpl::ForwardOutputs PRNetImpl::forward_impl_(const torch::Tensor& images,
                                                   const std::vector<torch::Tensor>& proto_features) {
  // 1) Encoder features
  auto feats_all = forward_encoder_(images);
  auto layer4 = feats_all[3];
  std::vector<torch::Tensor> feats = {feats_all[0], feats_all[1], feats_all[2]};

  // 2) Prototype + residual
  auto pfeats = get_prototype_features(feats, proto_features);
  auto rfeats = get_residual_features(feats, pfeats);

  // Residual heatmap (multi-scale prototype distance map)
  std::vector<torch::Tensor> residual_maps;
  residual_maps.reserve(rfeats.size());
  for (const auto& rf : rfeats) {
    auto rm = rf.mean(1, true);
    rm = torch::nn::functional::interpolate(
        rm,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{images.size(2), images.size(3)})
            .mode(torch::kBilinear)
            .align_corners(true));
    residual_maps.push_back(rm);
  }
  auto residual_heatmap = normalize_heatmap_minmax_((residual_maps[0] + residual_maps[1] + residual_maps[2]) / 3.0);

  // 3) Multi-scale fusion (twice, like python)
  auto fused = ms_fuser1->forward(feats[0], feats[1], feats[2]);
  auto rfused = ms_fuser1->forward(rfeats[0], rfeats[1], rfeats[2]);

  // 4) Concat input+residual per scale
  auto cfeats = get_concatenated_features(fused, rfused);

  // 5) Attention + second fusion
  auto attn = attn_module->forward(cfeats);
  auto fused2 = ms_fuser2->forward(attn[0], attn[1], attn[2]);

  // 6) Decoder
  auto l3 = up4_to_3->forward(layer4);
  l3 = torch::cat({fused2[2], l3}, 1);
  l3 = conv_block3->forward(l3);

  auto l2 = up3_to_2->forward(l3);
  l2 = torch::cat({fused2[1], l2}, 1);
  l2 = conv_block2->forward(l2);

  auto l1 = up2_to_1->forward(l2);
  l1 = torch::cat({fused2[0], l1}, 1);
  l1 = conv_block1->forward(l1);

  auto out_features = up1_to_0->forward(l1);
  auto logits = conv_out->forward(out_features);
  auto supervised_heatmap = logits_to_supervised_heatmap_(logits);

  ForwardOutputs out;
  out.logits = logits;
  out.residual_heatmap = residual_heatmap;
  out.supervised_heatmap = supervised_heatmap;
  return out;
}

torch::Tensor PRNetImpl::forward(const torch::Tensor& images,
                                 const std::vector<torch::Tensor>& proto_features) {
  return forward(images, proto_features, false);
}

torch::Tensor PRNetImpl::forward(const torch::Tensor& images,
                                 const std::vector<torch::Tensor>& proto_features,
                                 bool training_branch) {
  if (training_branch) {
    auto out = forward_impl_(images, proto_features);
    return out.logits;
  }

  torch::NoGradGuard no_grad;
  auto out = forward_impl_(images, proto_features);
  return out.logits;
}

torch::Tensor PRNetImpl::forward_heatmap(const torch::Tensor& images,
                                         const std::vector<torch::Tensor>& proto_features,
                                         HeatmapMode mode,
                                         bool training_branch) {
  ForwardOutputs out;
  if (training_branch) {
    out = forward_impl_(images, proto_features);
  } else {
    torch::NoGradGuard no_grad;
    out = forward_impl_(images, proto_features);
  }

  if (mode == HeatmapMode::Residual) {
    return out.residual_heatmap;
  }
  if (mode == HeatmapMode::Supervised) {
    return out.supervised_heatmap;
  }
  return 0.5 * (out.residual_heatmap + out.supervised_heatmap);
}

std::vector<torch::Tensor> PRNetImpl::extract_encoder_features(const torch::Tensor& images,
                                                               bool no_grad) {
  if (no_grad) {
    torch::NoGradGuard guard;
    return forward_encoder_(images);
  }
  return forward_encoder_(images);
}

}
#ifdef _MSC_VER
#pragma warning(pop)
#endif