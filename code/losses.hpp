#pragma once

#include <torch/torch.h>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

enum class Nonlinearity {
    None = 0,
    Softmax = 1,
    Sigmoid = 2,
};

class FocalLossImpl : public torch::nn::Module {
public:
    FocalLossImpl(
        Nonlinearity apply_nonlin = Nonlinearity::None,
        double alpha = 0.5,
        double gamma = 4.0,
        int64_t balance_index = 0,
        double smooth = 1e-5,
        bool size_average = true);

    torch::Tensor forward(const torch::Tensor& logit, const torch::Tensor& target);

private:
    Nonlinearity apply_nonlin_;
    double alpha_;
    double gamma_;
    int64_t balance_index_;
    double smooth_;
    bool size_average_;
};
TORCH_MODULE(FocalLoss);

class SmoothL1LossImpl : public torch::nn::Module {
public:
    SmoothL1LossImpl(double beta = 1.0, std::string reduction = "mean", double loss_weight = 1.0);

    torch::Tensor forward(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        const c10::optional<torch::Tensor>& weight = c10::nullopt,
        const c10::optional<double>& avg_factor = c10::nullopt,
        const c10::optional<std::string>& reduction_override = c10::nullopt);

private:
    double beta_;
    std::string reduction_;
    double loss_weight_;
};
TORCH_MODULE(SmoothL1Loss);

struct MaskSupervisionLoss {
    torch::Tensor focal;
    torch::Tensor smooth_l1;
    torch::Tensor total;
};

MaskSupervisionLoss compute_mask_supervision_loss(
    const torch::Tensor& pred,
    const torch::Tensor& mask_gt,
    double smooth_l1_weight = 0.5);

}
#ifdef _MSC_VER
#pragma warning(pop)
#endif 