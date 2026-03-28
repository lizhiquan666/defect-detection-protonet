#include "losses.hpp"

#include <limits>
#include <stdexcept>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

static torch::Tensor apply_nonlin(Nonlinearity nonlin, const torch::Tensor& x) {
    switch (nonlin) {
    case Nonlinearity::None:
        return x;
    case Nonlinearity::Softmax:
        return torch::softmax(x, 1);
    case Nonlinearity::Sigmoid:
        return torch::sigmoid(x);
    default:
        return x;
    }
}

FocalLossImpl::FocalLossImpl(
    Nonlinearity apply_nonlin,
    double alpha,
    double gamma,
    int64_t balance_index,
    double smooth,
    bool size_average)
    : apply_nonlin_(apply_nonlin),
      alpha_(alpha),
      gamma_(gamma),
      balance_index_(balance_index),
      smooth_(smooth),
      size_average_(size_average) {
    if (smooth_ < 0.0 || smooth_ > 1.0) {
        throw std::invalid_argument("FocalLoss smooth should be in [0,1]");
    }
}

torch::Tensor FocalLossImpl::forward(const torch::Tensor& logit_in, const torch::Tensor& target_in) {
    torch::Tensor logit = apply_nonlin(apply_nonlin_, logit_in);
    const int64_t num_class = logit.size(1);

    if (logit.dim() > 2) {
        logit = logit.view({logit.size(0), logit.size(1), -1});
        logit = logit.permute({0, 2, 1}).contiguous();
        logit = logit.view({-1, logit.size(-1)});
    }

    torch::Tensor target = target_in;
    if (target.dim() == 4 && target.size(1) == 1) {
        target = target.squeeze(1);
    }
    target = target.contiguous().view({-1, 1});

    torch::Tensor alpha;
    alpha = torch::ones({num_class, 1}, torch::TensorOptions().dtype(torch::kFloat));
    alpha = alpha * (1.0 - alpha_);
    if (balance_index_ < 0 || balance_index_ >= num_class) {
        throw std::invalid_argument("FocalLoss balance_index out of range");
    }
    alpha.index_put_({balance_index_, 0}, alpha_);

    alpha = alpha.to(logit.device());

    torch::Tensor idx = target.to(logit.device()).to(torch::kLong);

    torch::Tensor one_hot_key = torch::zeros({target.size(0), num_class}, logit.options().dtype(torch::kFloat));
    one_hot_key = one_hot_key.scatter(1, idx, 1.0);

    if (smooth_ > 0.0) {
        const double low = smooth_ / static_cast<double>(num_class - 1);
        const double high = 1.0 - smooth_;
        one_hot_key = torch::clamp(one_hot_key, low, high);
    }

    torch::Tensor pt = (one_hot_key * logit).sum(1) + smooth_;
    torch::Tensor logpt = torch::log(pt);

    torch::Tensor alpha_idx = alpha.index_select(0, idx.view({-1})).squeeze(1);

    torch::Tensor loss = -1.0 * alpha_idx * torch::pow((1.0 - pt), gamma_) * logpt;

    if (size_average_) {
        return loss.mean();
    }
    return loss.sum();
}

static torch::Tensor reduce_loss(const torch::Tensor& loss, const std::string& reduction) {
    if (reduction == "none") {
        return loss;
    }
    if (reduction == "mean") {
        return loss.mean();
    }
    if (reduction == "sum") {
        return loss.sum();
    }
    throw std::invalid_argument("Invalid reduction: " + reduction);
}

static torch::Tensor weight_reduce_loss(
    torch::Tensor loss,
    const c10::optional<torch::Tensor>& weight,
    const std::string& reduction,
    const c10::optional<double>& avg_factor) {
    if (weight.has_value()) {
        loss = loss * weight.value();
    }

    if (!avg_factor.has_value()) {
        return reduce_loss(loss, reduction);
    }

    if (reduction == "mean") {
        const double eps = std::numeric_limits<float>::epsilon();
        return loss.sum() / (avg_factor.value() + eps);
    }

    if (reduction != "none") {
        throw std::invalid_argument("avg_factor can not be used with reduction=\"sum\"");
    }

    return loss;
}

static torch::Tensor smooth_l1_loss_elementwise(const torch::Tensor& pred, const torch::Tensor& target, double beta) {
    if (target.numel() == 0) {
        return pred.sum() * 0;
    }
    if (!pred.sizes().equals(target.sizes())) {
        throw std::invalid_argument("SmoothL1Loss pred and target must have same shape");
    }

    torch::Tensor diff = torch::abs(pred - target);
    return torch::where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta);
}

SmoothL1LossImpl::SmoothL1LossImpl(double beta, std::string reduction, double loss_weight)
    : beta_(beta), reduction_(std::move(reduction)), loss_weight_(loss_weight) {
    if (beta_ <= 0.0) {
        throw std::invalid_argument("SmoothL1Loss beta must be > 0");
    }
    if (reduction_ != "none" && reduction_ != "mean" && reduction_ != "sum") {
        throw std::invalid_argument("SmoothL1Loss invalid reduction");
    }
}

torch::Tensor SmoothL1LossImpl::forward(
    const torch::Tensor& pred,
    const torch::Tensor& target,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<double>& avg_factor,
    const c10::optional<std::string>& reduction_override) {

    std::string reduction = reduction_;
    if (reduction_override.has_value()) {
        const auto& r = reduction_override.value();
        if (r != "none" && r != "mean" && r != "sum") {
            throw std::invalid_argument("SmoothL1Loss reduction_override invalid");
        }
        reduction = r;
    }

    torch::Tensor loss = smooth_l1_loss_elementwise(pred, target, beta_);
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor);
    return loss_weight_ * loss;
}

MaskSupervisionLoss compute_mask_supervision_loss(
    const torch::Tensor& pred,
    const torch::Tensor& mask_gt,
    double smooth_l1_weight) {

    if (pred.dim() != 4) {
        throw std::invalid_argument("compute_mask_supervision_loss: pred must be 4D [N,C,H,W]");
    }

    torch::Tensor gt = mask_gt;
    if (gt.dim() == 3) {
        gt = gt.unsqueeze(1);
    }
    if (gt.dim() != 4 || gt.size(1) != 1) {
        throw std::invalid_argument("compute_mask_supervision_loss: mask_gt must be [N,1,H,W] or [N,H,W]");
    }

    gt = gt.to(pred.device(), torch::kFloat32).clamp(0.0, 1.0);

    if (pred.size(0) != gt.size(0) || pred.size(2) != gt.size(2) || pred.size(3) != gt.size(3)) {
        throw std::invalid_argument("compute_mask_supervision_loss: pred and mask_gt spatial/batch shape mismatch");
    }

    torch::Tensor focal_input;
    torch::Tensor anomaly_prob;
    Nonlinearity focal_nonlin = Nonlinearity::None;

    if (pred.size(1) == 2) {
        focal_input = pred;
        focal_nonlin = Nonlinearity::Softmax;
        anomaly_prob = torch::softmax(pred, 1).slice(1, 1, 2);
    } else if (pred.size(1) == 1) {
        anomaly_prob = torch::sigmoid(pred);
        focal_input = torch::cat({1.0 - anomaly_prob, anomaly_prob}, 1);
        focal_nonlin = Nonlinearity::None;
    } else {
        throw std::invalid_argument("compute_mask_supervision_loss: pred channel must be 1 or 2");
    }

    torch::Tensor target_cls = gt.squeeze(1).round().to(torch::kLong);

    auto focal_loss_fn = FocalLoss(FocalLossImpl(focal_nonlin));
    auto smooth_l1_fn = SmoothL1Loss();

    torch::Tensor focal_loss = focal_loss_fn->forward(focal_input, target_cls);
    torch::Tensor smooth_l1 = smooth_l1_fn->forward(anomaly_prob, gt);
    torch::Tensor total_loss = focal_loss + smooth_l1_weight * smooth_l1;

    MaskSupervisionLoss out;
    out.focal = focal_loss;
    out.smooth_l1 = smooth_l1;
    out.total = total_loss;
    return out;
}

}
#ifdef _MSC_VER
#pragma warning(pop)
#endif