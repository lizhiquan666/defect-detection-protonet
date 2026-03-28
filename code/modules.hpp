#pragma once

#include <torch/torch.h>
#include <vector>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

struct UpsampleConvImpl : public torch::nn::Module {
    torch::nn::Upsample upsample{nullptr};
    torch::nn::Conv2d conv{nullptr};

    UpsampleConvImpl(int64_t in_channels, int64_t out_channels, int64_t scale_factor = 2) {
        upsample = register_module("upsample", torch::nn::Upsample(
            torch::nn::UpsampleOptions()
                .scale_factor(std::vector<double>{static_cast<double>(scale_factor),
                                                  static_cast<double>(scale_factor)})
                .mode(torch::kBilinear)
                .align_corners(true)));

        // 1x1 卷积调整通道数
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = upsample->forward(x);
        x = conv->forward(x);
        return x;
    }
};
TORCH_MODULE(UpsampleConv);

struct MultiScaleFusionImpl : public torch::nn::Module {
    UpsampleConv l2_to_l1{nullptr};
    UpsampleConv l3_to_l1{nullptr};

    torch::nn::Conv2d l1_to_l2{nullptr};   // depthwise stride=2
    UpsampleConv l3_to_l2{nullptr};

    torch::nn::Conv2d l1_to_l3{nullptr};   // depthwise stride=4 (kernel=5)
    torch::nn::Conv2d l2_to_l3{nullptr};   // depthwise stride=2

    MultiScaleFusionImpl(const std::vector<int64_t>& channels = {64, 128, 256}) {
        if (channels.size() != 3)
            throw std::runtime_error("MultiScaleFusion requires exactly 3 channel values");

        int64_t c1 = channels[0], c2 = channels[1], c3 = channels[2];

        // 上采样路径（l2/l3 → l1）
        l2_to_l1 = register_module("l2_to_l1", UpsampleConv(c2, c1, 2));
        l3_to_l1 = register_module("l3_to_l1", UpsampleConv(c3, c1, 4));

        // l1 → l2：depthwise stride=2
        l1_to_l2 = register_module("l1_to_l2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c1, c2, 3)
                .stride(2)
                .padding(1)
                .groups(c1)          
                .bias(false)));      

        l3_to_l2 = register_module("l3_to_l2", UpsampleConv(c3, c2, 2));

        // l1 → l3：depthwise stride=4，使用 kernel=5 pad=2 实现更大感受野（论文关键设计）
        l1_to_l3 = register_module("l1_to_l3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c1, c3, 5)
                .stride(4)
                .padding(2)
                .groups(c1)
                .bias(false)));

        // l2 → l3：depthwise stride=2
        l2_to_l3 = register_module("l2_to_l3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(c2, c3, 3)
                .stride(2)
                .padding(1)
                .groups(c2)
                .bias(false)));
    }

    // 返回三个增强后的特征图 (out1, out2, out3)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor layer1_x, torch::Tensor layer2_x, torch::Tensor layer3_x) {
        // out1 (layer1 scale)
        torch::Tensor out1 = layer1_x
                           + l2_to_l1->forward(layer2_x)
                           + l3_to_l1->forward(layer3_x);

        // out2 (layer2 scale)
        torch::Tensor out2 = layer2_x
                           + l1_to_l2->forward(layer1_x)
                           + l3_to_l2->forward(layer3_x);

        // out3 (layer3 scale)
        torch::Tensor out3 = layer3_x
                           + l1_to_l3->forward(layer1_x)
                           + l2_to_l3->forward(layer2_x);

        return {out1, out2, out3};
    }
};
TORCH_MODULE(MultiScaleFusion);

#ifdef _MSC_VER
#pragma warning(pop)
#endif