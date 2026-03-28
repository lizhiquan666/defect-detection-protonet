#pragma once

#include <torch/torch.h>
#include <vector>
#include <stdexcept>
#include <map>
#include <string>
#include <algorithm>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

// ====================== ResidualBlock ======================
// 论文中 MultiSizeAttention 最后的
//增强残差块（标准 BasicBlock 结构）
// 完全 inline 实现，确保 holder 类能正确转发 forward
struct ResidualBlockImpl : public torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d norm1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d norm2{nullptr};
    torch::nn::Conv2d downsample{nullptr};
    torch::nn::ReLU relu{nullptr};

    ResidualBlockImpl(int64_t in_channels, int64_t channels) {
        // 假设 in_channels == channels（PRNet 中所有调用均如此）
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, channels, 3).padding(1)));
        norm1 = register_module("norm1", torch::nn::BatchNorm2d(channels));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3).padding(1)));
        norm2 = register_module("norm2", torch::nn::BatchNorm2d(channels));
        downsample = register_module("downsample", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, channels, 1)));  // 1x1 shortcut
        relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = downsample->forward(x);

        torch::Tensor out = conv1->forward(x);
        out = norm1->forward(out);
        out = relu->forward(out);

        out = conv2->forward(out);
        out = norm2->forward(out);

        out += identity;
        out = relu->forward(out);
        return out;
    }
};
TORCH_MODULE(ResidualBlock);  // 生成 holder 类 ResidualBlock，支持 ->forward()

// ====================== MultiSizeAttention ======================
// 单层多尺寸自注意力（论文 3.3 节核心）
// 关键点：
// - patch_sizes = {h/4, h/2, h}（整除时）
// - 每个尺度独立 Conv embedding + MultiheadAttention + Linear projection
// - 多尺度输出相加 + 主残差 + ResidualBlock 增强
struct MultiSizeAttentionImpl : public torch::nn::Module {
    std::vector<int64_t> patch_sizes;
    std::map<int64_t, torch::nn::Conv2d> to_embed;
    std::map<int64_t, torch::nn::MultiheadAttention> attn;
    std::map<int64_t, torch::nn::Linear> to_patch;
    ResidualBlock residual{nullptr};

    int64_t embed_dim;
    int64_t num_heads = 8;  // 通道数 (128,256,512) 均可被 8 整除

    MultiSizeAttentionImpl(int64_t in_channels, int64_t height, int64_t width = -1) {
        if (width == -1) width = height;
        if (height != width) {
            throw std::runtime_error("MultiSizeAttention requires square feature maps (H == W)");
        }

        embed_dim = in_channels;  

        int64_t h = height;
        if (h % 4 == 0) patch_sizes.push_back(h / 4);
        if (h % 2 == 0 && (patch_sizes.empty() || patch_sizes.back() != h / 2)) {
            patch_sizes.push_back(h / 2);
        }
        if (patch_sizes.empty()) {
            patch_sizes.push_back(std::max<int64_t>(1, h));
        }

        for (auto ps : patch_sizes) {
            // Patch embedding: 非重叠 Conv (kernel=ps, stride=ps)
            to_embed.emplace(
                ps,
                register_module("to_embed_" + std::to_string(ps),
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, embed_dim, ps)
                        .stride(ps).bias(false))));

            // MultiheadAttention：C++ API 无 batch_first，手动 transpose 处理
            attn.emplace(
                ps,
                register_module("attn_" + std::to_string(ps),
                    torch::nn::MultiheadAttention(embed_dim, num_heads)));

            // Project back to patch volume (C × ps × ps)
            to_patch.emplace(
                ps,
                register_module("to_patch_" + std::to_string(ps),
                    torch::nn::Linear(embed_dim, in_channels * ps * ps)));
        }

        // 最后的 ResidualBlock 增强
        residual = register_module("residual", ResidualBlock(in_channels, in_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor input_residual = x;  // 主残差
        torch::Tensor out = torch::zeros_like(x);

        int64_t B = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);

        for (auto ps : patch_sizes) {
            if (H % ps != 0 || W % ps != 0) continue;  // 安全检查

            int64_t ph = H / ps;
            int64_t pw = W / ps;

            // 1. Patch embedding
            torch::Tensor embedded = to_embed.at(ps)->forward(x);  // (B, E, ph, pw)

            // 2. To tokens (B, ph*pw, E)
            torch::Tensor tokens = embedded.flatten(2).transpose(1, 2);  // (B, S, E)

            // 3. 转为 C++ API 要求的 (S, B, E)
            tokens = tokens.transpose(0, 1);  // (S, B, E)

            // 4. Self-attention
            auto [attn_out, _] = attn.at(ps)->forward(tokens, tokens, tokens);  // (S, B, E)

            // 5. 恢复为 (B, S, E)
            attn_out = attn_out.transpose(0, 1);  // (B, S, E)

            // 6. Project back
            torch::Tensor patch_vol = to_patch.at(ps)->forward(attn_out);  // (B, S, C*ps*ps)

            // 7. Unpatchify 还原特征图
            patch_vol = patch_vol.view({B, ph, pw, C, ps, ps});
            patch_vol = patch_vol.permute({0, 3, 1, 4, 2, 5}).contiguous();
            patch_vol = patch_vol.reshape({B, C, H, W});

            out += patch_vol;  // 多尺度相加
        }

        out += input_residual;           // 主残差连接
        out = residual->forward(out);    // 论文中额外卷积残差增强

        return out;
    }
};
TORCH_MODULE(MultiSizeAttention);

// ====================== MultiSizeAttentionModule ======================
// 多层堆叠（默认 3 层），对 layer1/2/3 特征独立处理
struct MultiSizeAttentionModuleImpl : public torch::nn::Module {
    torch::nn::Sequential layer1_msa{nullptr};
    torch::nn::Sequential layer2_msa{nullptr};
    torch::nn::Sequential layer3_msa{nullptr};

    MultiSizeAttentionModuleImpl(const std::vector<int64_t>& in_channels,
                                 const std::vector<int64_t>& heights,
                                 const std::vector<int64_t>& widths = {},
                                 int64_t num_layers = 3) {
        if (in_channels.size() != 3 || heights.size() != 3)
            throw std::runtime_error("MultiSizeAttentionModule expects 3 layers");

        std::vector<int64_t> w = widths.empty() ? heights : widths;

        layer1_msa = register_module("layer1_msa", torch::nn::Sequential());
        layer2_msa = register_module("layer2_msa", torch::nn::Sequential());
        layer3_msa = register_module("layer3_msa", torch::nn::Sequential());

        for (int i = 0; i < num_layers; ++i) {
            layer1_msa->push_back(MultiSizeAttention(in_channels[0], heights[0], w[0]));
            layer2_msa->push_back(MultiSizeAttention(in_channels[1], heights[1], w[1]));
            layer3_msa->push_back(MultiSizeAttention(in_channels[2], heights[2], w[2]));
        }
    }

    std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& features) {
        if (features.size() != 3)
            throw std::runtime_error("Expected 3 feature maps");

        torch::Tensor f1 = features[0];
        torch::Tensor f2 = features[1];
        torch::Tensor f3 = features[2];

        // Sequential 会自动逐层调用 forward
        f1 = layer1_msa->forward(f1);
        f2 = layer2_msa->forward(f2);
        f3 = layer3_msa->forward(f3);

        return {f1, f2, f3};
    }
};
TORCH_MODULE(MultiSizeAttentionModule);

#ifdef _MSC_VER
#pragma warning(pop)
#endif