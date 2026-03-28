#include "attention.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace prnet {

// ====================== 可视化单个特征图 ======================
void visualize_attention_feature(const torch::Tensor& tensor,
                                 const std::string& window_name,
                                 bool is_input) {
    // ==================== 强健性检查（新增）====================
    TORCH_CHECK(tensor.defined(), "visualize_attention_feature: tensor is undefined");
    TORCH_CHECK(tensor.dim() == 4,
                "visualize_attention_feature: expected 4D tensor (B,C,H,W), got dim=", tensor.dim());
    TORCH_CHECK(tensor.size(0) > 0, "visualize_attention_feature: batch size must > 0");
    TORCH_CHECK(tensor.size(1) > 0 && tensor.size(2) > 0 && tensor.size(3) > 0,
                "visualize_attention_feature: C, H, W must all > 0");

    // 取第一个样本的通道均值图
    torch::Tensor mean_channel = tensor[0].mean(0);  // (H, W)

    // 归一化到 [0,1]（防止 max==min 导致 NaN）
    auto min_val = mean_channel.min().item<float>();
    auto max_val = mean_channel.max().item<float>();
    torch::Tensor normalized = (mean_channel - min_val) / (max_val - min_val + 1e-8f);

    auto cpu_data = normalized.cpu().contiguous().data_ptr<float>();

    int H = static_cast<int>(mean_channel.size(0));
    int W = static_cast<int>(mean_channel.size(1));

    cv::Mat gray(H, W, CV_32F, cpu_data);

    // resize 到 256x256（可视化友好）
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);

    cv::Mat colored;
    resized.convertTo(resized, CV_8UC1, 255.0);
    cv::applyColorMap(resized, colored, cv::COLORMAP_JET);

    std::string label = is_input ? "Input" : "Output";
    cv::putText(colored, label + " (" + window_name + ")",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);

    cv::imshow(window_name, colored);
    cv::waitKey(1);  // 非阻塞
}

// ====================== 并排对比 Input | Output ======================
void visualize_attention_io(const torch::Tensor& input,
                            const torch::Tensor& output,
                            const std::string& window_name) {
    auto viz = [](const torch::Tensor& t) -> cv::Mat {
        TORCH_CHECK(t.defined() && t.dim() == 4 && t.size(0) > 0,
                    "viz lambda: invalid tensor (must be 4D with batch>0)");

        torch::Tensor mean_ch = t[0].mean(0);
        auto min_val = mean_ch.min().item<float>();
        auto max_val = mean_ch.max().item<float>();
        mean_ch = (mean_ch - min_val) / (max_val - min_val + 1e-8f);

        auto data = mean_ch.cpu().contiguous().data_ptr<float>();
        int H = static_cast<int>(mean_ch.size(0));
        int W = static_cast<int>(mean_ch.size(1));

        cv::Mat gray(H, W, CV_32F, data);
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(256, 256), 0, 0, cv::INTER_LINEAR);
        resized.convertTo(resized, CV_8UC1, 255.0);

        cv::Mat colored;
        cv::applyColorMap(resized, colored, cv::COLORMAP_JET);
        return colored;
    };

    cv::Mat in_viz = viz(input);
    cv::Mat out_viz = viz(output);

    cv::Mat side_by_side;
    cv::hconcat(in_viz, out_viz, side_by_side);

    cv::putText(side_by_side, "Input",  cv::Point(50, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    cv::putText(side_by_side, "Output", cv::Point(256 + 50, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    cv::putText(side_by_side, window_name, cv::Point(150, 280),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 2);

    cv::imshow(window_name + " [Input | Output]", side_by_side);
    cv::waitKey(1);
}

// ====================== 程序结束时销毁所有窗口 ======================
void destroy_all_windows() {
    try {
        cv::destroyAllWindows();
    } catch (...) {
        // 忽略任何可能的异常，确保主程序正常退出
    }
}

}
#ifdef _MSC_VER
#pragma warning(pop)
#endif