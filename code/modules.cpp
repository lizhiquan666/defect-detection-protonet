#include "modules.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

// ====================== OpenCV 可视化工具函数 ======================
// 功能：实时显示 MultiScaleFusion 输入/输出特征图（通道均值 + JET 伪彩色）
// - 用于调试：观察多尺度融合是否有效增强跨尺度异常一致性
// - 支持单层显示或三层拼接显示
// - 非阻塞（waitKey(1)），适合推理循环中插入
void visualize_fusion_layer(const torch::Tensor& tensor,
                            const std::string& layer_name) {
    torch::Tensor mean_ch = tensor[0].mean(0);  // (H, W)
    mean_ch = (mean_ch - mean_ch.min()) / (mean_ch.max() - mean_ch.min() + 1e-8);

    auto data = mean_ch.cpu().contiguous().data_ptr<float>();
    cv::Mat gray(mean_ch.sizes()[0], mean_ch.sizes()[1], CV_32F, data);
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(256, 256));

    cv::Mat colored;
    resized.convertTo(resized, CV_8UC1, 255.0);
    cv::applyColorMap(resized, colored, cv::COLORMAP_JET);

    cv::putText(colored, layer_name, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::imshow("Fusion " + layer_name, colored);
    cv::waitKey(1);
}

// 三层融合前后对比（推荐调试用）
void visualize_multi_scale_fusion(const std::vector<torch::Tensor>& inputs,
                                  const std::vector<torch::Tensor>& outputs,
                                  const std::string& stage_name) {
    auto viz = [](const torch::Tensor& t) -> cv::Mat {
        torch::Tensor mean_ch = t[0].mean(0);
        mean_ch = (mean_ch - mean_ch.min()) / (mean_ch.max() - mean_ch.min() + 1e-8);
        auto data = mean_ch.cpu().contiguous().data_ptr<float>();
        cv::Mat gray(mean_ch.sizes()[0], mean_ch.sizes()[1], CV_32F, data);
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(256, 256));
        resized.convertTo(resized, CV_8UC1, 255.0);
        cv::Mat colored;
        cv::applyColorMap(resized, colored, cv::COLORMAP_JET);
        return colored;
    };

    cv::Mat row1, row2;
    cv::hconcat(viz(inputs[0]), viz(inputs[1]), row1);
    cv::hconcat(row1, viz(inputs[2]), row1);

    cv::hconcat(viz(outputs[0]), viz(outputs[1]), row2);
    cv::hconcat(row2, viz(outputs[2]), row2);

    cv::Mat combined;
    cv::vconcat(row1, row2, combined);

    cv::putText(combined, "Input (L1 | L2 | L3)", cv::Point(50, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Output (L1 | L2 | L3)", cv::Point(50, 280 + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, stage_name, cv::Point(300, 550),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(200, 200, 200), 2);

    cv::imshow("MultiScaleFusion " + stage_name, combined);
    cv::waitKey(1);
}

// 可选：程序结束时销毁窗口
void destroy_fusion_windows() {
    cv::destroyAllWindows();
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif