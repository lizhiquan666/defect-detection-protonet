#pragma warning(disable:4267)   
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <limits>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "attention.hpp"
#include "dataset.hpp"
#include "losses.hpp"
#include "modules.hpp"
#include "prnet.hpp"
#include "prototype.hpp"
#include "utils.hpp"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    auto find_project_root = [](const fs::path& start) {
        std::error_code ec;
        fs::path cur = start;
        while (!cur.empty()) {
            if (fs::exists(cur / "CMakeLists.txt", ec)) {
                return cur;
            }
            fs::path parent = cur.parent_path();
            if (parent == cur) break;
            cur = parent;
        }
        return start;
    };

    std::error_code ec;
    fs::path exe_path = fs::absolute(fs::path(argv[0]), ec);
    if (ec) exe_path = fs::path(argv[0]);
    fs::path exe_dir = exe_path.parent_path();
    fs::path project_root = find_project_root(exe_dir);

    // Disable OpenCV parallel backend to avoid DLL loading crash
    cv::setNumThreads(1);
    cv::setUseOptimized(false);

    std::cout << "=== Qizhi Cup PRNet Demo (LibTorch 2.10.0 Debug + OpenCV 4.12.0) ===\n";

    std::string input_bmp = "";
    std::string val_root = "";
    std::string train_root = "";
    std::string proto_root = (project_root / "prototypes").string();
    std::string output_root = (project_root / "res").string();
    std::string class_name = "1";
    std::string weights_path = "";
    RBoxPostprocessOptions post_opt;
    bool train_mode = false;
    bool show_help = false;
    int epochs = 10;
    int batch_size = 4;
    bool batch_specified = false;
    double lr = 1e-4;
    int num_anomalies = 64;
    int steps_per_epoch = 0;
    std::string save_path = "";
    std::string device_arg = "auto";
    bool dynamic_proto = true;
    float proto_momentum = 0.05f;
    float val_ratio = 0.2f;
    float val_thresh = 0.5f;
    int early_stop_patience = 50;
    int min_epochs = 5;
    bool perlin_enable = false;
    float perlin_prob = 0.0f;
    int perlin_grid = 8;
    float perlin_thresh = 0.58f;
    float perlin_noise_std = 0.35f;
    bool typed_aug_enable = false;
    float scratch_prob = 0.0f;
    float missing_print_prob = 0.0f;
    float blob_prob = 0.0f;
    float scratch_strength = 0.20f;
    float missing_print_alpha = 0.70f;
    float blob_noise_std = 0.18f;

    auto print_usage = [&]() {
        std::cout << "Usage:\n"
                  << "  defect_detect --train --train_root <data_root> --class <class_name> [options]\n"
                  << "  defect_detect -i <image.bmp> [options]\n\n"
                  << "Common options:\n"
                  << "  --proto <dir>           Prototype root (default: prototypes)\n"
                  << "  --class <name>          Class name (default: 1)\n"
                  << "  --out <dir>             Output directory (default: res)\n"
                  << "  --help, -h              Show this help\n\n"
                  << "Training options:\n"
                  << "  --train                 Enable training mode\n"
                  << "  --train_root <dir>      Training dataset root\n"
                  << "  --epochs <int>          Epochs (default: 10)\n"
                  << "  --batch <int>           Batch size (default: 4)\n"
                  << "  --lr <float>            Learning rate (default: 1e-4)\n"
                  << "  --num_anomalies <int>   Number of NG samples to load (default: 64)\n"
                  << "  --steps_per_epoch <int> Batches per epoch (default: auto from train split)\n"
                  << "  --dynamic_proto <0|1>   Enable EMA prototype update (default: 1)\n"
                  << "  --proto_momentum <f>    Prototype EMA momentum in [0,1] (default: 0.05)\n"
                  << "  --val_ratio <float>     Validation split ratio in [0,0.5] (default: 0.2)\n"
                  << "  --val_thresh <float>    Validation pixel threshold (default: 0.5)\n"
                  << "  --patience <int>        Early stop patience (default: 50)\n"
                  << "  --min_epochs <int>      Minimum epochs before early stop (default: 5)\n"
                  << "  --perlin <0|1>          Enable Perlin synthetic anomaly (default: 0)\n"
                  << "  --perlin_prob <float>   Perlin apply probability [0,1] (default: 0.0)\n"
                  << "  --perlin_grid <int>     Perlin coarse grid size (default: 8)\n"
                  << "  --perlin_thresh <float> Perlin mask threshold [0,1] (default: 0.58)\n"
                  << "  --perlin_std <float>    Perlin noise std in normalized space (default: 0.35)\n"
                  << "  --typed_aug <0|1>       Enable type-directed augmentations (default: 0)\n"
                  << "  --scratch_prob <float>  Scratch augmentation probability weight (default: 0.0)\n"
                  << "  --missing_prob <float>  Missing-print augmentation probability weight (default: 0.0)\n"
                  << "  --blob_prob <float>     Blob-dirt augmentation probability weight (default: 0.0)\n"
                  << "  --scratch_strength <f>  Scratch perturbation strength (default: 0.20)\n"
                  << "  --missing_alpha <f>     Missing-print erase alpha (default: 0.70)\n"
                  << "  --blob_std <f>          Blob perturbation std (default: 0.18)\n"
                  << "  --device <auto|cuda|cpu> Device for train/infer (default: auto)\n"
                  << "  --weights <path>        Model weights to load (in infer mode, default: prototypes/<class>/prnet_semisup_weights.pt if exists)\n"
                  << "  --save <path>           Output weight path (default: prototypes/<class>/prnet_semisup_weights.pt)\n\n"
                  << "Inference options:\n"
                  << "  -i <image.bmp>          Single-image inference\n"
                  << "  --thresh <float>        Heatmap threshold (default: 0.5)\n"
                  << "  --min_score <float>     Min box mean score in [0,1] (default: 0.0)\n"
                  << "  --min_area <float>      Min RBox area in image px (default: 100)\n"
                  << "  --max_area <float>      Max RBox area in image px (default: 1e9)\n"
                  << "  --min_aspect <float>    Min long/short ratio (default: 1.0)\n"
                  << "  --max_aspect <float>    Max long/short ratio (default: 8.0)\n"
                  << "  --min_side <float>      Min short side in image px (default: 8)\n"
                  << "  --edge_margin <float>   Edge prior ratio in [0,0.45] (default: 0.0)\n"
                  << "  --ignore_tl_w <float>   Ignore top-left width ratio in [0,1] (default: 0.0)\n"
                  << "  --ignore_tl_h <float>   Ignore top-left height ratio in [0,1] (default: 0.0)\n"
                  << "  --ignore_x0 <float>     Ignore-rect x0 ratio in [0,1] (default: 0.0)\n"
                  << "  --ignore_y0 <float>     Ignore-rect y0 ratio in [0,1] (default: 0.0)\n"
                  << "  --ignore_x1 <float>     Ignore-rect x1 ratio in [0,1] (default: 0.0)\n"
                  << "  --ignore_y1 <float>     Ignore-rect y1 ratio in [0,1] (default: 0.0)\n"
                  << "  --nms_iou <float>       Rotated NMS IoU threshold (default: 0.2)\n"
                  << "  --morph_k <int>         Morphology kernel size (default: 3)\n"
                  << "  --close_iter <int>      Morph close iterations (default: 1)\n"
                  << "  --open_iter <int>       Morph open iterations (default: 1)\n";
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train") train_mode = true;
        else if (arg == "--help" || arg == "-h") show_help = true;
        else if (arg == "-i" && i + 1 < argc)          input_bmp = argv[++i];
        else if (arg == "--val_root" && i + 1 < argc) val_root = argv[++i];
        else if (arg == "--train_root" && i + 1 < argc) train_root = argv[++i];
        else if (arg == "--proto" && i + 1 < argc)   proto_root = argv[++i];
        else if (arg == "--out" && i + 1 < argc)     output_root = argv[++i];
        else if (arg == "--class" && i + 1 < argc)   class_name = argv[++i];
        else if (arg == "--thresh" && i + 1 < argc)  post_opt.thresh = std::stof(argv[++i]);
        else if (arg == "--min_score" && i + 1 < argc) post_opt.min_score = std::stof(argv[++i]);
        else if (arg == "--min_area" && i + 1 < argc) post_opt.min_area = std::stof(argv[++i]);
        else if (arg == "--max_area" && i + 1 < argc) post_opt.max_area = std::stof(argv[++i]);
        else if (arg == "--min_aspect" && i + 1 < argc) post_opt.min_aspect = std::stof(argv[++i]);
        else if (arg == "--max_aspect" && i + 1 < argc) post_opt.max_aspect = std::stof(argv[++i]);
        else if (arg == "--min_side" && i + 1 < argc) post_opt.min_side = std::stof(argv[++i]);
        else if (arg == "--edge_margin" && i + 1 < argc) post_opt.edge_margin_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_tl_w" && i + 1 < argc) post_opt.ignore_tl_width_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_tl_h" && i + 1 < argc) post_opt.ignore_tl_height_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_x0" && i + 1 < argc) post_opt.ignore_rect_x0_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_y0" && i + 1 < argc) post_opt.ignore_rect_y0_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_x1" && i + 1 < argc) post_opt.ignore_rect_x1_ratio = std::stof(argv[++i]);
        else if (arg == "--ignore_y1" && i + 1 < argc) post_opt.ignore_rect_y1_ratio = std::stof(argv[++i]);
        else if (arg == "--nms_iou" && i + 1 < argc) post_opt.nms_iou_thresh = std::stof(argv[++i]);
        else if (arg == "--morph_k" && i + 1 < argc) post_opt.morph_kernel = std::stoi(argv[++i]);
        else if (arg == "--close_iter" && i + 1 < argc) post_opt.morph_close_iter = std::stoi(argv[++i]);
        else if (arg == "--open_iter" && i + 1 < argc) post_opt.morph_open_iter = std::stoi(argv[++i]);
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::max(1, std::stoi(argv[++i]));
            batch_specified = true;
        }
        else if (arg == "--lr" && i + 1 < argc) lr = std::stod(argv[++i]);
        else if (arg == "--num_anomalies" && i + 1 < argc) num_anomalies = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--steps_per_epoch" && i + 1 < argc) steps_per_epoch = std::max(0, std::stoi(argv[++i]));
        else if (arg == "--dynamic_proto" && i + 1 < argc) dynamic_proto = (std::stoi(argv[++i]) != 0);
        else if (arg == "--proto_momentum" && i + 1 < argc) proto_momentum = std::stof(argv[++i]);
        else if (arg == "--val_ratio" && i + 1 < argc) val_ratio = std::stof(argv[++i]);
        else if (arg == "--val_thresh" && i + 1 < argc) val_thresh = std::stof(argv[++i]);
        else if (arg == "--patience" && i + 1 < argc) early_stop_patience = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--min_epochs" && i + 1 < argc) min_epochs = std::max(1, std::stoi(argv[++i]));
        else if (arg == "--perlin" && i + 1 < argc) perlin_enable = (std::stoi(argv[++i]) != 0);
        else if (arg == "--perlin_prob" && i + 1 < argc) perlin_prob = std::stof(argv[++i]);
        else if (arg == "--perlin_grid" && i + 1 < argc) perlin_grid = std::max(2, std::stoi(argv[++i]));
        else if (arg == "--perlin_thresh" && i + 1 < argc) perlin_thresh = std::stof(argv[++i]);
        else if (arg == "--perlin_std" && i + 1 < argc) perlin_noise_std = std::stof(argv[++i]);
        else if (arg == "--typed_aug" && i + 1 < argc) typed_aug_enable = (std::stoi(argv[++i]) != 0);
        else if (arg == "--scratch_prob" && i + 1 < argc) scratch_prob = std::stof(argv[++i]);
        else if (arg == "--missing_prob" && i + 1 < argc) missing_print_prob = std::stof(argv[++i]);
        else if (arg == "--blob_prob" && i + 1 < argc) blob_prob = std::stof(argv[++i]);
        else if (arg == "--scratch_strength" && i + 1 < argc) scratch_strength = std::stof(argv[++i]);
        else if (arg == "--missing_alpha" && i + 1 < argc) missing_print_alpha = std::stof(argv[++i]);
        else if (arg == "--blob_std" && i + 1 < argc) blob_noise_std = std::stof(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_arg = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--save" && i + 1 < argc) save_path = argv[++i];
        else {
            std::cerr << "ERROR: Unknown or incomplete argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (argc <= 1 || show_help) {
        print_usage();
        return 0;
    }

    std::transform(device_arg.begin(), device_arg.end(), device_arg.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const bool cuda_available = torch::cuda::is_available();
    torch::Device device = torch::kCPU;

    if (device_arg == "auto") {
        device = cuda_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    } else if (device_arg == "cuda") {
        if (!cuda_available) {
            std::cerr << "ERROR: --device cuda requested but CUDA is not available in current LibTorch/runtime.\n";
            std::cerr << "Hint: current workspace appears to use a +cpu LibTorch package; switch to CUDA LibTorch build.\n";
            return 1;
        }
        device = torch::Device(torch::kCUDA, 0);
    } else if (device_arg == "cpu") {
        device = torch::kCPU;
    } else {
        std::cerr << "ERROR: invalid --device value: " << device_arg << " (expected auto|cuda|cpu)\n";
        return 1;
    }

    std::cout << "[DEBUG] CUDA available: " << (cuda_available ? "YES" : "NO") << "\n";
    std::cout << "[DEBUG] Using device: " << (device.is_cuda() ? "CUDA:0" : "CPU") << "\n";

    if (train_mode && device.is_cuda() && !batch_specified && batch_size > 1) {
        batch_size = 1;
        std::cout << "[TRAIN] CUDA mode detected, auto-adjust batch size to 1 for stability."
                  << " Use --batch to override.\n";
    }

    prnet::PRNetImpl::Options opt;
    opt.encoder_ts_path = (project_root / "resnet18_encoder_ts.pt").string();

    prnet::PRNet model = nullptr;
    try {
        std::cout << "[DEBUG] Initializing PRNet ...\n";
        model = prnet::PRNet(opt);

        fs::path resolved_weights_path;
        bool should_load_weights = false;
        if (!weights_path.empty()) {
            resolved_weights_path = fs::path(weights_path);
            if (resolved_weights_path.is_relative()) {
                resolved_weights_path = project_root / resolved_weights_path;
            }
            should_load_weights = true;
        } else if (!train_mode) {
            fs::path default_weights = fs::path(proto_root) / class_name / "prnet_semisup_weights.pt";
            if (fs::exists(default_weights)) {
                resolved_weights_path = default_weights;
                should_load_weights = true;
            }
        }

        if (should_load_weights) {
            if (!fs::exists(resolved_weights_path)) {
                std::cerr << "ERROR: weight file not found: " << resolved_weights_path.string() << "\n";
                return 1;
            }
            torch::serialize::InputArchive in_archive;
            in_archive.load_from(resolved_weights_path.string(), c10::Device(c10::kCPU));
            model->load(in_archive);
            std::cout << "[DEBUG] Loaded model weights: " << resolved_weights_path.string() << "\n";
        } else if (!train_mode) {
            std::cout << "[DEBUG] No semisup weights found, using randomly initialized decoder/fusion weights.\n";
        }

        std::cout << "[DEBUG] Moving model to device ...\n";
        model->to(device);
        model->eval();
        std::cout << "[DEBUG] PRNet model loaded\n";
    } catch (const c10::Error& e) {
        std::cerr << "[MODEL][C10 ERROR] " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[MODEL][EXCEPTION] " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[MODEL][UNKNOWN ERROR] Unexpected exception during model initialization.\n";
        return 1;
    }

    // Load prototype (your generated file)
    std::vector<torch::Tensor> prototypes;
    try {
        std::string proto_path = (fs::path(proto_root) / class_name / "prototype_generator.pt").string();
        torch::load(prototypes, proto_path);
        for (auto& p : prototypes) p = p.to(device);
        std::cout << "[DEBUG] Prototype loaded successfully: " << proto_path << "\n";
    } catch (...) {
        std::cerr << "ERROR: Cannot load prototype! Check path: prototypes/" << class_name << "/prototype_generator.pt\n";
        return 1;
    }

    if (train_mode) {
        try {
            std::string train_data_root = train_root.empty() ? val_root : train_root;
            if (train_data_root.empty()) {
                std::cerr << "ERROR: train mode requires --train_root (or fallback --val_root).\n";
                return 1;
            }

            std::cout << "[TRAIN] Semi-supervised training mode enabled\n";
            std::cout << "[TRAIN] data_root=" << train_data_root
                      << " class=" << class_name
                      << " epochs=" << epochs
                      << " batch=" << batch_size
                      << " lr=" << lr
                      << " num_anomalies=" << num_anomalies
                      << " dynamic_proto=" << (dynamic_proto ? "1" : "0")
                      << " proto_momentum=" << proto_momentum
                      << " val_ratio=" << val_ratio
                      << " patience=" << early_stop_patience << "\n";

            val_ratio = std::clamp(val_ratio, 0.0f, 0.5f);
            val_thresh = std::clamp(val_thresh, 0.0f, 1.0f);
            proto_momentum = std::clamp(proto_momentum, 0.0f, 1.0f);
            perlin_prob = std::clamp(perlin_prob, 0.0f, 1.0f);
            perlin_thresh = std::clamp(perlin_thresh, 0.0f, 1.0f);
            perlin_noise_std = std::max(0.0f, perlin_noise_std);
            scratch_prob = std::max(0.0f, scratch_prob);
            missing_print_prob = std::max(0.0f, missing_print_prob);
            blob_prob = std::max(0.0f, blob_prob);
            scratch_strength = std::max(0.0f, scratch_strength);
            missing_print_alpha = std::clamp(missing_print_alpha, 0.0f, 0.98f);
            blob_noise_std = std::max(0.0f, blob_noise_std);

            prnet::DatasetConfig ds_cfg;
            ds_cfg.img_size = 256;
            ds_cfg.crp_size = 256;
            ds_cfg.msk_size = 256;
            ds_cfg.msk_crp_size = 256;
            ds_cfg.perlin_enable = perlin_enable;
            ds_cfg.perlin_prob = perlin_prob;
            ds_cfg.perlin_grid = perlin_grid;
            ds_cfg.perlin_thresh = perlin_thresh;
            ds_cfg.perlin_noise_std = perlin_noise_std;
            ds_cfg.typed_aug_enable = typed_aug_enable;
            ds_cfg.scratch_prob = scratch_prob;
            ds_cfg.missing_print_prob = missing_print_prob;
            ds_cfg.blob_prob = blob_prob;
            ds_cfg.scratch_strength = scratch_strength;
            ds_cfg.missing_print_alpha = missing_print_alpha;
            ds_cfg.blob_noise_std = blob_noise_std;

            prnet::TRAINMVTEC train_ds(train_data_root, class_name, true, ds_cfg, num_anomalies);
            const size_t n_samples = train_ds.size();
            if (n_samples == 0) {
                std::cerr << "ERROR: Training dataset is empty.\n";
                return 1;
            }

            model->train();
            torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));

            const size_t n_normal = train_ds.normal_size();
            const size_t n_anomaly = train_ds.anomaly_size();
            if (n_normal == 0) {
                std::cerr << "ERROR: No normal samples found, cannot train PRNet.\n";
                return 1;
            }

            std::vector<int64_t> normal_idx_all;
            std::vector<int64_t> anomaly_idx_all;
            normal_idx_all.reserve(n_normal);
            anomaly_idx_all.reserve(n_anomaly);
            for (size_t i = 0; i < n_normal; ++i) normal_idx_all.push_back(static_cast<int64_t>(i));
            for (size_t i = 0; i < n_anomaly; ++i) anomaly_idx_all.push_back(static_cast<int64_t>(n_normal + i));

            std::mt19937 split_rng(123);
            std::shuffle(normal_idx_all.begin(), normal_idx_all.end(), split_rng);
            std::shuffle(anomaly_idx_all.begin(), anomaly_idx_all.end(), split_rng);

            const size_t val_n_normal = std::min(normal_idx_all.size(), static_cast<size_t>(std::round(normal_idx_all.size() * val_ratio)));
            const size_t val_n_anomaly = std::min(anomaly_idx_all.size(), static_cast<size_t>(std::round(anomaly_idx_all.size() * val_ratio)));

            std::vector<int64_t> val_idx;
            std::vector<int64_t> train_normal_idx;
            std::vector<int64_t> train_anomaly_idx;

            val_idx.insert(val_idx.end(), normal_idx_all.begin(), normal_idx_all.begin() + static_cast<std::ptrdiff_t>(val_n_normal));
            train_normal_idx.insert(train_normal_idx.end(), normal_idx_all.begin() + static_cast<std::ptrdiff_t>(val_n_normal), normal_idx_all.end());

            val_idx.insert(val_idx.end(), anomaly_idx_all.begin(), anomaly_idx_all.begin() + static_cast<std::ptrdiff_t>(val_n_anomaly));
            train_anomaly_idx.insert(train_anomaly_idx.end(), anomaly_idx_all.begin() + static_cast<std::ptrdiff_t>(val_n_anomaly), anomaly_idx_all.end());

            if (train_normal_idx.empty()) {
                train_normal_idx = normal_idx_all;
            }
            if (n_anomaly > 0 && train_anomaly_idx.empty()) {
                train_anomaly_idx = anomaly_idx_all;
            }

            int64_t effective_steps = steps_per_epoch > 0
                                      ? static_cast<int64_t>(steps_per_epoch)
                                      : std::max<int64_t>(1, static_cast<int64_t>(std::ceil(static_cast<double>(train_normal_idx.size() + train_anomaly_idx.size()) /
                                                                                              static_cast<double>(std::max(batch_size, 1)))));

            prnet::BalancedBatchSampler::Config sampler_cfg;
            sampler_cfg.batch_size = batch_size;
            sampler_cfg.steps_per_epoch = effective_steps;
            sampler_cfg.num_anomalies = train_anomaly_idx.empty() ? 0 : static_cast<int64_t>(num_anomalies);

            prnet::BalancedBatchSampler sampler(sampler_cfg, train_normal_idx, train_anomaly_idx);

            auto score_from_logits = [](const torch::Tensor& logits) {
                if (logits.dim() == 4 && logits.size(1) == 2) {
                    return torch::softmax(logits, 1).slice(1, 1, 2);
                }
                return torch::sigmoid(logits);
            };

            auto update_prototypes_ema = [&](const torch::Tensor& images) {
                if (!dynamic_proto || proto_momentum <= 0.0f) {
                    return;
                }

                torch::NoGradGuard ng;
                auto feats_all = model->extract_encoder_features(images, true);
                if (feats_all.size() < 3 || prototypes.size() < 3) {
                    return;
                }

                for (size_t li = 0; li < 3; ++li) {
                    auto fi = feats_all[li].detach(); // (B,C,H,W)
                    auto pi = prototypes[li];         // (K,C,H,W)
                    if (fi.numel() == 0 || pi.numel() == 0) {
                        continue;
                    }

                    auto fi_flat = fi.view({fi.size(0), -1});
                    auto pi_flat = pi.view({pi.size(0), -1});

                    auto dist = torch::cdist(fi_flat, pi_flat, 2.0);
                    auto assign = std::get<1>(dist.min(1));

                    auto updated = pi.clone();
                    for (int64_t k = 0; k < pi.size(0); ++k) {
                        auto mask = (assign == k);
                        if (mask.sum().item<int64_t>() == 0) {
                            continue;
                        }
                        auto idx = torch::nonzero(mask).squeeze(1);
                        auto mean_feat = fi_flat.index_select(0, idx).mean(0).view_as(pi[k]);
                        updated[k] = (1.0f - proto_momentum) * pi[k] + proto_momentum * mean_feat;
                    }
                    prototypes[li] = updated;
                }
            };

            auto eval_val_f1 = [&](double& out_val_loss) {
                out_val_loss = 0.0;
                if (val_idx.empty()) {
                    return 0.0;
                }

                torch::NoGradGuard ng;
                model->eval();

                double tp = 0.0, fp = 0.0, fn = 0.0;
                int64_t n_batches = 0;

                for (size_t start = 0; start < val_idx.size(); start += static_cast<size_t>(batch_size)) {
                    const size_t end = std::min(start + static_cast<size_t>(batch_size), val_idx.size());

                    std::vector<torch::Tensor> images;
                    std::vector<torch::Tensor> masks;
                    images.reserve(end - start);
                    masks.reserve(end - start);

                    for (size_t j = start; j < end; ++j) {
                        auto s = train_ds.get(static_cast<size_t>(val_idx[j]));
                        images.push_back(s.image);
                        masks.push_back(s.mask);
                    }

                    auto batch_images = torch::stack(images, 0).to(device);
                    auto batch_masks = torch::stack(masks, 0).to(device);

                    auto logits = model->forward(batch_images, prototypes, true);
                    auto val_loss = prnet::compute_mask_supervision_loss(logits, batch_masks, 0.5);
                    out_val_loss += val_loss.total.item<double>();

                    auto score = score_from_logits(logits);
                    auto pred_bin = (score > val_thresh).to(torch::kFloat32);
                    auto gt_bin = (batch_masks > 0.5).to(torch::kFloat32);

                    tp += (pred_bin * gt_bin).sum().item<double>();
                    fp += (pred_bin * (1.0 - gt_bin)).sum().item<double>();
                    fn += ((1.0 - pred_bin) * gt_bin).sum().item<double>();
                    ++n_batches;
                }

                out_val_loss = out_val_loss / static_cast<double>(std::max<int64_t>(n_batches, 1));
                model->train();

                const double denom = 2.0 * tp + fp + fn + 1e-8;
                return denom > 0.0 ? (2.0 * tp / denom) : 0.0;
            };

            fs::path best_ckpt_path = fs::path(proto_root) / class_name / "prnet_semisup_weights_best_tmp.pt";
            fs::create_directories(best_ckpt_path.parent_path());

            double best_val_f1 = -std::numeric_limits<double>::infinity();
            int no_improve_epochs = 0;

            std::cout << "[TRAIN] split: train_normal=" << train_normal_idx.size()
                      << " train_anomaly=" << train_anomaly_idx.size()
                      << " val=" << val_idx.size()
                      << " steps_per_epoch=" << effective_steps
                      << " dynamic_proto=" << (dynamic_proto ? "ON" : "OFF")
                      << " perlin=" << (perlin_enable ? "ON" : "OFF")
                      << " perlin_prob=" << perlin_prob
                      << "\n";

            for (int ep = 0; ep < epochs; ++ep) {
                double epoch_total = 0.0;
                double epoch_focal = 0.0;
                double epoch_smooth = 0.0;

                for (int64_t step = 0; step < sampler.length(); ++step) {
                    std::vector<int64_t> batch_idx = sampler.next();
                    if (batch_idx.empty()) {
                        continue;
                    }

                    std::vector<torch::Tensor> images;
                    std::vector<torch::Tensor> masks;
                    images.reserve(batch_idx.size());
                    masks.reserve(batch_idx.size());

                    for (int64_t idx : batch_idx) {
                        auto s = train_ds.get(static_cast<size_t>(idx));
                        images.push_back(s.image);
                        masks.push_back(s.mask);
                    }

                    auto batch_images = torch::stack(images, 0).to(device);
                    auto batch_masks = torch::stack(masks, 0).to(device);

                    auto pred = model->forward(batch_images, prototypes, true);
                    auto loss_pack = prnet::compute_mask_supervision_loss(pred, batch_masks, 0.5);

                    optimizer.zero_grad();
                    loss_pack.total.backward();
                    optimizer.step();

                    // Memory-bank style dynamic prototype update using current normal-like features.
                    update_prototypes_ema(batch_images);

                    epoch_total += loss_pack.total.item<double>();
                    epoch_focal += loss_pack.focal.item<double>();
                    epoch_smooth += loss_pack.smooth_l1.item<double>();
                }

                const double denom = static_cast<double>(std::max<int64_t>(sampler.length(), 1));
                double val_loss = 0.0;
                const double val_f1 = eval_val_f1(val_loss);

                std::cout << "[TRAIN] Epoch " << (ep + 1) << "/" << epochs
                          << " total=" << (epoch_total / denom)
                          << " focal=" << (epoch_focal / denom)
                          << " smooth_l1=" << (epoch_smooth / denom)
                          << " val_loss=" << val_loss
                          << " val_f1=" << val_f1
                          << "\n";

                bool improved = val_f1 > best_val_f1;
                if (improved) {
                    best_val_f1 = val_f1;
                    no_improve_epochs = 0;
                    torch::serialize::OutputArchive best_archive;
                    model->save(best_archive);
                    best_archive.save_to(best_ckpt_path.string());
                } else {
                    ++no_improve_epochs;
                }

                if ((ep + 1) >= min_epochs && no_improve_epochs >= early_stop_patience) {
                    std::cout << "[TRAIN] Early stopping at epoch " << (ep + 1)
                              << " (best val_f1=" << best_val_f1 << ")\n";
                    break;
                }
            }

            if (fs::exists(best_ckpt_path)) {
                torch::serialize::InputArchive best_in;
                best_in.load_from(best_ckpt_path.string(), c10::Device(c10::kCPU));
                model->load(best_in);
                std::cout << "[TRAIN] Loaded best checkpoint (val_f1=" << best_val_f1 << ")\n";
            }

            fs::path out_model_path;
            if (!save_path.empty()) {
                out_model_path = fs::path(save_path);
            } else {
                out_model_path = fs::path(proto_root) / class_name / "prnet_semisup_weights.pt";
            }
            fs::create_directories(out_model_path.parent_path());

            try {
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to(out_model_path.string());
                std::cout << "[TRAIN] Saved trained weights: " << out_model_path.string() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "ERROR: Failed to save trained weights: " << e.what() << "\n";
                return 1;
            }

            std::cout << "[TRAIN] Training finished.\n";
            return 0;
        } catch (const c10::Error& e) {
            std::cerr << "[TRAIN][C10 ERROR] " << e.what() << "\n";
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "[TRAIN][EXCEPTION] " << e.what() << "\n";
            return 1;
        } catch (...) {
            std::cerr << "[TRAIN][UNKNOWN ERROR] Unexpected exception during training.\n";
            return 1;
        }
    }

    if (!input_bmp.empty()) {
        std::cout << "[DEBUG] Single image mode: " << input_bmp << "\n";

        cv::Mat orig = cv::imread(input_bmp);
        if (orig.empty()) {
            std::cerr << "ERROR: Cannot read image!\n";
            return 1;
        }
        std::cout << "[DEBUG] Image read OK, size = " << orig.rows << " x " << orig.cols << "\n";

        cv::Mat resized;
        cv::resize(orig, resized, cv::Size(256, 256));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        auto tensor = torch::from_blob(rgb.data, {256, 256, 3}, torch::kByte)
                  .permute({2, 0, 1}).to(torch::kFloat32).div_(255.0f)
                  .unsqueeze(0);
        auto mean = torch::tensor({0.485f, 0.456f, 0.406f}, tensor.options()).view({1, 3, 1, 1});
        auto std = torch::tensor({0.229f, 0.224f, 0.225f}, tensor.options()).view({1, 3, 1, 1});
        tensor = ((tensor - mean) / std).to(device);

        std::cout << "[DEBUG] Input tensor shape: " << tensor.sizes() << "\n";

        try {
            std::cout << "[DEBUG] Starting model->forward() ...\n";
            auto anomaly_map = model->forward_heatmap(
                tensor,
                prototypes,
                prnet::PRNetImpl::HeatmapMode::ResidualPlusSupervised,
                false);
            std::cout << "[DEBUG] forward() finished. Output shape: " << anomaly_map.sizes() << "\n";

            fs::create_directories(output_root);
            std::cout << "[DEBUG] Starting post-processing and saving results ...\n";
            process_and_save_results(input_bmp, anomaly_map, output_root, post_opt);
            std::cout << "Single image processing finished!\n";
        } catch (const std::exception& e) {
            std::cerr << "RUNTIME EXCEPTION: " << e.what() << "\n";
        } catch (...) {
            std::cerr << "UNKNOWN RUNTIME ERROR occurred!\n";
        }

        return 0;
    }

    std::cerr << "Please use --train for training, or -i for single image inference.\n";
    std::cerr << "Use --help to see full options.\n";
    return 1;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif