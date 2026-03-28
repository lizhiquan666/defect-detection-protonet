#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// MSVC 专属警告抑制（Debug 版 LibTorch 会产生大量 4251/4275）
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#pragma warning(disable: 4275)
#endif

namespace fs = std::filesystem;

namespace {

// Default paths:
// - For portability, we DO NOT hard-code a developer-specific absolute path.
// - We try to auto-discover common relative locations from the current working directory.
// - You can override via CLI args or environment variables.
static std::string auto_find_mvtec_root_dir() {
  const fs::path cwd = fs::current_path();
  const std::vector<fs::path> candidates = {
      cwd / "data",
      cwd / "dataset",
      cwd / "res",
      cwd.parent_path() / "data",
      cwd.parent_path() / "dataset",
      cwd.parent_path() / "res",
      cwd.parent_path().parent_path() / "data",
      cwd.parent_path().parent_path() / "dataset",
      cwd.parent_path().parent_path() / "res",
  };

  for (const auto& p : candidates) {
    std::error_code ec;
    if (fs::exists(p, ec) && fs::is_directory(p, ec)) {
      return p.lexically_normal().string();
    }
  }

  // Fallback (may not exist): keeps output/help stable without using absolute paths.
    return fs::path("data").string();
}

  static fs::path find_in_cwd_ancestors(const fs::path& relative_path, int max_levels = 6) {
    fs::path base = fs::current_path();
    for (int i = 0; i <= max_levels; ++i) {
      const fs::path candidate = base / relative_path;
      std::error_code ec;
      if (fs::exists(candidate, ec)) {
        return fs::weakly_canonical(candidate);
      }
      if (!base.has_parent_path()) {
        break;
      }
      base = base.parent_path();
    }
    return {};
  }

  static fs::path resolve_readable_path(const std::string& input_path,
                                       const std::vector<fs::path>& fallbacks = {}) {
    if (!input_path.empty()) {
      const fs::path raw(input_path);
      std::error_code ec;

      if (raw.is_absolute()) {
        if (fs::exists(raw, ec)) {
          return fs::weakly_canonical(raw);
        }
        throw std::runtime_error("Path does not exist: " + raw.string());
      }

      if (fs::exists(raw, ec)) {
        return fs::weakly_canonical(raw);
      }

      auto found = find_in_cwd_ancestors(raw);
      if (!found.empty()) {
        return found;
      }
    }

    for (const auto& fallback : fallbacks) {
      std::error_code ec;
      if (fs::exists(fallback, ec)) {
        return fs::weakly_canonical(fallback);
      }

      if (fallback.is_relative()) {
        auto found = find_in_cwd_ancestors(fallback);
        if (!found.empty()) {
          return found;
        }
      }
    }

    std::string msg = "Unable to resolve encoder path";
    if (!input_path.empty()) {
      msg += " from input: " + input_path;
    }
    throw std::runtime_error(msg);
  }

const std::string kDefaultOutRootDir = "prototypes";

constexpr int kInputSize = 256;

// ImageNet normalization
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

struct Args {
  std::string class_name;
  std::string encoder_ts_path;
  std::string mvtec_root_dir;
  std::string out_root_dir;
  float ratio = 0.1f;
  int num_clusters = -1;
  int batch_size = 32;
};

static void print_usage(const char* argv0) {
  std::cout << "Usage:\n"
            << "  " << argv0
            << " --class <mvtec_class> [--ratio 0.1] [--K 50] [--encoder <path_to_resnet18_encoder_ts.pt>] [--batch 32]\n"
            << "                 [--data <mvtec_root_dir>] [--out <output_root_dir>]\n\n"
            << "Env override (optional):\n"
            << "  PRNET_MVTEC_ROOT   Override dataset root\n";
}

static Args parse_args(int argc, char** argv) {
  Args args;

  if (const char* env = std::getenv("PRNET_MVTEC_ROOT")) {
    if (std::string(env).size() > 0) {
      args.mvtec_root_dir = env;
    }
  }
  if (args.mvtec_root_dir.empty()) {
    args.mvtec_root_dir = auto_find_mvtec_root_dir();
  }

  args.out_root_dir = kDefaultOutRootDir;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need_value = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };

    if (a == "--class") {
      args.class_name = need_value("--class");
    } else if (a == "--ratio") {
      args.ratio = std::stof(need_value("--ratio"));
    } else if (a == "--K" || a == "--k") {
      args.num_clusters = std::stoi(need_value("--K"));
    } else if (a == "--encoder") {
      args.encoder_ts_path = need_value("--encoder");
    } else if (a == "--data") {
      args.mvtec_root_dir = need_value("--data");
    } else if (a == "--out") {
      args.out_root_dir = need_value("--out");
    } else if (a == "--batch") {
      args.batch_size = std::stoi(need_value("--batch"));
    } else if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }

  if (args.class_name.empty()) {
    throw std::runtime_error("--class is required");
  }
  if (args.num_clusters > 0) {
    // --K provided: ratio is ignored
  } else if (args.ratio <= 0.0f || args.ratio > 1.0f) {
    throw std::runtime_error("--ratio must be in (0, 1]");
  }
  if (args.num_clusters == 0) {
    throw std::runtime_error("--K must be > 0");
  }
  if (args.batch_size <= 0) {
    throw std::runtime_error("--batch must be > 0");
  }
  if (args.mvtec_root_dir.empty()) {
    throw std::runtime_error("--data (mvtec_root_dir) must be non-empty");
  }
  if (args.out_root_dir.empty()) {
    throw std::runtime_error("--out (output_root_dir) must be non-empty");
  }

  {
    const fs::path p = args.mvtec_root_dir;
    if (!fs::exists(p) || !fs::is_directory(p)) {
      throw std::runtime_error(
          "MVTec root directory not found: " + p.string() +
          "\nProvide it via --data <mvtec_root_dir> or set PRNET_MVTEC_ROOT environment variable.");
    }
  }
  return args;
}

static std::vector<fs::path> list_train_good_images(const std::string& mvtec_root_dir,
                                                    const std::string& class_name) {
  const fs::path root = fs::path(mvtec_root_dir);

  // 支持两种结构：
  // 1) MVTec: <root>/<class>/train/good
  // 2) 启智杯: <root>/<class>-ok 或 <root>/<class>_ok
  std::vector<fs::path> candidates = {
      root,
      root / class_name / "train" / "good",
      root / (class_name + "-ok"),
      root / (class_name + "_ok"),
      root / class_name / (class_name + "-ok"),
      root / class_name / (class_name + "_ok")};

  fs::path dir;
  for (const auto& c : candidates) {
    std::error_code ec;
    if (fs::exists(c, ec) && fs::is_directory(c, ec)) {
      dir = c;
      break;
    }
  }

  if (dir.empty()) {
    std::string msg = "Cannot find normal-image directory. Tried:\n";
    for (const auto& c : candidates) {
      msg += "  - " + c.string() + "\n";
    }
    throw std::runtime_error(msg);
  }

  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto p = entry.path();
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
      paths.push_back(p);
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

static torch::Tensor preprocess_to_tensor_chw(const fs::path& image_path) {
  cv::Mat bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + image_path.string());
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_LINEAR);

  cv::Mat rgb_f32;
  resized.convertTo(rgb_f32, CV_32FC3, 1.0 / 255.0);

  // Normalize in-place: (x - mean) / std
  // Channel order is RGB (same as PyTorch after ToTensor()).
  std::vector<cv::Mat> ch(3);
  cv::split(rgb_f32, ch);
  for (int c = 0; c < 3; ++c) {
    ch[c] = (ch[c] - kMean[c]) / kStd[c];
  }
  cv::merge(ch, rgb_f32);

  // HWC -> CHW tensor
  auto t = torch::from_blob(rgb_f32.ptr<float>(), {kInputSize, kInputSize, 3}, torch::kFloat32);
  t = t.permute({2, 0, 1}).contiguous();
  // Important: from_blob does not own memory.
  return t.clone();
}

static std::vector<torch::Tensor> forward_encoder(torch::jit::Module& encoder, const torch::Tensor& batch) {
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(batch);

  auto out = encoder.forward(inputs);
  auto tup = out.toTuple();
  if (!tup) {
    throw std::runtime_error("encoder forward did not return a tuple");
  }
  const auto& elems = tup->elements();
  if (elems.size() < 3) {
    throw std::runtime_error("encoder must return at least 3 tensors (layer1, layer2, layer3)");
  }

  std::vector<torch::Tensor> feats;
  feats.reserve(3);
  for (size_t i = 0; i < 3; ++i) {
    feats.push_back(elems[i].toTensor());
  }
  return feats;
}

static cv::Mat tensor2d_to_cvmat_f32(const torch::Tensor& t2d) {
  if (t2d.dtype() != torch::kFloat32) {
    throw std::runtime_error("expected float32 tensor");
  }
  if (t2d.dim() != 2) {
    throw std::runtime_error("expected 2D tensor");
  }
  auto tc = t2d.contiguous();
  cv::Mat m(tc.size(0), tc.size(1), CV_32F);
  std::memcpy(m.data, tc.data_ptr<float>(), static_cast<size_t>(tc.numel()) * sizeof(float));
  return m;
}

static cv::Mat run_kmeans_deterministic(const cv::Mat& data, int K) {

  cv::setNumThreads(1);
  cv::theRNG().state = 0;

  cv::Mat labels;
  cv::Mat centers;

  const int max_iter = 300;      // sklearn KMeans default
  const double eps = 1e-4;       // sklearn tol default
  const int attempts = 10;       // sklearn n_init typical
  const int flags = cv::KMEANS_PP_CENTERS;

  const cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, max_iter, eps);
  cv::kmeans(data, K, labels, criteria, attempts, flags, centers);
  return centers;  // (K, dim), CV_32F
}

static void ensure_dir(const fs::path& p) {
  std::error_code ec;
  fs::create_directories(p, ec);
  if (ec) {
    throw std::runtime_error("failed to create directory: " + p.string() + ": " + ec.message());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);

    args.encoder_ts_path =
        resolve_readable_path(args.encoder_ts_path,
                  {"code/resnet18_encoder_ts.pt", "resnet18_encoder_ts.pt"})
            .string();

    // Torch determinism (CPU): keep thread counts fixed.
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::manual_seed(0);

    // 1) Load encoder TorchScript
    if (!fs::exists(args.encoder_ts_path)) {
      throw std::runtime_error(
          "TorchScript encoder not found: " + args.encoder_ts_path);
    }

    torch::jit::Module encoder = torch::jit::load(args.encoder_ts_path);
    encoder.eval();

    // 2) List images (single class only)
    const auto image_paths = list_train_good_images(args.mvtec_root_dir, args.class_name);
    const int64_t N = static_cast<int64_t>(image_paths.size());
    if (N <= 0) {
      throw std::runtime_error("no training images found");
    }

    int K = 0;
    if (args.num_clusters > 0) {
      K = args.num_clusters;
      if (K > N) {
        throw std::runtime_error("--K must be <= number of images. K=" + std::to_string(K) +
                                 ", images=" + std::to_string(N));
      }
    } else {
      K = std::max(1, static_cast<int>(static_cast<double>(N) * args.ratio));
    }
    std::cout << "Class: " << args.class_name << "\n";
    std::cout << "Images: " << N << "\n";
    if (args.num_clusters > 0) {
      std::cout << "K (clusters): " << K << " (--K)\n";
    } else {
      std::cout << "K (clusters): " << K << " (ratio=" << args.ratio << ")\n";
    }

    // 3) Extract features into 2D matrices: (N, C*H*W)
    // We need to know each layer's (C,H,W) for saving as (K,C,H,W)
    int64_t C1 = 0, H1 = 0, W1 = 0;
    int64_t C2 = 0, H2 = 0, W2 = 0;
    int64_t C3 = 0, H3 = 0, W3 = 0;

    cv::Mat layer1_data;
    cv::Mat layer2_data;
    cv::Mat layer3_data;

    torch::NoGradGuard no_grad;

    int64_t filled = 0;
    while (filled < N) {
      const int64_t remaining = N - filled;
      const int64_t bs = std::min<int64_t>(args.batch_size, remaining);

      std::vector<torch::Tensor> batch_tensors;
      batch_tensors.reserve(static_cast<size_t>(bs));
      for (int64_t i = 0; i < bs; ++i) {
        batch_tensors.push_back(preprocess_to_tensor_chw(image_paths[static_cast<size_t>(filled + i)]));
      }
      auto batch = torch::stack(batch_tensors, 0);  // (B,3,256,256)

      auto feats = forward_encoder(encoder, batch);
      // feats[i] shape: (B, C, H, W)
      if (C1 == 0) {
        C1 = feats[0].size(1);
        H1 = feats[0].size(2);
        W1 = feats[0].size(3);
        C2 = feats[1].size(1);
        H2 = feats[1].size(2);
        W2 = feats[1].size(3);
        C3 = feats[2].size(1);
        H3 = feats[2].size(2);
        W3 = feats[2].size(3);

        layer1_data = cv::Mat(static_cast<int>(N), static_cast<int>(C1 * H1 * W1), CV_32F);
        layer2_data = cv::Mat(static_cast<int>(N), static_cast<int>(C2 * H2 * W2), CV_32F);
        layer3_data = cv::Mat(static_cast<int>(N), static_cast<int>(C3 * H3 * W3), CV_32F);
      }

      for (int li = 0; li < 3; ++li) {
        auto flat = feats[li].contiguous().view({bs, -1}).to(torch::kFloat32);
        auto m = tensor2d_to_cvmat_f32(flat);

        cv::Mat* dst = nullptr;
        if (li == 0) dst = &layer1_data;
        if (li == 1) dst = &layer2_data;
        if (li == 2) dst = &layer3_data;

        cv::Mat roi = (*dst)(cv::Range(static_cast<int>(filled), static_cast<int>(filled + bs)), cv::Range::all());
        m.copyTo(roi);
      }

      filled += bs;
      std::cout << "Extracted: " << filled << "/" << N << "\r" << std::flush;
    }
    std::cout << "\nFeature extraction done.\n";

    // 4) KMeans per layer (OpenCV)
    std::cout << "Fitting layer1 kmeans...\n";
    cv::Mat c1 = run_kmeans_deterministic(layer1_data, K);
    std::cout << "Fitting layer2 kmeans...\n";
    cv::Mat c2 = run_kmeans_deterministic(layer2_data, K);
    std::cout << "Fitting layer3 kmeans...\n";
    cv::Mat c3 = run_kmeans_deterministic(layer3_data, K);

    // 5) Save .pt for PRNet::load_prototypes:
    //    vector<Tensor>{layer1, layer2, layer3}, each shaped (K, C, H, W).
    const fs::path out_dir = fs::path(args.out_root_dir) / args.class_name;
    ensure_dir(out_dir);

    auto t1 = torch::from_blob(c1.ptr<float>(), {K, C1 * H1 * W1}, torch::kFloat32)
            .clone()
            .view({K, C1, H1, W1});
    auto t2 = torch::from_blob(c2.ptr<float>(), {K, C2 * H2 * W2}, torch::kFloat32)
            .clone()
            .view({K, C2, H2, W2});
    auto t3 = torch::from_blob(c3.ptr<float>(), {K, C3 * H3 * W3}, torch::kFloat32)
            .clone()
            .view({K, C3, H3, W3});

    std::vector<torch::Tensor> prototypes = {t1, t2, t3};

    // 兼容现有 PRNet::load_prototypes 读取逻辑：
    // 将 3 层原型一次性保存为单文件 prototype_generator.pt（vector<Tensor>）。
    const fs::path out_pt = out_dir / "prototype_generator.pt";
    torch::save(prototypes, out_pt.string());

    // 额外导出 3 个分层文件，便于独立检查或与其他脚本对接。
    const fs::path out_l1 = out_dir / "layer1.pt";
    const fs::path out_l2 = out_dir / "layer2.pt";
    const fs::path out_l3 = out_dir / "layer3.pt";
    torch::save(t1, out_l1.string());
    torch::save(t2, out_l2.string());
    torch::save(t3, out_l3.string());

    std::cout << "Saved prototype file to: " << out_pt.string() << "\n";
    std::cout << "Saved layer files to: " << out_l1.string() << ", "
          << out_l2.string() << ", " << out_l3.string() << "\n";

    return 0;
  } catch (const c10::Error& e) {
    std::cerr << "LibTorch error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    print_usage(argv[0]);
    return 1;
  }
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif