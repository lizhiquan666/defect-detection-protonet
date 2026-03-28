⭐ If this project helps your work, please consider giving it a star.

# Defect-Detection-ProtoNet

## Abstract
Defect-Detection-ProtoNet is an industrial PCB defect detection and localization pipeline built on Prototypical Residual Networks (CVPR 2023).

The repository is designed for competition-grade and production-like batch inference on Windows, with emphasis on:
1. Reproducible prototype availability
2. Stable multi-class inference execution
3. Deterministic output normalization for submission

## Scope and Positioning
This repository focuses on inference-time engineering, not end-to-end model training from scratch.

Current implementation priorities are:
1. Operational robustness under missing prototype conditions
2. Script-driven automation with clear fallback semantics
3. Strict output naming and file filtering constraints

## Pipeline Logic (Implemented Behavior)
The effective workflow is orchestrated by batch scripts and follows a fail-fast policy.

1. Prototype readiness (remote first)
   - [bin/usage.bat](bin/usage.bat) calls [extra/libtorch.bat](extra/libtorch.bat) with `--prototypes-only`.
   - Prototypes are downloaded into `code/prototypes/<class>/prototype_generator.pt`.
2. Local prototype fallback
   - If remote download fails, [code/pt_builder.bat](code/pt_builder.bat) is invoked as fallback.
   - Fallback artifacts are synchronized into root/bin/build prototype paths.
3. Batch inference
   - [code/inference.bat](code/inference.bat) executes class-wise inference on `ok/ng` subsets with configured thresholds.
4. Output normalization
   - [code/clean_rename.bat](code/clean_rename.bat) renames `result_*.bmp/.txt` to `*_rst.bmp/.txt` and removes non-target files.

Engineering guarantees:
1. Dual fallback for prototype availability
2. Avoidance of redundant prototype rebuilding after successful fallback
3. Multi-directory prototype synchronization for runtime compatibility
4. Immediate termination when a required stage fails

## Key Entry Scripts 🧩
| Script | Role | Notes |
|---|---|---|
| [bin/usage.bat](bin/usage.bat) | 🚀 End-to-end orchestrator | Supports `--dry-run`, `/dryrun`, `-n` |
| [extra/libtorch.bat](extra/libtorch.bat) | 📦 Dependency/prototype utility | Full setup mode and `--prototypes-only` mode |
| [code/pt_builder.bat](code/pt_builder.bat) | 🧠 Local prototype generator | Default `K=75`, `ratio=0.2`, classes `1..10` |
| [code/inference.bat](code/inference.bat) | 🔍 Main inference runner | Default class set `2 4 6 8 10` |
| [code/clean_rename.bat](code/clean_rename.bat) | 🧹 Post-processing and cleanup | Keeps only standardized `.bmp/.txt` outputs |

## Quick Start (Windows) ⚡
```bash
# End-to-end pipeline
bin/usage.bat

# Dry run (no execution)
bin/usage.bat --dry-run

# Refresh prototypes only
extra/libtorch.bat --prototypes-only

# Full dependency setup (Administrator required for system PATH update)
extra/libtorch.bat
```

## Input/Output Specification 📁
Input conventions (default behavior):
1. Inference discovers dataset root automatically, or accepts it as an argument.
2. Data is traversed by class and subset:
   - `<IN_ROOT>/<class>/验证集ok/*.bmp`
   - `<IN_ROOT>/<class>/验证集ng/*.bmp`

Output conventions:
1. Inference output root: `res/<class>/<ok|ng>/`
2. End-to-end final retained files (after cleanup):
   - `*_rst.bmp`
   - `*_rst.txt`
3. The `res` directory stores inference-generated images and is treated as the final result set for delivery/submission.

## Reproducibility Notes ✅
Default detection parameters in [code/inference.bat](code/inference.bat) include:
1. `DET_THRESH=0.24`
2. `DET_MIN_AREA=220`
3. `DET_EDGE_MARGIN=0.034`
4. `DET_OPEN_ITER=2`
5. `DET_NMS_IOU=0.3`

Use fixed parameters and fixed class sets to obtain consistent benchmark behavior.

## Repository Layout
```text
Defect-Detection-ProtoNet/
|-- .vscode/
|-- bin/
|   |-- usage.bat
|   |-- defect_detect.exe
|   `-- prototype_generator.exe
|-- code/
|   |-- inference.bat
|   |-- pt_builder.bat
|   |-- clean_rename.bat
|   `-- *.cpp / *.hpp
|-- extra/
|   |-- libtorch.bat
|   |-- libtorch/
|   |-- opencv/
|   `-- build/
|-- models/
|-- res/   (inference images and final result outputs)
|-- LICENSE.txt
|-- .gitattributes
|-- .gitignore
`-- .gitmodules
```

## Citation 📚
Paper page: [Prototypical Residual Networks (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html)

Authors: <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html"><strong><span style="color:#1f6feb;">Hui Zhang</span></strong></a>, <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html"><strong><span style="color:#1f6feb;">Zuxuan Wu</span></strong></a>, <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html"><strong><span style="color:#1f6feb;">Zheng Wang</span></strong></a>, <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html"><strong><span style="color:#1f6feb;">Zhineng Chen</span></strong></a>, <a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Prototypical_Residual_Networks_for_Anomaly_Detection_and_Localization_CVPR_2023_paper.html"><strong><span style="color:#1f6feb;">Yu-Gang Jiang</span></strong></a>

```bash
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Hui and Wu, Zuxuan and Wang, Zheng and Chen, Zhineng and Jiang, Yu-Gang},
    title     = {Prototypical Residual Networks for Anomaly Detection and Localization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {16281-16291}
}
```

## Acknowledgments 🙏
This project is improved based on the open-source implementation: [PRNet](https://github.com/xcyao00/PRNet.git), and we sincerely thank the original authors for their contributions.

Special thanks to [**<span style="color:#1f6feb;">Xueli Zhang</span>**](https://github.com/Shelly-icecream) for support and inspiration.

Related work:
```bash
@misc{zhang2026motionlora,
  author       = {Xueli Zhang},
  title        = {AnimateDiff Motion Module LoRA for Slow Motion Generation},
  year         = {2026},
  howpublished = {\url{https://github.com/Shelly-icecream/AnimateDiff-Motion-Module-LoRA}},
  note         = {GitHub repository}
}
```

Project link: [AnimateDiff Motion Module LoRA](https://github.com/Shelly-icecream/AnimateDiff-Motion-Module-LoRA)

## License
This repository is released under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

Please refer to [LICENSE.txt](LICENSE.txt) for the complete legal text.

Academic and research use is permitted. Commercial use is strictly prohibited.