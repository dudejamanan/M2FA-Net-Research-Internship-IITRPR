# M²FA-Net: Multimodal Multi-Level Fusion Attention Network for Weed Segmentation

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./M2FA-Net.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-WeedyRice--RGBMS--DB-green)](https://data.mendeley.com/datasets/vt4s83pxx6/1)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

Official implementation of **"M²FA-Net: A Multimodal Multi-Level Fusion Attention Network for Weed Segmentation"**, submitted to *Artificial Intelligence Review* (Springer, under review).

M²FA-Net is a dual-branch, multi-level multimodal fusion framework for crop–weed segmentation that integrates **RGB imagery**, **multispectral bands** (Green, Red, Red Edge, NIR), and **vegetation indices** (NDVI, NDRE) using dual DeepLabV3-ResNet50 backbones, CBAM attention, deep supervision, and a composite Cross-Entropy + Dice + Boundary loss.

On the **WeedyRice-RGBMS-DB** dataset, M²FA-Net achieves **80.79% IoU** and **89.21% Dice**, outperforming eleven state-of-the-art CNN, transformer, and hybrid baselines.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Computational Complexity](#computational-complexity)
- [Installation](#installation)
- [Usage](#usage)
- [Baselines](#baselines)
- [Citation](#citation)
- [Authors](#authors)
- [Funding](#funding)
- [License](#license)

---

## Overview

Weedy rice is morphologically very similar to cultivated rice during early growth stages, making crop–weed discrimination difficult using RGB imagery alone. M²FA-Net addresses this by:

- Using **dual DeepLabV3-ResNet50 encoders** — one for RGB, one for a 6-channel multispectral input (G, R, RE, NIR, NDVI, NDRE).
- Performing **progressive multi-level feature fusion** at every encoder stage (not just the final layer), via channel-wise concatenation + 1×1 convolution.
- Applying **CBAM (Convolutional Block Attention Module)** at each fusion stage to refine channel and spatial features.
- Using **deep supervision** through auxiliary classifiers at intermediate fusion levels for improved gradient flow.
- Optimizing with a **composite loss**: `0.4 × Cross-Entropy + 0.4 × Dice + 0.2 × Boundary`.

## Architecture

<p align="center">
  <img src="./RP images/architecture.png" alt="M2FA-Net Architecture" width="800"/>
</p>

**Pipeline:**
1. RGB (3-channel) and multispectral (6-channel: G, R, RE, NIR, NDVI, NDRE) inputs are passed through two independent DeepLabV3-ResNet50 encoders pretrained on COCO-with-VOC labels.
2. At each of the four encoder stages (256 → 512 → 1024 → 2048 channels), RGB and multispectral features are concatenated and reduced via 1×1 convolution.
3. A CBAM module refines each fused representation (channel attention → spatial attention).
4. Auxiliary classifiers attached to intermediate fusion stages provide deep supervision.
5. The final fused representation passes through a segmentation head (Conv3×3 → BatchNorm → ReLU → Dropout → Conv3×3) followed by bilinear upsampling to produce the final weed segmentation mask.

> See `M2FA-Net.pdf` for the full architecture diagram (Fig. 2) and CBAM module details.

## Repository Structure

```
Research Internship IITRPR/
├── M2FA-Net.pdf                  # Full research paper
├── AIR_M_Net_2026/               # Manuscript source (LaTeX, Springer Nature template)
│   ├── sn-article.tex
│   ├── sn-article.pdf
│   ├── sn-bibliography.bib
│   ├── sn-jnl.cls
│   ├── sn-mathphys-num.bst
│   ├── user-manual.pdf           # Springer Nature template usage guide
│   └── Images/
│
├── Multi Level Fusion/           # ★ Proposed M²FA-Net implementation
│   ├── MultiLevelFusion.py       # Model definition, training & evaluation
│   ├── train_list.txt
│   ├── val_list.txt
│   └── test_list.txt
│
├── Dataset/
│   └── WeedyRice-RGBMS-DB/
│       ├── RGB/                  # 734 RGB field images (1280×960)
│       ├── Multispectral/        # 2936 spatially aligned band images (G, R, RE, NIR)
│       ├── Masks/                # 734 binary ground-truth segmentation masks
│       ├── Overlay/               # Overlay visualizations for qualitative inspection
│       ├── Metadata/             # GPS, altitude, timestamp, sensor, filename mappings
│       ├── train_list / val_list / test_list
│       └── readme.md
│
├── UNet/                         # Baseline: U-Net
├── UNet++/                       # Baseline: U-Net++
├── PSPNet/                       # Baseline: PSPNet
├── HRNet/                        # Baseline: HRNet
├── DenseUNet/                    # Baseline: DenseUNet
├── DeepLabV3/                    # Baseline: DeepLabV3
├── ViT transformer/               # Baseline: Vision Transformer (ViT)
├── Segformer/                    # Baseline: SegFormer
├── Swin+Trans/                   # Baseline: Swin Transformer
├── crfUnet/                      # Baseline: CRF-enhanced U-Net
├── CNN+Trans/                    # Baseline: Hybrid CNN + Transformer
│
├── complexity_analysis/          # FLOPs, parameter count, inference speed scripts/results
├── graphs/                       # Ablation, hyperparameter sensitivity, and training curve plots
├── RP images/                    # Figures used in the paper / README
└── Results/                      # Saved predictions, checkpoints, qualitative outputs
```

Each baseline model folder (`UNet`, `PSPNet`, `Segformer`, etc.) contains its own implementation code alongside the original reference paper for that architecture, so every method can be reproduced and compared under identical conditions.

## Dataset

Experiments use the **WeedyRice-RGBMS-DB** dataset — a UAV-based, multimodal, geo-referenced dataset for weedy rice segmentation, collected with a DJI M3M UAV over rice fields in An Giang Province, Vietnam (WGS 84 / UTM Zone 48N).

| Category | Count |
|---|---|
| RGB Images | 734 |
| Segmentation Masks | 734 |
| Overlay Images | 734 |
| Multispectral Images (Total) | 2936 |
| ↳ Green (G) | 734 |
| ↳ Red (R) | 734 |
| ↳ Red Edge (RE) | 734 |
| ↳ Near-Infrared (NIR) | 734 |
| **Training Set** | **438** |
| **Validation Set** | **148** |
| **Testing Set** | **148** |

- Images resized to **512 × 512** for uniform processing.
- Ground-truth masks: pixel value `255` = weedy rice, `0` = background.
- Multispectral input = 6 channels: `{G, R, RE, NIR, NDVI, NDRE}`, Z-score normalized using training-set statistics.

**Vegetation indices used:**

```
NDVI = (NIR − R) / (NIR + R)
NDRE = (NIR − RE) / (NIR + RE)
```

Dataset source: [Mendeley Data — vt4s83pxx6](https://data.mendeley.com/datasets/vt4s83pxx6/1)

To use the dataset, download it and place it under `Dataset/WeedyRice-RGBMS-DB/` following the structure above, or update the paths in `train_list.txt` / `val_list.txt` / `test_list.txt`.

## Results

All models trained under identical settings and evaluated with **Accuracy, Precision, Recall, IoU, and Dice** (mean ± std over 5 runs).

| Method | Accuracy (%) | Precision (%) | Recall (%) | IoU (%) | Dice (%) |
|---|---|---|---|---|---|
| U-Net | 89.68 ± 0.40 | 82.18 ± 2.59 | 80.44 ± 2.36 | 68.75 ± 0.46 | 80.97 ± 0.33 |
| ViT | 82.48 ± 0.57 | 66.99 ± 1.94 | 76.78 ± 2.45 | 56.08 ± 0.20 | 70.87 ± 0.19 |
| HRNet | 81.67 ± 0.62 | 63.48 ± 1.88 | 73.21 ± 2.76 | 52.72 ± 0.41 | 67.36 ± 0.43 |
| PSPNet | 89.87 ± 0.33 | 79.82 ± 1.35 | 83.27 ± 1.61 | 69.34 ± 0.59 | 81.36 ± 0.40 |
| DeepLabV3 | 88.89 ± 0.24 | 79.81 ± 2.11 | 80.23 ± 2.43 | 66.95 ± 0.54 | 79.62 ± 0.41 |
| CRF-U-Net | 88.91 ± 0.89 | 80.11 ± 2.01 | 79.76 ± 3.86 | 66.82 ± 2.55 | 79.54 ± 1.93 |
| SegFormer | 86.04 ± 0.63 | 78.65 ± 2.23 | 70.58 ± 3.54 | 59.04 ± 1.87 | 73.13 ± 1.56 |
| CNN + Transformer | 88.36 ± 0.52 | 74.39 ± 1.81 | 80.59 ± 1.53 | 63.55 ± 0.65 | 76.30 ± 0.55 |
| Swin Transformer | 90.58 ± 0.39 | 85.99 ± 2.19 | 86.13 ± 2.83 | 75.43 ± 1.03 | 85.99 ± 0.67 |
| U-Net++ | 90.41 ± 0.13 | 80.28 ± 2.16 | 78.26 ± 1.81 | 66.39 ± 0.27 | 78.56 ± 0.22 |
| DenseUNet | 90.75 ± 0.26 | 83.77 ± 2.88 | 81.38 ± 2.78 | 70.67 ± 0.31 | 82.27 ± 0.23 |
| **M²FA-Net (Proposed)** | **94.06 ± 0.07** | **90.72 ± 0.69** | **87.82 ± 0.84** | **80.79 ± 0.24** | **89.21 ± 0.15** |

**Key improvements over the strongest baselines:**
- +10.12% IoU over DenseUNet
- +10.65% Dice over U-Net++
- +13.84% IoU and +9.59% Dice over DeepLabV3
- Highest recall among all methods → fewer missed weed detections

## Ablation Study

Impact of each architectural component (mean ± std, %):

| Method | Accuracy | Precision | Recall | IoU | Dice |
|---|---|---|---|---|---|
| Baseline (Early Fusion) | 92.75 ± 0.09 | 90.28 ± 0.42 | 82.98 ± 0.43 | 76.48 ± 0.25 | 86.41 ± 0.15 |
| No Multi-Level Fusion | 92.86 ± 0.09 | 90.81 ± 0.56 | 82.80 ± 0.92 | 76.68 ± 0.42 | 86.55 ± 0.28 |
| No CBAM Attention | 93.10 ± 0.11 | 90.61 ± 0.80 | 84.10 ± 0.98 | 77.62 ± 0.38 | 87.18 ± 0.25 |
| No Deep Supervision | 93.06 ± 0.14 | 90.95 ± 0.67 | 83.36 ± 0.99 | 77.24 ± 0.51 | 86.92 ± 0.34 |
| No Warmup Scheduler | 92.78 ± 0.40 | 90.92 ± 1.80 | 83.10 ± 3.08 | 76.90 ± 1.38 | 86.72 ± 0.91 |
| No Augmentation | 93.26 ± 0.08 | 90.60 ± 0.28 | 84.78 ± 0.43 | 78.18 ± 0.24 | 87.55 ± 0.15 |
| **M²FA-Net (Full)** | **94.06 ± 0.07** | **90.72 ± 0.69** | **87.82 ± 0.84** | **80.79 ± 0.24** | **89.21 ± 0.15** |

Relative IoU contribution ranking: **Multi-Level Fusion (4.11%) > Warmup Scheduler (3.89%) > Deep Supervision (3.55%) > CBAM Attention (3.17%) > Data Augmentation (2.61%)**.

## Computational Complexity

Measured at 512×512 input resolution, batch size 1, on an NVIDIA RTX A5000 (24GB):

| Model | Parameters (M) | FLOPs (G) | Inference Speed (FPS) |
|---|---|---|---|
| DenseUNet | 17.60 | 68.60 | 47.00 |
| U-Net / CRF-UNet / PSPNet | ~48.80 | 19.53 (PSPNet) | 280.11 (PSPNet) |
| HRNet | 42.80 | – | – |
| U-Net++ | 53.00 | 148.29 | 40.81 |
| Swin Transformer | 62.00 | – | – |
| DeepLabV3 | 79.20 | – | – |
| CNN+Transformer | 92.00 | – | – |
| SegFormer | 3.70 | 6.77 | 194.58 |
| ViT | 1.24 | – | – |
| **M²FA-Net (Proposed)** | **64.30** | **267.68** | **40.22** |

M²FA-Net is more computationally demanding than lightweight architectures (PSPNet, SegFormer) but remains lighter in parameter count than several hybrid/transformer baselines while delivering substantially higher segmentation accuracy — a trade-off justified for precision-agriculture applications where segmentation quality is critical.

## Installation

```bash
git clone <this-repo-url>
cd "Research Internship IITRPR"

# create environment
conda create -n m2fanet python=3.10 -y
conda activate m2fanet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow opencv-python scikit-learn scikit-image \
            matplotlib tqdm pandas rasterio thop
```

**Reference hardware/software (as used in the paper):** 13th Gen Intel Core i9-13900K, NVIDIA RTX A5000 (24GB), 128GB RAM, Windows 11 Pro, Python 3.10.20, CUDA 12.4.

## Usage

### Train M²FA-Net

```bash
cd "Multi Level Fusion"
python MultiLevelFusion.py \
    --train_list train_list.txt \
    --val_list val_list.txt \
    --test_list test_list.txt \
    --batch_size 8 \
    --epochs 20 \
    --lr 2e-4 \
    --min_lr 1e-6 \
    --weight_decay 1e-3 \
    --warmup_epochs 3 \
    --aux_loss_weight 0.7 \
    --dropout 0.1 \
    --grad_clip 1.0 \
    --aug_prob 0.5 \
    --input_size 512
```

**Default training configuration (from the paper):**

| Batch Size | Epochs | Optimizer | Initial LR | Min LR | Weight Decay | Warmup Epochs | Aux. Loss λ | Dropout | Grad. Clip | Aug. Prob | Input Res. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 20 | AdamW | 2×10⁻⁴ | 1×10⁻⁶ | 1×10⁻³ | 3 | 0.7 | 0.1 | 1.0 | 0.5 | 512×512 |

### Evaluate a trained model

```bash
python MultiLevelFusion.py --evaluate --checkpoint <path_to_checkpoint> --test_list test_list.txt
```

### Run a baseline

Each baseline directory follows a similar structure with its own script and `train_list` / `val_list` / `test_list` files:

```bash
cd UNet   # or PSPNet, DeepLabV3, Segformer, Swin+Trans, etc.
python <model_script>.py --train_list train_list.txt --val_list val_list.txt --test_list test_list.txt
```

### Complexity analysis

```bash
cd complexity_analysis
python compute_complexity.py --model m2fanet --input_size 512
```

## Baselines

The following state-of-the-art segmentation models are implemented and evaluated for direct comparison, each with its original reference paper included in its respective folder:

- **CNN-based:** U-Net, U-Net++, PSPNet, HRNet, DenseUNet, DeepLabV3
- **Transformer-based:** Vision Transformer (ViT), SegFormer, Swin Transformer
- **Hybrid:** CRF-enhanced U-Net, CNN + Transformer

## Citation

The manuscript is currently **under review** at *Artificial Intelligence Review* (Springer). If you use this code or dataset pipeline, please cite:

```bibtex
@article{sanodiya2026m2fanet,
  title   = {M2FA-Net: A Multimodal Multi-Level Fusion Attention Network for Weed Segmentation},
  author  = {Sanodiya, Rakesh Kumar and Dudeja, Manan and R, Lekshmi and Khushi, Matloob},
  journal = {Artificial Intelligence Review},
  note    = {Manuscript under review},
  year    = {2026}
}
```

Please also cite the dataset:

```bibtex
@article{nguyen2025weedyrice,
  title   = {A dataset of aligned RGB and multispectral UAV imagery for semantic segmentation of weedy rice},
  author  = {Nguyen, V.-H. and Le, C.-D. and Truong, M.-T. and Bui, M.-P.T. and Le, T.-P.},
  journal = {Data in Brief},
  pages   = {112237},
  year    = {2025}
}
```

## Authors

- **Manan Dudeja** — Methodology, Software, Data curation, Validation, Writing (original draft) — Vellore Institute of Technology, Chennai
- **Rakesh Kumar Sanodiya** — Conceptualization, Data curation, Validation, Supervision, Project administration — IIT Ropar
- **Lekshmi R** — Formal analysis, Visualization, Investigation — Amrita Vishwa Vidyapeetham
- **Matloob Khushi** — Conceptualization, Validation, Supervision, Project administration, Writing (review & editing) — Brunel University of London

## Funding

This work was supported by the **Science and Engineering Research Board (SERB)** under the Core Research Grant Project (File No. `CRG/2023/001239`).

## License

This repository is released for academic and research purposes. Please refer to the dataset's own license terms on [Mendeley Data](https://data.mendeley.com/datasets/vt4s83pxx6/1) before redistribution. Code license: MIT (unless noted otherwise in individual baseline folders, which may retain their original authors' licenses).
