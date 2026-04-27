# VisionTriage: Cross-Attention Fusion for Fundus Image-Based Diabetic Retinopathy Triage

[![Triage](https://img.shields.io/badge/Vision-Triage-red.svg)]()
[![Thesis](https://img.shields.io/badge/Thesis-Sichuan%20University-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This codebase belongs to the thesis: *"Bidirectional Cross-Attention Fusion of Bilateral Fundus Images for Three-Tier Diabetic Retinopathy Urgency Triage (VisionTriage)"* **Sichuan University, 2026**

---


## Installation

```bash
git clone https://github.com/azizerorahman/VisionTriage.git
cd VisionTriage
pip install -r requirements.txt
```

**Requirements**

| Package | Version |
|---------|---------|
| Python  | 3.9+    |
| PyTorch | 2.0+    |
| CUDA    | 11.8+ (recommended) |

## Data Setup

### 1. Download Datasets

| Dataset | Source |
|---------|--------|
| IDRiD   | [IEEE Dataport](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| DODR    | [Kaggle](https://www.kaggle.com/datasets/mariaherrerot/eyepacsformat) |

### 2. Preprocess Images

Run the preprocessing pipeline:

```bash
python scripts/preprocess_dataset.py --dataset all
```

## Quick Start

### Configure

```bash
cp config/config.yaml.example config/config.yaml
```

Edit `config/config.yaml` with your dataset paths and hardware settings.  
See [`config/CONFIG_SETUP.md`](config/CONFIG_SETUP.md) for detailed instructions.

### Train

```bash
# M1 - Monocular baseline
python train.py --model baseline_cnn --seed 42

# M2 - Late concatenation (bilateral)
python train.py --model late_concat --seed 42

# M3 - Cross-attention (IDRiD only)
python train.py --model cross_attention --seed 42

# M3-MS - Cross-attention + multisource (IDRiD + DODR)
python train.py --model cross_attention --multisource --seed 42

# Seed robustness sweep
python train.py --model cross_attention --multisource --all-seeds
```

### Evaluate

**Single model evaluation**
```bash
python evaluate_checkpoint.py \
    --checkpoint checkpoints/cross_attention_MS_seed42/best_checkpoint.pth \
    --model cross_attention
```

**With Grad-CAM heatmap generation**
```bash
python evaluate_checkpoint.py \
    --checkpoint checkpoints/cross_attention_MS_seed42/best_checkpoint.pth \
    --model cross_attention \
    --gradcam \
    --gradcam-output results/gradcam/
```

**Statistical comparison (Wilcoxon + McNemar)**
```bash
python evaluate_checkpoint.py \
    --checkpoint checkpoints/cross_attention_MS_seed42/best_checkpoint.pth \
    --model cross_attention \
    --compare-checkpoint checkpoints/baseline_cnn_seed42/best_checkpoint.pth \
    --compare-model baseline_cnn
```

## Models

| ID    | Name                             | Input               | Fusion                  |
|:-----:|----------------------------------|---------------------|-------------------------|
| M1    | `baseline_cnn`                   | Monocular (left)    | —                       |
| M2    | `late_concat`                    | Bilateral           | Post-GAP concatenation  |
| M3    | `cross_attention`                | Bilateral           | Spatial cross-attention |
| M3-MS | `cross_attention --multisource`  | Bilateral           | Spatial cross-attention |

---

<div align="center">

**© 2026 Azizur Rahman || Sichuan University.** All rights reserved.

</div>
