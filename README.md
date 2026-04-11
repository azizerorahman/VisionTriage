# VisionTriage: Cross-Attention Fusion for Fundus Image-Based Diabetic Retinopathy Triage

Research implementation for urgency triage classification from fundus images using bilateral cross-attention fusion.

## Overview

VisionTriage addresses diabetic retinopathy (DR) screening bottlenecks by predicting clinical urgency directly from fundus images. The system classifies cases into three triage levels: **LOW**, **MEDIUM**, **HIGH** (aligned with NHS/AAO guidelines).

### Key Contributions

- **Baseline CNN:** Single ResNet-50 on monocular images
- **Late Concatenation:** Dual ResNet-50 with late feature fusion
- **Cross-Attention (Proposed):** Bidirectional bilateral fusion with multihead attention (L2R + R2L)

## Installation

```bash
git clone https://github.com/azizerorahman/VisionTriage.git
cd VisionTriage

pip install -r requirements.txt
```

## Quick Start

### Setup Configuration

```bash
cd config/
cp config.yaml.example config.yaml
cp label_mapping.yaml.example label_mapping.yaml
```

Edit `config.yaml` with your dataset paths and hardware settings.

### Train a Model

Train the cross-attention model on IDRiD dataset:

```bash
python train.py \
  --config config/config.yaml \
  --model cross_attention \
  --epochs 50 \
  --seed 42 \
  --auto_class_weights
```

### Evaluate Checkpoint

```bash
python evaluate_checkpoint.py \
  --model cross_attention \
  --checkpoint models/cross_attention/best_model.pth \
  --test_csv data/processed/test_triage_labels.csv \
  --test_img_dir data/processed/images/test \
  --output_json results/eval_metrics.json
```

## Supported Models

| Model | Input | Approach | Best F1 (IDRiD) |
|-------|-------|----------|-----------------|
| `baseline_cnn` | Monocular image | Single ResNet-50 | 0.7341 |
| `late_concat` | Bilateral images | Dual ResNet-50 + concatenation | 0.7523 |
| `cross_attention` | Bilateral images | Dual ResNet-50 + cross-attention | 0.7702 |

## Project Structure

```
src/
├── models/
│   ├── baseline_cnn.py      # Monocular baseline
│   ├── late_concat.py       # Late fusion baseline
│   └── cross_attention.py   # Proposed cross-attention model
├── data/
│   └── dataloader.py        # PyTorch Dataset and DataLoader
├── utils/
│   └── metrics.py           # Metrics tracking, checkpointing, early stopping
└── grad_cam.py              # Interpretability visualization

config/
├── config.yaml.example      # Hyperparameters and paths (copy and customize)
└── label_mapping.yaml.example # DR→Urgency clinical mapping

train.py                      # Main training script
evaluate_checkpoint.py        # Evaluation and metrics export
```

## Configuration

See `config/CONFIG_SETUP.md` for detailed setup instructions and environment-specific tuning.

### Example Training Variants

```bash
# Default cross-attention
python train.py --model cross_attention

# Ablation: reduced attention dimension
python train.py --model cross_attention --attn_dim 128 --run_tag ablation_d128

# Multisource training: IDRiD + DODR
python train.py --model cross_attention --train_csv data/processed/train_combined.csv
```

## Outputs

Training produces:

- **Checkpoints:** `models/<run_name>/best_model.pth`, `models/<run_name>/final_model.pth`
- **Metrics:** `models/<run_name>/training_history.json`
- **TensorBoard Logs:** `runs/<run_name>/`
- **Evaluation Results:** JSON file with predictions and metrics

## Reproducibility

All experiments use fixed random seeds and deterministic CUDA settings:

```python
python train.py --seed 42
```

Hyperparameters are loaded from YAML config to ensure repeatability across runs.

## Citation

This work is part of the thesis: *"Design and Evaluation of Cross-Attention Fusion for Fundus Image-Based Diabetic Retinopathy Triage"* (Sichuan University, 2026).

## License

See LICENSE file for details.
