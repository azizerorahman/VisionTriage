# Configuration Setup Guide

## Quick Setup

#### 1. Copy Example Config

```bash
cd config/
cp config.yaml.example config.yaml
```

#### 2. Edit Paths for Your Environment

Open `config/config.yaml` and update the dataset paths:

```yaml
dataset:
  # IDRiD raw images
  train_image_dir:  "data/raw/IDRiD/images/train"
  test_image_dir:   "data/raw/IDRiD/images/test"
  train_label_csv:  "data/raw/IDRiD/labels/train.csv"
  test_label_csv:   "data/raw/IDRiD/labels/test.csv"

  # DODR raw images
  dodr_image_dir:   "data/raw/DODR/images"
  dodr_label_csv:   "data/raw/DODR/labels.csv"

  # Preprocessed output
  processed_dir:    "data/processed"
```

#### 3. Run Preprocessing

Before any training, preprocess raw images once:

```bash
python scripts/preprocess_dataset.py --dataset all
```

#### 4. Verify Configuration

```bash
python -c "
import yaml
with open('config/config.yaml') as f:
    cfg = yaml.safe_load(f)
print('Dataset paths:')
for k, v in cfg['dataset'].items():
    print(f'  {k}: {v}')
print('Training:')
for k, v in cfg['training'].items():
    print(f'  {k}: {v}')
"
```

## Environment-Specific Tuning

```yaml
training:
  batch_size:   16
  num_workers:  4
  mixed_precision: true

model:
  attn_dim:   256
  num_heads:  8
  dropout:    0.5
```

For Limited GPU Memory reduce attention dimension and batch size (or according to the specifications)

## Full Configuration Reference

All available keys in `config.yaml`:

#### `dataset`

| Key | Default | Description |
|-----|---------|-------------|
| `train_image_dir` | `data/raw/IDRiD/images/train` | IDRiD train images |
| `test_image_dir` | `data/raw/IDRiD/images/test` | IDRiD test images |
| `train_label_csv` | `data/raw/IDRiD/labels/train.csv` | IDRiD train labels |
| `test_label_csv` | `data/raw/IDRiD/labels/test.csv` | IDRiD test labels |
| `dodr_image_dir` | `data/raw/DODR/images` | DODR images (M3-MS only) |
| `dodr_label_csv` | `data/raw/DODR/labels.csv` | DODR labels (M3-MS only) |
| `processed_dir` | `data/processed` | Preprocessed image cache |
| `num_classes` | `3` | Urgency tiers: LOW / MEDIUM / HIGH |
| `image_size` | `480` | Input resolution |

#### `preprocessing`

| Key | Default | Description |
|-----|---------|-------------|
| `apply_ben_graham` | `true` | Ben Graham circular crop |
| `apply_green_channel` | `true` | Green channel extraction |
| `apply_clahe` | `true` | CLAHE enhancement |
| `clahe_clip_limit` | `2.0` | CLAHE clip limit |
| `clahe_tile_size` | `[8, 8]` | CLAHE tile grid |

#### `model`

| Key | Default | Description |
|-----|---------|-------------|
| `backbone` | `resnet50` | Feature extractor |
| `pretrained` | `true` | ImageNet pretrained weights |
| `attn_dim` | `256` | Cross-attention projection dimension |
| `num_heads` | `8` | Multihead attention heads |
| `dropout` | `0.5` | Dropout probability |
| `num_classes` | `3` | Output classes |

#### `training`

| Key | Default | Description |
|-----|---------|-------------|
| `epochs` | `50` | Max training epochs |
| `batch_size` | `16` | Batch size |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `weight_decay` | `1e-5` | AdamW weight decay |
| `lr_scheduler` | `cosine` | LR schedule: cosine annealing |
| `t_max` | `50` | Cosine annealing T_max |
| `early_stopping_patience` | `10` | Early stopping patience |
| `auto_class_weights` | `true` | Inverse-frequency class weighting |
| `num_workers` | `4` | DataLoader worker threads |
| `mixed_precision` | `true` | AMP (torch.cuda.amp) |

#### `augmentation`

| Key | Default | Description |
|-----|---------|-------------|
| `random_horizontal_flip` | `true` | p=0.5 |
| `random_vertical_flip` | `true` | p=0.5 |
| `random_rotation` | `true` | ±15° |
| `color_jitter` | `true` | brightness/contrast ±0.2 |
| `random_erasing` | `true` | p=0.1 |
| `gaussian_blur` | `true` | p=0.1 |

#### `logging`

| Key | Default | Description |
|-----|---------|-------------|
| `checkpoint_dir` | `checkpoints/` | Model checkpoint save path |
| `tensorboard_dir` | `runs/` | TensorBoard log path |
| `save_every_n_epochs` | `5` | Epoch checkpoint frequency |
| `keep_best_only` | `false` | Delete non-best checkpoints |

#### `reproducibility`

| Key | Default | Description |
|-----|---------|-------------|
| `seed` | `42` | Global random seed |
| `deterministic` | `true` | `torch.backends.cudnn.deterministic` |
| `benchmark` | `false` | `torch.backends.cudnn.benchmark` |