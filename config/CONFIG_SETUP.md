# Configuration Setup Guide

## Quick Setup

### 1. Copy Example Configs

```bash
cd config/
cp config.yaml.example config.yaml
cp label_mapping.yaml.example label_mapping.yaml
```

### 2. Edit for Your Environment

#### config.yaml

Customize paths and cofigurations based on your hardware:

```yaml
dataset:
  raw_path: "/your/actual/path/IDRiD"
  processed_path: "/your/actual/path/data/processed"

models:
  proposed_crossattn:
    num_heads: 8
    attn_dim: 256

training:
  batch_size: 32           # Reduce if GPU memory limited
  learning_rate: 1e-4
  num_epochs: 50
  num_workers: 4           # Adjust to your CPU cores

gpu:
  device: "cuda"           # Use "cpu" if no GPU
  num_workers: 4
```

#### label_mapping.yaml

Usually no changes needed. Defines clinical DR grade to urgency mapping based on NHS and AAO guidelines.

### 3. Verify Configuration

```python
import yaml

with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
    print(config)
```

## Environment-Specific Tuning

### For GPU Training
```yaml
training:
  batch_size: 64
  num_workers: 4

gpu:
  device: "cuda"
  mixed_precision: true
```

### For CPU-Only
```yaml
training:
  batch_size: 8
  num_workers: 0

gpu:
  device: "cpu"
  mixed_precision: false
```

### For Limited GPU Memory

```yaml
models:
  proposed_crossattn:
    attn_dim: 128           # Reduce from 256
    num_heads: 4            # Reduce from 8

training:
  batch_size: 16           # Reduce from 32
```

## Configuration Reference

See config.yaml.example for all available options:

- dataset: paths and preprocessing settings
- models: architecture parameters for each model variant
- training: hyperparameters (batch size, learning rate, epochs)
- augmentation: data augmentation transforms
- logging: checkpoint and tensorboard directories
- gpu: device and worker configuration
- reproducibility: seed and determinism settings
