import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.models import get_model
from src.data.dataloader import get_dataloaders, get_multisource_dataloaders
from src.utils.metrics import (
    MetricsTracker,
    EarlyStopping,
    compute_seed_robustness,
    save_checkpoint,
    save_metrics_json,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  Seed set: {seed}")

def compute_class_weights(train_loader, device, num_classes=3):
    counts = torch.zeros(num_classes)

    for batch in train_loader:
        inputs, labels, _ = batch
        for c in range(num_classes):
            counts[c] += (labels == c).sum().item()

    max_count = counts.max()
    weights   = max_count / counts

    # Handle any class with zero samples
    weights[counts == 0] = 0.0

    print(f"  Class counts:  {counts.tolist()}")
    print(f"  Class weights: {[f'{w:.4f}' for w in weights.tolist()]}")

    return weights.to(device)

def forward_pass(model, batch, is_bilateral, device):
    inputs, labels, _ = batch
    labels = labels.to(device)

    if is_bilateral:
        # inputs is a tuple (x_left, x_right)
        if isinstance(inputs, (list, tuple)):
            x_left  = inputs[0].to(device)
            x_right = inputs[1].to(device)
        else:
            x_left  = inputs.to(device)
            x_right = torch.flip(x_left, dims=[3])

        logits = model(x_left, x_right)

    else:
        # M1
        if isinstance(inputs, (list, tuple)):
            x = inputs[0].to(device)
        else:
            x = inputs.to(device)

        logits = model(x)

    return logits, labels

def train_one_epoch(model, train_loader, criterion,
                    optimizer, device, epoch,
                    is_bilateral, writer=None):
    
    model.train()
    tracker = MetricsTracker(num_classes=3)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:>3} [Train]",
        leave=False
    )

    for step, batch in enumerate(pbar):
        logits, labels = forward_pass(
            model, batch, is_bilateral, device
        )

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        tracker.update(preds, labels, loss.item())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if writer is not None:
            global_step = (epoch - 1) * len(train_loader) + step
            writer.add_scalar('Train/StepLoss', loss.item(), global_step)

    metrics = tracker.compute()
    return metrics

def evaluate_epoch(model, loader, criterion,
                   device, epoch, is_bilateral,
                   split_name='Val'):

    model.eval()
    tracker = MetricsTracker(num_classes=3)

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:>3} [{split_name}]",
        leave=False
    )

    with torch.no_grad():
        for batch in pbar:
            logits, labels = forward_pass(
                model, batch, is_bilateral, device
            )
            loss  = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            tracker.update(preds, labels, loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    metrics = tracker.compute()
    return metrics

def train(
    config_path   = 'config/config.yaml',
    model_name    = 'baseline_cnn',
    seed          = 42,
    num_epochs    = 50,
    multisource   = False,
    save_dir      = 'checkpoints',
    run_tag       = None,
    auto_weights  = True,
):
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    ms_tag   = '_MS' if multisource else ''
    run_name = f"{model_name}{ms_tag}_seed{seed}"
    if run_tag:
        run_name = f"{run_name}_{run_tag}"

    print("=" * 70)
    print(f"  VisionTriage Training")
    print(f"  Model:       {model_name}{ms_tag}")
    print(f"  Seed:        {seed}")
    print(f"  Multisource: {multisource}")
    print(f"  Device:      {device}")
    print(f"  Run:         {run_name}")
    print("=" * 70)

    ds_cfg = config['dataset']

    _, is_bilateral_check = get_model(model_name)
    from src.models import BILATERAL_MODELS
    is_bilateral = BILATERAL_MODELS.get(model_name, False)

    print(f"\n  Loading data (bilateral={is_bilateral})...")

    if multisource:
        # M3-MS
        train_loader, test_loader, dataset_info = \
            get_multisource_dataloaders(
                idrid_train_csv     = ds_cfg['train_csv'],
                idrid_train_img_dir = ds_cfg['train_image_dir'],
                dodr_train_csv      = ds_cfg.get('dodr_train_csv', ''),
                dodr_train_img_dir  = ds_cfg.get('dodr_image_dir', ''),
                idrid_test_csv      = ds_cfg['test_csv'],
                idrid_test_img_dir  = ds_cfg['test_image_dir'],
                batch_size          = config['training']['batch_size'],
                num_workers         = config['gpu'].get('num_workers', 4),
                pin_memory          = config['gpu'].get('pin_memory', True),
                bilateral           = is_bilateral,
                label_col           = ds_cfg.get(
                    'diagnosis_column', 'Retinopathy grade'
                ),
            )
    else:
        # IDRiD-only
        train_loader, test_loader, dataset_info = get_dataloaders(
            train_csv     = ds_cfg['train_csv'],
            test_csv      = ds_cfg['test_csv'],
            train_img_dir = ds_cfg['train_image_dir'],
            test_img_dir  = ds_cfg['test_image_dir'],
            batch_size    = config['training']['batch_size'],
            num_workers   = config['gpu'].get('num_workers', 4),
            pin_memory    = config['gpu'].get('pin_memory', True),
            bilateral     = is_bilateral,
            label_col     = ds_cfg.get(
                'diagnosis_column', 'Retinopathy grade'
            ),
        )

    print(f"  Train samples: {dataset_info['train_size']}")
    print(f"  Test samples:  {dataset_info['test_size']}")

    print(f"\n  Building model: {model_name}...")

    model_cfg = {}
    if model_name in ('baseline_cnn', 'm1'):
        model_cfg = {
            'num_classes': 3,
            'pretrained':  config['models']['baseline_cnn']['pretrained'],
            'dropout':     config['models']['baseline_cnn'].get('dropout', 0.5),
        }
    elif model_name in ('late_concat', 'm2'):
        model_cfg = {
            'num_classes':   3,
            'pretrained':    config['models']['baseline_concat']['pretrained'],
            'fc_hidden_dim': config['models']['baseline_concat']['fc_hidden_dim'],
            'dropout':       config['models']['baseline_concat'].get('dropout', 0.5),
        }
    elif model_name in ('cross_attention', 'm3'):
        model_cfg = {
            'num_classes':   3,
            'pretrained':    config['models']['proposed_crossattn']['pretrained'],
            'attn_dim':      config['models']['proposed_crossattn']['attn_dim'],
            'num_heads':     config['models']['proposed_crossattn']['num_heads'],
            'fc_hidden_dim': config['models']['proposed_crossattn']['fc_hidden_dim'],
            'dropout':       config['models']['proposed_crossattn'].get('dropout', 0.5),
        }

    model, is_bilateral = get_model(model_name, model_cfg)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    print(f"\n  Computing class weights...")

    if auto_weights:
        class_weights = compute_class_weights(
            train_loader, device, num_classes=3
        )
    else:
        weights = config['training']['class_weights']
        class_weights = torch.tensor(
            weights, dtype=torch.float32
        ).to(device)
        print(f"  Class weights (config): {weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    lr           = float(config['training']['learning_rate'])   # 1e-4
    weight_decay = float(config['training']['weight_decay'])    # 1e-5

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )

    print(f"\n  Optimiser: AdamW (lr={lr}, wd={weight_decay})")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = num_epochs,   # Full training duration
        eta_min = 0.0,          # Reduce to zero
    )

    print(f"  Scheduler: CosineAnnealingLR (T_max={num_epochs}, eta_min=0.0)")

    early_stopping = EarlyStopping(
        patience  = config['training']['early_stopping']['patience'],
        mode      = 'max',
        min_delta = 0.001,
    )

    log_dir  = Path(config['logging']['tensorboard_dir']) / run_name
    save_dir = Path(save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"\n  Checkpoints: {save_dir}")
    print(f"  TensorBoard: {log_dir}")

    print(f"\n{'=' * 70}")
    print(f"  STARTING TRAINING — max {num_epochs} epochs")
    print(f"{'=' * 70}\n")

    best_f1       = 0.0
    best_epoch    = 0
    history       = {'train': [], 'val': []}

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion,
            optimizer, device, epoch,
            is_bilateral, writer
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_metrics = evaluate_epoch(
            model, test_loader, criterion,
            device, epoch, is_bilateral,
            split_name='Val'
        )

        val_f1 = val_metrics['f1_macro']

        writer.add_scalar('Train/MacroF1',  train_metrics['f1_macro'], epoch)
        writer.add_scalar('Train/Loss',     train_metrics['avg_loss'], epoch)
        writer.add_scalar('Val/MacroF1',    val_f1,                    epoch)
        writer.add_scalar('Val/Loss',       val_metrics['avg_loss'],   epoch)
        writer.add_scalar('Val/Kappa',      val_metrics['kappa'],      epoch)
        writer.add_scalar('LR',             current_lr,                epoch)

        print(
            f"Epoch {epoch:>3}/{num_epochs} | "
            f"Train F1: {train_metrics['f1_macro']:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Best: {best_f1:.4f}"
        )

        improved = early_stopping.step(val_f1)

        if improved:
            best_f1    = val_f1
            best_epoch = epoch

            # Save best checkpoint
            save_checkpoint(
                model      = model,
                optimizer  = optimizer,
                epoch      = epoch,
                metrics    = val_metrics,
                checkpoint_dir = str(save_dir),
                filename   = 'best_checkpoint.pth',
            )

            print(f"  New best Macro-F1: {best_f1:.4f} "
                  f"(epoch {best_epoch})")

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                str(save_dir),
                filename=f'checkpoint_epoch{epoch}.pth',
            )

        if early_stopping.stop:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(patience={early_stopping.patience})")
            break

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

    print(f"\n{'=' * 70}")
    print(f"  Training complete")
    print(f"  Best Val Macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"{'=' * 70}\n")

    # Save training history
    save_metrics_json(
        {'best_f1': best_f1, 'best_epoch': best_epoch,
         'history': {
             'train_f1': [m['f1_macro'] for m in history['train']],
             'val_f1':   [m['f1_macro'] for m in history['val']],
         }},
        str(save_dir),
        filename='training_history.json',
    )

    writer.close()
    return best_f1, run_name

def run_all_seeds(config_path, model_name, multisource=False,
                 seeds=None, **kwargs):
    if seeds is None:
        seeds = [42, 0, 123]

    print(f"\n{'=' * 70}")
    print(f"  THREE-SEED ROBUSTNESS RUN")
    print(f"  Model: {model_name} | Multisource: {multisource}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 70}\n")

    f1_scores  = []
    run_names  = []

    for seed in seeds:
        print(f"\n{'-' * 70}")
        print(f"  Running seed {seed}...")
        print(f"{'-' * 70}")

        best_f1, run_name = train(
            config_path = config_path,
            model_name  = model_name,
            seed        = seed,
            multisource = multisource,
            **kwargs,
        )

        f1_scores.append(best_f1)
        run_names.append(run_name)

        print(f"\n  Seed {seed} complete - Best F1: {best_f1:.4f}")

    # Compute robustness statistics
    robustness = compute_seed_robustness(f1_scores)

    print(f"\n{'=' * 70}")
    print(f"  SEED ROBUSTNESS RESULTS")
    print(f"{'=' * 70}")
    for seed, f1 in zip(seeds, f1_scores):
        print(f"  Seed {seed:>3}: {f1:.4f}")
    print(f"  {'-' * 30}")
    print(f"  Mean ± SD: {robustness['formatted']}")
    print(f"  Range:     {robustness['range']:.4f}")
    print(f"  CV:        {robustness['cv_pct']:.2f}%")
    print(f"{'=' * 70}\n")

    return robustness

def parse_args():
    parser = argparse.ArgumentParser(
        description='VisionTriage Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--model', type=str, default='baseline_cnn',
        choices=['baseline_cnn', 'late_concat', 'cross_attention',
                 'm1', 'm2', 'm3'],
        help='Model architecture to train'
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--multisource', action='store_true',
        help='Multisource training (M3-MS)'
    )
    parser.add_argument(
        '--all-seeds', action='store_true',
        help='Run all seeds'
    )
    parser.add_argument(
        '--save-dir', type=str, default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--tag', type=str, default=None,
        help='Optional run tag appended to run name'
    )
    parser.add_argument(
        '--no-auto-weights', action='store_true',
        help='Use config class weights instead of auto-computed'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.all_seeds:
        run_all_seeds(
            config_path = args.config,
            model_name  = args.model,
            multisource = args.multisource,
            num_epochs  = args.epochs,
            save_dir    = args.save_dir,
            run_tag     = args.tag,
            auto_weights = not args.no_auto_weights,
        )
    else:
        # Single seed run
        best_f1, run_name = train(
            config_path  = args.config,
            model_name   = args.model,
            seed         = args.seed,
            num_epochs   = args.epochs,
            multisource  = args.multisource,
            save_dir     = args.save_dir,
            run_tag      = args.tag,
            auto_weights = not args.no_auto_weights,
        )

        print(f"\nFinal best Macro-F1: {best_f1:.4f}")
        print(f"Run name: {run_name}")