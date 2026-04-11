import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
import random

import numpy as np
import pandas as pd

sys.path.append('/')

from src.models import get_baseline_cnn, get_late_concat_model, get_cross_attention_model
from src.data.dataloader import get_dataloaders
from src.utils.metrics import MetricsTracker, save_checkpoint, save_metrics_json, EarlyStopping


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_auto_class_weights(train_csv):
    # Compute inverse-frequency class weights
    df = pd.read_csv(train_csv)

    if 'urgency_label' in df.columns:
        labels = df['urgency_label'].astype(int)
    elif 'urgency' in df.columns:
        mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        labels = df['urgency'].astype(str).str.upper().map(mapping)
        if labels.isna().any():
            raise ValueError('Found unknown urgency values while computing class weights.')
        labels = labels.astype(int)
    else:
        raise ValueError('Training CSV must contain urgency_label or urgency column.')

    counts = labels.value_counts().to_dict()
    total = len(labels)
    num_classes = 3

    weights = []
    for class_idx in range(num_classes):
        class_count = counts.get(class_idx, 0)
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * class_count))

    return weights


def _build_model(model_name, config, overrides=None):
    if overrides is None:
        overrides = {}

    if model_name == 'baseline_cnn':
        model_config = config['models']['baseline_cnn']
        return get_baseline_cnn({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'fc_hidden_dim': model_config['fc_hidden_dim'],
            'dropout': 0.5,
        }), model_config['name']

    if model_name == 'late_concat':
        model_config = config['models']['baseline_concat']
        return get_late_concat_model({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'fc_hidden_dim': overrides.get('fc_hidden_dim', model_config['fc_hidden_dim']),
            'dropout': 0.5,
        }), model_config['name']

    if model_name == 'cross_attention':
        model_config = config['models']['proposed_crossattn']
        return get_cross_attention_model({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'attn_dim': overrides.get('attn_dim', model_config['attn_dim']),
            'num_heads': overrides.get('num_heads', model_config['num_heads']),
            'fc_hidden_dim': overrides.get('fc_hidden_dim', model_config['fc_hidden_dim']),
            'dropout': 0.5,
        }), model_config['name']

    raise ValueError(f'Unknown model_name: {model_name}')


def _forward_with_mode(model, model_name, images):
    if model_name == 'baseline_cnn':
        return model(images)

    right_images = torch.flip(images, dims=[3])
    return model(images, right_images)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, model_name):
    # Train for one epoch
    model.train()
    tracker = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = _forward_with_mode(model, model_name, images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        tracker.update(preds, labels, loss.item())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = tracker.compute()
    return metrics


def evaluate(model, test_loader, criterion, device, epoch, model_name):
    # Evaluate model
    model.eval()
    tracker = MetricsTracker()
    
    pbar = tqdm(test_loader, desc=f"Epoch {epoch} [Eval]")
    with torch.no_grad():
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = _forward_with_mode(model, model_name, images)
            loss = criterion(outputs, labels)
            
            preds = torch.argmax(outputs, dim=1)
            tracker.update(preds, labels, loss.item())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = tracker.compute()
    return metrics


def train(
    config_path='config/config.yaml',
    num_epochs=50,
    save_dir='models',
    model_name='baseline_cnn',
    run_tag=None,
    overrides=None,
    train_csv='data/processed/train_triage_labels.csv',
    test_csv='data/processed/test_triage_labels.csv',
    train_img_dir='data/processed/images/train',
    test_img_dir='data/processed/images/test',
    image_size=512,
    seed=42,
    auto_class_weights=False,
    checkpoint_interval=5,
    save_final_model=True,
):
    # Main training function
    print("=" * 70)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 70)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    
    print(f"\n Loading data...")
    train_loader, test_loader, dataset_info = get_dataloaders(
        train_csv=train_csv,
        test_csv=test_csv,
        train_img_dir=train_img_dir,
        test_img_dir=test_img_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['gpu'].get('num_workers', 4),
        pin_memory=config['gpu'].get('pin_memory', True),
        image_size=image_size,
    )
    
    print(f" Data loaded:")
    print(f"   Train: {dataset_info['train_size']} images")
    print(f"   Test: {dataset_info['test_size']} images")
    
    print(f"\n Creating model...")
    model, model_label = _build_model(model_name, config, overrides=overrides)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model created: {model_label}")
    print(f"   Parameters: {total_params:,}")
    if model_name != 'baseline_cnn':
        print("   Input mode: synthetic bilateral (left=original, right=horizontal-flip)")
    
    if auto_class_weights:
        auto_weights = compute_auto_class_weights(train_csv)
        class_weights = torch.tensor(auto_weights, dtype=torch.float32).to(device)
    else:
        class_weights = torch.tensor(config['training']['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        mode='max',
        min_delta=0.001
    )
    
    run_name = model_name if not run_tag else f"{model_name}_{run_tag}"

    log_dir = Path(config['logging']['tensorboard_dir']) / run_name
    writer = SummaryWriter(log_dir=log_dir)
    
    save_dir = Path(save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n Training configuration:")
    print(f"   Seed: {seed}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Train CSV: {train_csv}")
    print(f"   Test CSV: {test_csv}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay: {weight_decay}")
    if auto_class_weights:
        print(f"   Class weights (auto from train CSV): {class_weights.detach().cpu().tolist()}")
    else:
        print(f"   Class weights (config): {config['training']['class_weights']}")
        print(f"   Checkpoint interval: {checkpoint_interval}")
        print(f"   Save final model: {save_final_model}")
    print(f"   Early stopping patience: {config['training']['early_stopping']['patience']}")
    
    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print("=" * 70)
    
    best_f1 = 0.0
    history = {'train': [], 'test': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, model_name)
        
        test_metrics = evaluate(model, test_loader, criterion, device, epoch, model_name)
        
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_metrics['avg_loss'], epoch)
        writer.add_scalar('Loss/test', test_metrics['avg_loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/test', test_metrics['accuracy'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1_macro'], epoch)
        writer.add_scalar('F1/test', test_metrics['f1_macro'], epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n Train Metrics:")
        MetricsTracker().print_metrics(train_metrics, prefix="  ")
        print(f"\n Test Metrics:")
        MetricsTracker().print_metrics(test_metrics, prefix="  ")
        
        history['train'].append(train_metrics)
        history['test'].append(test_metrics)
        
        if test_metrics['f1_macro'] > best_f1:
            best_f1 = test_metrics['f1_macro']
            save_checkpoint(
                model, optimizer, epoch, test_metrics,
                save_dir, filename='best_model.pth'
            )
            print(f"New best F1: {best_f1:.4f}")
        
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, test_metrics,
                save_dir, filename=f'checkpoint_epoch_{epoch}.pth'
            )
        
        if early_stopping(test_metrics['f1_macro']):
            print(f"\n Early stopping at epoch {epoch}")
            break
    
    if save_final_model:
        save_checkpoint(
            model, optimizer, epoch, test_metrics,
            save_dir, filename='final_model.pth'
        )
    
    save_metrics_json(history, save_dir / 'training_history.json')
    
    writer.close()
    
    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n Best Test F1: {best_f1:.4f}")
    print(f" Models saved to: {save_dir}")
    print(f" TensorBoard logs: {log_dir}")
    print(f"\nTo view TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Triage Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--model', type=str, default='baseline_cnn',
                       choices=['baseline_cnn', 'late_concat', 'cross_attention'],
                       help='Model to train')
    parser.add_argument('--run_tag', type=str, default=None,
                       help='Optional run suffix for output dirs (e.g., ablation_h4)')
    parser.add_argument('--num_heads', type=int, default=None,
                       help='Override num_heads for cross_attention model')
    parser.add_argument('--attn_dim', type=int, default=None,
                       help='Override attn_dim for cross_attention model')
    parser.add_argument('--train_csv', type=str, default='data/processed/train_triage_labels.csv',
                       help='Path to training CSV')
    parser.add_argument('--test_csv', type=str, default='data/processed/test_triage_labels.csv',
                       help='Path to test CSV')
    parser.add_argument('--train_img_dir', type=str, default='data/processed/images/train',
                       help='Fallback train image directory when CSV has no image_path column')
    parser.add_argument('--test_img_dir', type=str, default='data/processed/images/test',
                       help='Fallback test image directory when CSV has no image_path column')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for resize before normalization')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--auto_class_weights', action='store_true',
                       help='Compute inverse-frequency class weights from training CSV')
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                       help='Save epoch checkpoints every N epochs (0 disables)')
    parser.add_argument('--no_final_model', action='store_true',
                       help='Do not save final_model.pth (best_model is still saved)')
    
    args = parser.parse_args()
    
    overrides = {}
    if args.num_heads is not None:
        overrides['num_heads'] = args.num_heads
    if args.attn_dim is not None:
        overrides['attn_dim'] = args.attn_dim

    model, history = train(
        config_path=args.config,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        model_name=args.model,
        run_tag=args.run_tag,
        overrides=overrides,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_img_dir=args.train_img_dir,
        test_img_dir=args.test_img_dir,
        image_size=args.image_size,
        seed=args.seed,
        auto_class_weights=args.auto_class_weights,
        checkpoint_interval=args.checkpoint_interval,
        save_final_model=not args.no_final_model,
    )
