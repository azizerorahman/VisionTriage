import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models import get_model, BILATERAL_MODELS
from src.data.dataloader import get_dataloaders, get_multisource_dataloaders
from src.utils.metrics import (
    MetricsTracker,
    compute_directional_errors,
    compute_seed_robustness,
    wilcoxon_significance_test,
    mcnemar_significance_test,
    save_metrics_json,
)
from src.utils.grad_cam import (
    get_gradcam_for_model,
    compute_iou,
    compute_iou_batch,
    compute_spatial_correlation,
    is_implausible_activation,
)

def load_checkpoint(checkpoint_path, model, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run train.py first to generate checkpoints."
        )

    print(f"  Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch   = checkpoint.get('epoch', -1)
    metrics = checkpoint.get('metrics', {})
    macro_f1 = checkpoint.get('macro_f1', metrics.get('f1_macro', 0.0))

    print(f"  Checkpoint epoch:    {epoch}")
    print(f"  Checkpoint Macro-F1: {macro_f1:.4f}")

    return epoch, metrics

def run_evaluation(model, test_loader, device,
                   is_bilateral, criterion=None):
    model.eval()
    tracker   = MetricsTracker(num_classes=3)
    all_preds  = []
    all_labels = []
    all_ids    = []

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    pbar = tqdm(test_loader, desc='  Evaluating', leave=False)

    with torch.no_grad():
        for batch in pbar:
            inputs, labels, image_ids = batch
            labels = labels.to(device)

            # Forward pass
            if is_bilateral:
                if isinstance(inputs, (list, tuple)):
                    x_left  = inputs[0].to(device)
                    x_right = inputs[1].to(device)
                else:
                    x_left  = inputs.to(device)
                    x_right = torch.flip(x_left, dims=[3])
                logits = model(x_left, x_right)
            else:
                if isinstance(inputs, (list, tuple)):
                    x = inputs[0].to(device)
                else:
                    x = inputs.to(device)
                logits = model(x)

            loss  = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            tracker.update(preds, labels, loss.item())

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_ids.extend(
                image_ids if isinstance(image_ids, list)
                else image_ids.tolist()
            )

    metrics = tracker.compute()
    return (
        np.array(all_preds),
        np.array(all_labels),
        all_ids,
        metrics,
    )

def run_gradcam_evaluation(model, test_loader, device,
                           is_bilateral, model_type,
                           output_dir, lesion_mask_dir=None):
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gradcam = get_gradcam_for_model(model, model_type)

    iou_scores         = []
    implausible_count  = 0
    heatmap_paths      = []
    correlation_scores = []

    print(f"\n  Generating Grad-CAM heatmaps → {output_dir}")

    pbar = tqdm(test_loader, desc='  Grad-CAM', leave=False)

    for batch in pbar:
        inputs, labels, image_ids = batch
        batch_size = labels.shape[0]

        for i in range(batch_size):
            img_id = (
                image_ids[i]
                if isinstance(image_ids, list)
                else str(image_ids[i].item())
            )

            if isinstance(inputs, (list, tuple)):
                x_left  = inputs[0][i:i+1].to(device)
                x_right = inputs[1][i:i+1].to(device) \
                    if is_bilateral else None
            else:
                x_left  = inputs[i:i+1].to(device)
                x_right = torch.flip(x_left, dims=[3]) \
                    if is_bilateral else None

            heatmap, pred_class, pred_probs = gradcam.generate(
                input_left  = x_left,
                class_idx   = None,
                input_right = x_right,
                stream      = 'left',
            )

            # ── IoU vs lesion mask ─────────────────────────────────
            if lesion_mask_dir is not None:
                mask_path = Path(lesion_mask_dir) / f"{img_id}_mask.png"
                if mask_path.exists():
                    lesion_mask = cv2.imread(
                        str(mask_path), cv2.IMREAD_GRAYSCALE
                    )
                    iou = compute_iou(heatmap, lesion_mask, threshold=0.5)
                    iou_scores.append(iou)

            implausible, border_ratio = is_implausible_activation(
                heatmap,
                np.zeros((480, 480, 3), dtype=np.uint8),
            )
            if implausible:
                implausible_count += 1

            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_colour = cv2.applyColorMap(
                heatmap_uint8, cv2.COLORMAP_JET
            )
            save_path = output_dir / f"{img_id}_gradcam.png"
            cv2.imwrite(str(save_path), heatmap_colour)
            heatmap_paths.append(str(save_path))

    gradcam.remove_hooks()

    # Compute summary stats
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    std_iou  = float(np.std(iou_scores))  if iou_scores else 0.0

    gradcam_results = {
        'mean_iou':          mean_iou,
        'std_iou':           std_iou,
        'iou_scores':        iou_scores,
        'n_evaluated':       len(iou_scores),
        'implausible_count': implausible_count,
        'implausible_rate':  implausible_count / max(len(heatmap_paths), 1),
        'heatmap_dir':       str(output_dir),
        'n_heatmaps':        len(heatmap_paths),
    }

    print(f"\n  Grad-CAM Results:")
    if iou_scores:
        print(f"    Mean IoU:          {mean_iou:.3f} ± {std_iou:.3f}")
    print(f"    Implausible maps:  "
          f"{implausible_count}/{len(heatmap_paths)} "
          f"({gradcam_results['implausible_rate']*100:.1f}%)")
    print(f"    Heatmaps saved:    {output_dir}")

    return gradcam_results

def run_statistical_comparison(preds_a, preds_b, labels,
                               name_a='Model A', name_b='Model B'):
    print(f"\n  Statistical Significance: {name_a} vs {name_b}")
    print(f"  {'─' * 50}")

    wilcoxon_result = wilcoxon_significance_test(
        preds_a, preds_b, labels, alpha=0.05
    )
    mcnemar_result  = mcnemar_significance_test(
        preds_a, preds_b, labels, alpha=0.05
    )

    print(f"  Wilcoxon: {wilcoxon_result['note']}")
    print(f"  McNemar:  {mcnemar_result['note']}")

    return {
        'wilcoxon': wilcoxon_result,
        'mcnemar':  mcnemar_result,
        'model_a':  name_a,
        'model_b':  name_b,
    }

def print_full_results(metrics, model_name, epoch):
    CLASS_NAMES = ['LOW', 'MEDIUM', 'HIGH']

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION RESULTS - {model_name} (epoch {epoch})")
    print(f"{'=' * 70}")

    print(f"\n  ── Aggregate Metrics ──────────────────────────────────")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:         {metrics['f1_macro']:.4f}  ← primary")
    print(f"  Weighted-F1:      {metrics['f1_weighted']:.4f}")
    print(f"  Cohen's Kappa:    {metrics['kappa']:.4f}")

    print(f"\n  ── Per-Class Metrics ──────────────────────────────────")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'─' * 42}")
    for cls in CLASS_NAMES:
        p = metrics['precision_per_class'].get(cls, 0.0)
        r = metrics['recall_per_class'].get(cls, 0.0)
        f = metrics['f1_per_class'].get(cls, 0.0)
        print(f"  {cls:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

    print(f"\n  ── Confusion Matrix ───────")
    import numpy as np
    cm = np.array(metrics['confusion_matrix'])
    header = f"  {'':>10}" + "".join(f"{n:>10}" for n in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = "".join(f"{cm[i,j]:>10}" for j in range(3))
        print(f"  {row_name:>10}{row_str}")

    print(f"\n  ── Error Analysis ─────────────────────────")
    de = metrics.get('directional_errors', {})
    if de:
        print(f"  High→Medium (under-triage): "
              f"{de.get('high_to_medium', 0):>3}  ⚠ dangerous")
        print(f"  High→Low    (under-triage): "
              f"{de.get('high_to_low', 0):>3}  ⚠ most dangerous")
        print(f"  Medium→Low  (under-triage): "
              f"{de.get('medium_to_low', 0):>3}")
        print(f"  Low→High    (over-triage):  "
              f"{de.get('low_to_high', 0):>3}  (acceptable)")
        print(f"  Total under-triage:         "
              f"{de.get('total_under_triage', 0):>3}")

    print(f"\n{'=' * 70}\n")

def evaluate(
    checkpoint_path,
    model_name,
    config_path         = 'config/config.yaml',
    multisource         = False,
    run_gradcam         = False,
    gradcam_output_dir  = 'results/gradcam',
    lesion_mask_dir     = None,
    compare_checkpoint  = None,
    compare_model_name  = None,
    output_dir          = 'results',
):

    # ── Config ────────────────────────────────────────────────────────

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("=" * 70)
    print(f"  VisionTriage Evaluation")
    print(f"  Model:      {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device:     {device}")
    print("=" * 70)

    # ── Data Loading ──────────────────────────────────────────────────

    ds_cfg       = config['dataset']
    is_bilateral = BILATERAL_MODELS.get(model_name, False)

    print(f"\n  Loading test data (bilateral={is_bilateral})...")

    _, test_loader, dataset_info = get_dataloaders(
        train_csv     = ds_cfg['train_csv'],
        test_csv      = ds_cfg['test_csv'],
        train_img_dir = ds_cfg['train_image_dir'],
        test_img_dir  = ds_cfg['test_image_dir'],
        batch_size    = config['training']['batch_size'],
        num_workers   = config['gpu'].get('num_workers', 4),
        pin_memory    = config['gpu'].get('pin_memory', True),
        bilateral     = is_bilateral,
        label_col     = ds_cfg.get('diagnosis_column', 'Retinopathy grade'),
    )

    print(f"  Test samples: {dataset_info['test_size']}")

    # ── Primary Model ─────────────────────────────────────────────────

    print(f"\n  Building model: {model_name}...")
    model, is_bilateral = get_model(model_name)
    model = model.to(device)

    epoch, ckpt_metrics = load_checkpoint(
        checkpoint_path, model, device
    )

    # ── Run Evaluation ────────────────────────────────────────────────

    print(f"\n  Running evaluation on test set...")
    criterion = nn.CrossEntropyLoss()

    preds_a, labels, image_ids, metrics = run_evaluation(
        model, test_loader, device, is_bilateral, criterion
    )

    print_full_results(metrics, model_name, epoch)

    # ── Grad-CAM ──────────────────────────────────────────────────────

    gradcam_results = None
    if run_gradcam:
        model_type_map = {
            'baseline_cnn':    'monocular',
            'm1':              'monocular',
            'late_concat':     'late_concat',
            'm2':              'late_concat',
            'cross_attention': 'cross_attention',
            'm3':              'cross_attention',
        }
        model_type = model_type_map.get(model_name, 'monocular')

        gradcam_results = run_gradcam_evaluation(
            model           = model,
            test_loader     = test_loader,
            device          = device,
            is_bilateral    = is_bilateral,
            model_type      = model_type,
            output_dir      = gradcam_output_dir,
            lesion_mask_dir = lesion_mask_dir,
        )

    # ── Statistical Comparison ────────────────────────────────────────
    
    stats_results = None
    if compare_checkpoint is not None and compare_model_name is not None:
        print(f"\n  Loading comparison model: {compare_model_name}...")

        is_bilateral_b = BILATERAL_MODELS.get(compare_model_name, False)
        model_b, _     = get_model(compare_model_name)
        model_b        = model_b.to(device)

        load_checkpoint(compare_checkpoint, model_b, device)

        if is_bilateral_b != is_bilateral:
            _, test_loader_b, _ = get_dataloaders(
                train_csv     = ds_cfg['train_csv'],
                test_csv      = ds_cfg['test_csv'],
                train_img_dir = ds_cfg['train_image_dir'],
                test_img_dir  = ds_cfg['test_image_dir'],
                batch_size    = config['training']['batch_size'],
                num_workers   = config['gpu'].get('num_workers', 4),
                pin_memory    = config['gpu'].get('pin_memory', True),
                bilateral     = is_bilateral_b,
                label_col     = ds_cfg.get(
                    'diagnosis_column', 'Retinopathy grade'
                ),
            )
        else:
            test_loader_b = test_loader

        preds_b, labels_b, _, metrics_b = run_evaluation(
            model_b, test_loader_b, device, is_bilateral_b, criterion
        )

        print_full_results(metrics_b, compare_model_name, -1)

        stats_results = run_statistical_comparison(
            preds_a    = preds_a,
            preds_b    = preds_b,
            labels     = labels,
            name_a     = model_name,
            name_b     = compare_model_name,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_name':     model_name,
        'checkpoint':     str(checkpoint_path),
        'epoch':          epoch,
        'metrics':        metrics,
        'gradcam':        gradcam_results,
        'statistics':     stats_results,
    }

    results_path = save_metrics_json(
        results,
        str(output_dir),
        filename=f'eval_{model_name}.json',
    )

    print(f"\n  Results saved {results_path}")
    print(f"\n  Primary Macro-F1: {metrics['f1_macro']:.4f}")

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='VisionTriage Checkpoint Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary model
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to primary model checkpoint (.pth)'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['baseline_cnn', 'late_concat', 'cross_attention',
                 'm1', 'm2', 'm3'],
        help='Primary model architecture'
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to config YAML'
    )
    parser.add_argument(
        '--multisource', action='store_true',
        help='Model was trained with multisource data'
    )

    # Grad-CAM
    parser.add_argument(
        '--gradcam', action='store_true',
        help='Generate Grad-CAM heatmaps'
    )
    parser.add_argument(
        '--gradcam-output', type=str, default='results/gradcam',
        help='Directory to save Grad-CAM heatmap PNGs'
    )
    parser.add_argument(
        '--lesion-masks', type=str, default=None,
        help='IDRiD lesion mask directory for IoU computation'
    )

    # Comparison model
    parser.add_argument(
        '--compare-checkpoint', type=str, default=None,
        help='Checkpoint of second model for statistical comparison'
    )
    parser.add_argument(
        '--compare-model', type=str, default=None,
        choices=['baseline_cnn', 'late_concat', 'cross_attention',
                 'm1', 'm2', 'm3', None],
        help='Architecture of comparison model'
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default='results',
        help='Directory to save evaluation JSON results'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    evaluate(
        checkpoint_path    = args.checkpoint,
        model_name         = args.model,
        config_path        = args.config,
        multisource        = args.multisource,
        run_gradcam        = args.gradcam,
        gradcam_output_dir = args.gradcam_output,
        lesion_mask_dir    = args.lesion_masks,
        compare_checkpoint = args.compare_checkpoint,
        compare_model_name = args.compare_model,
        output_dir         = args.output_dir,
    )