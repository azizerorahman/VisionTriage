import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
)
from scipy.stats import wilcoxon
from pathlib import Path
import json


CLASS_NAMES = ['LOW', 'MEDIUM', 'HIGH']


# ── Metrics Tracker ───────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes  = num_classes
        self.class_names  = class_names or CLASS_NAMES
        self.reset()

    def reset(self):
        self.all_preds  = []
        self.all_labels = []
        self.all_losses = []

    def update(self, preds, labels, loss=None):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())

        if loss is not None:
            self.all_losses.append(float(loss))

    def compute(self):
        preds  = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # ── Primary Metrics───────────────────

        # Overall accuracy
        accuracy = accuracy_score(labels, preds)

        f1_macro = f1_score(
            labels, preds,
            average='macro',
            zero_division=0
        )

        f1_weighted = f1_score(
            labels, preds,
            average='weighted',
            zero_division=0
        )

        kappa = cohen_kappa_score(labels, preds)

        # Per-class F1 scores
        f1_per_class_arr = f1_score(
            labels, preds,
            average=None,
            zero_division=0
        )

        # Per-class precision
        precision_arr = precision_score(
            labels, preds,
            average=None,
            zero_division=0
        )

        # Per-class recall
        recall_arr = recall_score(
            labels, preds,
            average=None,
            zero_division=0
        )

        cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

        # Average loss
        avg_loss = float(np.mean(self.all_losses)) \
            if self.all_losses else 0.0

        directional_errors = compute_directional_errors(cm)

        metrics = {
            # Aggregate
            'accuracy':    accuracy,
            'f1_macro':    f1_macro,
            'f1_weighted': f1_weighted,
            'kappa':       kappa,
            'avg_loss':    avg_loss,

            # Per-class (dict keyed by class name)
            'f1_per_class': {
                self.class_names[i]: float(f1_per_class_arr[i])
                for i in range(len(f1_per_class_arr))
            },
            'precision_per_class': {
                self.class_names[i]: float(precision_arr[i])
                for i in range(len(precision_arr))
            },
            'recall_per_class': {
                self.class_names[i]: float(recall_arr[i])
                for i in range(len(recall_arr))
            },

            # Confusion matrix
            'confusion_matrix': cm.tolist(),

            # Directional errors
            'directional_errors': directional_errors,
        }

        return metrics

    def print_metrics(self, metrics, prefix=''):
        """Summary:"""
        print(f"{prefix}{'-' * 50}")
        print(f"{prefix}Accuracy:         {metrics['accuracy']:.4f}")
        print(f"{prefix}Macro-F1:         {metrics['f1_macro']:.4f}")
        print(f"{prefix}Weighted-F1:      {metrics['f1_weighted']:.4f}")
        print(f"{prefix}Cohen's Kappa:    {metrics['kappa']:.4f}")
        print(f"{prefix}Avg Loss:         {metrics['avg_loss']:.4f}")

        print(f"{prefix}")
        print(f"{prefix}Per-Class Performance:")
        print(f"{prefix}  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"{prefix}  {'-'*42}")
        for cls in self.class_names:
            p = metrics['precision_per_class'][cls]
            r = metrics['recall_per_class'][cls]
            f = metrics['f1_per_class'][cls]
            print(f"{prefix}  {cls:<10} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

        print(f"{prefix}")
        print(f"{prefix}Error Analysis:")
        de = metrics['directional_errors']
        print(f"{prefix}  High>Medium (under-triage): {de['high_to_medium']:>3}")
        print(f"{prefix}  High>Low    (under-triage): {de['high_to_low']:>3}")
        print(f"{prefix}  Medium>Low  (under-triage): {de['medium_to_low']:>3}")
        print(f"{prefix}  Low>High    (over-triage):  {de['low_to_high']:>3}")
        print(f"{prefix}  Total under-triage:         {de['total_under_triage']:>3}")

        print(f"{prefix}")
        print(f"{prefix}Confusion Matrix (rows=true, cols=predicted):")
        cm = np.array(metrics['confusion_matrix'])
        header = f"{'':>10}" + "".join(
            f"{n:>10}" for n in self.class_names
        )
        print(f"{prefix}  {header}")
        for i, row_name in enumerate(self.class_names):
            row_str = "".join(f"{cm[i,j]:>10}" for j in range(3))
            print(f"{prefix}  {row_name:>10}{row_str}")
        print(f"{prefix}{'-' * 50}")

def compute_directional_errors(cm):
    cm = np.array(cm)

    high_to_medium   = int(cm[2, 1])
    high_to_low      = int(cm[2, 0])
    medium_to_low    = int(cm[1, 0])
    low_to_high      = int(cm[0, 2])
    low_to_medium    = int(cm[0, 1])
    medium_to_high   = int(cm[1, 2]) 

    total_under_triage = high_to_medium + high_to_low + medium_to_low

    return {
        'high_to_medium':     high_to_medium,
        'high_to_low':        high_to_low,
        'medium_to_low':      medium_to_low,
        'low_to_high':        low_to_high,
        'low_to_medium':      low_to_medium,
        'medium_to_high':     medium_to_high,
        'total_under_triage': total_under_triage,
    }


# ── Tests ───────────────────────────────────

def wilcoxon_significance_test(preds_a, preds_b, labels, alpha=0.05):
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels  = np.array(labels)

    correct_a = (preds_a == labels).astype(int)
    correct_b = (preds_b == labels).astype(int)

    diff = correct_a - correct_b

    if np.all(diff == 0):
        return {
            'statistic':   0.0,
            'p_value':     1.0,
            'significant': False,
            'note':        'No differences between models'
        }

    statistic, p_value = wilcoxon(correct_a, correct_b)

    return {
        'statistic':   float(statistic),
        'p_value':     float(p_value),
        'significant': bool(p_value < alpha),
        'alpha':       alpha,
        'note': (
            f"p={p_value:.4f} — "
            f"{'SIGNIFICANT' if p_value < alpha else 'not significant'} "
            f"at α={alpha}"
        )
    }


def mcnemar_significance_test(preds_a, preds_b, labels, alpha=0.05):
    from statsmodels.stats.contingency_tables import mcnemar

    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels  = np.array(labels)

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {
            'b':           b,
            'c':           c,
            'statistic':   0.0,
            'p_value':     1.0,
            'significant': False,
            'note':        'No discordant pairs found'
        }

    # Build contingency table
    table = np.array([
        [int(np.sum(correct_a & correct_b)),  b],
        [c, int(np.sum(~correct_a & ~correct_b))]
    ])

    result = mcnemar(table, exact=True)

    return {
        'b':           b,
        'c':           c,
        'statistic':   float(result.statistic),
        'p_value':     float(result.pvalue),
        'significant': bool(result.pvalue < alpha),
        'alpha':       alpha,
        'note': (
            f"p={result.pvalue:.4f} — "
            f"{'SIGNIFICANT' if result.pvalue < alpha else 'not significant'} "
            f"at α={alpha} | discordant pairs: b={b}, c={c}"
        )
    }


# ── Seed Robustness Analysis ──────────────────────────────────────────

def compute_seed_robustness(f1_scores):
    scores = np.array(f1_scores)
    mean   = float(np.mean(scores))
    std    = float(np.std(scores))
    cv     = float(std / mean * 100) if mean > 0 else 0.0

    return {
        'mean':   mean,
        'std':    std,
        'min':    float(np.min(scores)),
        'max':    float(np.max(scores)),
        'range':  float(np.max(scores) - np.min(scores)),
        'cv_pct': cv,
        'scores': scores.tolist(),
        'formatted': f"{mean:.4f} ± {std:.4f}",
    }


# ── Checkpoint Utilities ──────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=10, mode='max', min_delta=0.001):
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return True

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter    = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
            return False


def save_checkpoint(model, optimizer, epoch, metrics,
                    checkpoint_dir, filename='checkpoint.pth'):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch':               epoch,
        'model_state_dict':    model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics':             metrics,
        'macro_f1':            metrics.get('f1_macro', 0.0),
    }

    save_path = Path(checkpoint_dir) / filename
    torch.save(checkpoint, save_path)
    return str(save_path)


def save_metrics_json(metrics, save_dir, filename='metrics.json'):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / filename

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=convert)

    return str(save_path)