import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import json
from pathlib import Path


class MetricsTracker:
    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ['LOW', 'MEDIUM', 'HIGH']
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_losses = []

    def update(self, preds, labels, loss=None):
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())
        if loss is not None:
            self.all_losses.append(loss)

    def compute(self):
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        kappa = cohen_kappa_score(labels, preds)
        f1_per_class = f1_score(labels, preds, average=None)
        cm = confusion_matrix(labels, preds)
        avg_loss = np.mean(self.all_losses) if self.all_losses else 0.0

        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': kappa,
            'f1_per_class': {self.class_names[i]: f1_per_class[i] for i in range(len(f1_per_class))},
            'confusion_matrix': cm.tolist(),
            'avg_loss': avg_loss
        }

        return metrics

    def print_metrics(self, metrics, prefix=""):
        print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
        print(f"{prefix}F1 (Macro): {metrics['f1_macro']:.4f}")
        print(f"{prefix}F1 (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"{prefix}Cohen's Kappa: {metrics['kappa']:.4f}")
        print(f"{prefix}Avg Loss: {metrics['avg_loss']:.4f}")
        for class_name, f1 in metrics['f1_per_class'].items():
            print(f"{prefix}  {class_name}: {f1:.4f}")


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, filename='checkpoint.pth'):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    checkpoint_path = Path(checkpoint_dir) / filename
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    return epoch, metrics


def save_metrics_json(metrics, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0, monitor='val_loss', mode='min'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, current_score):
        if self.mode == 'min':
            if current_score < self.best_score - self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_score > self.best_score + self.delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
