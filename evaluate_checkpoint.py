import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.append('/')

from src.models import get_baseline_cnn, get_late_concat_model, get_cross_attention_model
from src.data.dataloader import get_dataloaders
from src.utils.metrics import MetricsTracker


def build_model(model_name, config, num_heads=None, attn_dim=None):
    if model_name == 'baseline_cnn':
        model_config = config['models']['baseline_cnn']
        return get_baseline_cnn({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'fc_hidden_dim': model_config['fc_hidden_dim'],
            'dropout': 0.5,
        })

    if model_name == 'late_concat':
        model_config = config['models']['baseline_concat']
        return get_late_concat_model({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'fc_hidden_dim': model_config['fc_hidden_dim'],
            'dropout': 0.5,
        })

    if model_name == 'cross_attention':
        model_config = config['models']['proposed_crossattn']
        return get_cross_attention_model({
            'num_classes': 3,
            'pretrained': model_config['pretrained'],
            'attn_dim': attn_dim if attn_dim is not None else model_config['attn_dim'],
            'num_heads': num_heads if num_heads is not None else model_config['num_heads'],
            'fc_hidden_dim': model_config['fc_hidden_dim'],
            'dropout': 0.5,
        })

    raise ValueError(f'Unsupported model_name: {model_name}')


def forward_with_mode(model, model_name, images):
    if model_name == 'baseline_cnn':
        return model(images)

    right_images = torch.flip(images, dims=[3])
    return model(images, right_images)


def evaluate(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = build_model(args.model, config, num_heads=args.num_heads, attn_dim=args.attn_dim)
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, test_loader, dataset_info = get_dataloaders(
        train_csv=args.test_csv,
        test_csv=args.test_csv,
        train_img_dir=args.test_img_dir,
        test_img_dir=args.test_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        image_size=args.image_size,
    )

    tracker = MetricsTracker(num_classes=3, class_names=dataset_info['class_names'])
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = forward_with_mode(model, args.model, images)
            preds = torch.argmax(outputs, dim=1)
            tracker.update(preds, labels)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = tracker.compute()

    out = {
        'model': args.model,
        'checkpoint': args.checkpoint,
        'test_csv': args.test_csv,
        'test_size': dataset_info['test_size'],
        'metrics': metrics,
        'labels': all_labels,
        'predictions': all_preds,
    }

    print('\nEvaluation Metrics')
    print('=' * 70)
    tracker.print_metrics(metrics, prefix='  ')

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nSaved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--model', type=str, required=True, choices=['baseline_cnn', 'late_concat', 'cross_attention'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--test_img_dir', type=str, default='data/processed/images/test')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--attn_dim', type=int, default=None)
    parser.add_argument('--output_json', type=str, required=True)

    args = parser.parse_args()
    evaluate(args)
