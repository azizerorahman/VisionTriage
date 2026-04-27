import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    preprocess_batch,
    validate_preprocessed_image,
    TARGET_SIZE,
)


# ── IDRiD ───────────────────────────────────────────────

def preprocess_idrid(config, overwrite=False):
    ds_cfg = config['dataset']

    splits = {
        'train': {
            'input_dir':  Path(ds_cfg['train_image_dir']),
            'output_dir': Path('data/processed/IDRiD/train'),
        },
        'test': {
            'input_dir':  Path(ds_cfg['test_image_dir']),
            'output_dir': Path('data/processed/IDRiD/test'),
        },
    }

    summary = {}

    for split_name, paths in splits.items():
        input_dir  = paths['input_dir']
        output_dir = paths['output_dir']

        if not input_dir.exists():
            print(f"IDRiD {split_name} dir not found: {input_dir}")
            summary[split_name] = {'success': 0, 'failed': 0, 'skipped': 0}
            continue

        # Collect all image files
        image_paths = sorted(
            list(input_dir.glob('*.jpg')) +
            list(input_dir.glob('*.jpeg')) +
            list(input_dir.glob('*.png'))
        )

        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            summary[split_name] = {'success': 0, 'failed': 0, 'skipped': 0}
            continue

        print(f"\n  IDRiD {split_name}: {len(image_paths)} images")
        print(f"    Input:  {input_dir}")
        print(f"    Output: {output_dir}")

        results = preprocess_batch(
            image_paths      = image_paths,
            output_dir       = output_dir,
            apply_ben_graham = True,
            apply_green_ch   = True,
            apply_clahe_flag = True,
            target_size      = TARGET_SIZE,
            overwrite        = overwrite,
        )

        n_success = sum(r['success'] for r in results)
        n_failed  = sum(not r['success'] for r in results)
        n_skipped = sum(r.get('skipped', False) for r in results)

        print(f"{n_success} success | "
              f"{n_failed} failed | "
              f"{n_skipped} skipped")

        if n_failed > 0:
            print(f"    Failed images:")
            for r in results:
                if not r['success']:
                    print(f"      {r['input']}: {r['error']}")

        summary[split_name] = {
            'success':    n_success,
            'failed':     n_failed,
            'skipped':    n_skipped,
            'output_dir': str(output_dir),
        }

    return summary


# ── DODR ────────────────────────────────────────────────

def preprocess_dodr(config, overwrite=False):
    ds_cfg    = config['dataset']
    input_dir = Path(ds_cfg.get('dodr_image_dir', 'data/raw/DODR/images'))

    if not input_dir.exists():
        print(f"DODR image dir not found: {input_dir}")
        return {'success': 0, 'failed': 0, 'skipped': 0}

    output_dir = Path('data/processed/DODR')

    image_paths = sorted(
        list(input_dir.glob('*.jpg')) +
        list(input_dir.glob('*.jpeg')) +
        list(input_dir.glob('*.png'))
    )

    if len(image_paths) == 0:
        print(f"No DODR images found in {input_dir}")
        return {'success': 0, 'failed': 0, 'skipped': 0}

    print(f"\n  DODR: {len(image_paths)} images")
    print(f"    Input:  {input_dir}")
    print(f"    Output: {output_dir}")

    results = preprocess_batch(
        image_paths      = image_paths,
        output_dir       = output_dir,
        apply_ben_graham = True,
        apply_green_ch   = True,
        apply_clahe_flag = True,
        target_size      = TARGET_SIZE,
        overwrite        = overwrite,
    )

    n_success = sum(r['success'] for r in results)
    n_failed  = sum(not r['success'] for r in results)
    n_skipped = sum(r.get('skipped', False) for r in results)

    print(f"{n_success} success | "
          f"{n_failed} failed | "
          f"{n_skipped} skipped")

    return {
        'success':    n_success,
        'failed':     n_failed,
        'skipped':    n_skipped,
        'output_dir': str(output_dir),
    }


# ── Validation ───────────────────────────────────────────────────

def validate_all_preprocessed(config):
    processed_root = Path('data/processed')

    if not processed_root.exists():
        print(f"Processed data directory not found: {processed_root}")
        print(f"Run preprocessing first.")
        return {}

    all_images = sorted(processed_root.rglob('*_preprocessed.png'))

    if len(all_images) == 0:
        print(f"No preprocessed images found in {processed_root}")
        return {}

    print(f"\n  Validating {len(all_images)} preprocessed images...")

    n_valid   = 0
    n_invalid = 0
    issues    = []

    for img_path in tqdm(all_images, desc='  Validating', leave=False):
        valid, img_issues = validate_preprocessed_image(img_path)
        if valid:
            n_valid += 1
        else:
            n_invalid += 1
            issues.append({
                'path':   str(img_path),
                'issues': img_issues,
            })

    print(f"\n Validation Results:")
    print(f"    Valid:   {n_valid}")
    print(f"    Invalid: {n_invalid}")

    if n_invalid > 0:
        print(f"\n  Invalid images:")
        for item in issues[:10]:   # Show first 10
            print(f"    {item['path']}")
            for issue in item['issues']:
                print(f"      → {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")

    return {
        'total':    len(all_images),
        'valid':    n_valid,
        'invalid':  n_invalid,
        'issues':   issues,
    }

def main(args):
    print("Dataset Preprocessing")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    start_time = time.time()

    # Validate only
    if args.validate_only:
        validation = validate_all_preprocessed(config)
        return

    metadata = {
        'target_size':      list(TARGET_SIZE),
        'apply_ben_graham': True,
        'apply_green_ch':   True,
        'apply_clahe':      True,
        'clahe_clip':       2.0,
        'clahe_tile':       [8, 8],
        'datasets':         {},
    }

    # Preprocess IDRiD
    if args.dataset in ('idrid', 'all'):
        print(f"\n{'-' * 70}")
        print(f"  Processing IDRiD dataset...")
        print(f"{'-' * 70}")
        idrid_summary = preprocess_idrid(config, overwrite=args.overwrite)
        metadata['datasets']['idrid'] = idrid_summary

    # Preprocess DODR
    if args.dataset in ('dodr', 'all'):
        print(f"\n{'-' * 70}")
        print(f"  Processing DODR dataset...")
        print(f"{'-' * 70}")
        dodr_summary = preprocess_dodr(config, overwrite=args.overwrite)
        metadata['datasets']['dodr'] = dodr_summary

    # Validate output
    print(f"\n{'-' * 70}")
    print(f"  Running validation pass...")
    print(f"{'-' * 70}")
    validation = validate_all_preprocessed(config)
    metadata['validation'] = validation

    # Save metadata
    metadata_path = Path('data/processed_metadata.json')
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time

    print(f"\n{'-' * 70}")
    print(f"  Preprocessing complete in {elapsed:.1f}s")
    print(f"  Metadata saved: {metadata_path}")
    print(f"{'-' * 70}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset Preprocessing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['idrid', 'dodr', 'all'],
        help='Which dataset to preprocess'
    )
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to config YAML'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing preprocessed images'
    )
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only validate existing preprocessed images'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())