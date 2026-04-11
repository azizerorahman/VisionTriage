import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms


class IDRiDTriageDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, augmentation=None, image_size=512):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.augmentation = augmentation
        self.image_size = image_size
        self.has_image_path = 'image_path' in self.df.columns
        self.has_urgency_label = 'urgency_label' in self.df.columns
        self.urgency_to_label = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.has_image_path:
            image_path = Path(row['image_path'])
            image_id = str(row['image_id']) if 'image_id' in self.df.columns else image_path.stem
        else:
            image_id = str(row['image_id'])
            image_path = self.image_dir / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.image_size is not None:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if self.augmentation is not None:
            image = self.augmentation(image)

        image = image.astype(np.float32) / 255.0

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)

        if self.has_urgency_label:
            label = int(row['urgency_label'])
        else:
            urgency = str(row['urgency']).strip().upper()
            label = self.urgency_to_label[urgency]

        return image, label, image_id


def get_transforms(config=None):
    if config is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = config.get('mean', [0.485, 0.456, 0.406])
        std = config.get('std', [0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


def get_dataloaders(train_csv, test_csv, train_img_dir, test_img_dir,
                   batch_size=32, num_workers=4, pin_memory=True, image_size=512):
    transform = get_transforms()

    train_dataset = IDRiDTriageDataset(
        csv_path=train_csv,
        image_dir=train_img_dir,
        transform=transform,
        augmentation=None,
        image_size=image_size
    )

    test_dataset = IDRiDTriageDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        transform=transform,
        augmentation=None,
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    def _label_counts(df):
        counts = {0: 0, 1: 0, 2: 0}
        if 'urgency_label' in df.columns:
            for key, value in df['urgency_label'].value_counts().to_dict().items():
                counts[int(key)] = int(value)
        elif 'urgency' in df.columns:
            mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            encoded = df['urgency'].astype(str).str.upper().map(mapping)
            for key, value in encoded.value_counts(dropna=True).to_dict().items():
                counts[int(key)] = int(value)
        return counts

    train_label_counts = _label_counts(train_df)
    test_label_counts = _label_counts(test_df)
    label_to_name = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}

    dataset_info = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'num_classes': 3,
        'class_names': ['LOW', 'MEDIUM', 'HIGH'],
        'train_distribution': {label_to_name[k]: v for k, v in train_label_counts.items()},
        'test_distribution': {label_to_name[k]: v for k, v in test_label_counts.items()},
        'train_label_counts': train_label_counts,
        'test_label_counts': test_label_counts
    }

    return train_loader, test_loader, dataset_info


if __name__ == '__main__':
    train_csv = 'data/processed/train_triage_labels.csv'
    test_csv = 'data/processed/test_triage_labels.csv'
    train_img_dir = 'data/processed/images/train'
    test_img_dir = 'data/processed/images/test'

    train_loader, test_loader, info = get_dataloaders(
        train_csv, test_csv, train_img_dir, test_img_dir,
        batch_size=8, num_workers=0
    )

    print(f"Train size: {info['train_size']}")
    print(f"Test size: {info['test_size']}")
    print(f"Classes: {info['num_classes']}")
    print(f"Train distribution: {info['train_distribution']}")

    images, labels, image_ids = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels.tolist()}")
