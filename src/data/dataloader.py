import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
from pathlib import Path


# ── Label Mapping ─────────────────────────────────────────────────────
IDRID_LABEL_MAP = {
    0: 0,   # No DR           → Low
    1: 0,   # Mild NPDR       → Low
    2: 1,   # Moderate NPDR   → Medium
    3: 2,   # Severe NPDR     → High
    4: 2,   # PDR             → High
}

DODR_LABEL_MAP = {
    0: 0,   # DR absent  → Low
    1: 2,   # DR present → High (Medium deliberately excluded)
}

CLASS_NAMES = ['LOW', 'MEDIUM', 'HIGH']


# ── Preprocessing Pipeline ────────────────────────────────────────────

def preprocess_fundus_image(image_bgr):
    # ── CLAHE in LAB colour space ────────────────────────────
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    image_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── Bilinear resize to 512x512 ───────────────────────────
    image_bgr = cv2.resize(
        image_bgr,
        (512, 512),
        interpolation=cv2.INTER_LINEAR   # NOT INTER_AREA
    )

    # ── Centre crop to 480x480 ───────────────────────────────
    start = (512 - 480) // 2   # = 16 pixels from each edge
    image_bgr = image_bgr[start:start + 480, start:start + 480]

    # ── BGR → RGB ─────────────────────────────────────────────
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb   # [480, 480, 3] uint8


def make_synthetic_right_eye(tensor):
    return torch.flip(tensor, dims=[2])   # Flip width dimension (W)


# ── Augmentation ───────────────────────────────────────────────

def get_training_augmentation(crop_size=480):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=30)
        ], p=0.7),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.0
            )
        ], p=0.5),
        transforms.RandomResizedCrop(
            size=crop_size,
            scale=(0.8, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ── Dataset Classes ───────────────────────────────────────────────────

class IDRiDTriageDataset(Dataset):
    def __init__(self, csv_path, image_dir,
                 augment=False, bilateral=False,
                 label_col='Retinopathy grade'):

        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.bilateral = bilateral
        self.label_col = label_col

        # Select appropriate transform
        if self.augment:
            self.transform = get_training_augmentation(crop_size=480)
        else:
            self.transform = get_eval_transform()

        # Validate CSV columns
        self._validate_columns()

    def _validate_columns(self):
        cols = self.df.columns.tolist()
        has_id = 'Image name' in cols or 'image_id' in cols or 'image_path' in cols
        has_label = (self.label_col in cols or
                     'urgency_label' in cols or
                     'urgency' in cols)
        if not has_id:
            raise ValueError(
                f"CSV must have 'Image name', 'image_id', or 'image_path' column. "
                f"Found: {cols}"
            )
        if not has_label:
            raise ValueError(
                f"CSV must have '{self.label_col}', 'urgency_label', or 'urgency' column. "
                f"Found: {cols}"
            )

    def _get_image_id(self, row):
        if 'Image name' in self.df.columns:
            return str(row['Image name'])
        elif 'image_id' in self.df.columns:
            return str(row['image_id'])
        else:
            return Path(str(row['image_path'])).stem

    def _get_label(self, row):
        # Map raw label to urgency tier (0/1/2)
        if 'urgency_label' in self.df.columns:
            return int(row['urgency_label'])

        # String urgency label
        if 'urgency' in self.df.columns:
            mapping = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
            return mapping[str(row['urgency']).strip().upper()]

        grade = int(row[self.label_col])
        return IDRID_LABEL_MAP[grade]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = self._get_image_id(row)
        label = self._get_label(row)

        # Load image
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            # Try without extension
            candidates = list(self.image_dir.glob(f"{image_id}.*"))
            if not candidates:
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_path = candidates[0]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"cv2 failed to load: {image_path}")

        # Apply preprocessing pipeline
        image_rgb = preprocess_fundus_image(image_bgr)

        if self.bilateral:
            # Each stream gets different random augmentation parameters
            x_left = self.transform(image_rgb)           # [3, 480, 480]

            # Generate right eye
            right_rgb = cv2.flip(image_rgb, 1)           # horizontal flip
            x_right = self.transform(right_rgb)          # [3, 480, 480]

            return (x_left, x_right), label, image_id

        else:
            x = self.transform(image_rgb)                # [3, 480, 480]
            return x, label, image_id


class DODRTriageDataset(Dataset):
    def __init__(self, csv_path, image_dir,
                 augment=False, bilateral=False):

        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.augment = augment
        self.bilateral = bilateral

        if self.augment:
            self.transform = get_training_augmentation(crop_size=480)
        else:
            self.transform = get_eval_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get image path
        if 'image_path' in self.df.columns:
            image_path = Path(str(row['image_path']))
        else:
            image_id = str(row['image_id'])
            image_path = self.image_dir / f"{image_id}.jpg"

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"cv2 failed to load: {image_path}")

        # Apply preprocessing
        image_rgb = preprocess_fundus_image(image_bgr)

        # Binary mapping
        binary_label = int(row['label']) if 'label' in self.df.columns \
            else int(row['diagnosis'])
        label = DODR_LABEL_MAP[binary_label]

        image_id = str(image_path.stem)

        if self.bilateral:
            x_left = self.transform(image_rgb)
            right_rgb = cv2.flip(image_rgb, 1)
            x_right = self.transform(right_rgb)
            return (x_left, x_right), label, image_id
        else:
            x = self.transform(image_rgb)
            return x, label, image_id


class MultisourceDataset(Dataset):
    def __init__(self, idrid_dataset, dodr_dataset):
        self.idrid = idrid_dataset
        self.dodr = dodr_dataset
        self.idrid_len = len(idrid_dataset)
        self.dodr_len = len(dodr_dataset)

    def __len__(self):
        return self.idrid_len + self.dodr_len

    def __getitem__(self, idx):
        if idx < self.idrid_len:
            return self.idrid[idx]
        else:
            return self.dodr[idx - self.idrid_len]


# ── DataLoader Factory ────────────────────────────────────────────────

def get_dataloaders(
    train_csv,
    test_csv,
    train_img_dir,
    test_img_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    bilateral=False,
    label_col='Retinopathy grade',
):

    train_dataset = IDRiDTriageDataset(
        csv_path=train_csv,
        image_dir=train_img_dir,
        augment=True,           # Augmentation ON for training
        bilateral=bilateral,
        label_col=label_col,
    )

    test_dataset = IDRiDTriageDataset(
        csv_path=test_csv,
        image_dir=test_img_dir,
        augment=False,          # Augmentation OFF for test
        bilateral=bilateral,
        label_col=label_col,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # Never shuffle test set
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    dataset_info = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'num_classes': 3,
        'class_names': CLASS_NAMES,
        'bilateral': bilateral,
    }

    return train_loader, test_loader, dataset_info


def get_multisource_dataloaders(
    idrid_train_csv,
    idrid_train_img_dir,
    dodr_train_csv,
    dodr_train_img_dir,
    idrid_test_csv,
    idrid_test_img_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    bilateral=False,
    label_col='Retinopathy grade',
):

    idrid_train = IDRiDTriageDataset(
        csv_path=idrid_train_csv,
        image_dir=idrid_train_img_dir,
        augment=True,
        bilateral=bilateral,
        label_col=label_col,
    )

    dodr_train = DODRTriageDataset(
        csv_path=dodr_train_csv,
        image_dir=dodr_train_img_dir,
        augment=True,
        bilateral=bilateral,
    )

    multisource_train = MultisourceDataset(idrid_train, dodr_train)

    idrid_test = IDRiDTriageDataset(
        csv_path=idrid_test_csv,
        image_dir=idrid_test_img_dir,
        augment=False,
        bilateral=bilateral,
        label_col=label_col,
    )

    train_loader = DataLoader(
        multisource_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        idrid_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    dataset_info = {
        'train_size': len(multisource_train),
        'idrid_train_size': len(idrid_train),
        'dodr_train_size': len(dodr_train),
        'test_size': len(idrid_test),
        'num_classes': 3,
        'class_names': CLASS_NAMES,
        'bilateral': bilateral,
        'mode': 'multisource',
    }

    return train_loader, test_loader, dataset_info