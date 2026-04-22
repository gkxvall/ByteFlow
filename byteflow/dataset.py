from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def scan_image_folder(root: Path) -> tuple[list[tuple[Path, int]], dict[str, int]]:
    """
    Walk class subfolders under root. Does not open or decode images.

    Returns ordered (path, label) pairs and class_to_idx (sorted class names).
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subfolders found under {root}")

    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    samples: list[tuple[Path, int]] = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        paths = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and _is_image_file(p)
        )
        samples.extend((p, label) for p in paths)

    if not samples:
        raise ValueError(f"No images with extensions {IMAGE_EXTENSIONS} under {root}")

    return samples, class_to_idx


def train_val_split(
    samples: list[tuple[Path, int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    rng = random.Random(seed)
    
    # Stratified split: group by label
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)
        
    train_samples = []
    val_samples = []
    
    for label, class_samples in by_label.items():
        rng.shuffle(class_samples)
        n_train = int(len(class_samples) * train_ratio)
        if len(class_samples) >= 2:
            n_train = max(1, min(n_train, len(class_samples) - 1))
        else:
            n_train = len(class_samples)
        train_samples.extend(class_samples[:n_train])
        val_samples.extend(class_samples[n_train:])
        
    # Shuffle the final lists so classes aren't clustered sequentially
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    
    return train_samples, val_samples


class ByteFlowImageFolder(Dataset):
    """
    Image folder classification dataset: only paths and labels are stored.
    Images are opened and decoded in ``__getitem__`` on demand.
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Missing image file: {path}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read or decode image: {path}") from e

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_val_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_datasets(
    dataset_root: str | Path,
    image_size: int,
    train_split: float,
    seed: int,
) -> tuple[ByteFlowImageFolder, ByteFlowImageFolder, dict[str, int]]:
    root = Path(dataset_root)
    samples, class_to_idx = scan_image_folder(root)
    train_samples, val_samples = train_val_split(samples, train_split, seed)
    train_tf = build_train_transforms(image_size)
    val_tf = build_val_transforms(image_size)
    train_ds = ByteFlowImageFolder(train_samples, class_to_idx, train_tf)
    val_ds = ByteFlowImageFolder(val_samples, class_to_idx, val_tf)
    return train_ds, val_ds, class_to_idx
