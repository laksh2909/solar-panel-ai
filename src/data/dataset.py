"""
PyTorch Dataset and DataLoader Pipeline for Solar Panel Fault Detection.

Supports:
- Loading from split manifest CSV (results/metrics/dataset_split.csv) or split directories (data/train, data/val, data/test)
- Clean, deterministic preprocessing (resize to 224x224, RGB conversion, ImageNet normalization)
- Optional training augmentation (disabled by default for baseline experiments)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

from src.preprocessing.pipeline import (
    load_image_rgb,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.preprocessing.augmentation import (
    get_training_augmentation,
    get_validation_pipeline,
    get_test_pipeline,
)
from src.utils.config import load_config, get_project_root
from src.utils.logger import setup_logger

logger = setup_logger("dataset")


class SolarPanelDataset(Dataset):
    """
    PyTorch Dataset representing solar panel images and associated fault labels.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        transform: Optional[A.Compose] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_path = str(row["file_path"])
        class_name = str(row["class_name"])
        label_idx = self.class_to_idx[class_name]

        # Robust RGB loading handling RGBA alpha compositing and grayscale
        image_rgb = load_image_rgb(image_path)

        if self.transform is not None:
            augmented = self.transform(image=image_rgb)
            tensor_img = augmented["image"]
        else:
            # Fallback deterministic transform if none provided
            pipe = get_validation_pipeline()
            tensor_img = pipe(image=image_rgb)["image"]

        return tensor_img, label_idx, image_path


def load_split_dataframe(split_name: str, root_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads DataFrame for a specific split from dataset_split.csv or disk directory.
    """
    root = root_dir or get_project_root()
    split_csv = root / "results" / "metrics" / "dataset_split.csv"

    if split_csv.exists():
        df = pd.read_csv(split_csv)
        subset = df[df["split"] == split_name].copy()
        if not subset.empty:
            return subset

    # Fallback to scanning data/<split_name>/
    split_dir = root / "data" / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Neither {split_csv} nor {split_dir} exists.")

    records = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for class_folder in split_dir.iterdir():
        if class_folder.is_dir() and not class_folder.name.startswith("."):
            c_name = class_folder.name
            for f in class_folder.iterdir():
                if f.is_file() and f.suffix.lower() in valid_exts:
                    records.append({
                        "file_path": str(f.resolve()),
                        "filename": f.name,
                        "class_name": c_name,
                        "split": split_name,
                    })

    return pd.DataFrame(records)


def create_dataloaders(
    config: Optional[Dict] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    use_augmentation: bool = False,  # False for clean baseline reference
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Builds Train, Validation, and Test DataLoaders.
    
    Args:
        config: Project configuration dictionary.
        batch_size: Mini-batch size (default 32).
        num_workers: Number of DataLoader worker threads (default 0 for Windows).
        use_augmentation: If False, uses purely deterministic preprocessing on all splits.
        image_size: Target image dimensions (default 224x224).
        
    Returns:
        (train_loader, val_loader, test_loader, class_to_idx)
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    classes = config.get("dataset", {}).get("classes", config.get("data", {}).get("classes", []))
    class_to_idx = {c: i for i, c in enumerate(sorted(classes))}

    train_df = load_split_dataframe("train", root)
    val_df = load_split_dataframe("val", root)
    test_df = load_split_dataframe("test", root)

    # Transforms setup
    if use_augmentation:
        train_transform = get_training_augmentation(image_size=image_size)
    else:
        # Pure deterministic preprocessing for clean baseline
        train_transform = get_validation_pipeline(image_size=image_size)

    val_transform = get_validation_pipeline(image_size=image_size)
    test_transform = get_test_pipeline(image_size=image_size)

    train_dataset = SolarPanelDataset(train_df, class_to_idx, transform=train_transform)
    val_dataset = SolarPanelDataset(val_df, class_to_idx, transform=val_transform)
    test_dataset = SolarPanelDataset(test_df, class_to_idx, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"DataLoaders ready: Train={len(train_dataset)}, Val={len(val_dataset)}, "
        f"Test={len(test_dataset)}, Batch Size={batch_size}, Augmentation={use_augmentation}"
    )
    return train_loader, val_loader, test_loader, class_to_idx
