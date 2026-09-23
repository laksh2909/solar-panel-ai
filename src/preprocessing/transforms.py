"""
Image Preprocessing and Data Augmentation Pipeline.

Tailored for solar panel defect detection:
- Geometric invariants: Horiz/Vert flips, slight affine rotations
- Photometric variations: Sun glares, shadows, dust haze, camera noise
- Tensor normalization using standard ImageNet mean and std
"""

from typing import Tuple, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    aug_params: Dict[str, Any] | None = None
) -> A.Compose:
    """
    Returns the training augmentation pipeline using Albumentations.
    """
    if aug_params is None:
        aug_params = {
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.3,
            "random_rotate_degrees": 15,
            "random_rotate_p": 0.5,
            "brightness_contrast_p": 0.4,
            "color_jitter_p": 0.3,
            "motion_blur_p": 0.2,
            "gaussian_noise_p": 0.2,
        }

    h, w = image_size

    transform_list = [
        A.Resize(h, w),
        A.HorizontalFlip(p=aug_params.get("horizontal_flip_p", 0.5)),
        A.VerticalFlip(p=aug_params.get("vertical_flip_p", 0.3)),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.1,
            rotate_limit=aug_params.get("random_rotate_degrees", 15),
            p=aug_params.get("random_rotate_p", 0.5),
            border_mode=0
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=aug_params.get("brightness_contrast_p", 0.4)
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=aug_params.get("color_jitter_p", 0.3)
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=aug_params.get("motion_blur_p", 0.2)),
        A.GaussNoise(std_range=(0.05, 0.15), p=aug_params.get("gaussian_noise_p", 0.2)),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]

    return A.Compose(transform_list)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Returns the deterministic validation / test evaluation preprocessing pipeline.
    """
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_transforms_for_split(
    split: str,
    image_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    aug_params: Dict[str, Any] | None = None
) -> A.Compose:
    """
    Helper function to get transforms by split name ('train', 'val', 'test').
    """
    if split.lower() == "train":
        return get_train_transforms(image_size, mean, std, aug_params)
    else:
        return get_val_transforms(image_size, mean, std)
