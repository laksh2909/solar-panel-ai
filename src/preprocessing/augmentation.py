"""
Data Augmentation Module using Albumentations for Solar Panel Fault Detection.

Domain-Specific Augmentations Designed for Solar Photovoltaic Imagery:
1. Horizontal Flip (p=0.5): Bilateral panel symmetry and reflection invariance.
2. Small Rotation (+/- 8 deg): Drone pitch/yaw and camera mounting angle variations without distortion.
3. Small Brightness Variation (limit=0.10): Variable solar irradiance and cloud shadow passing.
4. Small Contrast Variation (limit=0.10): Atmospheric haziness, glint, and diffuse skylight.
5. Mild Gaussian Blur (blur_limit=3, p=0.2): Minor motion blur from drone/rig vibration.
6. Mild Sensor Noise (GaussNoise std_range=(0.02, 0.06), p=0.2): Realistic CMOS sensor noise.
7. Small Image Quality / Compression Variation (quality=82-98, p=0.25): Telemetry/JPEG upload artifacts.
8. Mild Affine / Scale Transformation (scale=0.96-1.04, translation=+/-3%): Distance variation.

Strict Safety Constraints:
- NO 90/180/270 degree flips or heavy rotations that invert panel drainage and rack geometry.
- NO extreme color jittering/hue shifts that turn clean blue cells dusty or snow into bird drop.
- NO aggressive cropping or cutout that cuts away localized defect features (hotspots, cracks, droppings).
- Validation and Test sets REMAIN STRICTLY DETERMINISTIC without random transformations.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.preprocessing.pipeline import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_training_augmentation(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    include_normalization: bool = True,
    aug_config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """
    Creates the Albumentations training pipeline featuring realistic, domain-specific augmentations.
    """
    if aug_config is None:
        aug_config = {
            "horizontal_flip_p": 0.5,
            "rotation_limit": 8,
            "scale_limit": (0.96, 1.04),
            "translate_limit": (-0.03, 0.03),
            "affine_p": 0.4,
            "brightness_limit": 0.10,
            "contrast_limit": 0.10,
            "brightness_contrast_p": 0.4,
            "gaussian_blur_p": 0.2,
            "gaussian_noise_p": 0.2,
            "compression_p": 0.25,
            "compression_quality_range": (82, 98),
        }

    h, w = image_size

    pipeline = [
        # 0. Canonical resize
        A.Resize(h, w),

        # 1. Horizontal Flip (realistic bilateral symmetry)
        A.HorizontalFlip(p=aug_config.get("horizontal_flip_p", 0.5)),

        # 2 & 8. Controlled Affine (Small Rotation +/- 8 deg + Mild Scale 0.96-1.04 + Mild Shift +/- 3%)
        A.Affine(
            scale=aug_config.get("scale_limit", (0.96, 1.04)),
            translate_percent=aug_config.get("translate_limit", (-0.03, 0.03)),
            rotate=(-aug_config.get("rotation_limit", 8), aug_config.get("rotation_limit", 8)),
            p=aug_config.get("affine_p", 0.4),
            border_mode=cv2.BORDER_REFLECT_101,
        ),

        # 3 & 4. Photometric variations (Solar irradiance & atmospheric contrast variations)
        A.RandomBrightnessContrast(
            brightness_limit=aug_config.get("brightness_limit", 0.10),
            contrast_limit=aug_config.get("contrast_limit", 0.10),
            p=aug_config.get("brightness_contrast_p", 0.4),
        ),

        # 5. Mild Gaussian blur (drone/camera vibration)
        A.GaussianBlur(blur_limit=(3, 3), p=aug_config.get("gaussian_blur_p", 0.2)),

        # 6. Mild sensor noise (realistic CMOS sensor grain)
        A.GaussNoise(
            std_range=(0.02, 0.06),
            per_channel=False,
            p=aug_config.get("gaussian_noise_p", 0.2),
        ),

        # 7. Image quality / telemetry compression variation
        A.ImageCompression(
            quality_range=aug_config.get("compression_quality_range", (82, 98)),
            p=aug_config.get("compression_p", 0.25),
        ),
    ]

    # Normalization and tensor conversion
    if include_normalization:
        pipeline.extend([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return A.Compose(pipeline)


def get_visualization_augmentation(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    aug_config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """
    Training augmentation without normalization/tensor conversion, for rendering visual samples.
    """
    return get_training_augmentation(
        image_size=image_size,
        include_normalization=False,
        aug_config=aug_config,
    )


def get_validation_pipeline(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Deterministic validation preprocessing pipeline (NO random augmentation).
    """
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_test_pipeline(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> A.Compose:
    """
    Deterministic test preprocessing pipeline (NO random augmentation).
    """
    h, w = image_size
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_pipeline_by_split(
    split: str,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    use_augmentation_for_train: bool = True,
    aug_config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """
    Returns the appropriate pipeline for a given split ('train', 'val', 'test').
    """
    split_lower = split.strip().lower()
    if split_lower == "train":
        if use_augmentation_for_train:
            return get_training_augmentation(image_size=image_size, aug_config=aug_config)
        else:
            return get_validation_pipeline(image_size=image_size)
    elif split_lower in ("val", "validation"):
        return get_validation_pipeline(image_size=image_size)
    elif split_lower == "test":
        return get_test_pipeline(image_size=image_size)
    else:
        raise ValueError(f"Unknown split name: {split}. Expected 'train', 'val', or 'test'.")
