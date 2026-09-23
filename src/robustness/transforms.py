"""
Robustness Transformations Module for Solar Panel Fault Classification.

Implements 10 controlled, reproducible environmental and sensor transformations:
1. ORIGINAL: No transformation.
2. LOW_LIGHT: Moderately reduced illumination.
3. HIGH_BRIGHTNESS: Moderately increased illumination / glare.
4. LOW_CONTRAST: Reduced contrast / foggy / hazy atmospheric condition.
5. HIGH_CONTRAST: Increased contrast / harsh direct sunlight.
6. GAUSSIAN_BLUR: Mild blur simulating drone motion or camera vibration.
7. SENSOR_NOISE: Mild realistic Gaussian sensor grain.
8. JPEG_COMPRESSION: Compression artifacts from telemetry transmission.
9. REDUCED_RESOLUTION: Downscale (approx 78x78) and upscale back to 224x224.
10. SMALL_ROTATION: +10 degree rotation simulating aerial tilt.
"""

from typing import Callable, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import torch

ROBUSTNESS_CONDITIONS = [
    "ORIGINAL",
    "LOW_LIGHT",
    "HIGH_BRIGHTNESS",
    "LOW_CONTRAST",
    "HIGH_CONTRAST",
    "GAUSSIAN_BLUR",
    "SENSOR_NOISE",
    "JPEG_COMPRESSION",
    "REDUCED_RESOLUTION",
    "SMALL_ROTATION",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_robustness_transform(condition: str) -> A.Compose:
    """
    Returns an Albumentations Compose pipeline applying ONLY the specified
    visual robustness transformation without normalization/tensor conversion.
    Output is uint8 RGB of shape (224, 224, 3).
    """
    condition_upper = condition.upper()
    if condition_upper not in ROBUSTNESS_CONDITIONS:
        raise ValueError(
            f"Unknown robustness condition: '{condition}'. "
            f"Valid conditions: {ROBUSTNESS_CONDITIONS}"
        )

    if condition_upper == "ORIGINAL":
        t = A.NoOp()
    elif condition_upper == "LOW_LIGHT":
        # Moderate brightness reduction: shifts pixel values downwards by ~25%
        t = A.RandomBrightnessContrast(
            brightness_limit=(-0.25, -0.25),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        )
    elif condition_upper == "HIGH_BRIGHTNESS":
        # Moderate brightness increase: shifts pixel values upwards by ~25%
        t = A.RandomBrightnessContrast(
            brightness_limit=(0.25, 0.25),
            contrast_limit=(0.0, 0.0),
            p=1.0,
        )
    elif condition_upper == "LOW_CONTRAST":
        # Moderate contrast reduction: contracts pixel dynamic range by ~30%
        t = A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=(-0.30, -0.30),
            p=1.0,
        )
    elif condition_upper == "HIGH_CONTRAST":
        # Moderate contrast expansion: expands dynamic range by ~30%
        t = A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.0),
            contrast_limit=(0.30, 0.30),
            p=1.0,
        )
    elif condition_upper == "GAUSSIAN_BLUR":
        # Mild blur: kernel size 5, sigma ~1.5
        t = A.GaussianBlur(
            blur_limit=(5, 5),
            sigma_limit=(1.5, 1.5),
            p=1.0,
        )
    elif condition_upper == "SENSOR_NOISE":
        # Mild Gaussian noise representing sensor electronics
        t = A.GaussNoise(
            std_range=(0.04, 0.04),
            p=1.0,
        )
    elif condition_upper == "JPEG_COMPRESSION":
        # Moderate JPEG compression simulating drone telemetry compression
        t = A.ImageCompression(
            quality_range=(40, 40),
            p=1.0,
        )
    elif condition_upper == "REDUCED_RESOLUTION":
        # Downscale by factor 0.35 (~78x78) and resample back to 224x224
        t = A.Downscale(
            scale_range=(0.35, 0.35),
            p=1.0,
        )
    elif condition_upper == "SMALL_ROTATION":
        # +10 degrees rotation with border reflection to preserve panel structure
        t = A.Rotate(
            limit=(10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        )
    else:
        t = A.NoOp()

    return A.Compose([t])


def get_full_preprocessing_transform(condition: str) -> A.Compose:
    """
    Combines visual robustness transform with standard ImageNet normalization
    and PyTorch tensor conversion.
    """
    visual_t = get_robustness_transform(condition)
    pipeline = list(visual_t.transforms) + [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(pipeline)


def apply_condition_to_image(image_rgb: np.ndarray, condition: str) -> np.ndarray:
    """
    Applies the condition to an input RGB image (uint8, 224x224) and returns
    the transformed RGB uint8 image. Does not modify the input array.
    """
    img_copy = image_rgb.copy()
    transform = get_robustness_transform(condition)
    res = transform(image=img_copy)
    return res["image"]


def apply_condition_to_tensor(image_rgb: np.ndarray, condition: str) -> torch.Tensor:
    """
    Applies condition + ImageNet normalization + tensor conversion.
    Returns FloatTensor of shape [3, 224, 224].
    """
    img_copy = image_rgb.copy()
    transform = get_full_preprocessing_transform(condition)
    res = transform(image=img_copy)
    return res["image"]
