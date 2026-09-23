"""
Preprocessing and Augmentation Package for Solar Panel Imagery.
"""

from src.preprocessing.pipeline import (
    load_image_rgb,
    resize_image,
    normalize_image,
    preprocess_image,
    SolarPanelPreprocessor,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from src.preprocessing.augmentation import (
    get_training_augmentation,
    get_validation_pipeline,
    get_test_pipeline,
    get_pipeline_by_split,
)

__all__ = [
    "load_image_rgb",
    "resize_image",
    "normalize_image",
    "preprocess_image",
    "SolarPanelPreprocessor",
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_training_augmentation",
    "get_validation_pipeline",
    "get_test_pipeline",
    "get_pipeline_by_split",
]
