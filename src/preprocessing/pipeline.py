"""
Deterministic Image Preprocessing Module.

Features:
- Robust image loading from disk (handles JPEG, PNG, WEBP, BMP)
- Proper RGB color space conversion (including 4-channel RGBA and 1-channel Grayscale)
- Resizing to standardized model-ready spatial dimensions (default 224x224)
- Tensor conversion and ImageNet channel normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Completely separated from stochastic data augmentation
"""

from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch


# Standard ImageNet statistics for transfer learning and CNN backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = (224, 224)  # (Height, Width)


def load_image_rgb(image_source: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Loads an image from file path, PIL Image, or numpy array and converts it to a standard
    3-channel uint8 RGB numpy array (H, W, 3).
    
    Handles:
    - RGBA (4 channels) -> strips alpha or composites onto white background
    - Grayscale (1 channel) -> replicates to 3 channels
    - Standard RGB
    """
    if isinstance(image_source, (str, Path)):
        img_path = Path(image_source)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # Use PIL to reliably handle varied container formats and metadata
        with Image.open(img_path) as pil_img:
            if pil_img.mode == "RGBA":
                # Create white background to avoid transparent pixel blackouts
                bg = Image.new("RGB", pil_img.size, (255, 255, 255))
                bg.paste(pil_img, mask=pil_img.split()[3])
                rgb_arr = np.array(bg, dtype=np.uint8)
            elif pil_img.mode != "RGB":
                rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            else:
                rgb_arr = np.array(pil_img, dtype=np.uint8)
        return rgb_arr

    elif isinstance(image_source, Image.Image):
        if image_source.mode != "RGB":
            image_source = image_source.convert("RGB")
        return np.array(image_source, dtype=np.uint8)

    elif isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:  # Grayscale (H, W)
            return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
        elif image_source.ndim == 3:
            if image_source.shape[2] == 4:  # RGBA
                return cv2.cvtColor(image_source, cv2.COLOR_RGBA2RGB)
            elif image_source.shape[2] == 3:
                return image_source
            elif image_source.shape[2] == 1:
                return cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"Unsupported numpy array shape: {image_source.shape}")

    else:
        raise TypeError(f"Unsupported image input type: {type(image_source)}")


def resize_image(
    image_rgb: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resizes an RGB numpy image to target dimensions (height, width).
    """
    target_h, target_w = target_size
    return cv2.resize(image_rgb, (target_w, target_h), interpolation=interpolation)


def normalize_image(
    image_float: np.ndarray,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD
) -> np.ndarray:
    """
    Normalizes a float32 image in range [0, 1] using provided channel mean and standard deviation.
    """
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    return (image_float - mean_arr) / std_arr


def preprocess_image(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
    return_tensor: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Complete deterministic preprocessing pipeline:
    1. Load image and ensure 3-channel RGB uint8 format
    2. Resize to target dimensions (e.g. 224x224)
    3. Scale pixel values from [0, 255] to [0.0, 1.0]
    4. Normalize using ImageNet mean & std
    5. Convert to PyTorch Tensor of shape (3, H, W)
    """
    # 1. Load & RGB
    img_rgb = load_image_rgb(image_input)

    # 2. Resize
    resized = resize_image(img_rgb, target_size=target_size)

    # 3. Scale to [0, 1]
    scaled = resized.astype(np.float32) / 255.0

    # 4. Normalize
    normalized = normalize_image(scaled, mean=mean, std=std)

    # 5. Tensor conversion (H, W, C) -> (C, H, W)
    if return_tensor:
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        return tensor

    return normalized


class SolarPanelPreprocessor:
    """
    Callable preprocessor class for batch or single image processing.
    """
    def __init__(
        self,
        target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def __call__(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        return preprocess_image(
            image_input,
            target_size=self.target_size,
            mean=self.mean,
            std=self.std,
            return_tensor=True
        )
