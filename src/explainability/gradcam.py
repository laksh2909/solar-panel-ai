"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.

Provides visual explanations of model decisions by computing the gradients of a
target class score with respect to the feature maps of the final convolutional layer.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger("gradcam")


class GradCAM:
    """
    Grad-CAM implementation for PyTorch models (specialized for EfficientNet-B0).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ):
        self.model = model
        self.model.eval()

        # Find target layer automatically if not specified
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks = []
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """
        Locates the final convolutional layer of EfficientNet-B0.
        Default: model.features[-1] (Conv2dNormActivation with 1280 channels).
        """
        if hasattr(self.model, "features") and len(self.model.features) > 0:
            return self.model.features[-1]

        # Generic search for last Conv2d layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module

        if last_conv is None:
            raise ValueError("Could not automatically locate a convolutional layer in model.")
        return last_conv

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h_fwd, h_bwd])

    def remove_hooks(self):
        """Removes registered hooks from target layer."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates a normalized [0, 1] Grad-CAM heatmap for the input tensor.

        Args:
            input_tensor: Tensor of shape [1, 3, H, W] (already normalized)
            target_class: Target class index. If None, uses argmax predicted class.
            target_size: (Width, Height) tuple to resize heatmap to. If None, uses input_tensor size.

        Returns:
            Tuple of:
                - heatmap: 2D numpy array [H, W] normalized to [0, 1]
                - predicted_class: int
                - confidence: float
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())

        selected_class = target_class if target_class is not None else pred_idx

        # Backward pass with respect to selected class score
        score = logits[0, selected_class]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not captured by hooks.")

        # Compute importance weights via global average pooling of gradients
        # activations/gradients shape: [1, Channels, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU: only positive influence on class score
        cam = torch.relu(cam)

        # Determine spatial target size
        if target_size is not None:
            out_h, out_w = target_size[1], target_size[0]
        else:
            out_h, out_w = input_tensor.shape[2], input_tensor.shape[3]

        # Bilinear interpolation up to target resolution
        cam_upsampled = torch.nn.functional.interpolate(
            cam, size=(out_h, out_w), mode="bilinear", align_corners=False
        )
        cam_np = cam_upsampled.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        denom = cam_np.max() - cam_np.min()
        if denom > 1e-8:
            heatmap = (cam_np - cam_np.min()) / denom
        else:
            heatmap = np.zeros_like(cam_np, dtype=np.float32)

        # Clean zero gradients after backward pass to avoid side-effects
        self.model.zero_grad()

        return heatmap, pred_idx, pred_conf

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blends a Grad-CAM heatmap over an RGB image.

        Args:
            image_rgb: uint8 RGB numpy array (H, W, 3)
            heatmap: float numpy array (H, W) in [0, 1]
            alpha: Transparency factor for heatmap overlay (0 = image only, 1 = heatmap only)
            colormap: OpenCV colormap enum (e.g. cv2.COLORMAP_JET)

        Returns:
            uint8 RGB blended image of shape (H, W, 3)
        """
        h_img, w_img = image_rgb.shape[:2]
        h_cam, w_cam = heatmap.shape[:2]

        if (h_cam, w_cam) != (h_img, w_img):
            heatmap_resized = cv2.resize(heatmap, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap

        # Convert [0, 1] float to [0, 255] uint8
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))

        # Colorize using OpenCV
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend with original RGB image
        overlay = np.uint8(alpha * heatmap_rgb + (1.0 - alpha) * image_rgb)
        return overlay
