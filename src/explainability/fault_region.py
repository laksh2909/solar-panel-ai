"""
Approximate Visual Fault-Region Analysis Module.

Extracts connected high-activation saliency regions from Grad-CAM heatmaps to provide
approximate visual localization, bounding boxes, centroids, and area coverage metrics.

IMPORTANT SCIENTIFIC LIMITATION:
This module does NOT produce ground-truth segmentation. The bounding box represents
an approximate region of high model attention derived from Grad-CAM. It must NOT be
interpreted as an exact defect boundary or certified physical damage area.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class FaultRegionExtractor:
    """
    Extracts high-activation regions, bounding boxes, and spatial metrics
    from normalized Grad-CAM heatmaps.
    """

    def __init__(
        self,
        default_threshold: float = 0.60,
        min_area_pixels: int = 25,
    ):
        self.default_threshold = default_threshold
        self.min_area_pixels = min_area_pixels

    def threshold_heatmap(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        adaptive: bool = False,
    ) -> np.ndarray:
        """
        Thresholds a normalized [0, 1] heatmap into a binary mask (uint8: 0 or 255).

        Args:
            heatmap: 2D float numpy array in [0, 1]
            threshold: Saliency cutoff in [0, 1]. Defaults to self.default_threshold (0.60).
            adaptive: If True, uses Otsu's thresholding on the heatmap.

        Returns:
            uint8 binary mask with values {0, 255}.
        """
        thresh_val = threshold if threshold is not None else self.default_threshold

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))

        if adaptive:
            _, binary_mask = cv2.threshold(
                heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            cutoff_uint8 = int(thresh_val * 255)
            _, binary_mask = cv2.threshold(
                heatmap_uint8, cutoff_uint8, 255, cv2.THRESH_BINARY
            )

        return binary_mask

    def clean_mask(
        self,
        binary_mask: np.ndarray,
        min_area: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Removes small, noisy connected components from the binary mask.

        Args:
            binary_mask: uint8 binary mask (0 or 255)
            min_area: Minimum area in pixels to keep a connected component.

        Returns:
            Tuple of (cleaned_mask, num_components_remaining)
        """
        min_pixels = min_area if min_area is not None else self.min_area_pixels

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        valid_count = 0

        # Label 0 is the background
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_pixels:
                cleaned_mask[labels == i] = 255
                valid_count += 1

        return cleaned_mask, valid_count

    def extract_regions(
        self,
        heatmap: np.ndarray,
        threshold: Optional[float] = None,
        adaptive: bool = False,
        min_area: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Performs connected-component analysis on the thresholded heatmap and
        extracts the dominant high-activation fault region.

        Returns a dictionary containing:
            - num_regions: Total connected components meeting min_area threshold
            - bbox_x, bbox_y, bbox_width, bbox_height: Dominant region bounding box
            - centroid_x, centroid_y: Dominant region centroid
            - region_area_pixels: Area of dominant region in pixels
            - region_area_percent: Area as percentage of total image area
            - mean_activation: Average heatmap value within the dominant region
            - max_activation: Maximum heatmap value within the dominant region
            - threshold: Threshold applied
            - mask: 2D uint8 binary mask
            - all_regions: List of dicts for all detected regions
        """
        thresh_val = threshold if threshold is not None else self.default_threshold
        raw_mask = self.threshold_heatmap(heatmap, threshold=thresh_val, adaptive=adaptive)
        cleaned_mask, num_regions = self.clean_mask(raw_mask, min_area=min_area)

        total_image_pixels = heatmap.shape[0] * heatmap.shape[1]

        # Handle empty mask / no-region case safely
        if num_regions == 0:
            return {
                "num_regions": 0,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_width": 0,
                "bbox_height": 0,
                "centroid_x": 0.0,
                "centroid_y": 0.0,
                "region_area_pixels": 0,
                "region_area_percent": 0.0,
                "mean_activation": 0.0,
                "max_activation": 0.0,
                "threshold": thresh_val,
                "mask": cleaned_mask,
                "all_regions": [],
            }

        # Perform connected component extraction on cleaned mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask, connectivity=8
        )

        regions_list = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx = float(centroids[i][0])
            cy = float(centroids[i][1])

            comp_mask = (labels == i)
            comp_vals = heatmap[comp_mask]
            mean_act = float(np.mean(comp_vals)) if comp_vals.size > 0 else 0.0
            max_act = float(np.max(comp_vals)) if comp_vals.size > 0 else 0.0
            area_pct = round((area / total_image_pixels) * 100.0, 2)

            regions_list.append({
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": w,
                "bbox_height": h,
                "centroid_x": round(cx, 1),
                "centroid_y": round(cy, 1),
                "region_area_pixels": area,
                "region_area_percent": area_pct,
                "mean_activation": round(mean_act, 4),
                "max_activation": round(max_act, 4),
            })

        # Select dominant region: component with largest area
        dominant = max(regions_list, key=lambda r: r["region_area_pixels"])

        return {
            "num_regions": len(regions_list),
            "bbox_x": dominant["bbox_x"],
            "bbox_y": dominant["bbox_y"],
            "bbox_width": dominant["bbox_width"],
            "bbox_height": dominant["bbox_height"],
            "centroid_x": dominant["centroid_x"],
            "centroid_y": dominant["centroid_y"],
            "region_area_pixels": dominant["region_area_pixels"],
            "region_area_percent": dominant["region_area_percent"],
            "mean_activation": dominant["mean_activation"],
            "max_activation": dominant["max_activation"],
            "threshold": thresh_val,
            "mask": cleaned_mask,
            "all_regions": regions_list,
        }

    @staticmethod
    def draw_region_overlay(
        image_rgb: np.ndarray,
        mask: np.ndarray,
        bbox: Optional[Dict[str, int]] = None,
        centroid: Optional[Dict[str, float]] = None,
        mask_color: Tuple[int, int, int] = (255, 87, 34),  # Deep Orange tint
        bbox_color: Tuple[int, int, int] = (0, 230, 118),   # Neon Green bbox
        centroid_color: Tuple[int, int, int] = (213, 0, 0), # Red centroid
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Draws the binary activation region mask, bounding box, and centroid marker
        over the original RGB image.
        """
        h_img, w_img = image_rgb.shape[:2]
        if mask.shape[:2] != (h_img, w_img):
            mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        overlay = image_rgb.copy()

        # 1. Semi-transparent colored tint over active region
        region_pixels = (mask_resized > 0)
        if np.any(region_pixels):
            tint = np.full_like(overlay, mask_color, dtype=np.uint8)
            overlay[region_pixels] = np.uint8(
                (1.0 - alpha) * overlay[region_pixels] + alpha * tint[region_pixels]
            )

            # Draw contour boundary around the mask
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 171, 0), thickness=2)

        # 2. Draw Bounding Box
        if bbox is not None and bbox.get("bbox_width", 0) > 0 and bbox.get("bbox_height", 0) > 0:
            bx = bbox["bbox_x"]
            by = bbox["bbox_y"]
            bw = bbox["bbox_width"]
            bh = bbox["bbox_height"]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bbox_color, thickness=2)

        # 3. Draw Centroid Marker (Crosshair + Circle)
        if centroid is not None and centroid.get("centroid_x", 0) > 0 and centroid.get("centroid_y", 0) > 0:
            cx = int(round(centroid["centroid_x"]))
            cy = int(round(centroid["centroid_y"]))
            # Ensure within image bounds
            if 0 <= cx < w_img and 0 <= cy < h_img:
                cv2.circle(overlay, (cx, cy), radius=4, color=centroid_color, thickness=-1)
                cv2.circle(overlay, (cx, cy), radius=7, color=(255, 255, 255), thickness=1)
                # Crosshairs
                cv2.line(overlay, (cx - 10, cy), (cx + 10, cy), centroid_color, thickness=1)
                cv2.line(overlay, (cx, cy - 10), (cx, cy + 10), centroid_color, thickness=1)

        return overlay
