"""
Synthetic Benchmark Dataset Generator for Solar Panel Fault Detection.

Creates visual benchmark imagery simulating real-world photovoltaic panel defects:
- Clean: Standard blue/monocrystalline PV grid with busbars
- Dusty: Particulate haze and sand layer reducing irradiance
- Bird-drop: Localized high-contrast organic drops and splatters
- Electrical-damage: Thermal burn marks, hotspots, and charred busbars
- Physical-damage: Jagged fracture and crack lines across cells
- Snow-Covered: Thick white/snow cover occluding photovoltaic surface
"""

from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image

from src.utils.logger import setup_logger

logger = setup_logger("synthetic_generator")

TARGET_CLASSES = [
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-damage",
    "Snow-Covered",
]


def _draw_base_pv_cell(width: int = 256, height: int = 256) -> np.ndarray:
    """Renders a base photovoltaic solar panel with silicon cell grid and metallic busbars."""
    # Deep blue / navy silicon base with subtle gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    blue_base = np.random.randint(45, 75)
    img[:, :, 0] = blue_base  # B
    img[:, :, 1] = int(blue_base * 0.5)  # G
    img[:, :, 2] = int(blue_base * 0.2)  # R

    # Add subtle silicon wafer crystal texture
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw vertical busbars (typically 2 to 4 bright silver/white lines)
    num_busbars = 3
    spacing = width // (num_busbars + 1)
    for i in range(1, num_busbars + 1):
        x = i * spacing + np.random.randint(-2, 3)
        cv2.line(img, (x, 0), (x, height), (210, 210, 210), thickness=2)

    # Draw horizontal cell grid lines (fine grid wires)
    grid_spacing = height // 8
    for y in range(grid_spacing, height, grid_spacing):
        cv2.line(img, (0, y), (width, y), (120, 120, 120), thickness=1)

    return img


def generate_clean_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates an optimal, defect-free solar panel image."""
    img = _draw_base_pv_cell(width, height)
    # Subtle natural solar reflection gradient
    overlay = img.copy()
    cv2.circle(overlay, (width // 2, height // 2), width // 2, (100, 70, 30), -1)
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
    return img


def generate_dusty_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a solar panel with uniform or uneven dust/sand accumulation."""
    img = _draw_base_pv_cell(width, height)
    # Dust color in BGR: sandy brown/yellowish gray
    dust_layer = np.zeros_like(img)
    dust_layer[:, :, 0] = np.random.randint(90, 120)  # B
    dust_layer[:, :, 1] = np.random.randint(130, 170)  # G
    dust_layer[:, :, 2] = np.random.randint(160, 200)  # R

    # Non-uniform dust density mask
    dust_mask = np.random.uniform(0.35, 0.70, (height, width, 1)).astype(np.float32)
    dust_mask = cv2.GaussianBlur(dust_mask, (25, 25), 0)[:, :, np.newaxis]

    blended = (img.astype(np.float32) * (1.0 - dust_mask) + dust_layer.astype(np.float32) * dust_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_bird_drop_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a solar panel with localized organic bird drop contaminants."""
    img = _draw_base_pv_cell(width, height)
    num_splatters = random.randint(1, 4)

    for _ in range(num_splatters):
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
        radius = random.randint(12, 35)

        # Core drop (whitish-chalky)
        cv2.circle(img, (cx, cy), radius, (230, 235, 240), -1)
        # Irregular perimeter droplets
        for _ in range(random.randint(5, 12)):
            dx = cx + random.randint(-radius - 15, radius + 15)
            dy = cy + random.randint(-radius - 15, radius + 15)
            r_small = random.randint(2, 8)
            color_noise = random.randint(200, 245)
            cv2.circle(img, (dx, dy), r_small, (color_noise, color_noise, color_noise), -1)

        # Darker organic center
        cv2.circle(img, (cx + random.randint(-4, 4), cy + random.randint(-4, 4)), radius // 3, (80, 90, 95), -1)

    return img


def generate_electrical_damage_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a solar panel showing hot-spot burn marks and charred electrical junctions."""
    img = _draw_base_pv_cell(width, height)
    num_burns = random.randint(1, 3)

    for _ in range(num_burns):
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
        burn_radius = random.randint(15, 40)

        # Brownish halo ring
        cv2.circle(img, (cx, cy), burn_radius + 10, (15, 45, 95), -1)
        # Dark charred black core
        cv2.circle(img, (cx, cy), burn_radius, (10, 15, 20), -1)
        # Intense yellow/orange hotspot core
        cv2.circle(img, (cx, cy), max(3, burn_radius // 4), (30, 180, 255), -1)

    return img


def generate_physical_damage_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a solar panel with sharp fracture lines and cracked protective glass."""
    img = _draw_base_pv_cell(width, height)
    num_cracks = random.randint(2, 5)

    for _ in range(num_cracks):
        # Start point on an edge or interior
        start_x = random.randint(10, width - 10)
        start_y = random.randint(10, height - 10)
        curr_x, curr_y = start_x, start_y

        num_segments = random.randint(4, 10)
        for _ in range(num_segments):
            next_x = int(np.clip(curr_x + random.randint(-35, 35), 0, width - 1))
            next_y = int(np.clip(curr_y + random.randint(-35, 35), 0, height - 1))
            # Draw crack with bright reflective line and dark shadow line
            cv2.line(img, (curr_x, curr_y), (next_x, next_y), (250, 250, 250), thickness=2)
            cv2.line(img, (curr_x + 1, curr_y + 1), (next_x + 1, next_y + 1), (20, 20, 20), thickness=1)
            curr_x, curr_y = next_x, next_y

    return img


def generate_snow_sample(width: int = 256, height: int = 256) -> np.ndarray:
    """Generates a solar panel with heavy snow coverage."""
    img = _draw_base_pv_cell(width, height)
    # Snow layer: pure white / icy tint
    snow = np.zeros_like(img)
    snow[:, :, 0] = np.random.randint(235, 255)
    snow[:, :, 1] = np.random.randint(235, 255)
    snow[:, :, 2] = np.random.randint(235, 255)

    # Random smooth snow mask using Perlin-like blur
    random_noise = np.random.rand(height // 4, width // 4).astype(np.float32)
    snow_mask = cv2.resize(random_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    snow_mask = (snow_mask > 0.35).astype(np.float32)[:, :, np.newaxis]
    snow_mask = cv2.GaussianBlur(snow_mask, (21, 21), 0)[:, :, np.newaxis]

    blended = img.astype(np.float32) * (1.0 - snow_mask) + snow.astype(np.float32) * snow_mask
    return np.clip(blended, 0, 255).astype(np.uint8)


GENERATOR_MAP = {
    "Bird-drop": generate_bird_drop_sample,
    "Clean": generate_clean_sample,
    "Dusty": generate_dusty_sample,
    "Electrical-damage": generate_electrical_damage_sample,
    "Physical-damage": generate_physical_damage_sample,
    "Snow-Covered": generate_snow_sample,
}


def generate_synthetic_dataset(
    output_dir: str | Path = "data/raw",
    samples_per_class: int = 30,
    width: int = 256,
    height: int = 256,
    seed: int = 42,
) -> Path:
    """
    Generates a full synthetic benchmark dataset organized by class directories.
    
    Args:
        output_dir: Directory where class folders will be created.
        samples_per_class: Number of images per class.
        width: Image width.
        height: Image height.
        seed: Random seed.
        
    Returns:
        Path: Output directory path.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic dataset: {samples_per_class} samples/class into {output_path}")

    for class_name, gen_func in GENERATOR_MAP.items():
        class_folder = output_path / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            img_bgr = gen_func(width, height)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            file_path = class_folder / f"{class_name.lower()}_{i:04d}.jpg"
            pil_img.save(file_path, quality=95)

    total_images = len(TARGET_CLASSES) * samples_per_class
    logger.info(f"Synthetic dataset generation complete: {total_images} total images across {len(TARGET_CLASSES)} classes.")
    return output_path


if __name__ == "__main__":
    generate_synthetic_dataset("data/raw", samples_per_class=35)
