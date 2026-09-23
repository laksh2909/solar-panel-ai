"""
EfficientNet-B0 Baseline Model Architecture for Solar Panel Fault Detection.

Features:
- ImageNet-pretrained EfficientNet-B0 backbone (torchvision)
- 6-class classification head with standard dropout
- Model parameter inspection and memory footprint estimation
- Benchmark utility for inference latency and throughput (matching MobileNetV2 methodology)
"""

import time
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.utils.logger import setup_logger

logger = setup_logger("efficientnet_model")


def build_efficientnet_b0(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Constructs an EfficientNet-B0 model adapted for 6-class solar panel defect classification.
    
    Args:
        num_classes: Number of target fault classes (default 6).
        pretrained: If True, initializes with ImageNet-1K pretrained weights.
        dropout: Dropout rate before the final linear layer (default 0.2).
        
    Returns:
        nn.Module: Configured PyTorch EfficientNet-B0 model.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # In EfficientNet-B0, classifier is:
    # Sequential(
    #   (0): Dropout(p=0.2, inplace=True)
    #   (1): Linear(in_features=1280, out_features=1000, bias=True)
    # )
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )

    logger.info(
        f"Built EfficientNet-B0 (pretrained={pretrained}, num_classes={num_classes}, dropout={dropout})"
    )
    return model


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Calculates total parameters, trainable parameters, and approximate model size in MB.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Size in megabytes (float32 = 4 bytes per param)
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "architecture": "EfficientNet-B0",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "model_size_mb": round(size_mb, 2),
    }


def benchmark_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_warmup: int = 10,
    num_runs: int = 50,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Measures per-image inference latency and frames per second (FPS)
    using the exact same methodology as MobileNetV2.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_size, device=dev)

    # Warmup iterations
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    avg_latency_ms = sum(latencies) / len(latencies)
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "device": str(dev),
        "input_shape": list(input_size),
        "num_runs": num_runs,
        "average_latency_ms": round(avg_latency_ms, 3),
        "min_latency_ms": round(min(latencies), 3),
        "max_latency_ms": round(max(latencies), 3),
        "fps": round(fps, 1),
    }


if __name__ == "__main__":
    m = build_efficientnet_b0()
    summary = get_model_summary(m)
    print("Model summary:", summary)
    latency = benchmark_inference_latency(m, num_runs=20)
    print("Latency benchmark:", latency)
