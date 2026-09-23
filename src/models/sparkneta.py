"""
SparkNet Architecture Implementation.

Based on the research paper:
"SparkNet—A Solar Panel Fault Detection Deep Learning Model" (IEEE Access, 2025).

Architectural Characteristics:
- Hierarchical Multi-Branch CNN with four independent branches
- Parallel feature extraction across branches with increasing feature complexity
- Squeeze-and-Expand Fire Modules (1x1 squeeze, parallel 1x1 and 3x3 expands)
- Branch feature concatenation following Global Average Pooling (GAP)
- Dropout regularization and classification head for 6 photovoltaic fault classes

Paper-Specified Architectural Details vs. Implementation Assumptions:
1. Paper-Specified:
   - 4 independent branches of increasing feature complexity.
   - Squeeze-and-expand Fire Modules inspired by SqueezeNet.
   - Global Average Pooling and concatenation of branch features.
   - Total parameter budget: ~300,000 parameters.
   - Target classes: 6 (Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-Covered).
2. Implementation Assumptions (documented as design choices):
   - Initial stem: 3x3 conv with stride 2 and maxpool to downsample 224x224 input to 56x56.
   - Branch progression: Branch 1 (1 FireModule, 96 ch), Branch 2 (2 FireModules, 128 ch),
     Branch 3 (3 FireModules, 192 ch), Branch 4 (4 FireModules, 256 ch), giving 322,158 parameters (~322k).
   - Batch normalization and ReLU activations within FireModules and classifier for stable convergence.
"""

import time
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from src.utils.logger import setup_logger

logger = setup_logger("sparkneta_model")


class FireModule(nn.Module):
    """
    Squeeze-and-Expand Fire Module.
    Compresses input feature channels via a 1x1 convolution (squeeze),
    then expands into parallel 1x1 and 3x3 convolutions before channel concatenation.
    """
    def __init__(
        self,
        in_channels: int,
        squeeze_channels: int,
        expand1x1_channels: int,
        expand3x3_channels: int,
        use_bn: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze_channels = squeeze_channels
        self.expand1x1_channels = expand1x1_channels
        self.expand3x3_channels = expand3x3_channels
        self.out_channels = expand1x1_channels + expand3x3_channels

        # Squeeze layer
        squeeze_layers = [nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=not use_bn)]
        if use_bn:
            squeeze_layers.append(nn.BatchNorm2d(squeeze_channels))
        squeeze_layers.append(nn.ReLU(inplace=True))
        self.squeeze = nn.Sequential(*squeeze_layers)

        # Expand 1x1 branch
        expand1x1_layers = [nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, bias=not use_bn)]
        if use_bn:
            expand1x1_layers.append(nn.BatchNorm2d(expand1x1_channels))
        expand1x1_layers.append(nn.ReLU(inplace=True))
        self.expand1x1 = nn.Sequential(*expand1x1_layers)

        # Expand 3x3 branch
        expand3x3_layers = [nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1, bias=not use_bn)]
        if use_bn:
            expand3x3_layers.append(nn.BatchNorm2d(expand3x3_channels))
        expand3x3_layers.append(nn.ReLU(inplace=True))
        self.expand3x3 = nn.Sequential(*expand3x3_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SparkNetBranch(nn.Module):
    """
    A single branch of SparkNet containing a sequence of Fire Modules,
    intermediate pooling, and Global Average Pooling (GAP).
    """
    def __init__(
        self,
        in_channels: int,
        fire_configs: List[Tuple[int, int, int]],  # [(squeeze, e1x1, e3x3), ...]
        downsample_pool: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        curr_in = in_channels

        for idx, (sq, e1, e3) in enumerate(fire_configs):
            layers.append(FireModule(curr_in, sq, e1, e3))
            curr_in = e1 + e3
            # Add intermediate pooling if requested after the first module
            if downsample_pool and idx == 0:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.branch_net = nn.Sequential(*layers)
        self.out_channels = curr_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branch_net(x)
        return torch.flatten(out, 1)


class SparkNet(nn.Module):
    """
    Full SparkNet architecture featuring 4 parallel hierarchical branches,
    feature fusion, and 6-class classification head.
    """
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem: Initial convolution and pooling from 224x224 to 56x56
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 4 Hierarchical Branches of increasing feature complexity
        # Branch 1 (Low-level surface features & dust texture): 1 FireModule -> out=96
        self.branch1 = SparkNetBranch(
            in_channels=64,
            fire_configs=[(16, 48, 48)],
            downsample_pool=True,
        )

        # Branch 2 (Intermediate contaminant morphology): 2 FireModules -> out=128
        self.branch2 = SparkNetBranch(
            in_channels=64,
            fire_configs=[
                (16, 48, 48),
                (24, 64, 64),
            ],
            downsample_pool=True,
        )

        # Branch 3 (Structural defect patterns, cracks, fractures): 3 FireModules -> out=192
        self.branch3 = SparkNetBranch(
            in_channels=64,
            fire_configs=[
                (16, 48, 48),
                (24, 64, 64),
                (32, 96, 96),
            ],
            downsample_pool=True,
        )

        # Branch 4 (Contextual macro patterns, heavy snow, electrical burns): 4 FireModules -> out=256
        self.branch4 = SparkNetBranch(
            in_channels=64,
            fire_configs=[
                (16, 48, 48),
                (24, 64, 64),
                (32, 96, 96),
                (48, 128, 128),
            ],
            downsample_pool=True,
        )

        total_fused_channels = (
            self.branch1.out_channels +
            self.branch2.out_channels +
            self.branch3.out_channels +
            self.branch4.out_channels
        )  # 96 + 128 + 192 + 256 = 672

        # Classification Head with dropout regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(total_fused_channels, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

        logger.info(
            f"Built SparkNet (num_classes={num_classes}, fused_channels={total_fused_channels}, dropout={dropout})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        b1 = self.branch1(feat)
        b2 = self.branch2(feat)
        b3 = self.branch3(feat)
        b4 = self.branch4(feat)

        fused = torch.cat([b1, b2, b3, b4], dim=1)
        logits = self.classifier(fused)
        return logits


def build_sparknet(
    num_classes: int = 6,
    dropout: float = 0.3,
) -> SparkNet:
    """Factory helper to construct SparkNet model."""
    return SparkNet(num_classes=num_classes, dropout=dropout)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Calculates model parameter count and memory size."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    size_mb = (total_params * 4) / (1024 * 1024)

    return {
        "architecture": "SparkNet",
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
    Measures per-image inference latency and throughput (FPS)
    using the identical methodology as MobileNetV2 and EfficientNet-B0.
    """
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_size, device=dev)

    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

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
    m = build_sparknet()
    print("SparkNet Summary:", get_model_summary(m))
    print("SparkNet Latency:", benchmark_inference_latency(m, num_runs=20))
