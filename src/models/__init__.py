"""
Models Package for Solar Panel Fault Detection.
"""

from src.models.mobilenet import (
    build_mobilenet_v2,
    get_model_summary as get_mobilenet_summary,
    benchmark_inference_latency as benchmark_mobilenet_latency,
)

from src.models.efficientnet import (
    build_efficientnet_b0,
    get_model_summary as get_efficientnet_summary,
    benchmark_inference_latency as benchmark_efficientnet_latency,
)

from src.models.sparkneta import (
    FireModule,
    SparkNetBranch,
    SparkNet,
    build_sparknet,
    get_model_summary as get_sparknet_summary,
    benchmark_inference_latency as benchmark_sparknet_latency,
)

__all__ = [
    "build_mobilenet_v2",
    "get_mobilenet_summary",
    "benchmark_mobilenet_latency",
    "build_efficientnet_b0",
    "get_efficientnet_summary",
    "benchmark_efficientnet_latency",
    "FireModule",
    "SparkNetBranch",
    "SparkNet",
    "build_sparknet",
    "get_sparknet_summary",
    "benchmark_sparknet_latency",
]

