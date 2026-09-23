"""
Evaluation Package for Solar Panel Fault Detection.
"""

from src.evaluation.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_misclassifications,
)

__all__ = [
    "compute_classification_metrics",
    "plot_confusion_matrix",
    "plot_training_curves",
    "save_misclassifications",
]
