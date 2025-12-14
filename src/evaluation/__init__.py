"""Evaluation and metrics modules."""

from .metrics import (
    evaluate_model,
    compute_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
)
from .explainability import GradCAM, occlusion_sensitivity

__all__ = [
    "evaluate_model",
    "compute_metrics",
    "find_optimal_threshold",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
    "GradCAM",
    "occlusion_sensitivity",
]

