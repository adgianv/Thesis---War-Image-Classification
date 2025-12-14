"""Evaluation metrics and visualization."""

from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
)
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to evaluate on
        threshold: Classification threshold
    
    Returns:
        Tuple of (true_labels, predictions, probabilities)
    """
    model.eval()
    
    all_labels = []
    all_probs = []
    
    for inputs, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
        inputs = inputs.to(device)
        
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
    
    # Convert probabilities to predictions
    all_preds = [1 if p >= threshold else 0 for p in all_probs]
    
    return all_labels, all_preds, all_probs


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0  # Can't compute if only one class present
    
    return metrics


def find_optimal_threshold(
    y_true: List[int],
    y_prob: List[float],
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ("f1", "accuracy", "precision", "recall")
    
    Returns:
        Tuple of (best_threshold, best_metric_value)
    """
    best_threshold = 0.5
    best_value = 0.0
    
    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = [1 if p >= threshold else 0 for p in y_prob]
        
        if metric == "f1":
            value = f1_score(y_true, preds, zero_division=0)
        elif metric == "accuracy":
            value = accuracy_score(y_true, preds)
        elif metric == "precision":
            value = precision_score(y_true, preds, zero_division=0)
        elif metric == "recall":
            value = recall_score(y_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    return best_threshold, best_value


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels (default: ["Not War", "War"])
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    if labels is None:
        labels = ["Not War", "War"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel='Predicted Label',
        ylabel='True Label',
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: List[int],
    y_prob: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Find index of threshold closest to 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    # Mark threshold = 0.5
    ax.scatter(fpr[idx_05], tpr[idx_05], color='red', s=100, zorder=5, label='Threshold = 0.5')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Dictionary with 'train_loss' and 'val_loss' lists
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Training history saved to {save_path}")
    
    plt.show()


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Print metrics in a formatted way."""
    print(f"\n{title}")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"{name.capitalize():15s}: {value:.4f} ({value*100:.2f}%)")
    print("=" * 40)

