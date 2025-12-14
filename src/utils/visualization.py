"""Visualization utilities."""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from ..data.transforms import denormalize


def plot_images_grid(
    images: List[np.ndarray],
    labels: List[int] = None,
    predictions: List[int] = None,
    probabilities: List[float] = None,
    nrows: int = 2,
    ncols: int = 5,
    figsize: Tuple[int, int] = (20, 8),
    save_path: Optional[str] = None,
):
    """
    Plot a grid of images with optional labels and predictions.
    
    Args:
        images: List of image arrays (H, W, C)
        labels: Optional true labels
        predictions: Optional predicted labels
        probabilities: Optional prediction probabilities
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size
        save_path: Path to save figure
    """
    n_images = min(len(images), nrows * ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx in range(n_images):
        ax = axes[idx]
        img = images[idx]
        
        # Handle normalized images
        if img.max() <= 1.0:
            ax.imshow(img)
        else:
            ax.imshow(img.astype(np.uint8))
        
        # Build title
        title_parts = []
        if labels is not None:
            label_str = "War" if labels[idx] == 1 else "Not War"
            title_parts.append(f"True: {label_str}")
        
        if predictions is not None:
            pred_str = "War" if predictions[idx] == 1 else "Not War"
            title_parts.append(f"Pred: {pred_str}")
        
        if probabilities is not None:
            title_parts.append(f"Prob: {probabilities[idx]:.3f}")
        
        if title_parts:
            ax.set_title("\n".join(title_parts), fontsize=9)
        
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_predictions(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    probabilities: List[float],
    filter_type: str = "all",
    n_images: int = 10,
    figsize: Tuple[int, int] = (20, 10),
    save_path: Optional[str] = None,
):
    """
    Plot predictions with filtering options.
    
    Args:
        images: List of image arrays
        true_labels: True labels
        pred_labels: Predicted labels
        probabilities: Prediction probabilities
        filter_type: "all", "correct", "errors", "false_positives", "false_negatives"
        n_images: Number of images to show
        figsize: Figure size
        save_path: Path to save figure
    """
    # Filter images based on filter_type
    indices = []
    for i in range(len(images)):
        if filter_type == "all":
            indices.append(i)
        elif filter_type == "correct":
            if true_labels[i] == pred_labels[i]:
                indices.append(i)
        elif filter_type == "errors":
            if true_labels[i] != pred_labels[i]:
                indices.append(i)
        elif filter_type == "false_positives":
            if true_labels[i] == 0 and pred_labels[i] == 1:
                indices.append(i)
        elif filter_type == "false_negatives":
            if true_labels[i] == 1 and pred_labels[i] == 0:
                indices.append(i)
        
        if len(indices) >= n_images:
            break
    
    if not indices:
        print(f"No images found for filter_type='{filter_type}'")
        return
    
    # Get filtered data
    filtered_images = [images[i] for i in indices]
    filtered_labels = [true_labels[i] for i in indices]
    filtered_preds = [pred_labels[i] for i in indices]
    filtered_probs = [probabilities[i] for i in indices]
    
    # Calculate grid size
    n = len(filtered_images)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    
    plot_images_grid(
        filtered_images,
        labels=filtered_labels,
        predictions=filtered_preds,
        probabilities=filtered_probs,
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        save_path=save_path,
    )


def plot_class_distribution(
    labels: List[int],
    title: str = "Class Distribution",
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
):
    """
    Plot class distribution as a bar chart.
    
    Args:
        labels: List of labels
        title: Chart title
        class_names: Names for classes (default: ["Not War", "War"])
        figsize: Figure size
        save_path: Path to save figure
    """
    if class_names is None:
        class_names = ["Not War", "War"]
    
    # Count classes
    unique, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(class_names, counts)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{count}\n({count/sum(counts)*100:.1f}%)',
            ha='center',
            va='bottom',
        )
    
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")
    
    plt.show()

