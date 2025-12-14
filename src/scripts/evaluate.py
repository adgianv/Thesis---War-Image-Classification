#!/usr/bin/env python3
"""
Evaluate a trained model on the test set.

Usage:
    python scripts/evaluate.py --weights model_weights/best_model.pth
    python scripts/evaluate.py --weights model_weights/best_model.pth --config configs/config.yaml
"""

import os
import sys
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate(config: dict, weights_path: str, save_plots: bool = True):
    """
    Evaluate a trained model.
    
    Args:
        config: Configuration dictionary
        weights_path: Path to model weights
        save_plots: Whether to save evaluation plots
    """
    import torch
    from src.data.dataloader import create_dataloaders
    from src.models.resnet import load_model, get_device
    from src.evaluation.metrics import (
        evaluate_model,
        compute_metrics,
        find_optimal_threshold,
        print_metrics,
        plot_confusion_matrix,
        plot_roc_curve,
    )
    
    print("=" * 60)
    print("WAR IMAGE CLASSIFICATION - EVALUATION")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print()
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = load_model(
        path=weights_path,
        architecture=config['model']['architecture'],
        num_classes=config['model']['num_classes'],
        head_type=config['model']['head']['type'],
        hidden_dims=config['model']['head'].get('hidden_dims', [512, 256]),
        device=str(device),
    )
    
    # Create dataloaders
    print("Loading data...")
    _, val_loader, test_loader = create_dataloaders(
        images_dir=config['data']['images_dir'],
        train_labels=config['data']['train_labels'],
        val_labels=config['data']['val_labels'],
        test_labels=config['data']['test_labels'],
        batch_size=config['training']['batch_size'],
        image_size=config['preprocessing']['image_size'],
        use_weighted_sampler=False,  # Not needed for evaluation
    )
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Find optimal threshold on validation set
    print("\n" + "-" * 40)
    print("Finding optimal threshold on validation set...")
    val_labels, val_preds, val_probs = evaluate_model(model, val_loader, device)
    best_threshold, best_f1 = find_optimal_threshold(val_labels, val_probs)
    print(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Evaluate on validation set
    val_preds_optimal = [1 if p >= best_threshold else 0 for p in val_probs]
    val_metrics = compute_metrics(val_labels, val_preds_optimal, val_probs)
    print_metrics(val_metrics, title="Validation Set Metrics")
    
    # Evaluate on test set
    print("\n" + "-" * 40)
    print("Evaluating on test set...")
    test_labels, test_preds, test_probs = evaluate_model(model, test_loader, device, threshold=best_threshold)
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_metrics(test_metrics, title=f"Test Set Metrics (threshold={best_threshold:.2f})")
    
    # Also evaluate with default 0.5 threshold
    test_preds_05 = [1 if p >= 0.5 else 0 for p in test_probs]
    test_metrics_05 = compute_metrics(test_labels, test_preds_05, test_probs)
    print_metrics(test_metrics_05, title="Test Set Metrics (threshold=0.50)")
    
    # Generate plots
    if save_plots:
        plots_dir = config['logging'].get('plots_dir', 'report_charts')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        cm_path = os.path.join(plots_dir, "confusion_matrix.pdf")
        plot_confusion_matrix(test_labels, test_preds, save_path=cm_path)
        
        # ROC curve
        roc_path = os.path.join(plots_dir, "roc_curve.pdf")
        plot_roc_curve(test_labels, test_probs, save_path=roc_path)
    
    # Save metrics to JSON
    output_name = os.path.splitext(os.path.basename(weights_path))[0]
    metrics_path = os.path.join(plots_dir, f"{output_name}_evaluation.json")
    
    results = {
        'weights': weights_path,
        'optimal_threshold': best_threshold,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_metrics_threshold_05': test_metrics_05,
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {metrics_path}")
    
    print("\n✓ Evaluation complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights file",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Evaluate
    evaluate(config, args.weights, save_plots=not args.no_plots)


if __name__ == "__main__":
    main()

