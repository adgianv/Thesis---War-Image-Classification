#!/usr/bin/env python3
"""
Train the war image classification model.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --experiment my_experiment
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config: dict, experiment_name: str = None):
    """
    Train the model using configuration.
    
    Args:
        config: Configuration dictionary
        experiment_name: Optional name for this training run
    """
    import torch
    from src.data.dataloader import create_dataloaders
    from src.models.resnet import create_model, get_device, get_trainable_params, get_total_params
    from src.training.trainer import Trainer, create_optimizer, create_criterion
    from src.evaluation.metrics import plot_training_history
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"
    
    print("=" * 60)
    print("WAR IMAGE CLASSIFICATION - TRAINING")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print()
    
    # Get device
    device = get_device(config.get('device', 'auto'))
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=config['data']['images_dir'],
        train_labels=config['data']['train_labels'],
        val_labels=config['data']['val_labels'],
        test_labels=config['data']['test_labels'],
        batch_size=config['training']['batch_size'],
        image_size=config['preprocessing']['image_size'],
        use_weighted_sampler=config['training']['use_weighted_sampler'],
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        architecture=config['model']['architecture'],
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes'],
        head_type=config['model']['head']['type'],
        hidden_dims=config['model']['head'].get('hidden_dims', [512, 256]),
        unfreeze_layers=config['model']['unfreeze_layers'],
    )
    model = model.to(device)
    
    total_params = get_total_params(model)
    trainable_params = get_trainable_params(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # Create optimizer and criterion
    optimizer = create_optimizer(
        model,
        name=config['training']['optimizer']['name'],
        learning_rate=config['training']['optimizer']['learning_rate'],
        weight_decay=config['training']['optimizer'].get('weight_decay', 0.0),
    )
    criterion = create_criterion(config['training']['criterion'])
    
    print(f"Optimizer: {config['training']['optimizer']['name']}")
    print(f"Learning rate: {config['training']['optimizer']['learning_rate']}")
    print(f"Criterion: {config['training']['criterion']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config['training']['checkpoint_dir'],
        patience=config['training']['patience'],
        save_best_only=config['training']['save_best_only'],
    )
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    checkpoint_name = f"{experiment_name}.pth"
    model, history = trainer.train(
        num_epochs=config['training']['epochs'],
        checkpoint_name=checkpoint_name,
    )
    
    # Save training history
    history_path = os.path.join(config['training']['checkpoint_dir'], f"{experiment_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Plot training curves
    if config['logging'].get('save_plots', True):
        plots_dir = config['logging'].get('plots_dir', 'report_charts')
        os.makedirs(plots_dir, exist_ok=True)
        plot_path = os.path.join(plots_dir, f"{experiment_name}_training_curves.pdf")
        plot_training_history(history, save_path=plot_path)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    from src.evaluation.metrics import evaluate_model, compute_metrics, find_optimal_threshold, print_metrics
    
    # Get optimal threshold from validation set
    val_labels, val_preds, val_probs = evaluate_model(model, val_loader, device)
    best_threshold, best_f1 = find_optimal_threshold(val_labels, val_probs)
    print(f"Optimal threshold (from validation): {best_threshold:.2f} (F1: {best_f1:.4f})")
    
    # Evaluate on test set
    test_labels, test_preds, test_probs = evaluate_model(model, test_loader, device, threshold=best_threshold)
    metrics = compute_metrics(test_labels, test_preds, test_probs)
    print_metrics(metrics, title=f"Test Set Metrics (threshold={best_threshold:.2f})")
    
    # Save metrics
    metrics_path = os.path.join(config['training']['checkpoint_dir'], f"{experiment_name}_metrics.json")
    metrics['threshold'] = best_threshold
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n✓ Training complete!")
    print(f"Model checkpoint: {os.path.join(config['training']['checkpoint_dir'], checkpoint_name)}")
    
    return model, history, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train the war image classification model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name for this run",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config, experiment_name=args.experiment)


if __name__ == "__main__":
    main()

