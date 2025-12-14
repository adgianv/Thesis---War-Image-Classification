#!/usr/bin/env python3
"""
Run inference on unlabeled images.

Usage:
    python scripts/predict.py --weights model_weights/best_model.pth --input data/frames/la6 --output predictions/la6_predictions.csv
    python scripts/predict.py --weights model_weights/best_model.pth --input data/frames/la6 --apply-bias-correction
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def predict(
    config: dict,
    weights_path: str,
    input_dir: str,
    output_path: str,
    apply_bias_correction: bool = False,
):
    """
    Run inference on images in a directory.
    
    Args:
        config: Configuration dictionary
        weights_path: Path to model weights
        input_dir: Directory containing images
        output_path: Path to save predictions CSV
        apply_bias_correction: Whether to apply bias correction
    """
    import torch
    from src.models.resnet import load_model, get_device
    from src.inference.predictor import Predictor
    from src.inference.bias_correction import BiasCorrector
    
    print("=" * 60)
    print("WAR IMAGE CLASSIFICATION - INFERENCE")
    print("=" * 60)
    print(f"Model: {weights_path}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_path}")
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
    
    # Create predictor
    threshold = config['inference'].get('threshold', 0.5)
    predictor = Predictor(
        model=model,
        device=device,
        threshold=threshold,
        image_size=config['preprocessing']['image_size'],
    )
    
    print(f"Classification threshold: {threshold}")
    
    # Run predictions
    print(f"\nRunning inference on images in: {input_dir}")
    results = predictor.predict_directory(
        images_dir=input_dir,
        batch_size=config['inference'].get('batch_size', 32),
    )
    
    print(f"Processed {len(results)} images")
    
    # Summary statistics
    n_positive = results['predicted_class'].sum()
    n_negative = len(results) - n_positive
    print(f"\nPredictions summary:")
    print(f"  - War images: {n_positive} ({n_positive/len(results)*100:.1f}%)")
    print(f"  - Not war: {n_negative} ({n_negative/len(results)*100:.1f}%)")
    
    # Apply bias correction if requested
    if apply_bias_correction:
        print("\nApplying bias correction...")
        bc_config = config['bias_correction']
        corrector = BiasCorrector(
            sensitivity=bc_config['sensitivity'],
            specificity=bc_config['specificity'],
            precision=bc_config['precision'],
            npv=bc_config['npv'],
        )
        results = corrector.correct_dataframe(results, random_state=42)
        
        n_corrected = results['corrected_preds'].sum()
        print(f"Corrected predictions:")
        print(f"  - War images: {n_corrected} ({n_corrected/len(results)*100:.1f}%)")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save results
    results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on unlabeled images"
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
        "--input",
        type=str,
        required=True,
        help="Directory containing images to predict",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--apply-bias-correction",
        action="store_true",
        help="Apply bias correction to predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classification threshold (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override threshold if provided
    if args.threshold is not None:
        config['inference']['threshold'] = args.threshold
    
    # Run prediction
    predict(
        config=config,
        weights_path=args.weights,
        input_dir=args.input,
        output_path=args.output,
        apply_bias_correction=args.apply_bias_correction,
    )


if __name__ == "__main__":
    main()

