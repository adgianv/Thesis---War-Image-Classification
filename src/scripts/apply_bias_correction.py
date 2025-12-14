#!/usr/bin/env python3
"""
Apply bias correction to existing predictions.

Usage:
    python scripts/apply_bias_correction.py --input predictions/la6_predictions.csv --output predictions/la6_bias_corrected.csv
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


def main():
    parser = argparse.ArgumentParser(
        description="Apply bias correction to existing predictions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input predictions CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save corrected predictions CSV",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="predicted_class",
        help="Column name for predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    bc_config = config['bias_correction']
    
    print("=" * 60)
    print("BIAS CORRECTION")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print()
    print("Correction parameters:")
    print(f"  Sensitivity: {bc_config['sensitivity']:.3f}")
    print(f"  Specificity: {bc_config['specificity']:.3f}")
    print(f"  Precision: {bc_config['precision']:.3f}")
    print(f"  NPV: {bc_config['npv']:.3f}")
    print()
    
    from src.inference.bias_correction import apply_bias_correction
    
    apply_bias_correction(
        input_file=args.input,
        output_file=args.output,
        sensitivity=bc_config['sensitivity'],
        specificity=bc_config['specificity'],
        precision=bc_config['precision'],
        npv=bc_config['npv'],
        prediction_column=args.prediction_column,
        random_state=args.seed,
    )
    
    print("\n✓ Bias correction complete!")


if __name__ == "__main__":
    main()

