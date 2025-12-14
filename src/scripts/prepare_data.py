#!/usr/bin/env python3
"""
Prepare data by splitting into train/val/test sets.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --config configs/config.yaml
"""

import os
import sys
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(
    labels_file: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
    label_column: str = 'Binary Label',
):
    """
    Split labels into train/val/test sets.
    
    Args:
        labels_file: Path to the labels CSV file
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify by label
        label_column: Column name for labels
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Load labels
    print(f"Loading labels from: {labels_file}")
    df = pd.read_csv(labels_file)
    print(f"Total samples: {len(df)}")
    
    # Detect column names (handle different formats)
    print(f"Columns: {df.columns.tolist()}")
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        if col.lower() in ['image', 'filename']:
            column_mapping[col] = 'Image'
        elif col.lower() in ['channel']:
            column_mapping[col] = 'Channel'
        elif 'binary' in col.lower() or col.lower() == 'label':
            column_mapping[col] = 'Binary Label'
        elif 'multiclass' in col.lower():
            column_mapping[col] = 'Multiclass Label'
    
    df = df.rename(columns=column_mapping)
    
    # Check for required columns
    if 'Image' not in df.columns:
        raise ValueError("Could not find image column in labels file")
    if 'Binary Label' not in df.columns and label_column not in df.columns:
        raise ValueError("Could not find label column in labels file")
    
    # Use correct label column
    if label_column not in df.columns:
        label_column = 'Binary Label'
    
    # Print class distribution
    print(f"\nClass distribution:")
    print(df[label_column].value_counts())
    
    # Get stratification target
    stratify_col = df[label_column] if stratify else None
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_col,
    )
    
    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_col_tv = train_val_df[label_column] if stratify else None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=stratify_col_tv,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, "train_labels.csv")
    val_path = os.path.join(output_dir, "val_labels.csv")
    test_path = os.path.join(output_dir, "test_labels.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Data split complete!")
    print(f"  - Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%) -> {train_path}")
    print(f"  - Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%) -> {val_path}")
    print(f"  - Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%) -> {test_path}")
    
    # Print class distribution for each split
    print(f"\nClass distribution per split:")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df[label_column].value_counts()
        print(f"  {name}: {dict(counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data by splitting into train/val/test sets"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help="Path to labels file (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get settings
    labels_file = args.labels_file or config['data']['labels_file']
    output_dir = config['data']['processed_dir']
    split_config = config['data']['split']
    
    # Prepare data
    prepare_data(
        labels_file=labels_file,
        output_dir=output_dir,
        train_ratio=split_config['train'],
        val_ratio=split_config['val'],
        test_ratio=split_config['test'],
        random_seed=split_config['random_seed'],
        stratify=split_config['stratify'],
    )


if __name__ == "__main__":
    main()

