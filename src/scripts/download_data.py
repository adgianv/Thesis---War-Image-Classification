#!/usr/bin/env python3
"""
Download the War Images dataset from Kaggle.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --config configs/config.yaml
"""

import os
import sys
import shutil
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_dataset(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle using kagglehub.
    
    Args:
        dataset_name: Kaggle dataset identifier
        output_dir: Directory to save the dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)
    
    # Check for API token
    token = os.environ.get('KAGGLE_API_TOKEN')
    if not token:
        print("Warning: KAGGLE_API_TOKEN not set in environment.")
        print("You can set it with: export KAGGLE_API_TOKEN=your_token")
        print("Attempting to use default Kaggle credentials...")
    
    print(f"Downloading dataset: {dataset_name}")
    print("This may take a few minutes...")
    
    # Download dataset
    try:
        download_path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {download_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Set KAGGLE_API_TOKEN environment variable")
        print("2. Or create ~/.kaggle/kaggle.json with your credentials")
        print("3. Get your token from: https://www.kaggle.com/settings")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy files to project directory
    images_output = os.path.join(output_dir, "images")
    os.makedirs(images_output, exist_ok=True)
    
    # Find and copy files
    for item in os.listdir(download_path):
        src = os.path.join(download_path, item)
        
        if item.endswith('.csv'):
            # Copy labels file
            dst = os.path.join(output_dir, item)
            shutil.copy2(src, dst)
            print(f"Copied: {item} -> {dst}")
        
        elif os.path.isdir(src):
            # Copy image directories
            for subitem in os.listdir(src):
                subsrc = os.path.join(src, subitem)
                if subitem.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    subdst = os.path.join(images_output, subitem)
                    shutil.copy2(subsrc, subdst)
            print(f"Copied images from: {item}")
        
        elif item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Copy individual images
            dst = os.path.join(images_output, item)
            shutil.copy2(src, dst)
    
    # Count images
    n_images = len([f for f in os.listdir(images_output) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
    
    print(f"\n✓ Download complete!")
    print(f"  - Images: {n_images} files in {images_output}")
    print(f"  - Labels: {output_dir}/labels.csv")
    
    return download_path


def main():
    parser = argparse.ArgumentParser(
        description="Download the War Images dataset from Kaggle"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get settings
    dataset_name = config['data']['kaggle_dataset']
    output_dir = args.output_dir or config['data']['raw_dir']
    
    # Download
    download_dataset(dataset_name, output_dir)


if __name__ == "__main__":
    main()

