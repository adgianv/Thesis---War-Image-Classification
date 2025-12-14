"""PyTorch Dataset classes for war image classification."""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WarDataset(Dataset):
    """
    Dataset for labeled war images.
    
    Supports both binary (war/not_war) and multiclass labels.
    """
    
    # Label mapping for binary classification
    BINARY_LABEL_MAP = {
        'not_war': 0,
        'war': 1,
    }
    
    # Label mapping for multiclass (maps to binary)
    MULTICLASS_TO_BINARY_MAP = {
        'not_war': 0,
        'military': 1,
        'damaged_infrastructure': 1,
        'military&anchor': 1,
        'damaged_infrastructure&anchor': 1,
    }
    
    def __init__(
        self,
        images_dir: str,
        labels_file: str,
        transform=None,
        label_column: str = 'Binary Label',
        image_column: str = 'Image',
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing the images
            labels_file: Path to CSV file with labels
            transform: Optional transforms to apply
            label_column: Column name for labels ('Binary Label' or 'Multiclass Label')
            image_column: Column name for image filenames
        """
        self.images_dir = images_dir
        self.transform = transform
        self.label_column = label_column
        self.image_column = image_column
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Validate required columns exist
        if self.image_column not in self.labels_df.columns:
            # Try alternative column names
            if 'image' in self.labels_df.columns:
                self.image_column = 'image'
            elif 'Image' in self.labels_df.columns:
                self.image_column = 'Image'
            else:
                raise ValueError(f"Image column not found. Available: {self.labels_df.columns.tolist()}")
        
        if self.label_column not in self.labels_df.columns:
            # Try alternative column names
            if 'label_no_lc' in self.labels_df.columns:
                self.label_column = 'label_no_lc'
            elif 'Binary Label' in self.labels_df.columns:
                self.label_column = 'Binary Label'
            else:
                raise ValueError(f"Label column not found. Available: {self.labels_df.columns.tolist()}")
        
        # Filter out any rows with missing images
        self._validate_images()
    
    def _validate_images(self):
        """Remove rows where image file doesn't exist."""
        valid_rows = []
        for idx, row in self.labels_df.iterrows():
            img_path = os.path.join(self.images_dir, row[self.image_column])
            if os.path.exists(img_path):
                valid_rows.append(idx)
        
        if len(valid_rows) < len(self.labels_df):
            missing = len(self.labels_df) - len(valid_rows)
            print(f"Warning: {missing} images not found, using {len(valid_rows)} valid images")
            self.labels_df = self.labels_df.loc[valid_rows].reset_index(drop=True)
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        # Get image path and label
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.images_dir, row[self.image_column])
        label_str = row[self.label_column]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to numeric
        if label_str in self.BINARY_LABEL_MAP:
            label = self.BINARY_LABEL_MAP[label_str]
        elif label_str in self.MULTICLASS_TO_BINARY_MAP:
            label = self.MULTICLASS_TO_BINARY_MAP[label_str]
        else:
            # Try numeric label
            try:
                label = int(label_str)
            except ValueError:
                raise ValueError(f"Unknown label: {label_str}")
        
        return image, label
    
    def get_labels(self):
        """Get all labels as a list (for computing class weights)."""
        labels = []
        for idx in range(len(self)):
            label_str = self.labels_df.iloc[idx][self.label_column]
            if label_str in self.BINARY_LABEL_MAP:
                labels.append(self.BINARY_LABEL_MAP[label_str])
            elif label_str in self.MULTICLASS_TO_BINARY_MAP:
                labels.append(self.MULTICLASS_TO_BINARY_MAP[label_str])
            else:
                labels.append(int(label_str))
        return labels


class UnlabeledDataset(Dataset):
    """
    Dataset for unlabeled images (inference only).
    """
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    
    def __init__(self, images_dir: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing images
            transform: Optional transforms to apply
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Get list of image files
        self.image_paths = []
        for filename in sorted(os.listdir(images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in self.VALID_EXTENSIONS:
                self.image_paths.append(os.path.join(images_dir, filename))
        
        print(f"Found {len(self.image_paths)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, img_path

