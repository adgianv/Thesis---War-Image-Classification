"""DataLoader factory functions."""

from typing import Tuple, Optional
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import WarDataset, UnlabeledDataset
from .transforms import get_train_transforms, get_eval_transforms


def create_dataloaders(
    images_dir: str,
    train_labels: str,
    val_labels: str,
    test_labels: str,
    batch_size: int = 32,
    image_size: int = 256,
    use_weighted_sampler: bool = True,
    num_workers: int = 0,
    label_column: str = 'Binary Label',
    image_column: str = 'Image',
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        images_dir: Directory containing images
        train_labels: Path to training labels CSV
        val_labels: Path to validation labels CSV
        test_labels: Path to test labels CSV
        batch_size: Batch size for all loaders
        image_size: Target image size
        use_weighted_sampler: Whether to use weighted sampling for class imbalance
        num_workers: Number of data loading workers
        label_column: Column name for labels
        image_column: Column name for image filenames
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = WarDataset(
        images_dir=images_dir,
        labels_file=train_labels,
        transform=train_transform,
        label_column=label_column,
        image_column=image_column,
    )
    
    val_dataset = WarDataset(
        images_dir=images_dir,
        labels_file=val_labels,
        transform=eval_transform,
        label_column=label_column,
        image_column=image_column,
    )
    
    test_dataset = WarDataset(
        images_dir=images_dir,
        labels_file=test_labels,
        transform=eval_transform,
        label_column=label_column,
        image_column=image_column,
    )
    
    # Create weighted sampler for training (handles class imbalance)
    sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        train_labels_list = train_dataset.get_labels()
        class_counts = pd.Series(train_labels_list).value_counts()
        sample_weights = [1.0 / class_counts[label] for label in train_labels_list]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Can't use shuffle with sampler
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def create_inference_dataloader(
    images_dir: str,
    batch_size: int = 32,
    image_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a dataloader for inference on unlabeled images.
    
    Args:
        images_dir: Directory containing images
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader for inference
    """
    transform = get_eval_transforms(image_size=image_size)
    
    dataset = UnlabeledDataset(
        images_dir=images_dir,
        transform=transform,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader

