"""Image transformation pipelines."""

from torchvision import transforms


def get_train_transforms(image_size: int = 256, mean: list = None, std: list = None):
    """
    Get training transforms with data augmentation.
    
    Args:
        image_size: Target image size (square)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    
    Returns:
        torchvision.transforms.Compose: Training transform pipeline
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(image_size: int = 256, mean: list = None, std: list = None):
    """
    Get evaluation/inference transforms (no augmentation).
    
    Args:
        image_size: Target image size (square)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    
    Returns:
        torchvision.transforms.Compose: Evaluation transform pipeline
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def denormalize(image_tensor, mean: list = None, std: list = None):
    """
    Reverse normalization for visualization.
    
    Args:
        image_tensor: Normalized image tensor (C, H, W)
        mean: Normalization mean used
        std: Normalization std used
    
    Returns:
        numpy.ndarray: Denormalized image (H, W, C) in range [0, 1]
    """
    import numpy as np
    
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean)
    std = np.array(std)
    
    # Convert to numpy and transpose to (H, W, C)
    if hasattr(image_tensor, 'cpu'):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    if image_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        image_np = image_np.transpose(1, 2, 0)
    
    # Denormalize
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    
    return image_np

