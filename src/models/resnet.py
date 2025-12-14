"""ResNet model architecture for war image classification."""

from typing import List, Optional
import torch
import torch.nn as nn
from torchvision import models


def create_model(
    architecture: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 1,
    head_type: str = "custom",
    hidden_dims: List[int] = None,
    unfreeze_layers: List[str] = None,
) -> nn.Module:
    """
    Create a ResNet model for binary classification.
    
    Args:
        architecture: Model architecture ("resnet50", "resnet34", "resnet18")
        pretrained: Whether to use pretrained ImageNet weights
        num_classes: Number of output classes (1 for binary with sigmoid)
        head_type: Type of classification head ("base" or "custom")
        hidden_dims: Hidden layer dimensions for custom head (default: [512, 256])
        unfreeze_layers: List of layer names to unfreeze for fine-tuning
                        Options: "layer1", "layer2", "layer3", "layer4", "fc", "all"
    
    Returns:
        PyTorch model
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    if unfreeze_layers is None:
        unfreeze_layers = ["fc"]
    
    # Load pretrained model
    if architecture == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    elif architecture == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
    elif architecture == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Get input features for the final layer
    num_ftrs = model.fc.in_features
    
    # Replace the classification head
    if head_type == "base":
        # Simple linear layer
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif head_type == "custom":
        # Custom multi-layer head
        layers = []
        in_features = num_ftrs
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
            ])
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, num_classes))
        model.fc = nn.Sequential(*layers)
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")
    
    # Unfreeze specified layers
    _unfreeze_layers(model, unfreeze_layers)
    
    return model


def _unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """
    Unfreeze specified layers for fine-tuning.
    
    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    if "all" in layer_names:
        for param in model.parameters():
            param.requires_grad = True
        return
    
    for name in layer_names:
        if name == "fc":
            for param in model.fc.parameters():
                param.requires_grad = True
        elif name == "layer4":
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif name == "layer3":
            for param in model.layer3.parameters():
                param.requires_grad = True
        elif name == "layer2":
            for param in model.layer2.parameters():
                param.requires_grad = True
        elif name == "layer1":
            for param in model.layer1.parameters():
                param.requires_grad = True
        else:
            print(f"Warning: Unknown layer name '{name}', skipping")


def freeze_all_except_fc(model: nn.Module):
    """Freeze all layers except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> int:
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_params(model: nn.Module) -> int:
    """Get total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def save_model(model: nn.Module, path: str):
    """
    Save model weights to file.
    
    Args:
        model: PyTorch model
        path: Path to save weights
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(
    path: str,
    architecture: str = "resnet50",
    num_classes: int = 1,
    head_type: str = "custom",
    hidden_dims: List[int] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Load model weights from file.
    
    Args:
        path: Path to saved weights
        architecture: Model architecture
        num_classes: Number of output classes
        head_type: Type of classification head
        hidden_dims: Hidden layer dimensions for custom head
        device: Device to load model to
    
    Returns:
        Loaded PyTorch model
    """
    # Create model with same architecture
    model = create_model(
        architecture=architecture,
        pretrained=False,  # Don't need pretrained weights, we're loading our own
        num_classes=num_classes,
        head_type=head_type,
        hidden_dims=hidden_dims,
        unfreeze_layers=["all"],  # Unfreeze all for inference
    )
    
    # Load weights
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {path}")
    return model


def get_device(device_config: str = "auto") -> torch.device:
    """
    Get the appropriate device for training/inference.
    
    Args:
        device_config: "auto", "cuda", "mps", or "cpu"
    
    Returns:
        torch.device
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_config)

