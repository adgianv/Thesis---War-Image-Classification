"""Model explainability tools: GradCAM, Occlusion Sensitivity."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates heatmaps showing which regions of an image are important
    for the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = "layer4"):
        """
        Initialize GradCAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of the layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer '{self.target_layer}' not found in model")
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (default: predicted class)
        
        Returns:
            Heatmap as numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = 0  # For binary classification
        
        # Backward pass
        target = output[0][target_class]
        target.backward()
        
        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target layer.")
        
        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations.squeeze(0)
        for i in range(activations.shape[0]):
            activations[i] *= pooled_gradients[i]
        
        # Generate heatmap
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0, 1]
        
        return heatmap
    
    def visualize(
        self,
        image_path: str,
        transform,
        device: torch.device,
        alpha: float = 0.4,
        save_path: Optional[str] = None,
    ):
        """
        Generate and display Grad-CAM visualization.
        
        Args:
            image_path: Path to input image
            transform: Transform to apply to image
            device: Device to run model on
            alpha: Transparency of heatmap overlay
            save_path: Path to save visualization (optional)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(input_tensor)
        
        # Resize heatmap to image size
        image_np = np.array(image)
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
                (image_np.shape[1], image_np.shape[0]),
                Image.LANCZOS
            )
        ) / 255.0
        
        # Apply colormap
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        
        # Overlay on original image
        superimposed = heatmap_colored * alpha + image_np / 255.0 * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 1)
        
        # Display
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def occlusion_sensitivity(
    model: nn.Module,
    image_tensor: torch.Tensor,
    patch_size: int = 32,
    stride: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Perform occlusion sensitivity analysis.
    
    Slides a patch over the image and measures how much the prediction
    changes when each region is occluded.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Input image tensor (1, C, H, W)
        patch_size: Size of the occlusion patch
        stride: Stride for sliding the patch (default: patch_size)
        device: Device to run on
    
    Returns:
        Sensitivity heatmap as numpy array
    """
    if stride is None:
        stride = patch_size
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    _, _, img_height, img_width = image_tensor.shape
    image_tensor = image_tensor.to(device)
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(image_tensor)
        original_score = torch.sigmoid(original_output).item()
    
    # Initialize heatmap
    heatmap = np.zeros((img_height, img_width))
    count_map = np.zeros((img_height, img_width))
    
    # Slide patch over image
    for i in range(0, img_height - patch_size + 1, stride):
        for j in range(0, img_width - patch_size + 1, stride):
            # Create occluded image
            occluded = image_tensor.clone()
            occluded[:, :, i:i+patch_size, j:j+patch_size] = 0
            
            # Get prediction for occluded image
            with torch.no_grad():
                output = model(occluded)
                occluded_score = torch.sigmoid(output).item()
            
            # Sensitivity = change in score
            sensitivity = original_score - occluded_score
            
            # Update heatmap
            heatmap[i:i+patch_size, j:j+patch_size] += sensitivity
            count_map[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map
    
    return heatmap


def plot_occlusion_sensitivity(
    model: nn.Module,
    image_path: str,
    transform,
    device: torch.device,
    patch_size: int = 32,
    save_path: Optional[str] = None,
):
    """
    Generate and display occlusion sensitivity visualization.
    
    Args:
        model: Trained model
        image_path: Path to input image
        transform: Transform to apply
        device: Device to run on
        patch_size: Size of occlusion patch
        save_path: Path to save visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get heatmap
    heatmap = occlusion_sensitivity(model, input_tensor, patch_size, device=device)
    
    # Resize heatmap to original image size
    heatmap_resized = np.array(
        Image.fromarray(((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]),
            Image.LANCZOS
        )
    ) / 255.0
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(image_np)
    im = axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[1].set_title('Occlusion Sensitivity')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    
    plt.show()

