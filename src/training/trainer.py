"""Training loop with early stopping and checkpointing."""

import os
from typing import Tuple, List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Trainer class for model training with early stopping and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "model_weights",
        patience: int = 3,
        save_best_only: bool = True,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            patience: Early stopping patience
            save_best_only: Only save best model (by val loss)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.save_best_only = save_best_only
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        
        loop = tqdm(self.train_loader, desc="Training", unit="batch")
        for inputs, labels in loop:
            # Move to device and prepare labels for BCE
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            loop.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        running_loss = 0.0
        
        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
        
        val_loss = running_loss / len(self.val_loader.dataset)
        return val_loss
    
    def train(
        self,
        num_epochs: int,
        checkpoint_name: str = "best_model.pth",
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train the model for multiple epochs with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs
            checkpoint_name: Name for the saved checkpoint
        
        Returns:
            Tuple of (trained model, training history dict)
        """
        print(f"Training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✓ New best model saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    print("\n⚠ Early stopping triggered!")
                    break
        
        # Load best model
        print(f"\nLoading best model from {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        
        print("\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.model, history


def create_optimizer(
    model: nn.Module,
    name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: PyTorch model
        name: Optimizer name ("adam", "sgd", "adamw")
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
    
    Returns:
        PyTorch optimizer
    """
    # Only optimize parameters that require gradients
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if name.lower() == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif name.lower() == "adamw":
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif name.lower() == "sgd":
        return optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def create_criterion(name: str = "bce_with_logits") -> nn.Module:
    """
    Create a loss function.
    
    Args:
        name: Loss function name ("bce_with_logits", "bce", "cross_entropy")
    
    Returns:
        PyTorch loss function
    """
    if name.lower() == "bce_with_logits":
        return nn.BCEWithLogitsLoss()
    elif name.lower() == "bce":
        return nn.BCELoss()
    elif name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {name}")

