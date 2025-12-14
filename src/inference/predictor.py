"""Inference predictor for unlabeled images."""

from typing import List, Tuple, Optional
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import UnlabeledDataset
from ..data.transforms import get_eval_transforms


class Predictor:
    """
    Predictor class for running inference on unlabeled images.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
        image_size: int = 256,
    ):
        """
        Initialize the predictor.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            threshold: Classification threshold
            image_size: Image size for preprocessing
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.image_size = image_size
        self.transform = get_eval_transforms(image_size=image_size)
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def predict_directory(
        self,
        images_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> pd.DataFrame:
        """
        Run inference on all images in a directory.
        
        Args:
            images_dir: Directory containing images
            batch_size: Batch size for inference
            num_workers: Number of data loading workers
        
        Returns:
            DataFrame with columns: filename, probability, predicted_class
        """
        # Create dataset and dataloader
        dataset = UnlabeledDataset(images_dir, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Run inference
        all_paths = []
        all_probs = []
        
        for inputs, paths in tqdm(dataloader, desc="Predicting", unit="batch"):
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_paths.extend(paths)
            all_probs.extend(probs.tolist())
        
        # Create results DataFrame
        results = pd.DataFrame({
            'filename': [os.path.basename(p) for p in all_paths],
            'filepath': all_paths,
            'probability': all_probs,
            'predicted_class': [1 if p >= self.threshold else 0 for p in all_probs],
        })
        
        # Extract date from filename (format: frame_channel_noche_YYYY-MM-DD_sec_N.jpg)
        def extract_date(filename):
            try:
                parts = filename.split('_')
                return parts[3]  # YYYY-MM-DD
            except (IndexError, ValueError):
                return None
        
        results['date'] = results['filename'].apply(extract_date)
        
        return results
    
    @torch.no_grad()
    def predict_single(self, image_path: str) -> Tuple[int, float]:
        """
        Predict a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (predicted_class, probability)
        """
        from PIL import Image
        
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        output = self.model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= self.threshold else 0
        
        return pred, prob


def predict_batch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[List[str], List[float], List[int]]:
    """
    Run batch prediction on a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader (should return (images, paths))
        device: Device to run on
        threshold: Classification threshold
    
    Returns:
        Tuple of (paths, probabilities, predictions)
    """
    model.eval()
    model.to(device)
    
    all_paths = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_paths.extend(paths)
            all_probs.extend(probs.tolist())
    
    predictions = [1 if p >= threshold else 0 for p in all_probs]
    
    return all_paths, all_probs, predictions

