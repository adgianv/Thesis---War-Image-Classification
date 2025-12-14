"""Bias correction for model predictions.

Based on the confusion matrix of the classifier, we can correct for
systematic bias in predictions using conditional probabilities.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


class BiasCorrector:
    """
    Corrects for classifier bias using conditional probabilities.
    
    Given a classifier's performance metrics (sensitivity, specificity,
    precision, NPV), this class adjusts predictions to be unbiased
    estimates of the true class distribution.
    """
    
    def __init__(
        self,
        sensitivity: float,
        specificity: float,
        precision: float,
        npv: float,
    ):
        """
        Initialize the bias corrector.
        
        Args:
            sensitivity: TP / (TP + FN) - True Positive Rate / Recall
            specificity: TN / (TN + FP) - True Negative Rate
            precision: TP / (TP + FP) - Positive Predictive Value
            npv: TN / (TN + FN) - Negative Predictive Value
        """
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.precision = precision
        self.npv = npv
        
        # Validate values
        for name, value in [
            ("sensitivity", sensitivity),
            ("specificity", specificity),
            ("precision", precision),
            ("npv", npv),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
    
    def correct_predictions(
        self,
        predictions: List[int],
        random_state: Optional[int] = None,
    ) -> List[int]:
        """
        Apply bias correction to binary predictions.
        
        For each prediction, we sample from the posterior distribution
        of the true label given the predicted label.
        
        P(True=1 | Pred=1) = precision
        P(True=0 | Pred=0) = npv
        
        Args:
            predictions: List of binary predictions (0 or 1)
            random_state: Random seed for reproducibility
        
        Returns:
            List of corrected predictions
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        corrected = []
        for pred in predictions:
            if pred == 1:
                # P(True=1 | Pred=1) = precision
                corrected_label = np.random.choice(
                    [1, 0],
                    p=[self.precision, 1 - self.precision]
                )
            else:
                # P(True=0 | Pred=0) = npv
                corrected_label = np.random.choice(
                    [0, 1],
                    p=[self.npv, 1 - self.npv]
                )
            corrected.append(corrected_label)
        
        return corrected
    
    def correct_dataframe(
        self,
        df: pd.DataFrame,
        prediction_column: str = "predicted_class",
        output_column: str = "corrected_preds",
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Apply bias correction to a DataFrame.
        
        Args:
            df: DataFrame with predictions
            prediction_column: Name of column with predictions
            output_column: Name for corrected predictions column
            random_state: Random seed
        
        Returns:
            DataFrame with added corrected predictions column
        """
        predictions = df[prediction_column].tolist()
        corrected = self.correct_predictions(predictions, random_state)
        df = df.copy()
        df[output_column] = corrected
        return df
    
    def simulate_bias(
        self,
        n_samples: int = 1000,
        true_positive_rate: float = 0.21,
        n_simulations: int = 10000,
    ) -> dict:
        """
        Simulate the effect of bias and correction.
        
        Args:
            n_samples: Number of samples per simulation
            true_positive_rate: True proportion of positive class
            n_simulations: Number of simulations to run
        
        Returns:
            Dictionary with simulation results
        """
        results = {
            'true_counts': [],
            'predicted_counts': [],
            'corrected_counts': [],
        }
        
        for _ in range(n_simulations):
            # Generate ground truth
            ground_truth = np.random.choice(
                [1, 0],
                size=n_samples,
                p=[true_positive_rate, 1 - true_positive_rate]
            )
            
            # Simulate classifier predictions
            predictions = []
            for true_label in ground_truth:
                if true_label == 1:
                    # P(Pred=1 | True=1) = sensitivity
                    pred = np.random.choice([1, 0], p=[self.sensitivity, 1 - self.sensitivity])
                else:
                    # P(Pred=0 | True=0) = specificity
                    pred = np.random.choice([0, 1], p=[self.specificity, 1 - self.specificity])
                predictions.append(pred)
            
            # Apply correction
            corrected = self.correct_predictions(predictions)
            
            # Store counts
            results['true_counts'].append(sum(ground_truth))
            results['predicted_counts'].append(sum(predictions))
            results['corrected_counts'].append(sum(corrected))
        
        return results
    
    def __repr__(self):
        return (
            f"BiasCorrector("
            f"sensitivity={self.sensitivity:.3f}, "
            f"specificity={self.specificity:.3f}, "
            f"precision={self.precision:.3f}, "
            f"npv={self.npv:.3f})"
        )


def apply_bias_correction(
    input_file: str,
    output_file: str,
    sensitivity: float,
    specificity: float,
    precision: float,
    npv: float,
    prediction_column: str = "predicted_class",
    random_state: Optional[int] = 42,
):
    """
    Apply bias correction to a predictions CSV file.
    
    Args:
        input_file: Path to input CSV with predictions
        output_file: Path to save corrected predictions
        sensitivity: Classifier sensitivity
        specificity: Classifier specificity
        precision: Classifier precision
        npv: Classifier negative predictive value
        prediction_column: Column name for predictions
        random_state: Random seed
    """
    # Load predictions
    df = pd.read_csv(input_file)
    
    # Create corrector and apply
    corrector = BiasCorrector(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        npv=npv,
    )
    
    df = corrector.correct_dataframe(
        df,
        prediction_column=prediction_column,
        random_state=random_state,
    )
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"Bias-corrected predictions saved to {output_file}")
    
    # Print summary
    original_rate = df[prediction_column].mean()
    corrected_rate = df['corrected_preds'].mean()
    print(f"Original positive rate: {original_rate:.4f}")
    print(f"Corrected positive rate: {corrected_rate:.4f}")

