"""
NFL Big Data Bowl 2026 - Model Ensemble
========================================
Ensemble multiple models for improved prediction accuracy.
"""

import numpy as np
import torch


class ModelEnsemble:
    """
    Ensemble multiple trained models with weighted averaging.
    
    Args:
        models: List of trained models
        weights: Optional list of weights for each model (defaults to equal weights)
    """
    def __init__(self, models, weights=None):
        self.models = models
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            total = sum(weights)
            self.weights = [w / total for w in weights]  # Normalize
    
    def predict(self, X_test, scalers, device):
        """
        Generate ensemble predictions.
        
        Args:
            X_test: List of test sequences
            scalers: List of scalers (one per model)
            device: Device to run inference on
            
        Returns:
            numpy array: Averaged predictions of shape (n_samples, horizon, 2)
        """
        all_preds = []
        
        for model, scaler, weight in zip(self.models, scalers, self.weights):
            # Scale features
            X_scaled = [scaler.transform(s) for s in X_test]
            X_tensor = torch.tensor(np.stack(X_scaled).astype(np.float32)).to(device)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                preds = model(X_tensor).cpu().numpy()
            
            all_preds.append(preds * weight)
        
        # Weighted average
        ensemble_preds = np.sum(all_preds, axis=0)
        return ensemble_preds