"""
NFL Big Data Bowl 2026 - Dual Coordinate Transformer
=====================================================
Separate X and Y prediction models matching competition notebook architecture.

This architecture uses separate models for predicting dx and dy,
which matches the competition notebook's approach.
"""

import torch
import torch.nn as nn

from config import Config


class ResidualBlock(nn.Module):
    """Standard residual block with Pre-LN."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.net(x) + x)


class ResidualMLP(nn.Module):
    """MLP with residual blocks matching notebook architecture."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        
        # First layer: project to hidden dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Residual blocks
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for single-coordinate prediction.
    
    This matches the notebook's SpatioTemporalTransformer architecture exactly.
    It predicts a single coordinate (either X or Y) with cumsum output.
    
    Args:
        input_dim: Number of input features
        horizon: Prediction horizon (number of future frames)
        hidden_dim: Hidden dimension size (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 4)
        dropout: Dropout probability (default: 0.1)
    """
    def __init__(self, input_dim, horizon, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Learnable temporal positional encoding
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, Config.WINDOW_SIZE, hidden_dim))
        
        # Transformer encoder with norm_first=True (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN transformer (important!)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Prediction head (outputs horizon values for single coordinate)
        self.prediction_head = ResidualMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=horizon,  # Single coordinate output
            num_layers=3,
            dropout=dropout
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(horizon)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform (matching notebook)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            
        Returns:
            Tensor of shape (batch, horizon) containing coordinate predictions
        """
        batch_size, window_size, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.temporal_pos_encoding[:, :window_size, :]
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Attention pooling over time
        attention_weights = torch.softmax(torch.mean(x, dim=-1), dim=-1)
        x_pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        # Predict deltas
        pred = self.prediction_head(x_pooled)
        pred = self.output_norm(pred)
        
        # Cumulative sum to convert deltas to positions
        pred = torch.cumsum(pred, dim=1)
        
        return pred


class ImprovedSeqModel(nn.Module):
    """
    Wrapper around SpatioTemporalTransformer for single-coordinate prediction.
    
    This matches the notebook's ImprovedSeqModel exactly.
    """
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.horizon = horizon
        config = Config()
        self.model = SpatioTemporalTransformer(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=config.HIDDEN_DIM,
            num_heads=config.N_HEADS,
            num_layers=config.N_LAYERS,
            dropout=0.1
        )
    
    def forward(self, x):
        return self.model(x)


class DualCoordinatePredictor:
    """
    Container for separate X and Y prediction models.
    
    This class manages a pair of SpatioTemporalTransformer models,
    one for predicting dx and one for predicting dy.
    
    Usage:
        predictor = DualCoordinatePredictor(input_dim, horizon, device)
        predictor.train()  # Set to training mode
        dx = predictor.model_x(X_batch)
        dy = predictor.model_y(X_batch)
    """
    def __init__(self, input_dim, horizon, device=None):
        self.input_dim = input_dim
        self.horizon = horizon
        self.device = device or Config.DEVICE
        
        # Create separate models for X and Y
        self.model_x = ImprovedSeqModel(input_dim, horizon).to(self.device)
        self.model_y = ImprovedSeqModel(input_dim, horizon).to(self.device)
    
    def train(self):
        """Set both models to training mode."""
        self.model_x.train()
        self.model_y.train()
    
    def eval(self):
        """Set both models to evaluation mode."""
        self.model_x.eval()
        self.model_y.eval()
    
    def parameters(self):
        """Return parameters from both models."""
        return list(self.model_x.parameters()) + list(self.model_y.parameters())
    
    def to(self, device):
        """Move both models to device."""
        self.device = device
        self.model_x = self.model_x.to(device)
        self.model_y = self.model_y.to(device)
        return self
    
    def state_dict(self):
        """Return state dict for both models."""
        return {
            'model_x': self.model_x.state_dict(),
            'model_y': self.model_y.state_dict()
        }
    
    def load_state_dict(self, state):
        """Load state dict for both models."""
        self.model_x.load_state_dict(state['model_x'])
        self.model_y.load_state_dict(state['model_y'])
    
    def predict(self, x):
        """
        Predict both dx and dy.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            
        Returns:
            Tuple of (dx, dy) tensors, each of shape (batch, horizon)
        """
        with torch.no_grad():
            dx = self.model_x(x)
            dy = self.model_y(x)
        return dx, dy
