"""
NFL Big Data Bowl 2026 - GRU Model
===================================
GRU-based architecture for trajectory prediction.

This is a simpler alternative to the Transformer model that can be
used for ensembling or as a standalone predictor.
"""

import torch
import torch.nn as nn

from config import Config


class GRUModel(nn.Module):
    """
    GRU-based model for trajectory prediction.
    
    Architecture:
    1. Input projection to hidden dimension
    2. Bidirectional GRU layers
    3. Attention pooling over time
    4. MLP head for trajectory prediction
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for GRU
        horizon: Prediction horizon (number of future frames)
        n_layers: Number of GRU layers (default: 2)
        dropout: Dropout probability (default: 0.1)
        bidirectional: Whether to use bidirectional GRU (default: True)
    """
    def __init__(self, input_dim, hidden_dim, horizon, n_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention pooling
        gru_output_dim = hidden_dim * self.n_directions
        self.attention = nn.Linear(gru_output_dim, 1)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * 2)
        )
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier uniform."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)
            
        Returns:
            Tensor of shape (batch, horizon, 2) containing (dx, dy) predictions
        """
        B, S, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = self.input_ln(x)
        
        # GRU encoding
        gru_out, _ = self.gru(x)  # (B, S, hidden_dim * n_directions)
        
        # Attention pooling
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)  # (B, S, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)  # (B, hidden_dim * n_directions)
        
        # Output prediction
        out = self.output_head(context)  # (B, horizon * 2)
        out = out.view(B, self.horizon, 2)
        
        # Cumulative sum to convert deltas to absolute positions
        out = torch.cumsum(out, dim=1)
        
        return out