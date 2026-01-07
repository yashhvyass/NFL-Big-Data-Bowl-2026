"""
NFL Big Data Bowl 2026 - Neural Network Layers
===============================================
Reusable neural network components for trajectory prediction models.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    A standard residual block: FFN + skip connection.
    
    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Pre-normalization with residual connection
        return x + self.ffn(self.norm(x))


class ResidualMLPHead(nn.Module):
    """
    MLP head with residual connections for robust prediction.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        n_res_blocks: Number of residual blocks
        dropout: Dropout probability
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_res_blocks=2, dropout=0.2):
        super().__init__()
        
        # Project from input to hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(n_res_blocks)]
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_norm(x)
        x = self.output_layer(x)
        return x