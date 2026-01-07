"""
NFL Big Data Bowl 2026 - Spatio-Temporal Transformer
=====================================================
Transformer-based architecture for trajectory prediction.
"""

import torch
import torch.nn as nn

from config import Config
from src.models.layers import ResidualMLPHead


class STTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for player trajectory prediction.
    
    Architecture:
    1. Input projection: Maps features to hidden dimension
    2. Positional encoding: Learnable temporal positions
    3. Transformer encoder: Multi-head self-attention layers (Pre-LN)
    4. Attention pooling: Aggregate temporal information
    5. Residual MLP head: Predict trajectory deltas
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension size
        horizon: Prediction horizon (number of future frames)
        window_size: Input sequence length
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        dropout: Dropout probability
    """
    def __init__(self, input_dim, hidden_dim, horizon, window_size, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # 1. Spatial: Feature embedding
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. Temporal: Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, window_size, hidden_dim)) 
        self.embed_dropout = nn.Dropout(dropout)

        # 3. Transformer Encoder (with norm_first=True for Pre-LN, matching notebook)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-LN transformer (matches competition notebook)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 4. Attention Pooling
        self.pool_ln = nn.LayerNorm(hidden_dim)
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads=n_heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim)) 

        # 5. Output Head (ResidualMLPHead)
        # Get MLP hyperparameters from config
        config = Config()
        self.head = ResidualMLPHead(
            input_dim=hidden_dim,                   # 128
            hidden_dim=config.MLP_HIDDEN_DIM,       # 256
            output_dim=horizon * 2,                 # 188 (94 frames * 2 coordinates)
            n_res_blocks=config.N_RES_BLOCKS,       # 2
            dropout=0.2
        )
        
        # Apply weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initialize weights using Xavier uniform initialization.
        
        This matches the competition notebook's initialization strategy.
        """
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
            Tensor of shape (batch, horizon, 2) containing (dx, dy) predictions
        """
        B, S, _ = x.shape
        
        # Embed features and add positional encoding
        x_embed = self.input_projection(x) 
        x = x_embed + self.pos_embed[:, :S, :] 
        x = self.embed_dropout(x)
        
        # Process through transformer
        h = self.transformer_encoder(x) 

        # Pool temporal information
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        ctx = ctx.squeeze(1) 

        # Predict trajectory
        out = self.head(ctx)
        out = out.view(B, self.horizon, 2)
        
        # Cumulative sum to convert deltas to absolute positions
        out = torch.cumsum(out, dim=1)
        
        return out