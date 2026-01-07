"""
NFL Big Data Bowl 2026 - Models Package
========================================
Neural network architectures for trajectory prediction.

Available models:
- STTransformer: Joint (dx, dy) prediction in single model
- SpatioTemporalTransformer: Single coordinate prediction (matches notebook)
- ImprovedSeqModel: Wrapper for SpatioTemporalTransformer
- DualCoordinatePredictor: Container for separate X/Y models
- GRUModel: GRU-based alternative
- ModelEnsemble: Ensemble multiple models
"""

from src.models.layers import ResidualBlock, ResidualMLPHead
from src.models.transformer import STTransformer
from src.models.dual_transformer import (
    SpatioTemporalTransformer,
    ImprovedSeqModel,
    DualCoordinatePredictor
)
from src.models.gru import GRUModel
from src.models.ensemble import ModelEnsemble

__all__ = [
    'ResidualBlock',
    'ResidualMLPHead',
    'STTransformer',
    'SpatioTemporalTransformer',
    'ImprovedSeqModel',
    'DualCoordinatePredictor',
    'GRUModel',
    'ModelEnsemble'
]