"""
NFL Big Data Bowl 2026 - Source Package
========================================
Main package for NFL trajectory prediction.

Modules:
- utils: Core utility functions (set_seed, data type helpers)
- features: Feature engineering pipeline with geometric priors
- training: Training loops and loss functions
- inference: NFLPredictor class for model inference
- models: Neural network architectures (Transformer, GRU, etc.)
"""

# Lazy imports to avoid circular dependencies
# Import modules directly when needed rather than at package level

__all__ = ['utils', 'features', 'training', 'inference', 'models']