"""
NFL Big Data Bowl 2026 - Utility Functions
===========================================
Core utility functions including set_seed and data type helpers.

Note: Feature engineering functions (get_velocity, height_to_feet, 
      geometric features) have been consolidated into features.py
      to avoid code duplication.
"""

import random
import os
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def canonicalize_key_dtypes(df):
    """
    Ensure key columns have consistent int64 types for proper joins.
    
    Args:
        df: DataFrame with game_id, play_id, nfl_id columns
        
    Returns:
        DataFrame with standardized key column types
    """
    import pandas as pd
    
    df = df.copy()
    for c in ('game_id', 'play_id', 'nfl_id'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Drop rows with missing keys
    df = df.dropna(subset=['game_id', 'play_id', 'nfl_id'])
    
    # Unify to int64
    df['game_id'] = df['game_id'].astype('int64')
    df['play_id'] = df['play_id'].astype('int64')
    df['nfl_id'] = df['nfl_id'].astype('int64')
    
    return df