"""
NFL Big Data Bowl 2026 - Configuration
========================================
Central configuration for the NFL trajectory prediction project.
"""

from pathlib import Path
import torch
import os


class Config:
    """Main configuration class for the NFL prediction project."""
    
    # ============================================================================
    # PATHS (with local development fallback)
    # ============================================================================
    # Kaggle paths
    _KAGGLE_DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    _KAGGLE_LOAD_DIR = Path("/kaggle/input/nfl-gnn-a43/outputs/models")
    
    # Local development paths (relative to project root)
    _LOCAL_DATA_DIR = Path("./data")
    _LOCAL_OUTPUT_DIR = Path("./outputs")
    
    # Determine which paths to use based on environment
    if _KAGGLE_DATA_DIR.exists():
        DATA_DIR = _KAGGLE_DATA_DIR
        LOAD_DIR = str(_KAGGLE_LOAD_DIR)
    else:
        DATA_DIR = _LOCAL_DATA_DIR
        LOAD_DIR = str(_LOCAL_OUTPUT_DIR / "models")
    
    OUTPUT_DIR = Path("./outputs")
    
    # Create directories only when they're actually used (lazy creation)
    @classmethod
    def ensure_output_dir(cls):
        """Ensure output directory exists."""
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        return cls.OUTPUT_DIR
    
    @classmethod
    def ensure_model_dir(cls):
        """Ensure model directory exists."""
        cls.ensure_output_dir()
        model_dir = cls.OUTPUT_DIR / "models"
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    # Property for MODEL_DIR (lazy creation)
    @classmethod
    @property
    def MODEL_DIR(cls):
        return cls.ensure_model_dir()
    
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = True
    
    # ============================================================================
    # TRAINING PARAMETERS
    # ============================================================================
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 256
    EPOCHS = 200
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    WINDOW_SIZE = 12
    HIDDEN_DIM = 256
    MAX_FUTURE_HORIZON = 94
    
    # Transformer hyperparameters
    N_HEADS = 8
    N_LAYERS = 4
    
    # ResidualMLP Head hyperparameters
    MLP_HIDDEN_DIM = 512
    N_RES_BLOCKS = 3
    
    # ============================================================================
    # FIELD DIMENSIONS
    # ============================================================================
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    # ============================================================================
    # FEATURE ENGINEERING
    # ============================================================================
    K_NEIGH = 6      # Number of neighbors for GNN
    RADIUS = 30.0    # Radius for neighbor search
    TAU = 8.0        # Temperature for attention weights
    N_ROUTE_CLUSTERS = 7  # Number of route pattern clusters
    
    # ============================================================================
    # SYSTEM
    # ============================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEBUG = False
    
    # Debug mode settings
    if DEBUG:
        N_FOLDS = 2