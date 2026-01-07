# NFL Big Data Bowl 2026

Trajectory prediction for NFL players during pass plays. Given player positions and velocities before a pass, predict where each player will be when the ball arrives.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train.py

# Run inference (called by Kaggle API)
from src.inference import predict
```

## Project Structure

```
NFL_BDB_2026/
├── config.py              # Hyperparameters and paths
├── train.py               # Training entrypoint
├── src/
│   ├── features.py        # Feature engineering (180+ features)
│   ├── inference.py       # NFLPredictor class
│   ├── training.py        # Training loops
│   ├── utils.py           # Utilities
│   └── models/
│       ├── dual_transformer.py   # Main model (separate X/Y)
│       ├── transformer.py        # Joint STTransformer
│       ├── gru.py                # GRU alternative
│       └── layers.py             # ResidualMLP, etc.
└── outputs/models/        # Saved checkpoints
```

## Approach

### Pipeline Architecture

```
Raw Tracking Data
      │
      ▼
┌─────────────────┐
│ Direction Fix   │  Normalize all plays to same direction
└────────┬────────┘
         ▼
┌─────────────────┐
│ Feature Eng.    │  180+ features (kinematic, spatial, opponent, GNN, geometric)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Sequence Build  │  Create (window_size, features) sequences per player
└────────┬────────┘
         ▼
┌─────────────────┐
│ Normalize       │  StandardScaler per fold
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────┐
│              Model Ensemble                     │
│                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐     │
│  │ ST-Trans  │ │ GRU+Attn  │ │ Conv+GRU  │     │
│  │  (X, Y)   │ │  (X, Y)   │ │  (X, Y)   │     │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘     │
│        └─────────────┼───────────────┘         │
│                      ▼                         │
│              Weighted Average                  │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────┐
│ Post-process    │  Cumsum deltas, clip to field, restore direction
└────────┬────────┘
         ▼
   Predicted (x, y)
```

### Model Architecture

We use separate SpatioTemporalTransformer models for X and Y coordinate prediction:

- **Input**: Sequence of player features over 12 frames
- **Encoder**: Transformer with 8 attention heads, 4 layers, 256 hidden dim
- **Output**: Cumulative position deltas for 94 future frames

### Feature Engineering

180+ features including:
- **Kinematic**: velocity, acceleration, momentum
- **Spatial**: distance to ball, angle to target, field position
- **Opponent**: nearest defender distance, coverage assignment
- **Route**: clustered route patterns (K-Means)
- **GNN**: neighbor-weighted positional embeddings
- **Geometric**: required velocity to intercept, alignment errors

### Training

- 5-fold cross-validation (grouped by game)
- Temporal Huber loss with exponential decay
- AdamW optimizer with learning rate scheduling
- Early stopping (patience=30)

## Results

Validation RMSE: ~0.49 yards per frame

## Key Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters in one place |
| `src/features.py` | Feature engineering pipeline |
| `src/models/dual_transformer.py` | Main model architecture |
| `src/inference.py` | Kaggle submission interface |

## Configuration

Key hyperparameters in `config.py`:

```python
HIDDEN_DIM = 256
N_HEADS = 8
N_LAYERS = 4
WINDOW_SIZE = 12
N_FOLDS = 5
```
