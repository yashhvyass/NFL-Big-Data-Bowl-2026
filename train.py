"""
NFL Big Data Bowl 2026 - Training Script
========================================
Main entry point for training dual coordinate models.

Usage:
    python train.py

This script will:
1. Load training data from weeks 1-18
2. Engineer 180+ features including geometric priors
3. Train 5-fold cross-validated dual models (separate X and Y)
4. Save model checkpoints and artifacts
5. Print validation statistics
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import pickle

from config import Config
from src.utils import set_seed
from src.features import prepare_sequences_geometric
from src.training import train_dual_models, compute_val_rmse_dual


def main():
    """Main training function."""
    
    # Initialize
    config = Config()
    set_seed(config.SEED)
    
    # Ensure output directories exist
    model_dir = config.ensure_model_dir()
    
    print("="*80)
    print("NFL BIG DATA BOWL 2026 - DUAL COORDINATE MODEL TRAINING")
    print("="*80)
    print(f"\nConfiguration (matching competition notebook):")
    print(f"  Device: {config.DEVICE}")
    print(f"  Folds: {config.N_FOLDS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Window Size: {config.WINDOW_SIZE}")
    print(f"  Model: SpatioTemporalTransformer (Hidden={config.HIDDEN_DIM}, Heads={config.N_HEADS}, Layers={config.N_LAYERS})")
    print(f"  Architecture: DUAL (Separate X and Y models)")
    print(f"  Debug Mode: {config.DEBUG}")
    print()
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("[1/4] Loading training data...")
    
    train_input_files = [
        config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)
    ]
    train_output_files = [
        config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)
    ]
    
    if config.DEBUG:
        # Debug mode: use only first 100 plays for quick testing
        print("  DEBUG MODE: Loading subset of data...")
        sample_train_input = pd.read_csv(train_input_files[0])
        sample_train_output = pd.read_csv(train_output_files[0])
        sample_plays = sample_train_output[['game_id', 'play_id']].drop_duplicates().head(100)
        
        train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
        train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
        
        train_input = train_input.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
        train_output = train_output.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
        print(f"  Loaded {len(sample_plays)} unique plays")
    else:
        # Full dataset
        print("  Loading all weeks (1-18)...")
        train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
        train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
        print(f"  Loaded {len(train_input)} input frames, {len(train_output)} output frames")
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING
    # ========================================================================
    print("\n[2/4] Engineering features (180+ total: proven + geometric + advanced)...")
    
    result = prepare_sequences_geometric(
        train_input, 
        train_output, 
        is_training=True, 
        window_size=config.WINDOW_SIZE
    )
    
    sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, \
        geo_x, geo_y, route_kmeans, route_scaler = result
    
    sequences = list(sequences)
    targets_dx = list(targets_dx)
    targets_dy = list(targets_dy)
    
    print(f"  Created {len(sequences)} sequences")
    print(f"  Feature dimension: {sequences[0].shape[-1]}")
    
    # ========================================================================
    # STEP 3: CROSS-VALIDATION TRAINING (DUAL MODELS)
    # ========================================================================
    print(f"\n[3/4] Training {config.N_FOLDS}-fold cross-validated DUAL models...")
    
    groups = np.array([d['game_id'] for d in sequence_ids])
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    
    models_x = []
    models_y = []
    scalers = []
    fold_losses = []
    fold_rmses = []
    
    for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{config.N_FOLDS}")
        print(f"{'='*60}")
        
        # Split data
        X_tr = [sequences[i] for i in tr]
        X_va = [sequences[i] for i in va]
        y_tr_dx = [targets_dx[i] for i in tr]
        y_va_dx = [targets_dx[i] for i in va]
        y_tr_dy = [targets_dy[i] for i in tr]
        y_va_dy = [targets_dy[i] for i in va]
        
        print(f"  Training samples: {len(X_tr)}")
        print(f"  Validation samples: {len(X_va)}")
        
        # Scale features
        scaler = StandardScaler()
        scaler.fit(np.vstack([s for s in X_tr]))
        X_tr_sc = [scaler.transform(s) for s in X_tr]
        X_va_sc = [scaler.transform(s) for s in X_va]
        
        # Train dual models
        print(f"\n  Training separate X and Y models...")
        model_x, model_y, loss = train_dual_models(
            X_tr_sc, y_tr_dx, y_tr_dy,
            X_va_sc, y_va_dx, y_va_dy,
            X_tr[0].shape[-1], 
            config.MAX_FUTURE_HORIZON, 
            config
        )
        
        # Compute validation RMSE
        print(f"\n  Computing validation RMSE...")
        val_rmse = compute_val_rmse_dual(
            model_x, model_y, X_va_sc, y_va_dx, y_va_dy, 
            config.MAX_FUTURE_HORIZON, config.DEVICE, config.BATCH_SIZE
        )
        
        models_x.append(model_x)
        models_y.append(model_y)
        scalers.append(scaler)
        fold_losses.append(loss)
        fold_rmses.append(val_rmse)
        
        print(f"\n  Fold {fold} Results:")
        print(f"    Loss: {loss:.5f}")
        print(f"    RMSE: {val_rmse:.5f}")
        
        # Save artifacts
        if config.SAVE_ARTIFACTS:
            try:
                model_x_path = model_dir / f"model_x_fold{fold}.pt"
                model_y_path = model_dir / f"model_y_fold{fold}.pt"
                scaler_path = model_dir / f"scaler_fold{fold}.pkl"
                
                # Save model state dicts on CPU
                state_x = {k: v.cpu() for k, v in model_x.state_dict().items()}
                state_y = {k: v.cpu() for k, v in model_y.state_dict().items()}
                torch.save(state_x, str(model_x_path))
                torch.save(state_y, str(model_y_path))
                
                # Save scaler
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                
                print(f"    Saved: {model_x_path.name}, {model_y_path.name}, {scaler_path.name}")
            except Exception as e:
                print(f"    Warning: Failed to save artifacts - {e}")
    
    # ========================================================================
    # STEP 4: SUMMARY STATISTICS
    # ========================================================================
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    avg_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    avg_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    
    print(f"\nOverall Performance:")
    print(f"  Average Loss:      {avg_loss:.5f} ± {std_loss:.5f}")
    print(f"  Average RMSE:      {avg_rmse:.5f} ± {std_rmse:.5f}")
    
    print(f"\nPer-Fold Results:")
    print(f"  {'Fold':<6} {'Loss':<10} {'RMSE':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10}")
    for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
        print(f"  {i:<6} {loss:<10.5f} {rmse:<10.5f}")
    
    # Save route clustering objects
    if config.SAVE_ARTIFACTS:
        try:
            with open(model_dir / "route_kmeans.pkl", "wb") as f:
                pickle.dump(route_kmeans, f)
            with open(model_dir / "route_scaler.pkl", "wb") as f:
                pickle.dump(route_scaler, f)
            print(f"\nSaved route artifacts: route_kmeans.pkl, route_scaler.pkl")
        except Exception as e:
            print(f"\nWarning: Failed to save route artifacts - {e}")
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nModel artifacts saved to: {model_dir}")
    print(f"Total models trained: {len(models_x)} X models + {len(models_y)} Y models")
    print(f"\nTo use these models for inference, set:")
    print(f"  Config.LOAD_ARTIFACTS = True")
    print(f"  Config.LOAD_DIR = '{model_dir}'")
    print()


if __name__ == "__main__":
    main()