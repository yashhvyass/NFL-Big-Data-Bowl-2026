"""
NFL Big Data Bowl 2026 - Inference Module
=========================================
NFLPredictor class for model inference and submission generation.

Uses separate X and Y models (matching competition notebook architecture).
"""

import torch
import numpy as np
import pandas as pd
import polars as pl
import pickle
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from config import Config
from src.utils import set_seed
from src.features import (
    prepare_sequences_geometric, 
    build_play_direction_map,
    apply_direction_to_df,
    invert_to_original_direction,
    unify_left_direction
)
from src.training import train_dual_models, compute_val_rmse_dual
from src.models.dual_transformer import ImprovedSeqModel

warnings.filterwarnings('ignore')


class NFLPredictor:
    """
    Main predictor class for NFL player trajectory prediction.
    
    This class uses SEPARATE X and Y models (matching competition notebook):
    - models_x: List of trained models for X coordinate prediction
    - models_y: List of trained models for Y coordinate prediction
    - scalers: List of feature scalers (one per fold, shared by X and Y models)
    
    Attributes:
        config: Configuration object
        models_x: List of X prediction models (one per fold)
        models_y: List of Y prediction models (one per fold)
        scalers: List of feature scalers (one per fold)
        route_kmeans: Route clustering model
        route_scaler: Route feature scaler
    """
    
    def __init__(self):
        """Initialize predictor, loading pre-trained models or training new ones."""
        warnings.filterwarnings('ignore')
        self.config = Config()
        set_seed(self.config.SEED)

        # Try to load pre-saved artifacts
        load_dir = Path(self.config.LOAD_DIR) if self.config.LOAD_DIR is not None else self.config.ensure_model_dir()
        
        # Check for dual model artifacts (model_x_fold* and model_y_fold*)
        artifacts_present = all(
            (load_dir / f"model_x_fold{f}.pt").exists() and 
            (load_dir / f"model_y_fold{f}.pt").exists()
            for f in range(1, self.config.N_FOLDS + 1)
        ) and (
            (load_dir / "route_kmeans.pkl").exists() and 
            (load_dir / "route_scaler.pkl").exists()
        )

        if self.config.LOAD_ARTIFACTS and artifacts_present:
            print(f"\n[1/2] Loading trained artifacts from disk (from {load_dir})...")
            self.models_x = []
            self.models_y = []
            self.scalers = []
            
            # Load route objects
            try:
                with open(load_dir / "route_kmeans.pkl", "rb") as f:
                    self.route_kmeans = pickle.load(f)
                with open(load_dir / "route_scaler.pkl", "rb") as f:
                    self.route_scaler = pickle.load(f)
            except Exception as e:
                print("Failed to load route artifacts:", e)
                self.route_kmeans = None
                self.route_scaler = None

            # Infer input_dim from first scaler
            dummy_input_dim = 180  # fallback (larger due to new features)
            input_dim = dummy_input_dim
            try:
                with open(load_dir / "scaler_fold1.pkl", "rb") as f:
                    scaler1 = pickle.load(f)
                input_dim = scaler1.mean_.shape[0]
            except Exception:
                pass

            # Load models and scalers
            for fold in range(1, self.config.N_FOLDS + 1):
                model_x_path = load_dir / f"model_x_fold{fold}.pt"
                model_y_path = load_dir / f"model_y_fold{fold}.pt"
                scaler_path = load_dir / f"scaler_fold{fold}.pkl"

                # Load scaler
                try:
                    with open(scaler_path, "rb") as f:
                        scaler = pickle.load(f)
                except Exception:
                    scaler = None

                # Instantiate and load X model
                model_x = ImprovedSeqModel(
                    input_dim=input_dim,
                    horizon=self.config.MAX_FUTURE_HORIZON
                ).to(self.config.DEVICE)
                
                try:
                    state_x = torch.load(model_x_path, map_location=self.config.DEVICE)
                    model_x.load_state_dict(state_x)
                except Exception as e:
                    print(f"Failed to load model_x fold {fold}:", e)

                # Instantiate and load Y model
                model_y = ImprovedSeqModel(
                    input_dim=input_dim,
                    horizon=self.config.MAX_FUTURE_HORIZON
                ).to(self.config.DEVICE)
                
                try:
                    state_y = torch.load(model_y_path, map_location=self.config.DEVICE)
                    model_y.load_state_dict(state_y)
                except Exception as e:
                    print(f"Failed to load model_y fold {fold}:", e)

                model_x.eval()
                model_y.eval()

                self.models_x.append(model_x)
                self.models_y.append(model_y)
                self.scalers.append(scaler)

            print(f"Loaded {len(self.models_x)} X models and {len(self.models_y)} Y models")
            print("[2/2] Ready for inference.")
            return

        # If not loading, proceed with training
        print("[1/4] Loading data for training...")
        train_input_files = [
            self.config.DATA_DIR / f"train/input_2023_w{w:02d}.csv" for w in range(1, 19)
        ]
        train_output_files = [
            self.config.DATA_DIR / f"train/output_2023_w{w:02d}.csv" for w in range(1, 19)
        ]
        
        if self.config.DEBUG:
            sample_train_input = pd.read_csv(train_input_files[0])
            sample_train_output = pd.read_csv(train_output_files[0])
            sample_plays = sample_train_output[['game_id', 'play_id']].drop_duplicates().head(100)
            
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])
            
            train_input = train_input.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            train_output = train_output.merge(sample_plays, on=['game_id', 'play_id'], how='inner')
            print(f"Reduced to {len(sample_plays)} unique plays.")
        else:
            train_input = pd.concat([pd.read_csv(f) for f in train_input_files if f.exists()])
            train_output = pd.concat([pd.read_csv(f) for f in train_output_files if f.exists()])

        print("\n[2/4] Preparing geometric sequences and feature scalers...")
        result = prepare_sequences_geometric(
            train_input, train_output, is_training=True, window_size=self.config.WINDOW_SIZE
        )
        sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids, \
            geo_x, geo_y, route_kmeans, route_scaler = result

        self.route_kmeans = route_kmeans
        self.route_scaler = route_scaler
        
        sequences = list(sequences)
        targets_dx = list(targets_dx)
        targets_dy = list(targets_dy)

        model_dir = self.config.ensure_model_dir()

        print("\n[3/4] Training dual coordinate models...")
        groups = np.array([d['game_id'] for d in sequence_ids])
        gkf = GroupKFold(n_splits=self.config.N_FOLDS)

        self.models_x, self.models_y, self.scalers = [], [], []
        fold_losses = []
        fold_rmses = []

        for fold, (tr, va) in enumerate(gkf.split(sequences, groups=groups), 1):
            print(f"\n--- Fold {fold}/{self.config.N_FOLDS} ---")
            
            X_tr = [sequences[i] for i in tr]
            X_va = [sequences[i] for i in va]
            y_tr_dx = [targets_dx[i] for i in tr]
            y_va_dx = [targets_dx[i] for i in va]
            y_tr_dy = [targets_dy[i] for i in tr]
            y_va_dy = [targets_dy[i] for i in va]
            
            scaler = StandardScaler()
            scaler.fit(np.vstack([s for s in X_tr]))
            X_tr_sc = [scaler.transform(s) for s in X_tr]
            X_va_sc = [scaler.transform(s) for s in X_va]
            
            # Train dual models
            model_x, model_y, loss = train_dual_models(
                X_tr_sc, y_tr_dx, y_tr_dy,
                X_va_sc, y_va_dx, y_va_dy,
                X_tr[0].shape[-1], self.config.MAX_FUTURE_HORIZON, self.config
            )
            
            # Compute validation RMSE
            val_rmse = compute_val_rmse_dual(
                model_x, model_y, X_va_sc, y_va_dx, y_va_dy, 
                self.config.MAX_FUTURE_HORIZON, self.config.DEVICE, self.config.BATCH_SIZE
            )
            
            self.models_x.append(model_x)
            self.models_y.append(model_y)
            self.scalers.append(scaler)
            fold_losses.append(loss)
            fold_rmses.append(val_rmse)
            
            print(f"\nFold {fold} - Loss: {loss:.5f}, Validation RMSE: {val_rmse:.5f}")
            
            # Save artifacts
            if self.config.SAVE_ARTIFACTS:
                try:
                    model_x_path = model_dir / f"model_x_fold{fold}.pt"
                    model_y_path = model_dir / f"model_y_fold{fold}.pt"
                    scaler_path = model_dir / f"scaler_fold{fold}.pkl"
                    
                    state_x = {k: v.cpu() for k, v in model_x.state_dict().items()}
                    state_y = {k: v.cpu() for k, v in model_y.state_dict().items()}
                    torch.save(state_x, str(model_x_path))
                    torch.save(state_y, str(model_y_path))
                    
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)
                    
                    print(f"  Saved: {model_x_path.name}, {model_y_path.name}, {scaler_path.name}")
                except Exception as e:
                    print(f"  Warning: failed to save artifacts for fold {fold}:", e)

        # Print summary
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        avg_loss = np.mean(fold_losses)
        std_loss = np.std(fold_losses)
        avg_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        
        print(f"Average Loss:            {avg_loss:.5f} ± {std_loss:.5f}")
        print(f"Average Validation RMSE: {avg_rmse:.5f} ± {std_rmse:.5f}")
        print(f"\nPer-Fold Results:")
        for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
            print(f"  Fold {i}: Loss={loss:.5f}, RMSE={rmse:.5f}")
        print("="*60 + "\n")

        # Save route clustering objects
        if self.config.SAVE_ARTIFACTS:
            try:
                with open(model_dir / "route_kmeans.pkl", "wb") as f:
                    pickle.dump(self.route_kmeans, f)
                with open(model_dir / "route_scaler.pkl", "wb") as f:
                    pickle.dump(self.route_scaler, f)
                print(f"Saved route artifacts -> route_kmeans.pkl, route_scaler.pkl")
            except Exception as e:
                print("Warning: failed to save route artifacts:", e)

        for mx in self.models_x:
            mx.eval()
        for my in self.models_y:
            my.eval()
        
        print("\nDual Coordinate Model is ready for inference!")

    def predict(self, test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
        """
        Inference function called by the Kaggle API for each time step.
        
        Uses ensemble of separate X and Y models with direction inversion.
        """
        test_input_pd = test_input.to_pandas()
        test_template_pd = test.to_pandas()
        
        dir_map = build_play_direction_map(test_input_pd)
        
        if 'play_direction' not in test_template_pd.columns:
            dir_df = dir_map.reset_index()
            test_template_pd = test_template_pd.merge(
                dir_df, on=['game_id', 'play_id'], how='left', validate='many_to_one'
            )
        
        test_seq, test_ids, test_geo_x, test_geo_y = prepare_sequences_geometric(
            test_input_pd, 
            test_template=test_template_pd, 
            is_training=False,
            window_size=self.config.WINDOW_SIZE,
            route_kmeans=self.route_kmeans, 
            route_scaler=self.route_scaler
        )

        X_test = list(test_seq)
        x_last = np.array([s[-1, 0] for s in X_test])
        y_last = np.array([s[-1, 1] for s in X_test])

        # Ensemble prediction with separate X and Y models
        all_dx, all_dy = [], []
        H = self.config.MAX_FUTURE_HORIZON

        for mx, my, sc in zip(self.models_x, self.models_y, self.scalers):
            if sc is None:
                X_sc = [s for s in X_test]
            else:
                X_sc = [sc.transform(s) for s in X_test]

            X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(self.config.DEVICE)

            mx.eval()
            my.eval()
            with torch.no_grad():
                dx = mx(X_t).cpu().numpy()  # (N, H)
                dy = my(X_t).cpu().numpy()  # (N, H)

            all_dx.append(dx)
            all_dy.append(dy)

        # Ensemble average
        ens_dx = np.mean(all_dx, axis=0)
        ens_dy = np.mean(all_dy, axis=0)

        # Format submission with direction inversion
        rows = []
        for i, sid in enumerate(test_ids):
            fids = test_template_pd[
                (test_template_pd['game_id'] == sid['game_id']) &
                (test_template_pd['play_id'] == sid['play_id']) &
                (test_template_pd['nfl_id'] == sid['nfl_id'])
            ]['frame_id'].sort_values().tolist()
            
            play_dir = sid.get('play_direction', None)
            if play_dir is None:
                play_dir = test_template_pd[
                    (test_template_pd['game_id'] == sid['game_id']) &
                    (test_template_pd['play_id'] == sid['play_id'])
                ]['play_direction'].iloc[0] if 'play_direction' in test_template_pd.columns else 'left'
            
            play_dir_right = (play_dir == 'right')
            
            for t, fid in enumerate(fids):
                tt = min(t, H - 1)
                
                # Compute unified position
                x_u = np.clip(x_last[i] + ens_dx[i, tt], 0, self.config.FIELD_X_MAX)
                y_u = np.clip(y_last[i] + ens_dy[i, tt], 0, self.config.FIELD_Y_MAX)
                
                # Invert to original direction
                x_orig, y_orig = invert_to_original_direction(x_u, y_u, play_dir_right)

                rows.append({
                    'x': x_orig,
                    'y': y_orig
                })

        submission = pd.DataFrame(rows)
        return submission


# ============================================================================
# KAGGLE API INTERFACE
# ============================================================================

_predictor = None


def _get_predictor():
    """Lazy initialization of predictor."""
    global _predictor
    if _predictor is None:
        _predictor = NFLPredictor()
    return _predictor


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """
    Main prediction function called by the Kaggle evaluation API.
    """
    predictor = _get_predictor()
    return predictor.predict(test, test_input)