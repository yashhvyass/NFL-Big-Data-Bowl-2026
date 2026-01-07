"""
NFL Big Data Bowl 2026 - Training Module
========================================
Training loops, loss functions, and validation metrics.

Supports both:
- Joint model (STTransformer) predicting (dx, dy) together
- Dual model (ImprovedSeqModel x2) predicting dx and dy separately
"""

import torch
import torch.nn as nn
import numpy as np
from config import Config
from src.models.transformer import STTransformer
from src.models.dual_transformer import ImprovedSeqModel


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class TemporalHuber(nn.Module):
    """
    Temporal Huber Loss with exponential time decay for joint (dx, dy) prediction.
    
    Combines Huber loss (robust to outliers) with temporal weighting
    that emphasizes accuracy on earlier predictions.
    """
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        """
        Compute temporal Huber loss.
        
        Args:
            pred: Predicted trajectories (batch, horizon, 2)
            target: Target trajectories (batch, horizon, 2)
            mask: Validity mask (batch, horizon)
        """
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1) * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)


class TemporalHuber1D(nn.Module):
    """
    Temporal Huber Loss for single coordinate prediction (matching notebook).
    """
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay
    
    def forward(self, pred, target, mask):
        """
        Compute temporal Huber loss for single coordinate.
        
        Args:
            pred: Predicted values (batch, horizon)
            target: Target values (batch, horizon)
            mask: Validity mask (batch, horizon)
        """
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(abs_err <= self.delta, 0.5 * err * err, 
                           self.delta * (abs_err - 0.5 * self.delta))
        
        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L)
            huber = huber * weight
            mask = mask * weight
        
        return (huber * mask).sum() / (mask.sum() + 1e-8)


# ============================================================================
# TARGET PREPARATION
# ============================================================================

def prepare_targets(batch_dx, batch_dy, max_h):
    """
    Prepare targets with padding for variable-length sequences (joint model).
    
    Returns:
        Tuple of (targets, masks) tensors
        - targets: (batch, max_h, 2) with x and y deltas
        - masks: (batch, max_h) with 1 for valid frames, 0 for padding
    """
    tensors_x, tensors_y, masks = [], [], []
    
    for dx, dy in zip(batch_dx, batch_dy):
        L = len(dx)
        padded_x = np.pad(dx, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_y = np.pad(dy, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        
        tensors_x.append(torch.tensor(padded_x))
        tensors_y.append(torch.tensor(padded_y))
        masks.append(torch.tensor(mask))
    
    targets = torch.stack([torch.stack(tensors_x), torch.stack(tensors_y)], dim=-1)
    return targets, torch.stack(masks)


def prepare_targets_1d(batch_d, max_h):
    """
    Prepare targets for single coordinate (dual model).
    
    Returns:
        Tuple of (targets, masks) tensors
        - targets: (batch, max_h) 
        - masks: (batch, max_h)
    """
    tensors, masks = [], []
    
    for d in batch_d:
        L = len(d)
        padded = np.pad(d, (0, max_h - L), constant_values=0).astype(np.float32)
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        
        tensors.append(torch.tensor(padded))
        masks.append(torch.tensor(mask))
    
    return torch.stack(tensors), torch.stack(masks)


# ============================================================================
# VALIDATION METRICS
# ============================================================================

def compute_val_rmse(model, X_val, y_val_dx, y_val_dy, horizon, device, batch_size=256):
    """
    Compute validation RMSE for joint model (predicts (dx, dy) together).
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32)).to(device)
            
            pred = model(bx).cpu().numpy()  # Shape: (batch, horizon, 2)
            
            for j, idx in enumerate(range(i, end)):
                dx_true = y_val_dx[idx]
                dy_true = y_val_dy[idx]
                n_steps = len(dx_true)
                
                dx_pred = pred[j, :n_steps, 0]
                dy_pred = pred[j, :n_steps, 1]
                
                sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
                all_errors.extend(sq_errors)
    
    return np.sqrt(np.mean(all_errors))


def compute_val_rmse_dual(model_x, model_y, X_val, y_val_dx, y_val_dy, horizon, device, batch_size=256):
    """
    Compute validation RMSE for dual model (separate X and Y models).
    
    This matches the notebook's per-dimension RMSE calculation.
    """
    model_x.eval()
    model_y.eval()
    
    X_t = torch.tensor(np.stack(X_val).astype(np.float32)).to(device)
    
    with torch.no_grad():
        pdx = model_x(X_t).cpu().numpy()  # (N, H)
        pdy = model_y(X_t).cpu().numpy()  # (N, H)
    
    # Prepare targets
    ydx, m = prepare_targets_1d(y_val_dx, horizon)
    ydy, _ = prepare_targets_1d(y_val_dy, horizon)
    ydx, ydy, m = ydx.numpy(), ydy.numpy(), m.numpy()
    
    # Compute squared errors
    se_sum2d = ((pdx - ydx)**2 + (pdy - ydy)**2) * m
    denom = m.sum() + 1e-8
    
    # Per-dimension RMSE (matching notebook's formula)
    return float(np.sqrt(se_sum2d.sum() / (2.0 * denom)))


# ============================================================================
# TRAINING LOOPS
# ============================================================================

def train_model(X_train, y_train_dx, y_train_dy, X_val, y_val_dx, y_val_dy, 
                input_dim, horizon, config):
    """
    Train joint model (predicts dx, dy together) with early stopping.
    
    This is the original training function for STTransformer.
    """
    device = config.DEVICE
    
    model = STTransformer(
        input_dim=input_dim,
        hidden_dim=config.HIDDEN_DIM,
        horizon=horizon,
        window_size=config.WINDOW_SIZE,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS
    ).to(device)
    
    criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=False)
    
    # Prepare batches
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            horizon
        )
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by, bm))
    
    best_loss, best_state, bad = float('inf'), None, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_losses = []
        for bx, by, bm in train_batches:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            pred = model(bx)
            loss = criterion(pred, by, bm)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by, bm in val_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                pred = model(bx)
                val_losses.append(criterion(pred, by, bm).item())
        
        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_loss


def train_dual_models(X_train, y_train_dx, y_train_dy, X_val, y_val_dx, y_val_dy,
                      input_dim, horizon, config):
    """
    Train separate X and Y models (matching notebook architecture).
    
    This trains two ImprovedSeqModel instances, one for dx and one for dy.
    
    Returns:
        Tuple of (model_x, model_y, best_loss)
    """
    device = config.DEVICE
    
    # Create separate models
    model_x = ImprovedSeqModel(input_dim, horizon).to(device)
    model_y = ImprovedSeqModel(input_dim, horizon).to(device)
    
    # Separate criteria for each coordinate
    criterion = TemporalHuber1D(delta=0.5, time_decay=0.03)
    
    # Separate optimizers
    optimizer_x = torch.optim.AdamW(model_x.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    optimizer_y = torch.optim.AdamW(model_y.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler_x = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_x, patience=5, factor=0.5, verbose=False)
    scheduler_y = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_y, patience=5, factor=0.5, verbose=False)
    
    # Prepare batches
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        tdx, tm = prepare_targets_1d([y_train_dx[j] for j in range(i, end)], horizon)
        tdy, _ = prepare_targets_1d([y_train_dy[j] for j in range(i, end)], horizon)
        train_batches.append((bx, tdx, tdy, tm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        tdx, tm = prepare_targets_1d([y_val_dx[j] for j in range(i, end)], horizon)
        tdy, _ = prepare_targets_1d([y_val_dy[j] for j in range(i, end)], horizon)
        val_batches.append((bx, tdx, tdy, tm))
    
    best_loss_x, best_loss_y = float('inf'), float('inf')
    best_state_x, best_state_y = None, None
    bad_x, bad_y = 0, 0
    
    for epoch in range(1, config.EPOCHS + 1):
        # Training
        model_x.train()
        model_y.train()
        train_losses_x, train_losses_y = [], []
        
        for bx, tdx, tdy, tm in train_batches:
            bx = bx.to(device)
            tdx, tdy, tm = tdx.to(device), tdy.to(device), tm.to(device)
            
            # Train X model
            pred_x = model_x(bx)
            loss_x = criterion(pred_x, tdx, tm)
            optimizer_x.zero_grad()
            loss_x.backward()
            nn.utils.clip_grad_norm_(model_x.parameters(), 1.0)
            optimizer_x.step()
            train_losses_x.append(loss_x.item())
            
            # Train Y model
            pred_y = model_y(bx)
            loss_y = criterion(pred_y, tdy, tm)
            optimizer_y.zero_grad()
            loss_y.backward()
            nn.utils.clip_grad_norm_(model_y.parameters(), 1.0)
            optimizer_y.step()
            train_losses_y.append(loss_y.item())
        
        # Validation
        model_x.eval()
        model_y.eval()
        val_losses_x, val_losses_y = [], []
        
        with torch.no_grad():
            for bx, tdx, tdy, tm in val_batches:
                bx = bx.to(device)
                tdx, tdy, tm = tdx.to(device), tdy.to(device), tm.to(device)
                
                pred_x = model_x(bx)
                pred_y = model_y(bx)
                val_losses_x.append(criterion(pred_x, tdx, tm).item())
                val_losses_y.append(criterion(pred_y, tdy, tm).item())
        
        train_loss_x, val_loss_x = np.mean(train_losses_x), np.mean(val_losses_x)
        train_loss_y, val_loss_y = np.mean(train_losses_y), np.mean(val_losses_y)
        
        scheduler_x.step(val_loss_x)
        scheduler_y.step(val_loss_y)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: X(train={train_loss_x:.4f}, val={val_loss_x:.4f}) "
                  f"Y(train={train_loss_y:.4f}, val={val_loss_y:.4f})")
        
        # Early stopping for X
        if val_loss_x < best_loss_x:
            best_loss_x = val_loss_x
            best_state_x = {k: v.cpu().clone() for k, v in model_x.state_dict().items()}
            bad_x = 0
        else:
            bad_x += 1
        
        # Early stopping for Y
        if val_loss_y < best_loss_y:
            best_loss_y = val_loss_y
            best_state_y = {k: v.cpu().clone() for k, v in model_y.state_dict().items()}
            bad_y = 0
        else:
            bad_y += 1
        
        # Stop if both converged
        if bad_x >= config.PATIENCE and bad_y >= config.PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break
    
    # Load best states
    if best_state_x:
        model_x.load_state_dict(best_state_x)
    if best_state_y:
        model_y.load_state_dict(best_state_y)
    
    combined_loss = (best_loss_x + best_loss_y) / 2
    return model_x, model_y, combined_loss