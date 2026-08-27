import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple, List
import os
from itertools import islice
from scipy.stats import pearsonr, spearmanr 
import pandas as pd
import multiprocessing as mp
import random
import warnings
import sys

# --- WARNING SUPPRESSION ---
if sys.version_info[:2] >= (3, 11):
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid")
warnings.filterwarnings("ignore", category=FutureWarning)

from ddsrn_agnostic import create_agnostic_model 
from dataloader import create_dynamic_dataloader 

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==============================================================================
# FOCAL LOSS IMPLEMENTATION
# ==============================================================================
class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate standard BCE but DO NOT reduce it to a single number yet
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate the probabilities (equivalent to applying Sigmoid)
        pt = torch.exp(-bce_loss) 
        
        # Apply the Focal Loss formula to penalize easy background pixels
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

class PrecisionCorrelationLoss(nn.Module):
    """
    Balanced Loss for Low MAE, High Correlation, and Multi-Scale FPN Objectness (Focal Loss).
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=10.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.precision_loss = nn.SmoothL1Loss(beta=0.1) 
        
        # CHANGED: Replaced vanilla BCE with Focal Loss
        self.heatmap_loss = FocalLossWithLogits(alpha=0.25, gamma=2.0)

    def forward(self, pred, target, spatial_preds=None, spatial_target=None):
        prec_loss = self.precision_loss(pred, target)
        
        # Prevent NaN crashes. (With drop_last=True and BS=2, this will always run)
        if pred.numel() > 1 and pred.std() > 1e-6 and target.std() > 1e-6:
            pred_centered = pred - pred.mean()
            target_centered = target - target.mean()
            pearson_sim = F.cosine_similarity(pred_centered, target_centered, dim=0, eps=1e-8)
            pearson_loss = 1.0 - pearson_sim
        else:
            pearson_loss = torch.tensor(0.0, device=pred.device)
        
        total_loss = (self.alpha * prec_loss) + (self.beta * pearson_loss)
        hm_loss_val = 0.0
        
        if spatial_preds is not None and spatial_target is not None:
            # Ensure spatial_target has a channel dimension
            if spatial_target.dim() == 3:
                spatial_target = spatial_target.unsqueeze(1)
                
            # If model returns a single tensor, convert to list for iteration
            if not isinstance(spatial_preds, (list, tuple)):
                spatial_preds = [spatial_preds]
                
            total_hm_loss = 0.0
            
            # Loop through FPN multi-scale outputs
            for sp_pred in spatial_preds:
                # Downsample the target to match this specific FPN stride prediction
                sp_target_resized = F.interpolate(
                    spatial_target, 
                    size=(sp_pred.shape[2], sp_pred.shape[3]), 
                    mode='bilinear', 
                    align_corners=False
                )
                total_hm_loss += self.heatmap_loss(sp_pred, sp_target_resized)
                
            # Average the Focal loss across all FPN levels
            hm_loss = total_hm_loss / len(spatial_preds)
            total_loss += (self.gamma * hm_loss)
            hm_loss_val = hm_loss.item()
            
        return total_loss, prec_loss.item(), pearson_loss.item(), hm_loss_val

# ==============================================================================
# 1. OPTIMIZED AGNOSTIC TRAINER
# ==============================================================================
class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 3e-4, 
        checkpoint_dir: Optional[str] = None,
        num_epochs: int = 100,
        try_run: bool = False,
        use_online_wandb=True,
        attempt: str = "0",
        batch_size: int = 2,
    ):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.try_run = try_run
        self.use_online_wandb = use_online_wandb
        self.attempt = attempt
        self.batch_size = batch_size
        self.total_epochs = num_epochs
        self.base_lr = learning_rate

        self.model = create_agnostic_model().to(device)

        # Gamma 10.0 works well, but keep an eye on BCE loss scale compared to MSE
        self.loss = PrecisionCorrelationLoss(alpha=1.0, beta=0.5, gamma=10.0).to(device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=0.05, 
        )
        self.scaler = amp.GradScaler() 

        if self.try_run:
            steps_per_epoch = 50
        else:
            steps_per_epoch = len(self.train_loader)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.base_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
            anneal_strategy='cos'
        )

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_hm_loss = 0.0
        all_preds = []
        all_targets = []
        
        # --- GRADIENT ACCUMULATION SETTINGS ---
        # BS(2) * accum_steps(8) = Effective Batch Size of 16
        accum_steps = 8 
        self.optimizer.zero_grad()

        iterator = self.train_loader
        if self.try_run:
            iterator = islice(iterator, 50)
            num_batches = 50
        else:
            num_batches = len(self.train_loader)
        
        pbar = tqdm(iterator, total=num_batches, desc="Training")

        for batch_idx, batch in enumerate(pbar):
            gt = batch["gt"].to(self.device, non_blocking=True)
            distorted = batch["distorted"].to(self.device, non_blocking=True)
            target_scores = batch["score"].to(self.device, non_blocking=True)
            target_heatmaps = batch["heatmap"].to(self.device, non_blocking=True) 

            with amp.autocast():
                # CHANGED: Request ALL outputs to extract obj_logits correctly
                global_score, deg_map, obj_logits, f_gt, f_mod = self.model(gt, distorted, return_all=True)
                predictions = global_score.view(-1)
                
                if torch.isnan(predictions).any():
                    print(f"\n[CRITICAL] Batch {batch_idx}: Model output contained NaN! Skipping.")
                    continue

                # CHANGED: Pass obj_logits to the loss function instead of the degradation map
                loss, prec_val, pearson_val, hm_val = self.loss(
                    predictions, target_scores.view(-1), obj_logits, target_heatmaps
                )
                
                # Scale the loss by the number of accumulation steps
                scaled_loss = loss / accum_steps
            
            # Backpropagate the scaled loss
            self.scaler.scale(scaled_loss).backward()
            
            # Update weights only after accumulating 'accum_steps' gradients
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                scale_after = self.scaler.get_scale()

                if scale_after < scale_before:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        self.scheduler.step()
                else:
                    self.scheduler.step()

                # Reset gradients for the next accumulation cycle
                self.optimizer.zero_grad()

            loss_val = loss.item()
            if not np.isnan(loss_val):
                running_loss += loss_val
                running_hm_loss += hm_val
            
            all_preds.extend(predictions.detach().float().cpu().numpy())
            all_targets.extend(target_scores.detach().float().cpu().numpy())
            
            # --- CALCULATE RUNNING CORRELATION ---
            if len(all_preds) > 1 and np.std(all_preds) > 1e-6 and np.std(all_targets) > 1e-6:
                run_pearson, _ = pearsonr(all_preds, all_targets)
            else:
                run_pearson = 0.0

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss_val:.3f}",
                "Prec": f"{prec_val:.3f}",
                "CorrL": f"{pearson_val:.3f}",
                "HMap": f"{hm_val:.3f}",
                "Corr": f"{run_pearson:.3f}",
                "lr": f"{current_lr:.5f}"
            })

        if len(all_preds) > 1:
            try:
                train_pearson, _ = pearsonr(all_preds, all_targets)
                train_spearman, _ = spearmanr(all_preds, all_targets)
            except Exception:
                train_pearson, train_spearman = 0.0, 0.0
        else:
            train_pearson, train_spearman = 0.0, 0.0

        return {
            "train_loss": running_loss / num_batches if num_batches > 0 else 0,
            "train_hm_loss": running_hm_loss / num_batches if num_batches > 0 else 0,
            "train_correlation": train_pearson,
            "train_spearman": train_spearman,
        }

    @torch.no_grad()
    def validate(self, current_epoch, val_log_file) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_hm_loss = 0.0
        all_preds = []
        all_targets = []

        iterator = self.val_loader
        if self.try_run:
            iterator = islice(iterator, 15)
            num_batches = 15
        else:
            num_batches = len(self.val_loader)
        
        pbar = tqdm(iterator, total=num_batches, desc="Validating")
        
        for batch in pbar:
            gt = batch["gt"].to(self.device, non_blocking=True)
            distorted = batch["distorted"].to(self.device, non_blocking=True)
            target_scores = batch["score"].to(self.device, non_blocking=True)
            target_heatmaps = batch["heatmap"].to(self.device, non_blocking=True)

            with amp.autocast():
                # CHANGED: Request ALL outputs here as well
                global_score, deg_map, obj_logits, f_gt, f_mod = self.model(gt, distorted, return_all=True)
                predictions = global_score.view(-1)
                
                loss, _, _, hm_val = self.loss(
                    predictions, target_scores.view(-1), obj_logits, target_heatmaps
                )
            
            running_loss += loss.item()
            running_hm_loss += hm_val
            all_preds.extend(predictions.float().cpu().numpy())
            all_targets.extend(target_scores.float().cpu().numpy())

            # --- CALCULATE RUNNING CORRELATION FOR VALIDATION ---
            if len(all_preds) > 1 and np.std(all_preds) > 1e-6 and np.std(all_targets) > 1e-6:
                run_pearson, _ = pearsonr(all_preds, all_targets)
            else:
                run_pearson = 0.0

            # --- UPDATED POSTFIX FOR VALIDATION ---
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "HMap": f"{hm_val:.3f}",
                "Corr": f"{run_pearson:.3f}"
            })

        if len(all_preds) > 1:
            val_pearson, _ = pearsonr(all_preds, all_targets)
            val_spearman, _ = spearmanr(all_preds, all_targets)
            epoch_mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
        else:
            val_pearson, val_spearman, epoch_mae = 0.0, 0.0, 0.0
            
        # Free memory aggressively after validation
        torch.cuda.empty_cache()

        return {
            "val_loss": running_loss / num_batches if num_batches > 0 else 0,
            "val_hm_loss": running_hm_loss / num_batches if num_batches > 0 else 0,
            "val_correlation": val_pearson,
            "val_spearman": val_spearman,
            "val_mae": epoch_mae,
        }

    @torch.no_grad()
    def compute_test_metrics(self) -> Tuple[float, float, float]:
        self.model.eval()
        all_preds = []
        all_targets = []

        iterator = self.test_loader
        if self.try_run:
            iterator = islice(iterator, 10)
            num_batches = 10
        else:
            num_batches = len(self.test_loader)
        
        for batch in tqdm(iterator, total=num_batches, desc="Testing"):
            gt = batch["gt"].to(self.device, non_blocking=True)
            distorted = batch["distorted"].to(self.device, non_blocking=True)
            target_scores = batch["score"].to(self.device, non_blocking=True)

            with amp.autocast():
                global_score, _ = self.model(gt, distorted)
                predictions = global_score.view(-1)
                
            all_preds.extend(predictions.float().cpu().numpy())
            all_targets.extend(target_scores.float().cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        if len(all_preds) > 1:
            pearson_corr, _ = pearsonr(all_preds, all_targets)
            spearman_corr, _ = spearmanr(all_preds, all_targets)
            mae = np.mean(np.abs(all_preds - all_targets))
        else:
            pearson_corr, spearman_corr, mae = 0.0, 0.0, 0.0

        return pearson_corr, spearman_corr, mae

    def save_test_metrics_to_csv(self, pearson_corr: float, spearman_corr: float, mae: float) -> None:
        if not self.checkpoint_dir:
            return
        csv_path = self.checkpoint_dir / "test_metrics.csv"
        metrics_data = {
            'metric': ['pearson_correlation', 'spearman_correlation', 'mae', 'num_samples', 'attempt', 'timestamp'],
            'value': [
                pearson_corr, spearman_corr, mae,
                len(self.test_loader.dataset), 
                self.attempt, 
                "AGNOSTIC",
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(metrics_data).to_csv(csv_path, index=False)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        if not self.checkpoint_dir:
            return
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "scaler": self.scaler.state_dict()
        }
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

    def train(self, num_epochs: int, early_stopping_patience: int = 25) -> None:
        wandb.init(
            project="DDSRN",
            mode="offline" if (self.try_run or not self.use_online_wandb) else "online",
            name=f"Kitti_Visdrone",
            config={
                "learning_rate": self.base_lr,
                "batch_size": self.train_loader.batch_size,
                "model_type": "DDSRN_Spatial_FPN",
                "loss_gamma_heatmap": self.loss.gamma 
            },
        )
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            if hasattr(self.train_loader.dataset, 'set_epoch'):
                self.train_loader.dataset.set_epoch(epoch)

            train_metrics = self.train_epoch()
            val_metrics = self.validate(current_epoch=epoch + 1, val_log_file=None)

            current_lr = self.optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": current_lr, **train_metrics, **val_metrics})

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(HM: {train_metrics['train_hm_loss']:.4f}) | "
                  f"Val Loss: {val_metrics['val_loss']:.4f} "
                  f"(HM: {val_metrics['val_hm_loss']:.4f}) | "
                  f"SRCC: {val_metrics['val_spearman']:.4f} | "
                  f"MAE: {val_metrics['val_mae']:.4f}")

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # --- LOAD AND TEST FINAL MODEL ---
        if self.checkpoint_dir:
            best_model_path = self.checkpoint_dir / "best_model.pt"
            if best_model_path.exists():
                print(f"\n[INFO] Loading best model from {best_model_path} for testing...")
                
                # Fetch checkpoint, assign missing keys directly for safety
                checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
                
                # Catch unexpected FPN layers vs checkpoint mismatches if any
                try:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                except RuntimeError as e:
                    print(f"[WARN] Strict loading failed. Attempting non-strict loading.\n{e}")
                    self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                print("\n[WARN] Best model checkpoint not found. Proceeding with last epoch weights.")

        pearson_corr, spearman_corr, mae = self.compute_test_metrics()
        self.save_test_metrics_to_csv(pearson_corr, spearman_corr, mae)
        
        wandb.log({
            "test_pearson_correlation": pearson_corr, 
            "test_spearman_correlation": spearman_corr,
            "test_mae": mae
        })
        wandb.finish()

# ==============================================================================
# LOADER HELPERS
# ==============================================================================
def create_dataloaders_for_coco2017_splits(
    dataset_root: str, 
    batch_size: int, 
    num_workers: int = 8, 
    **kwargs
):
    dataset_path = Path(dataset_root)
    
    paths = {
        "train": dataset_path / "train/images",
        "val": dataset_path / "validation/images",
        "test": dataset_path / "test/images"
    }
    
    if not paths["train"].exists():
        if (dataset_path / "train").exists():
             paths = {
                 "train": dataset_path / "train",
                 "val": dataset_path / "val",
                 "test": dataset_path / "test"
             }
        else:
             paths = {"train": dataset_path}

    def make_loader(split_name, bs, determ, shuffle, drop_last):
        if split_name not in paths or not paths[split_name].exists():
            print(f"[WARN] Split {split_name} not found at {paths.get(split_name)}")
            return None
            
        return create_dynamic_dataloader(
            dataset_root=str(paths[split_name]), 
            batch_size=bs, 
            deterministic=determ, 
            num_workers=num_workers,
            persistent_workers=(num_workers > 0), 
            drop_last=drop_last, 
            **kwargs
        )
    
    train_loader = make_loader("train", batch_size, False, True, drop_last=True)
    val_loader = make_loader("val", batch_size, True, False, drop_last=False)
    test_loader = make_loader("test", batch_size, True, False, drop_last=False)
    
    return train_loader, val_loader, test_loader

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    mp.set_start_method('spawn', force=True) 
    torch.set_float32_matmul_precision('medium')
    
    GPU_ID = 0
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    DATASET_ROOT = "Kitti_Visdrone_Dataset" 
    
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-4  
    
    ATTEMPT = "Agnostic_Kitti_Visdrone_FPN" 
    DIR = "v1"
    CHECKPOINT_DIR = f"checkpoints/attempt{ATTEMPT}_{DIR}"
    
    TRY_RUN = False
    USE_ONLINE_WANDB = True
    
    NUM_EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 15
    print(f"[INFO] Setting NUM_EPOCHS = {NUM_EPOCHS} with EARLY_STOPPING_PATIENCE = {EARLY_STOPPING_PATIENCE}\n")

    set_global_seed(42)
    
    train_loader, val_loader, test_loader = create_dataloaders_for_coco2017_splits(
        dataset_root=DATASET_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=16, 
        seed=42, 
    )

    if train_loader is None:
        print("Error: Train loader could not be initialized.")
        return

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=CHECKPOINT_DIR,
        num_epochs=NUM_EPOCHS,
        try_run=TRY_RUN,
        use_online_wandb=USE_ONLINE_WANDB,
        batch_size=BATCH_SIZE,
    )

    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE)

if __name__ == "__main__":
    main()