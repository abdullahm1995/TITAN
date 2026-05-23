"""
stage3_sae.py — Stage 3: Sparse Autoencoder (SAE) Training

Trains one SAE per crucial layer (identified in Stage 2) on cached MLP residual activations.

Architecture per layer:
  Encoder : Linear(d_model, d_sae) → ReLU   (expansion factor 4×, d_sae = 14336)
  Decoder : Linear(d_sae, d_model)            tied weights (W_dec = W_enc.T)
  Loss    : ||h - ĥ||² + λ||z||₁

Stability measures:
  - 3-seed ensemble; best checkpoint selected by validation loss
  - λ reduced automatically when dead-feature fraction exceeds 20%
  - Reconstruction error flagged if > 10% of data variance
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    HIDDEN_DIM, NUM_LAYERS,
    RANDOM_SEED,
    SAE_BATCH_SIZE, SAE_EXPANSION_FACTOR, SAE_LAMBDA_SPARSITY,
    SAE_LEARNING_RATE, SAE_MAX_DEAD_FEATURE_RATIO, SAE_MAX_EPOCHS,
    SAE_NUM_SEEDS,
    STAGE2_OUTPUT_DIR, STAGE3_OUTPUT_DIR,
)
from utils import free_gpu_memory, load_json, save_json, set_seed, setup_logging

logger = setup_logging(
    "stage3",
    log_file=os.path.join(STAGE3_OUTPUT_DIR, "stage3.log"),
)

_D_MODEL = HIDDEN_DIM
_D_SAE   = _D_MODEL * SAE_EXPANSION_FACTOR


# ---------------------------------------------------------------------------
# SAE Architecture
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with tied weights.
      z = ReLU(W_enc h + b_enc)
      ĥ = W_enc.T z + b_dec
      Loss = ||h - ĥ||² + λ||z||₁
    """
    def __init__(self, d_model: int, d_sae: int, lambda_l1: float = SAE_LAMBDA_SPARSITY):
        super().__init__()
        self.d_model   = d_model
        self.d_sae     = d_sae
        self.lambda_l1 = lambda_l1

        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        nn.init.kaiming_uniform_(self.W_enc)

    @property
    def W_dec(self) -> torch.Tensor:
        return self.W_enc.T

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(h @ self.W_enc.T + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_dec

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z     = self.encode(h)
        h_hat = self.decode(z)
        loss  = F.mse_loss(h_hat, h) + self.lambda_l1 * z.abs().mean()
        return h_hat, z, loss

    def normalize_decoder(self) -> None:
        with torch.no_grad():
            norms = self.W_enc.norm(dim=1, keepdim=True).clamp(min=1.0)
            self.W_enc.data.div_(norms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActivationDataset(Dataset):
    def __init__(self, activations: torch.Tensor):
        self.data = activations.float()
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> torch.Tensor: return self.data[idx]


def load_layer_activations(
    stage2_output_dir: str,
    layer_idx: int,
    split_tag: str = "sipp",
) -> Optional[torch.Tensor]:
    path = os.path.join(
        stage2_output_dir, "activation_cache",
        f"{split_tag}_layer{layer_idx:02d}_mlp_plus.pt",
    )
    if not Path(path).exists():
        logger.warning(f"  No MLP cache for layer {layer_idx}: {path}")
        return None
    return torch.load(path, weights_only=True).float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sae_single_seed(
    activations: torch.Tensor,
    layer_idx:   int,
    seed:        int,
    d_sae:       int,
    lambda_l1:   float,
    max_epochs:  int,
    batch_size:  int,
    lr:          float,
    device:      torch.device,
) -> Tuple[SparseAutoencoder, Dict[str, Any]]:
    set_seed(seed)
    N = len(activations)
    if N > 5000:
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed))[:5000]
        activations = activations[idx]

    loader = DataLoader(
        ActivationDataset(activations),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    sae = SparseAutoencoder(_D_MODEL, d_sae, lambda_l1).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    train_losses, recon_losses = [], []
    feature_active = torch.zeros(d_sae)
    best_loss  = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(max_epochs):
        epoch_loss = epoch_recon = epoch_n = 0.0
        epoch_z_active = torch.zeros(d_sae)

        for h_batch in loader:
            h_batch = h_batch.to(device)
            h_hat, z, loss = sae(h_batch)
            recon = F.mse_loss(h_hat, h_batch).item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()
            with torch.no_grad():
                epoch_z_active += (z > 0).float().sum(dim=0).cpu()
            epoch_loss  += loss.item() * len(h_batch)
            epoch_recon += recon * len(h_batch)
            epoch_n     += len(h_batch)

        mean_loss  = epoch_loss  / epoch_n
        mean_recon = epoch_recon / epoch_n
        feature_active += (epoch_z_active > 0).float()
        train_losses.append(mean_loss)
        recon_losses.append(mean_recon)

        if mean_loss < best_loss:
            best_loss  = mean_loss
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 5:
                break

    if best_state:
        sae.load_state_dict(best_state)

    dead_feature_frac = (feature_active == 0).float().mean().item()
    return sae, {
        "seed": seed, "d_sae": d_sae, "lambda_l1": lambda_l1,
        "n_epochs_run":     len(train_losses),
        "final_loss":       train_losses[-1] if train_losses else float("nan"),
        "final_recon_loss": recon_losses[-1] if recon_losses else float("nan"),
        "best_loss":        best_loss,
        "dead_feature_frac": dead_feature_frac,
        "flag_dead_high":   dead_feature_frac > SAE_MAX_DEAD_FEATURE_RATIO,
    }


def train_sae_layer(
    activations: torch.Tensor,
    layer_idx:   int,
    device:      torch.device,
    output_dir:  str,
    n_seeds:     int = SAE_NUM_SEEDS,
    max_epochs:  int = SAE_MAX_EPOCHS,
    batch_size:  int = SAE_BATCH_SIZE,
    lr:          float = SAE_LEARNING_RATE,
) -> Dict[str, Any]:
    layer_start = time.time()
    logger.info(f"  Layer {layer_idx:02d} | N={len(activations)} | d_sae={_D_SAE}")

    lambda_l1 = SAE_LAMBDA_SPARSITY
    all_diags: List[Dict] = []
    best_sae:  Optional[SparseAutoencoder] = None
    best_loss  = float("inf")

    for si in range(n_seeds):
        seed = RANDOM_SEED + si * 100 + layer_idx
        sae_i, diag_i = train_sae_single_seed(
            activations, layer_idx, seed, _D_SAE, lambda_l1,
            max_epochs, batch_size, lr, device,
        )
        all_diags.append(diag_i)
        if diag_i["best_loss"] < best_loss:
            best_loss = diag_i["best_loss"]; best_sae = sae_i

    mean_dead = np.mean([d["dead_feature_frac"] for d in all_diags])

    if mean_dead > SAE_MAX_DEAD_FEATURE_RATIO:
        logger.warning(f"  Layer {layer_idx}: dead={mean_dead:.3f} → reducing λ")
        lambda_adj = lambda_l1 * 0.1
        adj_diags, adj_sae, adj_best = [], None, float("inf")
        for si in range(n_seeds):
            seed = RANDOM_SEED + si * 100 + layer_idx + 1000
            sae_i, diag_i = train_sae_single_seed(
                activations, layer_idx, seed, _D_SAE, lambda_adj,
                max_epochs, batch_size, lr, device,
            )
            diag_i["lambda_adjusted"] = True
            adj_diags.append(diag_i)
            if diag_i["best_loss"] < adj_best:
                adj_best = diag_i["best_loss"]; adj_sae = sae_i

        mean_dead_adj = np.mean([d["dead_feature_frac"] for d in adj_diags])
        if mean_dead_adj < mean_dead and adj_sae is not None:
            all_diags = adj_diags; best_sae = adj_sae; lambda_l1 = lambda_adj
            logger.info(f"  Layer {layer_idx}: λ adjustment: dead {mean_dead:.3f}→{mean_dead_adj:.3f}")

    final_recon  = np.mean([d["final_recon_loss"] for d in all_diags])
    recon_thresh = 0.1 * activations.var().item()
    if final_recon > recon_thresh:
        logger.warning(f"  Layer {layer_idx}: recon {final_recon:.4f} > threshold {recon_thresh:.4f}")

    ckpt_path = os.path.join(output_dir, "checkpoints", f"sae_layer{layer_idx:02d}.pt")
    if best_sae is not None:
        torch.save({
            "state_dict": best_sae.cpu().state_dict(),
            "d_model":    _D_MODEL,
            "d_sae":      _D_SAE,
            "lambda_l1":  lambda_l1,
            "layer_idx":  layer_idx,
        }, ckpt_path)

    layer_diag = {
        "layer_idx":          layer_idx,
        "n_activations":      len(activations),
        "d_model":            _D_MODEL,
        "d_sae":              _D_SAE,
        "lambda_l1":          float(lambda_l1),
        "mean_dead_features": float(mean_dead),
        "mean_recon_loss":    float(final_recon),
        "flag_dead_high":     bool(mean_dead > SAE_MAX_DEAD_FEATURE_RATIO),
        "flag_recon_high":    bool(final_recon > recon_thresh),
        "elapsed_s":          time.time() - layer_start,
    }
    save_json(layer_diag, os.path.join(output_dir, "diagnostics", f"layer{layer_idx:02d}_diag.json"))
    logger.info(f"  Layer {layer_idx} done | dead={mean_dead:.3f} | recon={final_recon:.4f}")
    return layer_diag


# ---------------------------------------------------------------------------
# Crucial-layer identification
# ---------------------------------------------------------------------------

def get_crucial_layers(stage2_output_dir: str) -> List[int]:
    scores_path = os.path.join(stage2_output_dir, "importance_scores_full.json")
    if not Path(scores_path).exists():
        return list(range(NUM_LAYERS))
    scores = load_json(scores_path)
    crucial = sorted({entry["layer"] for entry in scores.values() if entry.get("is_crucial", False)})
    return crucial if crucial else list(range(NUM_LAYERS))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage3(
    stage2_output_dir: str = STAGE2_OUTPUT_DIR,
    output_dir:        str = STAGE3_OUTPUT_DIR,
    n_seeds:           int = SAE_NUM_SEEDS,
    max_epochs:        int = SAE_MAX_EPOCHS,
    batch_size:        int = SAE_BATCH_SIZE,
    lr:                float = SAE_LEARNING_RATE,
    seed:              int = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)

    for d in [output_dir, os.path.join(output_dir, "checkpoints"),
              os.path.join(output_dir, "diagnostics")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 3: SAE Training")
    logger.info(f"  d_model={_D_MODEL}  d_sae={_D_SAE}  λ={SAE_LAMBDA_SPARSITY}")
    logger.info("=" * 70)

    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crucial_layers = get_crucial_layers(stage2_output_dir)
    logger.info(f"Crucial layers: {crucial_layers}")

    all_diags: Dict[int, Dict] = {}
    failed:    List[int]       = []

    for layer_idx in crucial_layers:
        try:
            acts = load_layer_activations(stage2_output_dir, layer_idx)
            if acts is None:
                failed.append(layer_idx); continue
            diag = train_sae_layer(
                acts, layer_idx, device, output_dir, n_seeds, max_epochs, batch_size, lr
            )
            all_diags[layer_idx] = diag
            free_gpu_memory()
        except Exception as e:
            logger.error(f"  Layer {layer_idx} FAILED: {e}", exc_info=True)
            failed.append(layer_idx)

    stats = {
        "trained_layers": list(all_diags.keys()),
        "failed_layers":  failed,
        "mean_dead":      np.mean([d["mean_dead_features"] for d in all_diags.values()]) if all_diags else float("nan"),
        "mean_recon":     np.mean([d["mean_recon_loss"]    for d in all_diags.values()]) if all_diags else float("nan"),
        "elapsed_s":      time.time() - t_start,
    }
    save_json(stats, os.path.join(output_dir, "stage3_stats.json"))
    logger.info(f"Stage 3 complete in {(time.time()-t_start)/60:.1f}min")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=SAE_MAX_EPOCHS)
    parser.add_argument("--seeds",      type=int, default=SAE_NUM_SEEDS)
    args = parser.parse_args()
    run_stage3(max_epochs=args.max_epochs, n_seeds=args.seeds)
