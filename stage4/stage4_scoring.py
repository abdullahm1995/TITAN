"""
stage4_scoring.py — Stage 4: SAE Feature Scoring

Scores each SAE feature in each crucial layer (identified in Stage 2) on:
  lift        — mean activation ratio (translation text vs permuted baseline)
  selectivity — mean(source activations) − mean(target activations)
  script_sel  — selectivity on target-script-character samples

Feature classifications (exactly one per feature):
  SHARED       — activates for both source and target language
  SRC_SPECIFIC — activates primarily for source language input
  TGT_SPECIFIC — activates primarily for target language input
  SCRIPT       — selective on target-script characters (set TGT_SCRIPT_CHARS=[])
                 to disable if source/target share a script
  INDICATOR    — strong language-direction signal
  NOISE        — low lift + low selectivity (excluded from fine-tuning)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DIR_FWD, DIR_BWD,
    FEATURE_CLASSES,
    HIDDEN_DIM,
    INDICATOR_HIGH, LIFT_HIGH,
    RANDOM_SEED,
    SCRIPT_HIGH, SELECTIVITY_HIGH, SELECTIVITY_MIN_SAMPLES,
    TGT_SCRIPT_CHARS,
    STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR, STAGE3_OUTPUT_DIR, STAGE4_OUTPUT_DIR,
)
from stage3.stage3_sae import SparseAutoencoder
from utils import free_gpu_memory, load_json, load_jsonl, save_json, set_seed, setup_logging

logger = setup_logging(
    "stage4",
    log_file=os.path.join(STAGE4_OUTPUT_DIR, "stage4.log"),
)

_D_MODEL     = HIDDEN_DIM
_SCRIPT_CHARS = frozenset(TGT_SCRIPT_CHARS)


# ---------------------------------------------------------------------------
# SAE loader
# ---------------------------------------------------------------------------

def load_sae(
    layer_idx:         int,
    stage3_output_dir: str,
    device:            torch.device,
) -> Optional[SparseAutoencoder]:
    ckpt_path = os.path.join(
        stage3_output_dir, "checkpoints", f"sae_layer{layer_idx:02d}.pt"
    )
    if not Path(ckpt_path).exists():
        logger.warning(f"SAE checkpoint not found: {ckpt_path}")
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae  = SparseAutoencoder(ckpt["d_model"], ckpt["d_sae"], ckpt["lambda_l1"])
    sae.load_state_dict(ckpt["state_dict"])
    return sae.to(device).eval()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_features(
    sae:         SparseAutoencoder,
    activations: torch.Tensor,
    device:      torch.device,
    batch_size:  int = 256,
) -> torch.Tensor:
    all_z = []
    sae.eval()
    with torch.no_grad():
        for i in range(0, len(activations), batch_size):
            z = sae.encode(activations[i:i+batch_size].to(device))
            all_z.append(z.cpu())
    return torch.cat(all_z, dim=0)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def compute_lift_score(
    z_translation: torch.Tensor,
    z_random:      torch.Tensor,
) -> torch.Tensor:
    """Lift = mean activation on translation text / (mean on random baseline + ε)."""
    return z_translation.mean(dim=0) / (z_random.mean(dim=0) + 1e-8)


def compute_selectivity(
    z_src: torch.Tensor,
    z_tgt: torch.Tensor,
) -> torch.Tensor:
    """Selectivity = mean_src − mean_tgt. Positive → source-specific."""
    return z_src.mean(dim=0) - z_tgt.mean(dim=0)


def compute_script_selectivity(
    z_script:    torch.Tensor,
    z_no_script: torch.Tensor,
) -> torch.Tensor:
    """Script selectivity = mean activation on target-script samples − non-script."""
    if len(z_script) == 0 or len(z_no_script) == 0:
        return torch.zeros(z_script.shape[1] if z_script.ndim > 1 else 1)
    return z_script.mean(dim=0) - z_no_script.mean(dim=0)


def classify_feature(
    lift:       float,
    sel:        float,
    script_sel: float,
) -> str:
    if lift < LIFT_HIGH:
        return "NOISE"
    if abs(sel) >= INDICATOR_HIGH:
        return "INDICATOR"
    if _SCRIPT_CHARS and script_sel >= SCRIPT_HIGH:
        return "SCRIPT"
    if sel >= SELECTIVITY_HIGH:
        return "SRC_SPECIFIC"
    if sel <= -SELECTIVITY_HIGH:
        return "TGT_SPECIFIC"
    return "SHARED"


# ---------------------------------------------------------------------------
# Random activation baseline
# ---------------------------------------------------------------------------

def build_random_activations(
    activations: torch.Tensor,
    n_random:    int = 200,
    seed:        int = RANDOM_SEED,
) -> torch.Tensor:
    g   = torch.Generator(); g.manual_seed(seed)
    idx = torch.randperm(len(activations), generator=g)[:n_random]
    shuffled = activations[idx].clone()
    for i in range(len(shuffled)):
        perm = torch.randperm(shuffled.shape[1], generator=g)
        shuffled[i] = shuffled[i][perm]
    return shuffled


# ---------------------------------------------------------------------------
# Per-layer scoring
# ---------------------------------------------------------------------------

def score_layer(
    layer_idx:         int,
    sae:               SparseAutoencoder,
    samples_fwd:       List[Dict],
    samples_bwd:       List[Dict],
    stage2_output_dir: str,
    device:            torch.device,
    split_tag:         str = "sipp",
) -> Dict[str, Any]:
    cache_dir  = os.path.join(stage2_output_dir, "activation_cache")
    mlp_p_path = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_mlp_plus.pt")
    if not Path(mlp_p_path).exists():
        logger.warning(f"  Layer {layer_idx}: no activation cache — skipping")
        return {}

    acts_all = torch.load(mlp_p_path, weights_only=True).float()
    n_total  = len(acts_all)
    n_half   = n_total // 2

    n_each = min(len(samples_fwd), len(samples_bwd), n_half, 1000)
    if n_each < SELECTIVITY_MIN_SAMPLES:
        logger.warning(f"  Layer {layer_idx}: only {n_each} samples per direction — low confidence")

    acts_src = acts_all[:n_each]
    acts_tgt = acts_all[n_half : n_half + n_each]

    # Script-character samples from target-language direction
    script_mask    = []
    for s in samples_bwd[:n_each]:
        src_text = s.get("source_text", "")
        script_mask.append(bool(_SCRIPT_CHARS and any(c in _SCRIPT_CHARS for c in src_text)))

    script_idx    = [i for i, m in enumerate(script_mask) if m]
    no_script_idx = [i for i, m in enumerate(script_mask) if not m]

    acts_script    = acts_tgt[script_idx]    if script_idx    else torch.zeros(0, acts_tgt.shape[1])
    acts_no_script = acts_tgt[no_script_idx] if no_script_idx else acts_tgt

    acts_random = build_random_activations(acts_all, n_random=min(200, n_total))

    z_all    = get_features(sae, acts_all,        device)
    z_src    = get_features(sae, acts_src,         device)
    z_tgt    = get_features(sae, acts_tgt,         device)
    z_script = get_features(sae, acts_script,      device) if len(acts_script) > 0 else z_tgt[:0]
    z_nosc   = get_features(sae, acts_no_script,   device)
    z_random = get_features(sae, acts_random,      device)

    lift       = compute_lift_score(z_all, z_random)
    sel        = compute_selectivity(z_src, z_tgt)
    script_sel = compute_script_selectivity(z_script, z_nosc)

    feature_scores: Dict[str, Any] = {}
    d_sae = sae.d_sae

    for feat_id in range(d_sae):
        l  = float(lift[feat_id])
        s  = float(sel[feat_id])
        sc = float(script_sel[feat_id]) if feat_id < len(script_sel) else 0.0
        feature_scores[str(feat_id)] = {
            "lift":              l,
            "selectivity":       s,
            "script_selectivity": sc,
            "classification":    classify_feature(l, s, sc),
        }

    class_counts = {cls: 0 for cls in FEATURE_CLASSES}
    for v in feature_scores.values():
        c = v["classification"]
        if c in class_counts:
            class_counts[c] += 1

    logger.info(
        f"  Layer {layer_idx}: {class_counts} | "
        f"lift mean={float(lift.mean()):.4f} | sel std={float(sel.std()):.4f}"
    )
    return {
        "features":      feature_scores,
        "class_counts":  class_counts,
        "n_src_samples": n_each,
        "n_tgt_samples": n_each,
        "n_script":      len(script_idx),
        "lift_mean":     float(lift.mean()),
        "lift_max":      float(lift.max()),
        "sel_std":       float(sel.std()),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage4(
    stage1_output_dir: str = STAGE1_OUTPUT_DIR,
    stage2_output_dir: str = STAGE2_OUTPUT_DIR,
    stage3_output_dir: str = STAGE3_OUTPUT_DIR,
    output_dir:        str = STAGE4_OUTPUT_DIR,
    seed:              int = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 4: SAE Feature Scoring")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_samples  = load_jsonl(os.path.join(stage1_output_dir, "analysis_contrastive.jsonl"))
    samples_fwd  = [s for s in all_samples if s["direction_label"] == DIR_FWD]
    samples_bwd  = [s for s in all_samples if s["direction_label"] == DIR_BWD]
    logger.info(f"Samples — fwd:{len(samples_fwd)} bwd:{len(samples_bwd)}")

    scores_path = os.path.join(stage2_output_dir, "importance_scores_full.json")
    if Path(scores_path).exists():
        stage2_scores  = load_json(scores_path)
        crucial_layers = sorted({v["layer"] for v in stage2_scores.values() if v.get("is_crucial")})
    else:
        logger.warning("No Stage 2 scores — running on all layers")
        from config import NUM_LAYERS
        crucial_layers = list(range(NUM_LAYERS))

    logger.info(f"Scoring SAE features for layers: {crucial_layers}")

    all_layer_results: Dict[int, Dict] = {}

    for layer_idx in crucial_layers:
        logger.info(f"Layer {layer_idx:02d} ...")
        sae = load_sae(layer_idx, stage3_output_dir, device)
        if sae is None:
            logger.warning(f"  No SAE for layer {layer_idx} — skipping")
            continue

        result = score_layer(
            layer_idx, sae, samples_fwd, samples_bwd, stage2_output_dir, device,
        )
        if result:
            all_layer_results[layer_idx] = result
            save_json(
                result["features"],
                os.path.join(output_dir, f"feature_scores_layer{layer_idx:02d}.json"),
            )
        del sae
        free_gpu_memory()

    class_totals = {cls: 0 for cls in FEATURE_CLASSES}
    for res in all_layer_results.values():
        for cls, cnt in res.get("class_counts", {}).items():
            if cls in class_totals:
                class_totals[cls] += cnt

    stats = {
        "crucial_layers": crucial_layers,
        "layers_scored":  list(all_layer_results.keys()),
        "class_totals":   class_totals,
        "elapsed_s":      time.time() - t_start,
        "thresholds": {
            "selectivity_high": SELECTIVITY_HIGH,
            "lift_high":        LIFT_HIGH,
            "indicator_high":   INDICATOR_HIGH,
            "script_high":      SCRIPT_HIGH,
        },
    }
    save_json(stats, os.path.join(output_dir, "stage4_stats.json"))

    logger.info(f"Stage 4 complete in {(time.time()-t_start)/60:.1f}min")
    logger.info(f"Class totals: {class_totals}")
    return stats


if __name__ == "__main__":
    run_stage4()
