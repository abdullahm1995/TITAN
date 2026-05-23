"""
stage5_mapping.py — Stage 5: Component Mapping

Consolidates Stage 2 (importance scores) and Stage 4 (SAE feature
classifications) into a canonical components.json that drives Stages 6 & 7.

Each entry in components.json:
  component_id, layer_index, is_mlp, head_index, delta_score, role,
  dominant_feature_class, include_in_finetuning

Exclusion rule: components whose SAE features are all NOISE are excluded.
Validates that trainable parameter fraction stays below SFT_MAX_TRAINABLE_PARAM_RATIO.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    FEATURE_CLASSES,
    HEAD_DIM, HIDDEN_DIM, MLP_DIM,
    NUM_HEADS, NUM_LAYERS,
    RANDOM_SEED,
    S2_IMPORTANCE_THRESHOLD,
    SFT_MAX_TRAINABLE_PARAM_RATIO,
    STAGE2_OUTPUT_DIR, STAGE4_OUTPUT_DIR, STAGE5_OUTPUT_DIR,
)
from utils import load_json, save_json, set_seed, setup_logging

logger = setup_logging(
    "stage5",
    log_file=os.path.join(STAGE5_OUTPUT_DIR, "stage5.log"),
)

_QUALIFYING_CLASSES: Set[str] = {
    "SHARED", "SRC_SPECIFIC", "TGT_SPECIFIC", "SCRIPT", "INDICATOR"
}

_PARAMS_PER_MLP:  int = 3 * HIDDEN_DIM * MLP_DIM          # gate + up + down projections
_PARAMS_PER_HEAD: int = 2 * HIDDEN_DIM * HEAD_DIM          # q_proj + o_proj slices
_TOTAL_PARAMS:    int = 9_000_000_000                       # total backbone parameters


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stage2_scores(stage2_output_dir: str) -> Dict[str, Dict]:
    path = os.path.join(stage2_output_dir, "importance_scores_full.json")
    if not Path(path).exists():
        raise FileNotFoundError(f"Stage 2 full scores not found: {path}")
    return load_json(path)


def load_stage4_features(
    stage4_output_dir: str,
    crucial_layers:    List[int],
) -> Dict[int, Dict[str, str]]:
    result: Dict[int, Dict[str, str]] = {}
    for layer_idx in crucial_layers:
        path = os.path.join(stage4_output_dir, f"feature_scores_layer{layer_idx:02d}.json")
        if not Path(path).exists():
            logger.warning(f"  No Stage 4 scores for layer {layer_idx}")
            continue
        scores = load_json(path)
        result[layer_idx] = {fid: v["classification"] for fid, v in scores.items()}
    return result


# ---------------------------------------------------------------------------
# Component building
# ---------------------------------------------------------------------------

def get_dominant_feature_class(feature_classes: Dict[str, str]) -> str:
    counts: Dict[str, int] = {}
    for cls in feature_classes.values():
        if cls != "NOISE":
            counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return "NOISE"
    return max(counts, key=counts.__getitem__)


def build_components(
    stage2_scores:   Dict[str, Dict],
    stage4_features: Dict[int, Dict[str, str]],
    role_labels:     Dict[str, str],
) -> List[Dict[str, Any]]:
    components: List[Dict[str, Any]] = []

    for comp_id, info in stage2_scores.items():
        if not info.get("is_crucial", False):
            continue

        layer_idx = info["layer"]
        is_mlp    = info["is_mlp"]
        delta     = info["delta_score"]

        layer_features = stage4_features.get(layer_idx, {})
        dominant_class = get_dominant_feature_class(layer_features) if layer_features else "SHARED"
        include        = dominant_class in _QUALIFYING_CLASSES
        head_idx       = None if is_mlp else int(comp_id[4:6])

        components.append({
            "component_id":           comp_id,
            "layer_index":            layer_idx,
            "is_mlp":                 is_mlp,
            "head_index":             head_idx,
            "delta_score":            delta,
            "attention_type":         info.get("attention_type"),
            "role":                   role_labels.get(comp_id, info.get("role", "unclassified")),
            "dominant_feature_class": dominant_class,
            "include_in_finetuning":  include,
        })

    components.sort(key=lambda x: abs(x["delta_score"]), reverse=True)
    return components


def estimate_trainable_params(components: List[Dict[str, Any]]) -> Tuple[int, float]:
    trainable = 0
    for comp in components:
        if not comp["include_in_finetuning"]:
            continue
        trainable += _PARAMS_PER_MLP if comp["is_mlp"] else _PARAMS_PER_HEAD
    return trainable, trainable / _TOTAL_PARAMS


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage5(
    stage2_output_dir: str = STAGE2_OUTPUT_DIR,
    stage4_output_dir: str = STAGE4_OUTPUT_DIR,
    output_dir:        str = STAGE5_OUTPUT_DIR,
    seed:              int = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 5: Component Mapping")
    logger.info("=" * 70)

    stage2_scores = load_stage2_scores(stage2_output_dir)
    logger.info(f"Stage 2 scores loaded: {len(stage2_scores)} components")

    crucial_layers = sorted({
        v["layer"] for v in stage2_scores.values() if v.get("is_crucial")
    })
    logger.info(f"Crucial layers: {crucial_layers}")

    stage4_features = load_stage4_features(stage4_output_dir, crucial_layers)
    logger.info(f"Stage 4 features loaded for {len(stage4_features)} layers")

    role_labels_path = os.path.join(stage2_output_dir, "component_roles.json")
    role_labels = load_json(role_labels_path) if Path(role_labels_path).exists() else {}

    components = build_components(stage2_scores, stage4_features, role_labels)
    included   = [c for c in components if c["include_in_finetuning"]]

    logger.info(f"Total crucial components    : {len(components)}")
    logger.info(f"Included for fine-tuning   : {len(included)}")

    trainable, ratio = estimate_trainable_params(components)
    logger.info(f"Trainable parameters: {trainable:,} = {100*ratio:.2f}% of total")

    if ratio > SFT_MAX_TRAINABLE_PARAM_RATIO:
        logger.warning(
            f"Trainable ratio {100*ratio:.2f}% exceeds {SFT_MAX_TRAINABLE_PARAM_RATIO*100:.0f}% limit"
        )

    components_path = os.path.join(output_dir, "components.json")
    save_json(components, components_path)
    logger.info(f"Saved: {components_path}")

    dominant_classes = {}
    for c in included:
        dc = c["dominant_feature_class"]
        dominant_classes[dc] = dominant_classes.get(dc, 0) + 1

    stats = {
        "total_crucial":               len(components),
        "included_components":         len(included),
        "excluded_components":         len(components) - len(included),
        "trainable_params":            trainable,
        "trainable_ratio":             ratio,
        "dominant_class_distribution": dominant_classes,
        "role_distribution": {
            r: sum(1 for c in included if c["role"] == r)
            for r in {"source", "indicator", "positional", "unclassified", "mlp"}
        },
        "elapsed_s": time.time() - t_start,
    }
    save_json(stats, os.path.join(output_dir, "stage5_stats.json"))

    logger.info("Top included components:")
    for comp in included[:10]:
        logger.info(
            f"  {comp['component_id']:10s} | δ={comp['delta_score']:+.4f} | "
            f"role={comp['role']:12s} | class={comp['dominant_feature_class']}"
        )

    logger.info(f"Stage 5 complete in {(time.time()-t_start):.1f}s")
    return stats


if __name__ == "__main__":
    run_stage5()
