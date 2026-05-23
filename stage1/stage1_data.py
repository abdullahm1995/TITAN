"""
stage1_data.py — Stage 1: Contrastive Dataset Construction

Reads sft_train.csv, samples n_pairs, builds contrastive prompts for both
translation directions (X+ = valid prompt, X- = counterfactual), and saves
analysis_contrastive.jsonl consumed by Stages 2–4.

Output:
  stage1/outputs/analysis_contrastive.jsonl
  stage1/outputs/stage1_stats.json
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    COUNTERFACTUAL_TEMPLATES,
    DIR_FWD, DIR_BWD,
    LANG_PAIR,
    RANDOM_SEED,
    SFT_TRAIN_PATH,
    STAGE1_OUTPUT_DIR,
    TEMPLATE_FWD, TEMPLATE_BWD,
)
from utils import (
    classify_word_type, count_shared_vocab,
    normalise_text, save_json, save_jsonl, set_seed, setup_logging,
)

logger = setup_logging(
    "stage1",
    log_file=os.path.join(STAGE1_OUTPUT_DIR, "stage1.log"),
)

_SRC_LANG = LANG_PAIR["src_lang"]
_TGT_LANG = LANG_PAIR["tgt_lang"]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_pairs(csv_path: str, n: int = None) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path, dtype=str).dropna(subset=["src", "tgt"])
    src = [normalise_text(t) for t in df["src"].tolist()]
    tgt = [normalise_text(t) for t in df["tgt"].tolist()]
    valid = [(s, t) for s, t in zip(src, tgt) if s and t]
    if n is not None:
        valid = valid[:n]
    src_out, tgt_out = zip(*valid) if valid else ([], [])
    return list(src_out), list(tgt_out)


# ---------------------------------------------------------------------------
# Contrastive pair construction
# ---------------------------------------------------------------------------

def build_contrastive_sample(
    source: str,
    target: str,
    direction: str,
    pair_idx: int,
    rng: random.Random,
) -> Dict[str, Any]:
    if direction == DIR_FWD:
        positive_prompt = TEMPLATE_FWD.format(source=source)
        lang = _SRC_LANG
    else:
        positive_prompt = TEMPLATE_BWD.format(source=source)
        lang = _TGT_LANG

    template_key = rng.choice(list(COUNTERFACTUAL_TEMPLATES.keys()))
    counterfactual_text = COUNTERFACTUAL_TEMPLATES[template_key].format(
        lang=lang, source=source,
    )

    source_words  = source.split()
    word_types    = [classify_word_type(w) for w in source_words]
    vocab_overlap = count_shared_vocab(source_words, target.split())

    return {
        "pair_idx":            pair_idx,
        "direction_label":     direction,
        "source_text":         source,
        "target_text":         target,
        "positive_prompt":     positive_prompt,
        "counterfactual_text": counterfactual_text,
        "counterfactual_type": template_key,
        "word_count":          len(source_words),
        "word_type_counts":    dict(Counter(word_types)),
        "vocab_overlap":       vocab_overlap,
    }


def build_contrastive_dataset(
    src_lines: List[str],
    tgt_lines: List[str],
    seed: int = RANDOM_SEED,
) -> List[Dict[str, Any]]:
    """Each pair produces 2 samples (forward + backward direction)."""
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
        samples.append(build_contrastive_sample(src, tgt, DIR_FWD, idx * 2,     rng))
        samples.append(build_contrastive_sample(tgt, src, DIR_BWD, idx * 2 + 1, rng))

    rng.shuffle(samples)
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_stage1(
    sft_csv_path: str = SFT_TRAIN_PATH,
    output_dir:   str = STAGE1_OUTPUT_DIR,
    seed:         int = RANDOM_SEED,
    n_pairs:      int = 5_000,   # × 2 directions = 10k contrastive samples
) -> Dict[str, Any]:
    set_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 1: Contrastive Data Construction")
    logger.info(f"  Source : {sft_csv_path}")
    logger.info(f"  Pairs  : {n_pairs} → {n_pairs * 2} contrastive samples")
    logger.info("=" * 70)

    src_all, tgt_all = load_csv_pairs(sft_csv_path)
    rng = random.Random(seed)
    indices = list(range(len(src_all)))
    rng.shuffle(indices)
    indices = indices[:n_pairs]
    src_sipp = [src_all[i] for i in indices]
    tgt_sipp = [tgt_all[i] for i in indices]

    logger.info(f"Sampled {len(src_sipp)} pairs from {len(src_all)} available")

    samples = build_contrastive_dataset(src_sipp, tgt_sipp, seed=seed)
    logger.info(f"Built {len(samples)} contrastive samples")

    out_path = os.path.join(output_dir, "analysis_contrastive.jsonl")
    save_jsonl(samples, out_path)
    logger.info(f"Saved → {out_path}")

    dir_counts = Counter(s["direction_label"] for s in samples)
    cf_counts  = Counter(s["counterfactual_type"] for s in samples)

    stats = {
        "n_source_pairs":       len(src_sipp),
        "n_contrastive":        len(samples),
        "direction_counts":     dict(dir_counts),
        "counterfactual_types": dict(cf_counts),
        "src_mean_words":       sum(len(s.split()) for s in src_sipp) / len(src_sipp),
        "tgt_mean_words":       sum(len(t.split()) for t in tgt_sipp) / len(tgt_sipp),
        "seed":                 seed,
    }
    save_json(stats, os.path.join(output_dir, "stage1_stats.json"))

    logger.info(f"Direction balance    : {dict(dir_counts)}")
    logger.info(f"Counterfactual types : {dict(cf_counts)}")
    logger.info(f"Sample positive      : {samples[0]['positive_prompt'][:80]}")
    logger.info(f"Sample counter       : {samples[0]['counterfactual_text'][:80]}")
    logger.info("Stage 1 complete.")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-csv", default=SFT_TRAIN_PATH)
    parser.add_argument("--n-pairs", type=int, default=5_000)
    parser.add_argument("--seed",    type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run_stage1(sft_csv_path=args.sft_csv, n_pairs=args.n_pairs, seed=args.seed)
