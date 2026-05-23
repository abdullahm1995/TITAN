"""
data_prep.py — Dataset preparation for the pipeline.

Supports two data sources:
  1. HuggingFace datasets (streaming, e.g. WMT14)
  2. Local CSV file with configurable column names

Output (always written to DATA_DIR):
  val.csv       — validation split   (standardised src/tgt columns)
  sft_train.csv — SFT training split
  dpo_train.csv — DPO training split

Usage:
  # From HuggingFace (configured in LANG_PAIR):
  python data_prep.py

  # From a local CSV:
  python data_prep.py --from-csv /path/to/data.csv --src-col source_lang --tgt-col target_lang

  # Custom sizes:
  python data_prep.py --n-val 1000 --n-sft 20000 --n-dpo 15000 --seed 42
"""

import argparse
import random
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import DATA_DIR, LANG_PAIR, N_DPO, N_SFT, N_VAL, RANDOM_SEED

_MIN_WORDS = 3
_MAX_WORDS = 80


# ---------------------------------------------------------------------------
# Text validation
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def is_valid(src: str, tgt: str, tgt_script_chars: Optional[set] = None) -> bool:
    if not src or not tgt:
        return False
    sw = len(src.split())
    tw = len(tgt.split())
    if sw < _MIN_WORDS or tw < _MIN_WORDS:
        return False
    if sw > _MAX_WORDS or tw > _MAX_WORDS:
        return False
    if src.lower() == tgt.lower():
        return False
    if tgt_script_chars and not any(c in tgt_script_chars for c in tgt):
        return False
    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_hf(
    n_total: int,
    seed: int,
    hf_dataset: str,
    hf_config: str,
    hf_src_key: str,
    hf_tgt_key: str,
    tgt_script_chars: Optional[set] = None,
) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError:
        print("[data_prep] ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"[data_prep] Streaming {hf_dataset}/{hf_config} ...")
    ds = load_dataset(hf_dataset, hf_config, split="train", streaming=True)

    rng = random.Random(seed)
    buf_src: List[str] = []
    buf_tgt: List[str] = []
    buffer_limit = n_total * 4

    for i, item in enumerate(ds):
        pair = item.get("translation", item)
        src = normalise(pair.get(hf_src_key, ""))
        tgt = normalise(pair.get(hf_tgt_key, ""))
        if is_valid(src, tgt, tgt_script_chars):
            buf_src.append(src)
            buf_tgt.append(tgt)
            if len(buf_src) >= buffer_limit:
                break
        if (i + 1) % 500_000 == 0:
            print(f"  scanned {(i+1)//1_000_000:.1f}M rows | buffered {len(buf_src):,}", flush=True)

    print(f"[data_prep] Buffered {len(buf_src):,} clean pairs — sampling {n_total:,} ...")
    indices = list(range(len(buf_src)))
    rng.shuffle(indices)
    indices = indices[:n_total]

    df = pd.DataFrame({
        "src": [buf_src[i] for i in indices],
        "tgt": [buf_tgt[i] for i in indices],
    })
    if len(df) < n_total:
        print(f"[data_prep] WARNING: only {len(df):,} pairs available (target {n_total:,}).")
    return df


def load_from_csv(
    csv_path: str,
    src_col: str,
    tgt_col: str,
    n_total: int,
    seed: int,
    tgt_script_chars: Optional[set] = None,
) -> pd.DataFrame:
    print(f"[data_prep] Loading from CSV: {csv_path}")
    raw = pd.read_csv(csv_path, dtype=str).dropna(subset=[src_col, tgt_col])
    records = []
    for _, row in raw.iterrows():
        src = normalise(row[src_col])
        tgt = normalise(row[tgt_col])
        if is_valid(src, tgt, tgt_script_chars):
            records.append({"src": src, "tgt": tgt})

    df = pd.DataFrame(records)
    print(f"[data_prep] Valid pairs: {len(df):,}")
    if len(df) > n_total:
        df = df.sample(n=n_total, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Split & save
# ---------------------------------------------------------------------------

def split_and_save(
    df: pd.DataFrame,
    output_dir: Path,
    n_val: int,
    n_sft: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_val = min(n_val, len(df))
    n_sft = min(n_sft, len(df) - n_val)

    val_df = df.iloc[:n_val].reset_index(drop=True)
    sft_df = df.iloc[n_val : n_val + n_sft].reset_index(drop=True)
    dpo_df = df.iloc[n_val + n_sft :].reset_index(drop=True)

    val_df.to_csv(output_dir / "val.csv",       index=False)
    sft_df.to_csv(output_dir / "sft_train.csv", index=False)
    dpo_df.to_csv(output_dir / "dpo_train.csv", index=False)

    print("\n=== Data Split Complete ===")
    print(f"  val.csv       : {len(val_df):>7,} pairs")
    print(f"  sft_train.csv : {len(sft_df):>7,} pairs")
    print(f"  dpo_train.csv : {len(dpo_df):>7,} pairs")

    print("\n--- Sample pairs ---")
    for _, row in val_df.head(3).iterrows():
        print(f"  SRC: {row['src'][:90]}")
        print(f"  TGT: {row['tgt'][:90]}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--from-csv",  default=None,      help="Path to input CSV (skips HuggingFace)")
    parser.add_argument("--src-col",   default=LANG_PAIR["src_col"], help="Source column name in input CSV")
    parser.add_argument("--tgt-col",   default=LANG_PAIR["tgt_col"], help="Target column name in input CSV")
    parser.add_argument("--n-val",     type=int, default=N_VAL)
    parser.add_argument("--n-sft",     type=int, default=N_SFT)
    parser.add_argument("--n-dpo",     type=int, default=N_DPO)
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    n_total = args.n_val + args.n_sft + args.n_dpo
    tgt_chars = set(LANG_PAIR["tgt_script_chars"]) if LANG_PAIR["tgt_script_chars"] else None

    if args.from_csv:
        df = load_from_csv(
            args.from_csv, args.src_col, args.tgt_col,
            n_total, args.seed, tgt_chars,
        )
    else:
        df = load_from_hf(
            n_total, args.seed,
            LANG_PAIR["hf_dataset"], LANG_PAIR["hf_config"],
            LANG_PAIR["hf_src_key"], LANG_PAIR["hf_tgt_key"],
            tgt_chars,
        )

    split_and_save(df, Path(args.output_dir), args.n_val, args.n_sft, args.seed)

    for fname in ("val.csv", "sft_train.csv", "dpo_train.csv"):
        loaded = pd.read_csv(Path(args.output_dir) / fname)
        assert list(loaded.columns) == ["src", "tgt"], f"Column mismatch in {fname}"
        print(f"[data_prep] Verified: {fname} ({len(loaded):,} rows)")

    print("\n[data_prep] Done. Run stages 1→7 to complete the pipeline.")


if __name__ == "__main__":
    main()
