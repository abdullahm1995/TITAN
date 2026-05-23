"""
utils.py — Shared utilities for the pipeline.
"""

import gc
import json
import logging
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """NFC-normalise and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def has_script_chars(text: str, script_chars: frozenset) -> bool:
    """Return True if text contains at least one character from script_chars."""
    return any(c in script_chars for c in text)


def classify_word_type(word: str) -> str:
    """
    Heuristic word-type annotation for contrastive pair metadata.
    Returns one of: 'noun', 'verb', 'adjective', 'other'.
    This is surface-level and works best for Latin-script languages;
    it is used only as metadata and does not affect circuit discovery.
    """
    w = word.lower().strip(".,!?;:\"'")
    if w.endswith(("ing", "tion", "er", "ir", "re", "ize", "ise", "ify")):
        return "verb"
    if w.endswith(("ful", "less", "ous", "ive", "al", "ic", "eux", "euse", "able", "ible")):
        return "adjective"
    if word[0].isupper() and len(word) > 1:
        return "noun"
    return "other"


def count_shared_vocab(src_words: List[str], tgt_words: List[str]) -> float:
    """Lowercased Jaccard similarity between source and target word sets."""
    src_set = set(w.lower() for w in src_words)
    tgt_set = set(w.lower() for w in tgt_words)
    if not src_set or not tgt_set:
        return 0.0
    return len(src_set & tgt_set) / len(src_set | tgt_set)


# ---------------------------------------------------------------------------
# JSON / JSONL I/O
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_gpu_memory(logger: logging.Logger, tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc  = torch.cuda.memory_allocated(i) / 1e9
        reserv = torch.cuda.memory_reserved(i) / 1e9
        total  = torch.cuda.get_device_properties(i).total_memory / 1e9
        logger.debug(
            f"GPU{i} [{tag}]: {alloc:.1f}/{total:.1f} GB alloc, {reserv:.1f} GB reserved"
        )
