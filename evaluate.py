"""
evaluate.py — Translation evaluation for the pipeline.

Metrics: BLEU, chrF++, TER, BERTScore.
Left-pads inputs for decoder-only models.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    BERTSCORE_MODEL, DIR_FWD, DIR_BWD,
    LANG_PAIR, MAX_NEW_TOKENS_EVAL,
    TEMPLATE_FWD, TEMPLATE_BWD,
)
from utils import normalise_text, setup_logging

logger = setup_logging("evaluate")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_translations(
    model,
    tokenizer,
    source_texts: List[str],
    direction: str = DIR_FWD,
    max_new_tokens: int = MAX_NEW_TOKENS_EVAL,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> List[str]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    translations: List[str] = []
    template = TEMPLATE_FWD if direction == DIR_FWD else TEMPLATE_BWD

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(source_texts), batch_size):
        batch_src = source_texts[i : i + batch_size]
        prompts   = [template.format(source=s) for s in batch_src]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = out_ids[:, prompt_len:]
        for seq in generated:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            translations.append(normalise_text(text))

    return translations


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    try:
        from sacrebleu.metrics import BLEU
        return float(BLEU(effective_order=True).corpus_score(hypotheses, [references]).score)
    except ImportError:
        logger.warning("sacrebleu not installed — BLEU skipped")
        return float("nan")


def compute_chrf(hypotheses: List[str], references: List[str]) -> float:
    try:
        from sacrebleu.metrics import CHRF
        return float(CHRF(word_order=2).corpus_score(hypotheses, [references]).score)
    except ImportError:
        logger.warning("sacrebleu not installed — chrF++ skipped")
        return float("nan")


def compute_ter(hypotheses: List[str], references: List[str]) -> float:
    try:
        from sacrebleu.metrics import TER
        return float(TER().corpus_score(hypotheses, [references]).score)
    except ImportError:
        logger.warning("sacrebleu not installed — TER skipped")
        return float("nan")


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str = LANG_PAIR["tgt_lang_code"],
) -> Dict[str, float]:
    try:
        from bert_score import score as bs_score
        P, R, F1 = bs_score(
            hypotheses, references,
            model_type=BERTSCORE_MODEL,
            lang=lang,
            verbose=False,
        )
        return {
            "precision": float(P.mean()),
            "recall":    float(R.mean()),
            "f1":        float(F1.mean()),
        }
    except ImportError:
        logger.warning("bert_score not installed — BERTScore skipped")
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    direction: str,
    run_tag: str = "",
) -> Dict[str, Any]:
    logger.info(f"Evaluating {direction} [{run_tag}] on {len(samples)} samples ...")

    source_texts = [s["source_text"] for s in samples]
    references   = [normalise_text(s["target_text"]) for s in samples]
    hypotheses   = generate_translations(model, tokenizer, source_texts, direction=direction)

    bleu   = compute_bleu(hypotheses, references)
    chrf   = compute_chrf(hypotheses, references)
    ter    = compute_ter(hypotheses, references)
    bscore = compute_bertscore(hypotheses, references)

    results = {
        "direction": direction,
        "run_tag":   run_tag,
        "n_samples": len(samples),
        "bleu":      bleu,
        "chrf":      chrf,
        "ter":       ter,
        "bertscore": bscore,
    }

    logger.info(
        f"  [{run_tag}] BLEU={bleu:.2f} chrF++={chrf:.2f} "
        f"TER={ter:.2f} BERTScore-F1={bscore['f1']:.4f}"
    )
    for i in range(min(3, len(samples))):
        logger.info(f"  src: {source_texts[i][:70]}")
        logger.info(f"  ref: {references[i][:70]}")
        logger.info(f"  hyp: {hypotheses[i][:70]}")
        logger.info("  ---")

    return results


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    max_length: int = 256,
) -> float:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        n_tok = enc["attention_mask"].sum().item()
        total_nll    += out.loss.item() * n_tok
        total_tokens += n_tok

    return float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))))
