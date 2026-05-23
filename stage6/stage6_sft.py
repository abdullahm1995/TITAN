"""
stage6_sft.py — Stage 6: Targeted Supervised Fine-Tuning

Inputs : components.json from Stage 5
         Data/sft_train.csv  (parallel src/tgt pairs)
         Data/val.csv        (parallel src/tgt pairs)

Trainable: ONLY components from components.json (< 5% of total params)
All other parameters: frozen

Curriculum (three phases):
  Phase 1: Word-level   (single word pairs)
  Phase 2: Phrase-level (2-5 word pairs)
  Phase 3: Sentence-level (> 5 words)

Regularisation:
  KL divergence vs frozen base model (weight = SFT_KL_WEIGHT)
  Label smoothing = 0.1
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DIR_FWD, DIR_BWD,
    HEAD_DIM, HIDDEN_DIM,
    LANG_PAIR,
    MODEL_NAME,
    TEMPLATE_FWD, TEMPLATE_BWD,
    RANDOM_SEED,
    SFT_BATCH_SIZE, SFT_EARLY_STOPPING_PATIENCE, SFT_GRAD_ACCUMULATION_STEPS,
    SFT_KL_WEIGHT, SFT_LABEL_SMOOTHING, SFT_LEARNING_RATE, SFT_MAX_EPOCHS_PER_PHASE,
    SFT_MAX_TRAINABLE_PARAM_RATIO,
    SFT_TRAIN_PATH, SFT_VAL_PATH,
    STAGE5_OUTPUT_DIR, STAGE6_OUTPUT_DIR,
)
from utils import free_gpu_memory, load_json, normalise_text, save_json, set_seed, setup_logging

logger = setup_logging(
    "stage6",
    log_file=os.path.join(STAGE6_OUTPUT_DIR, "stage6.log"),
)

_SRC_COL = LANG_PAIR["src_col"]
_TGT_COL = LANG_PAIR["tgt_col"]


# ---------------------------------------------------------------------------
# Data loading from CSV
# ---------------------------------------------------------------------------

def load_csv_pairs(csv_path: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(csv_path, dtype=str).dropna(subset=[_SRC_COL, _TGT_COL])
    src = [normalise_text(t) for t in df[_SRC_COL].tolist()]
    tgt = [normalise_text(t) for t in df[_TGT_COL].tolist()]
    return src, tgt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        tokenizer,
        direction: str = "both",
        max_length: int = 256,
    ):
        self.pairs: List[Tuple[str, str]] = []
        self.tokenizer  = tokenizer
        self.max_length = max_length

        for src, tgt in zip(src_lines, tgt_lines):
            if direction in (DIR_FWD, "both"):
                self.pairs.append((TEMPLATE_FWD.format(source=src), tgt))
            if direction in (DIR_BWD, "both"):
                self.pairs.append((TEMPLATE_BWD.format(source=tgt), src))

    def __len__(self) -> int: return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, target = self.pairs[idx]
        eos = self.tokenizer.eos_token or ""
        full_text = prompt + " " + target + eos
        enc = self.tokenizer(
            full_text, max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
        prompt_len = len(prompt_enc["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len + 1] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_len   = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        l = item["input_ids"].shape[0]
        input_ids[i, :l] = item["input_ids"]
        attn_mask[i, :l] = item["attention_mask"]
        labels[i, :l]    = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def split_by_word_count(
    src: List[str], tgt: List[str]
) -> Dict[str, Tuple[List[str], List[str]]]:
    word, phrase, sent = [], [], []
    word_t, phrase_t, sent_t = [], [], []
    for s, t in zip(src, tgt):
        wc = len(s.split())
        if wc == 1:
            word.append(s); word_t.append(t)
        elif 2 <= wc <= 5:
            phrase.append(s); phrase_t.append(t)
        else:
            sent.append(s); sent_t.append(t)
    return {
        "word":     (word,   word_t),
        "phrase":   (phrase, phrase_t),
        "sentence": (sent,   sent_t),
    }


# ---------------------------------------------------------------------------
# Parameter freezing with gradient masking
# ---------------------------------------------------------------------------

def freeze_model_except_components(
    model: nn.Module,
    components: List[Dict[str, Any]],
) -> Tuple[int, int, float, List]:
    """
    Freeze all parameters except those in components list.
    Returns (trainable_params, total_params, ratio, hook_handles).

    Hook placement assumes a decoder-only transformer layout:
      model.model.layers[l].self_attn.{q_proj, o_proj}
      model.model.layers[l].mlp
    Adjust these attribute paths for other model families.
    """
    for param in model.parameters():
        param.requires_grad = False

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    hook_handles: List = []
    _head_masks: Dict[Tuple[int, str], torch.Tensor] = {}

    for comp in components:
        if not comp.get("include_in_finetuning", True):
            continue
        layer_idx = comp["layer_index"]
        is_mlp    = comp["is_mlp"]
        head_idx  = comp.get("head_index")
        layer     = model.model.layers[layer_idx]

        if is_mlp:
            for param in layer.mlp.parameters():
                param.requires_grad = True
                trainable_params += param.numel()
        else:
            q_proj = layer.self_attn.q_proj
            o_proj = layer.self_attn.o_proj
            q_proj.weight.requires_grad = True
            o_proj.weight.requires_grad = True
            trainable_params += 2 * HIDDEN_DIM * HEAD_DIM

            start, end = head_idx * HEAD_DIM, (head_idx + 1) * HEAD_DIM

            q_key = (layer_idx, "q")
            if q_key not in _head_masks:
                _head_masks[q_key] = torch.zeros(q_proj.weight.shape[0])
            _head_masks[q_key][start:end] = 1.0

            o_key = (layer_idx, "o")
            if o_key not in _head_masks:
                _head_masks[o_key] = torch.zeros(o_proj.weight.shape[1])
            _head_masks[o_key][start:end] = 1.0

    for (layer_idx, proj_type), mask_1d in _head_masks.items():
        layer = model.model.layers[layer_idx]
        if proj_type == "q":
            param  = layer.self_attn.q_proj.weight
            mask2d = mask_1d.unsqueeze(1).to(device=param.device, dtype=param.dtype)
            h = param.register_hook(lambda g, m=mask2d: g * m)
        else:
            param  = layer.self_attn.o_proj.weight
            mask2d = mask_1d.unsqueeze(0).to(device=param.device, dtype=param.dtype)
            h = param.register_hook(lambda g, m=mask2d: g * m)
        hook_handles.append(h)

    ratio = trainable_params / total_params
    return trainable_params, total_params, ratio, hook_handles


# ---------------------------------------------------------------------------
# KL divergence loss
# ---------------------------------------------------------------------------

def kl_divergence_loss(
    policy_logits:    torch.Tensor,
    reference_logits: torch.Tensor,
    attention_mask:   torch.Tensor,
) -> torch.Tensor:
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_probs        = F.softmax(reference_logits, dim=-1)
    kl = F.kl_div(policy_log_probs, ref_probs, reduction="none").sum(dim=-1)
    return kl[attention_mask.bool()].mean()


# ---------------------------------------------------------------------------
# Single-phase training
# ---------------------------------------------------------------------------

def train_one_phase(
    model,
    ref_model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase_name: str,
    output_dir: str,
    device: torch.device,
    max_epochs: int = SFT_MAX_EPOCHS_PER_PHASE,
    lr: float = SFT_LEARNING_RATE,
    grad_accum: int = SFT_GRAD_ACCUMULATION_STEPS,
    kl_weight: float = SFT_KL_WEIGHT,
    patience: int = SFT_EARLY_STOPPING_PATIENCE,
) -> Dict[str, Any]:
    logger.info(f"  === Phase: {phase_name} | {len(train_loader.dataset)} train samples ===")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs * len(train_loader),
    )

    best_val_loss  = float("inf")
    patience_count = 0
    train_log: List[Dict] = []
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = epoch_kl = epoch_ce = 0.0
        epoch_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            labels     = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=SFT_LABEL_SMOOTHING,
            )

            with torch.no_grad():
                ref_out = ref_model(input_ids=input_ids, attention_mask=attn_mask)
            kl_loss = kl_divergence_loss(out.logits, ref_out.logits, attn_mask)

            total_loss = ce_loss + kl_weight * kl_loss
            (total_loss / grad_accum).backward()

            epoch_loss  += total_loss.item()
            epoch_kl    += kl_loss.item()
            epoch_ce    += ce_loss.item()
            epoch_steps += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                val_loss += F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                ).item()
                val_steps += 1

        val_loss /= max(val_steps, 1)
        train_loss_avg = epoch_loss / max(epoch_steps, 1)

        log_entry = {
            "epoch":      epoch + 1,
            "train_loss": train_loss_avg,
            "val_loss":   val_loss,
            "kl_loss":    epoch_kl / max(epoch_steps, 1),
            "ce_loss":    epoch_ce / max(epoch_steps, 1),
        }
        train_log.append(log_entry)
        logger.info(
            f"    Epoch {epoch+1}/{max_epochs} | "
            f"train={train_loss_avg:.4f} val={val_loss:.4f} kl={log_entry['kl_loss']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break

    if best_state:
        cur = model.state_dict()
        cur.update(best_state)
        model.load_state_dict(cur)

    ckpt_path = os.path.join(output_dir, "checkpoints", f"sft_{phase_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"  Phase {phase_name} checkpoint: {ckpt_path}")

    return {"phase": phase_name, "best_val_loss": best_val_loss,
            "train_log": train_log, "epochs_run": len(train_log)}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage6(
    stage5_output_dir: str = STAGE5_OUTPUT_DIR,
    output_dir: str        = STAGE6_OUTPUT_DIR,
    sft_csv_path: str      = SFT_TRAIN_PATH,
    val_csv_path: str      = SFT_VAL_PATH,
    model_name: str        = MODEL_NAME,
    seed: int              = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)

    for d in [output_dir, os.path.join(output_dir, "checkpoints"),
              os.path.join(output_dir, "logs")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"STAGE 6: Targeted SFT — {LANG_PAIR['src_lang']}→{LANG_PAIR['tgt_lang']}")
    logger.info("=" * 70)

    components = load_json(os.path.join(stage5_output_dir, "components.json"))
    logger.info(f"Loaded {len(components)} components from Stage 5")

    src_train, tgt_train = load_csv_pairs(sft_csv_path)
    src_val,   tgt_val   = load_csv_pairs(val_csv_path)
    logger.info(f"Training pairs: {len(src_train)} | Val pairs: {len(src_val)}")

    buckets = split_by_word_count(src_train, tgt_train)
    logger.info("Curriculum buckets: " + " | ".join(f"{k}:{len(v[0])}" for k, v in buckets.items()))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.config.use_cache = False
    device = model.model.embed_tokens.weight.device

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    trainable, total, ratio, _hook_handles = freeze_model_except_components(model, components)
    logger.info(f"Trainable: {trainable:,}/{total:,} = {100*ratio:.2f}%")
    if ratio > SFT_MAX_TRAINABLE_PARAM_RATIO:
        logger.warning(f"Ratio {100*ratio:.2f}% exceeds {SFT_MAX_TRAINABLE_PARAM_RATIO*100:.0f}% limit!")

    val_ds     = TranslationDataset(src_val, tgt_val, tokenizer)
    val_loader = DataLoader(val_ds, batch_size=SFT_BATCH_SIZE, collate_fn=collate_fn)

    all_phase_logs: List[Dict] = []

    phase_specs = [
        ("word",     buckets["word"],     SFT_LEARNING_RATE,       SFT_MAX_EPOCHS_PER_PHASE),
        ("phrase",   buckets["phrase"],   SFT_LEARNING_RATE * 0.5, SFT_MAX_EPOCHS_PER_PHASE),
        ("sentence", buckets["sentence"], SFT_LEARNING_RATE * 0.1, 3),
    ]

    for phase_name, (phase_src, phase_tgt), phase_lr, phase_epochs in phase_specs:
        if not phase_src:
            logger.info(f"  Phase {phase_name}: no samples — skipping")
            continue

        ckpt_path = os.path.join(output_dir, "checkpoints", f"sft_{phase_name}.pt")
        if Path(ckpt_path).exists():
            logger.info(f"  Phase {phase_name}: checkpoint found — loading")
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
            free_gpu_memory()
            all_phase_logs.append({"phase": phase_name, "resumed": True})
            continue

        train_ds     = TranslationDataset(phase_src, phase_tgt, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=SFT_BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn)

        phase_log = train_one_phase(
            model, ref_model, tokenizer,
            train_loader, val_loader, phase_name,
            output_dir, device, lr=phase_lr, max_epochs=phase_epochs,
        )
        all_phase_logs.append(phase_log)
        free_gpu_memory()

    for h in _hook_handles:
        h.remove()

    final_path = os.path.join(output_dir, "checkpoints", "sft_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final SFT checkpoint: {final_path}")

    from evaluate import evaluate_all
    test_fwd = [{"source_text": s, "target_text": t} for s, t in zip(src_val[:200], tgt_val[:200])]
    test_bwd = [{"source_text": t, "target_text": s} for s, t in zip(src_val[:200], tgt_val[:200])]
    eval_fwd = evaluate_all(model, tokenizer, test_fwd, DIR_FWD, "sft_fwd")
    eval_bwd = evaluate_all(model, tokenizer, test_bwd, DIR_BWD, "sft_bwd")

    stats = {
        "language_pair":    f"{LANG_PAIR['src_lang']}→{LANG_PAIR['tgt_lang']}",
        "seed":             seed,
        "n_train":          len(src_train),
        "trainable_params": trainable,
        "trainable_ratio":  ratio,
        "phase_logs":       all_phase_logs,
        "eval_fwd":         eval_fwd,
        "eval_bwd":         eval_bwd,
        "elapsed_s":        time.time() - t_start,
    }
    save_json(stats, os.path.join(output_dir, "stage6_stats.json"))
    logger.info(f"Stage 6 complete in {(time.time()-t_start)/60:.1f}min")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-csv", default=SFT_TRAIN_PATH)
    parser.add_argument("--val-csv", default=SFT_VAL_PATH)
    args = parser.parse_args()
    run_stage6(sft_csv_path=args.sft_csv, val_csv_path=args.val_csv)
