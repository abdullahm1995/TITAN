"""
stage7_dpo.py — Stage 7: Targeted Direct Preference Optimisation (DPO)

CRITICAL: KL reference = SFT checkpoint, NOT base model.
           This preserves SFT translation improvements.

DPO objective (Rafailov et al. 2023):
  L = -E[log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                - β · (log π_θ(y_l|x) - log π_ref(y_l|x)))]

Trainable: same crucial components as Stage 6 (< 5% of params)
Data:      Data/dpo_train.csv  (must include src, tgt, and base_tgt columns)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DIR_FWD, DIR_BWD,
    DPO_BATCH_SIZE, DPO_BETA, DPO_EARLY_STOPPING_PATIENCE,
    DPO_GRAD_ACCUMULATION_STEPS, DPO_LEARNING_RATE, DPO_MAX_EPOCHS,
    DPO_TRAIN_PATH,
    LANG_PAIR,
    MODEL_NAME,
    TEMPLATE_FWD, TEMPLATE_BWD,
    RANDOM_SEED,
    SFT_VAL_PATH,
    STAGE5_OUTPUT_DIR, STAGE6_OUTPUT_DIR, STAGE7_OUTPUT_DIR,
)
from utils import free_gpu_memory, load_json, normalise_text, save_json, set_seed, setup_logging

logger = setup_logging(
    "stage7",
    log_file=os.path.join(STAGE7_OUTPUT_DIR, "stage7.log"),
)

_SRC_COL = LANG_PAIR["src_col"]
_TGT_COL = LANG_PAIR["tgt_col"]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv_pairs(
    csv_path: str,
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Returns (src, tgt, base_tgt).
    base_tgt is None if the CSV has no 'base_tgt' column.
    """
    df = pd.read_csv(csv_path, dtype=str).dropna(subset=[_SRC_COL, _TGT_COL])
    src = [normalise_text(t) for t in df[_SRC_COL].tolist()]
    tgt = [normalise_text(t) for t in df[_TGT_COL].tolist()]
    base_tgt = None
    if "base_tgt" in df.columns:
        base_tgt = [normalise_text(str(t)) for t in df["base_tgt"].tolist()]
    return src, tgt, base_tgt


# ---------------------------------------------------------------------------
# DPO Dataset
# ---------------------------------------------------------------------------

class DPODataset(Dataset):
    """
    Batch-tokenizes all samples at once in __init__.
    __getitem__ is a pure dict lookup — no tokenizer calls at runtime.
    """
    def __init__(
        self,
        src_lines: List[str],
        tgt_lines: List[str],
        tokenizer,
        direction: str = DIR_FWD,
        max_length: int = 256,
        base_rejections: Optional[List[str]] = None,
    ):
        self.data: List[Dict[str, torch.Tensor]] = []

        template = TEMPLATE_FWD if direction == DIR_FWD else TEMPLATE_BWD

        prompts:   List[str] = []
        chosens:   List[str] = []
        rejecteds: List[str] = []

        if base_rejections is None:
            raise ValueError(
                "DPO training requires a 'base_tgt' column in the CSV with "
                "pre-generated rejected translations."
            )

        for idx, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            if direction == DIR_FWD:
                prompt = template.format(source=src)
                chosen = tgt
            else:
                prompt = template.format(source=tgt)
                chosen = src

            rejected = base_rejections[idx] if idx < len(base_rejections) else chosen

            prompts.append(prompt)
            chosens.append(prompt + " " + chosen)
            rejecteds.append(prompt + " " + rejected)

        p_enc = tokenizer(prompts,   add_special_tokens=False)
        c_enc = tokenizer(chosens,   max_length=max_length, truncation=True)
        r_enc = tokenizer(rejecteds, max_length=max_length, truncation=True)

        for i in range(len(prompts)):
            p_len = len(p_enc["input_ids"][i])

            c_ids  = torch.tensor(c_enc["input_ids"][i],      dtype=torch.long)
            c_mask = torch.tensor(c_enc["attention_mask"][i], dtype=torch.long)
            r_ids  = torch.tensor(r_enc["input_ids"][i],      dtype=torch.long)
            r_mask = torch.tensor(r_enc["attention_mask"][i], dtype=torch.long)

            c_lbl = c_ids.clone(); c_lbl[:p_len + 1] = -100
            r_lbl = r_ids.clone(); r_lbl[:p_len + 1] = -100

            self.data.append({
                "chosen_input_ids":        c_ids,
                "chosen_attention_mask":   c_mask,
                "chosen_labels":           c_lbl,
                "rejected_input_ids":      r_ids,
                "rejected_attention_mask": r_mask,
                "rejected_labels":         r_lbl,
            })

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


def dpo_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    def pad(tensors: List[torch.Tensor], val: int = 0) -> torch.Tensor:
        max_l = max(t.shape[0] for t in tensors)
        out   = torch.full((len(tensors), max_l), val, dtype=torch.long)
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out
    return {
        "chosen_input_ids":        pad([b["chosen_input_ids"]        for b in batch]),
        "chosen_attention_mask":   pad([b["chosen_attention_mask"]   for b in batch]),
        "chosen_labels":           pad([b["chosen_labels"]           for b in batch], -100),
        "rejected_input_ids":      pad([b["rejected_input_ids"]      for b in batch]),
        "rejected_attention_mask": pad([b["rejected_attention_mask"] for b in batch]),
        "rejected_labels":         pad([b["rejected_labels"]         for b in batch], -100),
    }


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Shift logits and labels before gathering so that logits[:, t, :]
    is compared against labels[:, t+1].  Returns (batch,) sum of
    log-probs over response tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_labels = shift_labels.to(shift_logits.device)

    log_probs   = F.log_softmax(shift_logits, dim=-1)
    labels_safe = shift_labels.clamp(min=0)
    token_lp    = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
    mask        = (shift_labels != -100).float()
    return (token_lp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_lp:   torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp:      torch.Tensor,
    ref_rejected_lp:    torch.Tensor,
    beta: float = DPO_BETA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standard DPO loss — Rafailov et al. 2023."""
    chosen_rewards   = beta * (policy_chosen_lp   - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss, chosen_rewards.mean(), rejected_rewards.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_dpo(
    policy_model,
    ref_model,
    tokenizer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    output_dir:   str,
    device:       torch.device,
    max_epochs:   int   = DPO_MAX_EPOCHS,
    lr:           float = DPO_LEARNING_RATE,
    grad_accum:   int   = DPO_GRAD_ACCUMULATION_STEPS,
    beta:         float = DPO_BETA,
    patience:     int   = DPO_EARLY_STOPPING_PATIENCE,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [p for p in policy_model.parameters() if p.requires_grad],
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
        policy_model.train()
        e_loss = e_cr = e_rr = e_acc = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            cids = batch["chosen_input_ids"].to(device)
            cmsk = batch["chosen_attention_mask"].to(device)
            clbl = batch["chosen_labels"].to(device)
            rids = batch["rejected_input_ids"].to(device)
            rmsk = batch["rejected_attention_mask"].to(device)
            rlbl = batch["rejected_labels"].to(device)

            pol_c = compute_log_probs(policy_model(cids, attention_mask=cmsk).logits, clbl)
            pol_r = compute_log_probs(policy_model(rids, attention_mask=rmsk).logits, rlbl)

            with torch.no_grad():
                ref_c = compute_log_probs(ref_model(cids, attention_mask=cmsk).logits, clbl)
                ref_r = compute_log_probs(ref_model(rids, attention_mask=rmsk).logits, rlbl)

            loss, cr, rr = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
            (loss / grad_accum).backward()

            acc = ((pol_c - ref_c) > (pol_r - ref_r)).float().mean()
            e_loss += loss.item()
            e_cr   += cr.item()
            e_rr   += rr.item()
            e_acc  += acc.item()
            n_steps += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy_model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        policy_model.eval()
        v_loss = 0.0; v_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                cids  = batch["chosen_input_ids"].to(device)
                cmsk  = batch["chosen_attention_mask"].to(device)
                clbl  = batch["chosen_labels"].to(device)
                rids  = batch["rejected_input_ids"].to(device)
                rmsk  = batch["rejected_attention_mask"].to(device)
                rlbl  = batch["rejected_labels"].to(device)
                pc    = compute_log_probs(policy_model(cids, attention_mask=cmsk).logits, clbl)
                pr    = compute_log_probs(policy_model(rids, attention_mask=rmsk).logits, rlbl)
                rc    = compute_log_probs(ref_model(cids,    attention_mask=cmsk).logits, clbl)
                rr_v  = compute_log_probs(ref_model(rids,    attention_mask=rmsk).logits, rlbl)
                vl, _, _ = dpo_loss(pc, pr, rc, rr_v, beta=beta)
                v_loss += vl.item(); v_steps += 1

        v_loss /= max(v_steps, 1)
        entry = {
            "epoch":           epoch + 1,
            "train_loss":      e_loss / max(n_steps, 1),
            "val_loss":        v_loss,
            "chosen_reward":   e_cr   / max(n_steps, 1),
            "rejected_reward": e_rr   / max(n_steps, 1),
            "reward_margin":   (e_cr - e_rr) / max(n_steps, 1),
            "reward_accuracy": e_acc  / max(n_steps, 1),
        }
        train_log.append(entry)
        logger.info(
            f"  Epoch {epoch+1}/{max_epochs} | train={entry['train_loss']:.4f} "
            f"val={v_loss:.4f} margin={entry['reward_margin']:.4f} "
            f"acc={entry['reward_accuracy']:.4f}"
        )

        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_state = {k: v.cpu().clone() for k, v in policy_model.named_parameters()
                          if v.requires_grad}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        cur = policy_model.state_dict()
        cur.update(best_state)
        policy_model.load_state_dict(cur)

    return train_log


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage7(
    stage5_output_dir: str = STAGE5_OUTPUT_DIR,
    stage6_output_dir: str = STAGE6_OUTPUT_DIR,
    output_dir:        str = STAGE7_OUTPUT_DIR,
    dpo_csv_path:      str = DPO_TRAIN_PATH,
    val_csv_path:      str = SFT_VAL_PATH,
    model_name:        str = MODEL_NAME,
    seed:              int = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)

    for d in [output_dir, os.path.join(output_dir, "checkpoints"),
              os.path.join(output_dir, "logs")]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"STAGE 7: Targeted DPO — {LANG_PAIR['src_lang']}→{LANG_PAIR['tgt_lang']}")
    logger.info(f"  KL reference: SFT checkpoint (NOT base model)")
    logger.info(f"  β = {DPO_BETA}")
    logger.info("=" * 70)

    components = load_json(os.path.join(stage5_output_dir, "components.json"))

    src_dpo, tgt_dpo, base_tgt_rejections = load_csv_pairs(dpo_csv_path)
    src_val, tgt_val, _                   = load_csv_pairs(val_csv_path)
    n = len(src_dpo)
    logger.info(f"DPO pairs: {n} | Val pairs: {len(src_val)}")

    if base_tgt_rejections is None:
        raise ValueError(
            "dpo_train.csv must contain a 'base_tgt' column with rejected translations."
        )
    logger.info(f"DPO pairs loaded: {len(base_tgt_rejections)} rejected translations")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sft_ckpt = os.path.join(stage6_output_dir, "checkpoints", "sft_final.pt")
    sft_state = None
    if Path(sft_ckpt).exists():
        logger.info(f"Loading SFT checkpoint from {sft_ckpt} ...")
        sft_state = torch.load(sft_ckpt, map_location="cpu", weights_only=True)
        logger.info("SFT checkpoint loaded into CPU RAM.")
    else:
        logger.warning("SFT checkpoint not found — using base model weights (suboptimal)")

    logger.info("Loading policy model ...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    if sft_state is not None:
        policy_model.load_state_dict(sft_state, strict=False)
        logger.info("SFT weights applied to policy model.")

    logger.info("Loading reference model (frozen SFT) ...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    if sft_state is not None:
        ref_model.load_state_dict(sft_state, strict=False)
    del sft_state
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    from stage6.stage6_sft import freeze_model_except_components
    trainable, total, ratio, _hook_handles = freeze_model_except_components(policy_model, components)
    logger.info(f"Policy trainable: {100*ratio:.2f}%")

    logger.info(f"Building train dataset ({n} pairs) — batch tokenising ...")
    t0 = time.time()
    train_ds = DPODataset(src_dpo, tgt_dpo, tokenizer, DIR_FWD, base_rejections=base_tgt_rejections)
    logger.info(f"  Train dataset ready in {time.time()-t0:.1f}s")
    t0 = time.time()
    val_ds   = DPODataset(src_val, tgt_val, tokenizer, DIR_FWD)
    logger.info(f"  Val dataset ready in {time.time()-t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=DPO_BATCH_SIZE,
                              shuffle=True, collate_fn=dpo_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=DPO_BATCH_SIZE,
                              collate_fn=dpo_collate_fn)

    train_log = train_dpo(
        policy_model, ref_model, tokenizer,
        train_loader, val_loader, output_dir, device,
    )

    for h in _hook_handles:
        h.remove()

    final_path = os.path.join(output_dir, "checkpoints", "dpo_final.pt")
    torch.save(policy_model.state_dict(), final_path)
    logger.info(f"Final DPO checkpoint: {final_path}")

    from evaluate import evaluate_all
    test_fwd = [{"source_text": s, "target_text": t} for s, t in zip(src_val[:200], tgt_val[:200])]
    test_bwd = [{"source_text": t, "target_text": s} for s, t in zip(src_val[:200], tgt_val[:200])]
    eval_fwd = evaluate_all(policy_model, tokenizer, test_fwd, DIR_FWD, "dpo_fwd")
    eval_bwd = evaluate_all(policy_model, tokenizer, test_bwd, DIR_BWD, "dpo_bwd")

    stats = {
        "language_pair":   f"{LANG_PAIR['src_lang']}→{LANG_PAIR['tgt_lang']}",
        "seed":            seed,
        "n_dpo_pairs":     n,
        "reference_model": "sft_checkpoint",
        "beta":            DPO_BETA,
        "trainable_ratio": ratio,
        "train_log":       train_log,
        "eval_fwd":        eval_fwd,
        "eval_bwd":        eval_bwd,
        "elapsed_s":       time.time() - t_start,
    }
    save_json(stats, os.path.join(output_dir, "stage7_stats.json"))
    logger.info(f"Stage 7 complete in {(time.time()-t_start)/60:.1f}min")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-csv", default=DPO_TRAIN_PATH)
    parser.add_argument("--val-csv", default=SFT_VAL_PATH)
    args = parser.parse_args()
    run_stage7(dpo_csv_path=args.dpo_csv, val_csv_path=args.val_csv)
