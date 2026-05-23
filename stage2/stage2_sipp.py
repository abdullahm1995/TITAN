"""
stage2_sipp.py — Stage 2: Causal Translation Circuit Identification

Implements:
  Algorithm 1 — Task Steering Subspace Identification
  Algorithm 2 — Importance Scoring via Path Patching

Hook placement targets a decoder-only transformer architecture
(self_attn.o_proj and mlp per layer).
Update hook registration if using a different model backbone.
"""

import gc
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import (
    DIR_FWD, DIR_BWD,
    HEAD_DIM, HIDDEN_DIM, KV_HEADS,
    NUM_HEADS, NUM_LAYERS, MODEL_NAME,
    INDICATOR_TOKENS,
    RANDOM_SEED,
    S2_BATCH_SIZE, S2_EPS, S2_IMPORTANCE_THRESHOLD,
    S2_N_SCORE, S2_N_SUBSPACE, S2_NULL_EDIT_THRESHOLD,
    S2_RANDOM_ABLATION_BUDGET, S2_SVD_RANK, S2_KNOCKOUT_STEPS,
    STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR,
)
from utils import (
    free_gpu_memory, load_json, load_jsonl, log_gpu_memory,
    save_json, save_jsonl, set_seed, setup_logging,
)

logger = setup_logging(
    "stage2",
    log_file=os.path.join(STAGE2_OUTPUT_DIR, "stage2.log"),
)

_HEAD_DIM   = HEAD_DIM
_NUM_HEADS  = NUM_HEADS
_HIDDEN_DIM = HIDDEN_DIM
_NUM_LAYERS = NUM_LAYERS


# ---------------------------------------------------------------------------
# Algorithm 1: Task Steering Subspace Identification
# ---------------------------------------------------------------------------

def compute_steering_subspace(
    diff_matrix: torch.Tensor,
    r: int = S2_SVD_RANK,
) -> torch.Tensor:
    """
    Algorithm 1 — translation-steering direction S_c.

    diff_matrix: M_c ∈ R^(d × N) — Δa_c^(i) = a_c(X+) - a_c(X-)
    Returns: S_c ∈ R^(d,) normalised unit vector (on CPU)
    """
    d, N = diff_matrix.shape
    device = diff_matrix.device

    s_bar     = diff_matrix.mean(dim=1)
    M_centered = diff_matrix - s_bar.unsqueeze(1)

    try:
        U, _, _ = torch.linalg.svd(M_centered, full_matrices=False)
        E = U[:, :r]
    except torch.linalg.LinAlgError:
        logger.warning("SVD failed — using mean direction directly")
        E = torch.zeros(d, r, device=device)

    s_orth = s_bar - E @ (E.T @ s_bar)
    norm = torch.norm(s_orth)
    if norm < 1e-8:
        s_orth = s_bar
        norm = torch.norm(s_orth)

    return (s_orth / (norm + 1e-8)).cpu()


# ---------------------------------------------------------------------------
# Equation 2: Subspace Projection Patching
# ---------------------------------------------------------------------------

def compute_patched_activation(
    a_plus:  torch.Tensor,
    a_minus: torch.Tensor,
    S_c:     torch.Tensor,
) -> torch.Tensor:
    """ã_c = P_{S_c} a_minus + P_{S_c⊥} a_plus"""
    S_c      = S_c / (torch.norm(S_c) + 1e-8)
    proj_minus = torch.einsum("...d,d->...", a_minus, S_c).unsqueeze(-1) * S_c
    proj_plus  = torch.einsum("...d,d->...", a_plus,  S_c).unsqueeze(-1) * S_c
    return a_plus - proj_plus + proj_minus


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

def _register_capture_hooks(
    model,
    storage_attn: Dict[int, torch.Tensor],
    storage_mlp:  Dict[int, torch.Tensor],
) -> List:
    handles = []
    for l in range(_NUM_LAYERS):
        layer = model.model.layers[l]

        def _attn_hook(lidx):
            def h(module, args):
                storage_attn[lidx] = args[0][:, -1, :].detach().cpu()
                return args
            return h

        def _mlp_hook(lidx):
            def h(module, inp, out):
                storage_mlp[lidx] = out[:, -1, :].detach().cpu()
            return h

        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn_hook(l)))
        handles.append(layer.mlp.register_forward_hook(_mlp_hook(l)))
    return handles


def _remove_hooks(handles: List) -> None:
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Null-edit identity test
# ---------------------------------------------------------------------------

def null_edit_identity_test(
    model, tokenizer, sample_prompts: List[str], device: torch.device,
) -> bool:
    logger.info("=== NULL-EDIT IDENTITY TEST ===")
    model.eval()

    inputs = tokenizer(
        sample_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=256,
    ).to(device)

    with torch.no_grad():
        baseline_out = model(**inputs).logits.detach().cpu()

    storage_attn: Dict[int, torch.Tensor] = {}
    storage_mlp:  Dict[int, torch.Tensor] = {}
    hooks = _register_capture_hooks(model, storage_attn, storage_mlp)

    with torch.no_grad():
        hooked_out = model(**inputs).logits.detach().cpu()

    _remove_hooks(hooks)

    max_diff = (baseline_out - hooked_out).abs().max().item()
    logger.info(f"  Max |diff| with capture hooks: {max_diff:.2e}")

    if max_diff > S2_NULL_EDIT_THRESHOLD:
        logger.error(
            f"NULL-EDIT TEST FAILED: {max_diff:.2e} > {S2_NULL_EDIT_THRESHOLD}. "
            "Hooks incorrectly placed — STOPPING."
        )
        return False

    if len(storage_attn) != _NUM_LAYERS or len(storage_mlp) != _NUM_LAYERS:
        logger.error("NULL-EDIT TEST FAILED: not all layers captured.")
        return False

    logger.info("NULL-EDIT IDENTITY TEST PASSED")
    return True


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_batch(tokenizer, prompts: List[str], device: torch.device, max_length: int = 256) -> Dict:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in enc.items()}


def get_first_target_token_ids(tokenizer, target_texts: List[str]) -> List[int]:
    ids = []
    for text in target_texts:
        text = text.strip()
        if not text:
            ids.append(tokenizer.eos_token_id)
            continue
        toks = tokenizer.encode(text, add_special_tokens=False)
        ids.append(toks[0] if toks else tokenizer.eos_token_id)
    return ids


# ---------------------------------------------------------------------------
# Phase 1: Activation extraction and caching
# ---------------------------------------------------------------------------

def extract_and_cache_activations(
    model, tokenizer,
    samples: List[Dict[str, Any]],
    cache_dir: str,
    device: torch.device,
    batch_size: int = S2_BATCH_SIZE,
    split_tag: str = "sipp",
) -> Dict[str, Any]:
    logger.info(f"=== PHASE 1: Activation Extraction ({len(samples)} samples) ===")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    N = len(samples)
    target_ids = get_first_target_token_ids(tokenizer, [s["target_text"] for s in samples])

    attn_plus:  Dict[int, List[torch.Tensor]] = defaultdict(list)
    attn_minus: Dict[int, List[torch.Tensor]] = defaultdict(list)
    mlp_plus:   Dict[int, List[torch.Tensor]] = defaultdict(list)
    mlp_minus:  Dict[int, List[torch.Tensor]] = defaultdict(list)
    y_orig_list: List[torch.Tensor] = []

    n_batches = (N + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_samples = samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        pos_prompts   = [s["positive_prompt"]     for s in batch_samples]
        neg_prompts   = [s["counterfactual_text"] for s in batch_samples]
        batch_tids    = target_ids[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{n_batches}")

        # X+ pass
        st_a: Dict[int, torch.Tensor] = {}
        st_m: Dict[int, torch.Tensor] = {}
        hooks_p = _register_capture_hooks(model, st_a, st_m)
        enc_p = tokenize_batch(tokenizer, pos_prompts, device)
        with torch.no_grad():
            out_p = model(**enc_p)
        end_logits = out_p.logits[:, -1, :]
        batch_y = torch.tensor(
            [end_logits[i, tid].item() for i, tid in enumerate(batch_tids)], dtype=torch.float32
        )
        y_orig_list.append(batch_y)
        _remove_hooks(hooks_p)
        for l in range(_NUM_LAYERS):
            if l in st_a: attn_plus[l].append(st_a[l].to(torch.bfloat16))
            if l in st_m: mlp_plus[l].append(st_m[l].to(torch.bfloat16))
        del out_p, enc_p, st_a, st_m
        free_gpu_memory()

        # X- pass
        st_a2: Dict[int, torch.Tensor] = {}
        st_m2: Dict[int, torch.Tensor] = {}
        hooks_m = _register_capture_hooks(model, st_a2, st_m2)
        enc_m = tokenize_batch(tokenizer, neg_prompts, device)
        with torch.no_grad():
            _ = model(**enc_m)
        _remove_hooks(hooks_m)
        for l in range(_NUM_LAYERS):
            if l in st_a2: attn_minus[l].append(st_a2[l].to(torch.bfloat16))
            if l in st_m2: mlp_minus[l].append(st_m2[l].to(torch.bfloat16))
        del enc_m, st_a2, st_m2
        free_gpu_memory()

    logger.info("  Saving activation cache ...")
    y_orig = torch.cat(y_orig_list, dim=0)
    torch.save(y_orig, os.path.join(cache_dir, f"{split_tag}_y_orig.pt"))
    torch.save(
        torch.tensor(target_ids, dtype=torch.long),
        os.path.join(cache_dir, f"{split_tag}_target_ids.pt"),
    )

    for l in range(_NUM_LAYERS):
        for tag, store in [
            ("attn_plus", attn_plus), ("attn_minus", attn_minus),
            ("mlp_plus",  mlp_plus),  ("mlp_minus",  mlp_minus),
        ]:
            if store[l]:
                torch.save(
                    torch.cat(store[l], dim=0),
                    os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_{tag}.pt"),
                )

    return {"N": N, "y_orig_mean": y_orig.mean().item()}


# ---------------------------------------------------------------------------
# Phase 2: Steering subspaces
# ---------------------------------------------------------------------------

def compute_all_steering_subspaces(
    cache_dir:    str,
    subspace_dir: str,
    split_tag:    str = "sipp",
    r:            int = S2_SVD_RANK,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    logger.info("=== PHASE 2: Steering Subspace Computation (Algorithm 1) ===")
    Path(subspace_dir).mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subspaces: Dict[str, torch.Tensor] = {}

    for l in range(_NUM_LAYERS):
        # Attention heads
        ap = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_attn_plus.pt")
        am = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_attn_minus.pt")
        if Path(ap).exists() and Path(am).exists():
            all_cached = all(
                Path(os.path.join(subspace_dir, f"L{l:02d}H{h:02d}_Sc.pt")).exists()
                for h in range(_NUM_HEADS)
            )
            if all_cached:
                for h in range(_NUM_HEADS):
                    comp_id = f"L{l:02d}H{h:02d}"
                    subspaces[comp_id] = torch.load(
                        os.path.join(subspace_dir, f"{comp_id}_Sc.pt"), weights_only=True
                    )
            else:
                attn_p = torch.load(ap, weights_only=True).float().to(device)
                attn_m = torch.load(am, weights_only=True).float().to(device)
                for h in range(_NUM_HEADS):
                    comp_id   = f"L{l:02d}H{h:02d}"
                    save_path = os.path.join(subspace_dir, f"{comp_id}_Sc.pt")
                    if Path(save_path).exists():
                        subspaces[comp_id] = torch.load(save_path, weights_only=True)
                        continue
                    s, e = h * _HEAD_DIM, (h + 1) * _HEAD_DIM
                    diff = (attn_p[:, s:e] - attn_m[:, s:e]).T
                    S_c = compute_steering_subspace(diff, r) if diff.norm() >= 1e-8 \
                          else torch.zeros(_HEAD_DIM)
                    subspaces[comp_id] = S_c
                    torch.save(S_c, save_path)
                del attn_p, attn_m
                torch.cuda.empty_cache()

        # MLP
        mp = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_mlp_plus.pt")
        mm = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_mlp_minus.pt")
        if Path(mp).exists() and Path(mm).exists():
            comp_id   = f"L{l:02d}MLP"
            save_path = os.path.join(subspace_dir, f"{comp_id}_Sc.pt")
            if Path(save_path).exists():
                subspaces[comp_id] = torch.load(save_path, weights_only=True)
            else:
                mlp_p = torch.load(mp, weights_only=True).float().to(device)
                mlp_m = torch.load(mm, weights_only=True).float().to(device)
                diff  = (mlp_p - mlp_m).T
                S_c   = compute_steering_subspace(diff, r) if diff.norm() >= 1e-8 \
                        else torch.zeros(_HIDDEN_DIM)
                subspaces[comp_id] = S_c
                torch.save(S_c, save_path)
                del mlp_p, mlp_m
                torch.cuda.empty_cache()

        logger.info(f"  Layer {l:02d}/{_NUM_LAYERS-1} done ({len(subspaces)} subspaces total)")

    logger.info(f"Phase 2 complete. {len(subspaces)} subspaces computed.")
    return subspaces


# ---------------------------------------------------------------------------
# Phase 3: path-patching importance scoring
# ---------------------------------------------------------------------------

def score_component(
    model, tokenizer,
    samples: List[Dict[str, Any]],
    layer_idx: int,
    component: str,
    S_c: torch.Tensor,
    y_orig_cached: torch.Tensor,
    target_ids: List[int],
    cache_dir: str,
    device: torch.device,
    batch_size: int = S2_BATCH_SIZE,
    split_tag: str = "sipp",
) -> float:
    N      = len(samples)
    S_c    = S_c.to(device)
    is_mlp = (component == "mlp")

    if is_mlp:
        plus_path  = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_mlp_plus.pt")
        minus_path = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_mlp_minus.pt")
    else:
        head_idx   = int(component.split("_")[1])
        plus_path  = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_attn_plus.pt")
        minus_path = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_attn_minus.pt")

    if not Path(plus_path).exists() or not Path(minus_path).exists():
        return 0.0

    all_plus  = torch.load(plus_path,  weights_only=True).float()[:N]
    all_minus = torch.load(minus_path, weights_only=True).float()[:N]

    if not is_mlp:
        start, end = head_idx * _HEAD_DIM, (head_idx + 1) * _HEAD_DIM
        all_plus   = all_plus[:, start:end]
        all_minus  = all_minus[:, start:end]

    all_patched = compute_patched_activation(
        all_plus.to(device), all_minus.to(device), S_c
    ).cpu()

    delta_list: List[float] = []
    n_batches = (N + batch_size - 1) // batch_size

    for b in range(n_batches):
        sl            = slice(b * batch_size, (b + 1) * batch_size)
        batch_samples = samples[sl]
        pos_prompts   = [s["positive_prompt"] for s in batch_samples]
        batch_tids    = target_ids[sl]
        batch_y_orig  = y_orig_cached[sl].to(device)
        batch_patched = all_patched[sl].to(device)

        enc = tokenize_batch(tokenizer, pos_prompts, device)

        if is_mlp:
            def _mlp_patch(pt):
                def hook(module, inp, out):
                    r = out.clone(); r[:, -1, :] = pt; return r
                return hook
            h = model.model.layers[layer_idx].mlp.register_forward_hook(
                _mlp_patch(batch_patched)
            )
        else:
            s_i, e_i = head_idx * _HEAD_DIM, (head_idx + 1) * _HEAD_DIM
            def _attn_patch(pt, s, e):
                def hook(module, args):
                    m = args[0].clone(); m[:, -1, s:e] = pt; return (m,)
                return hook
            h = model.model.layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                _attn_patch(batch_patched, s_i, e_i)
            )

        with torch.no_grad():
            out_new = model(**enc)
        h.remove()

        logits_new = out_new.logits[:, -1, :]
        y_new = torch.tensor(
            [logits_new[i, tid].item() for i, tid in enumerate(batch_tids)],
            device=device, dtype=torch.float32,
        )
        delta = (y_new - batch_y_orig) / (batch_y_orig.abs() + S2_EPS)
        delta_list.extend(delta.cpu().tolist())

        del out_new, logits_new, enc
        free_gpu_memory()

    return float(np.mean(delta_list))


def fast_proxy_scores(
    subspaces: Dict[str, torch.Tensor],
    cache_dir:  str,
    split_tag:  str = "sipp",
    top_k:      int = 50,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Zero-forward-pass pre-filter.
    proxy_c = mean_i |S_c · (a+_i - a-_i)|
    Returns the top_k candidate component IDs.
    """
    logger.info(f"=== PHASE 3a: Fast Proxy Pre-filter (top-{top_k}) ===")
    proxy: Dict[str, float] = {}

    for l in range(_NUM_LAYERS):
        ap = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_attn_plus.pt")
        am = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_attn_minus.pt")
        if Path(ap).exists() and Path(am).exists():
            attn_p = torch.load(ap, weights_only=True).float()
            attn_m = torch.load(am, weights_only=True).float()
            diff_attn = attn_p - attn_m
            for h in range(_NUM_HEADS):
                comp_id = f"L{l:02d}H{h:02d}"
                S_c = subspaces.get(comp_id, torch.zeros(_HEAD_DIM))
                s, e = h * _HEAD_DIM, (h + 1) * _HEAD_DIM
                proxy[comp_id] = (diff_attn[:, s:e] @ S_c).abs().mean().item()

        mp = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_mlp_plus.pt")
        mm = os.path.join(cache_dir, f"{split_tag}_layer{l:02d}_mlp_minus.pt")
        if Path(mp).exists() and Path(mm).exists():
            comp_id = f"L{l:02d}MLP"
            S_c = subspaces.get(comp_id, torch.zeros(_HIDDEN_DIM))
            mlp_p = torch.load(mp, weights_only=True).float()
            mlp_m = torch.load(mm, weights_only=True).float()
            proxy[comp_id] = ((mlp_p - mlp_m) @ S_c).abs().mean().item()

    sorted_comps = sorted(proxy, key=lambda c: proxy[c], reverse=True)
    candidates   = sorted_comps[:top_k]
    logger.info(f"  Proxy computed for {len(proxy)} components. Top-{top_k}: {candidates[:5]} ...")
    return candidates, proxy


def run_sipp_scoring(
    model, tokenizer,
    score_samples, subspaces,
    cache_dir, output_dir, device,
    batch_size=32, split_tag="sipp",
    top_k_candidates: int = 50,
) -> Dict[str, float]:
    N = len(score_samples)
    logger.info(f"=== PHASE 3: Importance Scoring ({N} samples) ===")

    candidates, proxy_scores = fast_proxy_scores(
        subspaces, cache_dir, split_tag, top_k=top_k_candidates
    )

    y_orig = torch.load(
        os.path.join(cache_dir, f"{split_tag}_y_orig.pt"), weights_only=True
    ).float()[:N]
    target_ids_raw = torch.load(
        os.path.join(cache_dir, f"{split_tag}_target_ids.pt"), weights_only=True
    ).tolist()[:N]

    candidate_scores: Dict[str, float] = {}
    total = len(candidates)
    t0 = time.time()

    for done, comp_id in enumerate(candidates, 1):
        l      = int(comp_id[1:3])
        is_mlp = "MLP" in comp_id
        component = "mlp" if is_mlp else f"head_{int(comp_id[4:6])}"
        S_c = subspaces.get(comp_id, torch.zeros(_HIDDEN_DIM if is_mlp else _HEAD_DIM))

        candidate_scores[comp_id] = score_component(
            model, tokenizer, score_samples, l, component,
            S_c, y_orig, target_ids_raw, cache_dir, device, batch_size, split_tag,
        )
        eta = (time.time() - t0) / done * (total - done)
        logger.info(
            f"  [{done:02d}/{total}] {comp_id} δ={candidate_scores[comp_id]:+.4f} | "
            f"ETA {eta/60:.1f}min"
        )
        free_gpu_memory()

    all_ids = (
        [f"L{l:02d}H{h:02d}" for l in range(_NUM_LAYERS) for h in range(_NUM_HEADS)] +
        [f"L{l:02d}MLP"       for l in range(_NUM_LAYERS)]
    )
    importance_scores = {cid: candidate_scores.get(cid, 0.0) for cid in all_ids}

    save_json(importance_scores, os.path.join(output_dir, "importance_scores.json"))
    save_json(proxy_scores,      os.path.join(output_dir, "proxy_scores.json"))
    logger.info(f"Phase 3 complete. {len(candidate_scores)} components fully scored.")
    return importance_scores


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

def classify_component_roles(
    model, tokenizer,
    samples: List[Dict[str, Any]],
    crucial_heads: List[Tuple[int, int]],
    device: torch.device,
    batch_size: int = S2_BATCH_SIZE,
    n_samples: int = 100,
) -> Dict[str, str]:
    logger.info(f"=== ROLE CLASSIFICATION ({len(crucial_heads)} crucial heads) ===")
    role_labels: Dict[str, str] = {}
    classify_samples = samples[:min(n_samples, len(samples))]

    head_stats: Dict[Tuple[int, int], Dict[str, List[float]]] = {
        (l, h): {"ind": [], "src": [], "adj": []} for l, h in crucial_heads
    }
    layers_needed = sorted(set(l for l, h in crucial_heads))
    n_batches = (len(classify_samples) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch = classify_samples[batch_idx*batch_size : (batch_idx+1)*batch_size]
        pos_prompts = [s["positive_prompt"] for s in batch]

        storage: Dict[int, torch.Tensor] = {}
        hooks = []
        for l in layers_needed:
            def _cap(ll):
                def h(module, args):
                    storage[ll] = args[0].detach().cpu()
                    return args
                return h
            hooks.append(
                model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(_cap(l))
            )

        enc = tokenize_batch(tokenizer, pos_prompts, device)
        input_ids = enc["input_ids"]
        seq_len = input_ids.shape[1]
        with torch.no_grad():
            _ = model(**enc)
        _remove_hooks(hooks)

        ind_token_ids = set()
        for t in INDICATOR_TOKENS:
            ind_token_ids.update(tokenizer.encode(t, add_special_tokens=False))

        for l, h in crucial_heads:
            if l not in storage:
                continue
            pre_wo   = storage[l]
            head_out = pre_wo[:, :, h*_HEAD_DIM:(h+1)*_HEAD_DIM]
            head_norms = head_out.norm(dim=-1)

            for si in range(head_norms.shape[0]):
                norms    = head_norms[si]
                total    = norms.sum() + 1e-8
                norm_attn = (norms / total).tolist()
                tokens   = input_ids[si].tolist()

                ind_mass = sum(norm_attn[p] for p, tid in enumerate(tokens)
                               if tid in ind_token_ids and p < seq_len)
                adj_mass = sum(norm_attn[p] for p in range(1, min(3, seq_len)))
                src_mass = max(0.0, 1.0 - ind_mass - adj_mass)

                head_stats[(l, h)]["ind"].append(ind_mass)
                head_stats[(l, h)]["src"].append(src_mass)
                head_stats[(l, h)]["adj"].append(adj_mass)

        del storage, enc
        free_gpu_memory()

    for l, h in crucial_heads:
        stats = head_stats[(l, h)]
        if not stats["ind"]:
            role_labels[f"L{l:02d}H{h:02d}"] = "source"
            continue
        mi, ms, ma = np.mean(stats["ind"]), np.mean(stats["src"]), np.mean(stats["adj"])
        if mi >= ms and mi >= ma:
            role = "indicator"
        elif ma >= ms and ma >= mi:
            role = "positional"
        else:
            role = "source"
        role_labels[f"L{l:02d}H{h:02d}"] = role
        logger.info(f"  L{l:02d}H{h:02d}: ind={mi:.3f} src={ms:.3f} adj={ma:.3f} → {role}")

    return role_labels


# ---------------------------------------------------------------------------
# Knockout validation
# ---------------------------------------------------------------------------

def run_knockout_validation(
    model, tokenizer,
    val_samples, crucial_components, subspaces,
    cache_dir, output_dir, device,
    batch_size=S2_BATCH_SIZE,
    n_knockout_steps=S2_KNOCKOUT_STEPS,
    n_random=S2_RANDOM_ABLATION_BUDGET,
    split_tag="sipp",
) -> Dict[str, Any]:
    logger.info(f"=== KNOCKOUT VALIDATION ({len(crucial_components)} crucial) ===")

    mean_ablation: Dict[str, torch.Tensor] = {}
    for cid in crucial_components[:n_knockout_steps]:
        layer_idx = int(cid[1:3])
        is_mlp = "MLP" in cid
        if is_mlp:
            path = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_mlp_minus.pt")
            if Path(path).exists():
                mean_ablation[cid] = torch.load(path, weights_only=True).float().mean(dim=0)
        else:
            head_idx = int(cid[4:6])
            path = os.path.join(cache_dir, f"{split_tag}_layer{layer_idx:02d}_attn_minus.pt")
            if Path(path).exists():
                data = torch.load(path, weights_only=True).float()
                s, e = head_idx * _HEAD_DIM, (head_idx + 1) * _HEAD_DIM
                mean_ablation[cid] = data[:, s:e].mean(dim=0)

    def compute_acc(component_ids_to_ablate):
        correct = 0
        N_eval  = len(val_samples)
        n_b     = (N_eval + batch_size - 1) // batch_size
        for b in range(n_b):
            sl          = slice(b * batch_size, (b + 1) * batch_size)
            batch       = val_samples[sl]
            pos_prompts = [s["positive_prompt"] for s in batch]
            batch_tids  = get_first_target_token_ids(tokenizer, [s["target_text"] for s in batch])
            enc         = tokenize_batch(tokenizer, pos_prompts, device)

            ablation_hooks = []
            for cid in component_ids_to_ablate:
                if cid not in mean_ablation:
                    continue
                mv   = mean_ablation[cid].to(device)
                lidx = int(cid[1:3])
                if "MLP" in cid:
                    def _mh(mv_):
                        def hook(m, inp, out):
                            r = out.clone()
                            r[:, -1, :] = mv_.unsqueeze(0).expand(out.shape[0], -1)
                            return r
                        return hook
                    ablation_hooks.append(
                        model.model.layers[lidx].mlp.register_forward_hook(_mh(mv))
                    )
                else:
                    hi = int(cid[4:6])
                    si, ei = hi * _HEAD_DIM, (hi + 1) * _HEAD_DIM
                    def _ah(mv_, s, e):
                        def hook(m, args):
                            mod = args[0].clone()
                            mod[:, -1, s:e] = mv_.unsqueeze(0).expand(mod.shape[0], -1)
                            return (mod,)
                        return hook
                    ablation_hooks.append(
                        model.model.layers[lidx].self_attn.o_proj.register_forward_pre_hook(
                            _ah(mv, si, ei)
                        )
                    )

            with torch.no_grad():
                out = model(**enc)
            _remove_hooks(ablation_hooks)
            preds = out.logits[:, -1, :].argmax(dim=-1).cpu().tolist()
            correct += sum(p == t for p, t in zip(preds, batch_tids))
            del out, enc
            free_gpu_memory()
        return correct / N_eval

    baseline_acc = compute_acc([])
    logger.info(f"  Baseline accuracy: {baseline_acc:.4f}")

    crucial_curve = [baseline_acc]
    ablated = []
    for step in range(min(n_knockout_steps, len(crucial_components))):
        ablated.append(crucial_components[step])
        acc = compute_acc(ablated)
        crucial_curve.append(acc)
        logger.info(f"  Crucial step {step+1}: ablated={ablated[-1]} acc={acc:.4f}")

    import random as _rng_mod
    rng = _rng_mod.Random(RANDOM_SEED)
    all_ids    = (
        [f"L{l:02d}H{h:02d}" for l in range(_NUM_LAYERS) for h in range(_NUM_HEADS)] +
        [f"L{l:02d}MLP"       for l in range(_NUM_LAYERS)]
    )
    non_crucial  = [c for c in all_ids if c not in set(crucial_components)]
    random_comps = rng.sample(non_crucial, min(n_random, len(non_crucial)))

    random_curve = [baseline_acc]
    ablated_r = []
    for step in range(min(n_knockout_steps, len(random_comps))):
        ablated_r.append(random_comps[step])
        acc = compute_acc(ablated_r)
        random_curve.append(acc)

    results = {
        "baseline_accuracy":      baseline_acc,
        "crucial_accuracy_curve": crucial_curve,
        "random_accuracy_curve":  random_curve,
        "validation_passed": (baseline_acc - crucial_curve[-1]) > (baseline_acc - random_curve[-1]) + 0.02,
    }
    save_json(results, os.path.join(output_dir, "knockout_results.json"))
    logger.info(f"Knockout passed: {results['validation_passed']}")
    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_stage2(
    model_name:        str = MODEL_NAME,
    stage1_output_dir: str = STAGE1_OUTPUT_DIR,
    output_dir:        str = STAGE2_OUTPUT_DIR,
    n_subspace:        int = S2_N_SUBSPACE,
    n_score:           int = S2_N_SCORE,
    batch_size:        int = 32,
    seed:              int = RANDOM_SEED,
) -> Dict[str, Any]:
    t_start = time.time()
    set_seed(seed)

    for d in [
        output_dir,
        os.path.join(output_dir, "activation_cache"),
        os.path.join(output_dir, "steering_subspaces"),
        os.path.join(output_dir, "knockout"),
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 2: Causal Translation Circuit Identification")
    logger.info(f"  Model     : {model_name}")
    logger.info(f"  Threshold : |δ| > {S2_IMPORTANCE_THRESHOLD*100:.1f}%")
    logger.info("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log_gpu_memory(logger)

    all_samples   = load_jsonl(os.path.join(stage1_output_dir, "analysis_contrastive.jsonl"))
    fwd_samples   = [s for s in all_samples if s["direction_label"] == DIR_FWD]
    bwd_samples   = [s for s in all_samples if s["direction_label"] == DIR_BWD]

    subspace_samples = fwd_samples + bwd_samples
    score_samples    = fwd_samples[:500] + bwd_samples[:500]
    val_samples      = fwd_samples[-100:] + bwd_samples[-100:]

    logger.info(
        f"Samples — subspace:{len(subspace_samples)} "
        f"score:{len(score_samples)} val:{len(val_samples)}"
    )

    passed = null_edit_identity_test(
        model, tokenizer, [subspace_samples[0]["positive_prompt"]], device
    )
    if not passed:
        raise RuntimeError("Null-edit identity test failed — fix hook placement.")

    cache_dir  = os.path.join(output_dir, "activation_cache")
    last_file  = os.path.join(cache_dir, f"sipp_layer{_NUM_LAYERS-1:02d}_mlp_plus.pt")
    cache_ok   = False
    if Path(last_file).exists():
        cached_n = torch.load(last_file, weights_only=True).shape[0]
        cache_ok = (cached_n >= len(subspace_samples))

    if not cache_ok:
        if Path(last_file).exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        extract_and_cache_activations(
            model, tokenizer, subspace_samples,
            cache_dir, device, batch_size=batch_size, split_tag="sipp",
        )
    else:
        logger.info(f"Phase 1 cache complete ({cached_n} samples) — skipping extraction.")

    subspace_dir = os.path.join(output_dir, "steering_subspaces")
    subspaces = compute_all_steering_subspaces(
        cache_dir, subspace_dir, split_tag="sipp", device=device
    )

    importance_scores = run_sipp_scoring(
        model, tokenizer, score_samples, subspaces,
        cache_dir, output_dir, device,
        batch_size=batch_size, split_tag="sipp",
        top_k_candidates=50,
    )

    crucial_list = sorted(
        [(cid, d) for cid, d in importance_scores.items() if abs(d) >= S2_IMPORTANCE_THRESHOLD],
        key=lambda x: abs(x[1]), reverse=True,
    )
    crucial_ids = [c[0] for c in crucial_list]

    n_total   = len(importance_scores)
    n_crucial = len(crucial_ids)
    logger.info(f"Crucial: {n_crucial}/{n_total} ({100*n_crucial/n_total:.1f}%)")

    crucial_attn = [(int(cid[1:3]), int(cid[4:6])) for cid in crucial_ids if "H" in cid]
    role_labels  = {}
    if crucial_attn:
        role_labels = classify_component_roles(
            model, tokenizer, fwd_samples[:100], crucial_attn, device,
            batch_size=batch_size, n_samples=100,
        )
    save_json(role_labels, os.path.join(output_dir, "component_roles.json"))

    knockout_results = run_knockout_validation(
        model, tokenizer, val_samples, crucial_ids, subspaces,
        cache_dir, os.path.join(output_dir, "knockout"),
        device, batch_size=batch_size,
        n_knockout_steps=min(S2_KNOCKOUT_STEPS, len(crucial_ids)),
    )

    full_output = {}
    for comp_id, delta in importance_scores.items():
        layer_idx = int(comp_id[1:3])
        is_mlp    = "MLP" in comp_id
        full_output[comp_id] = {
            "layer":          layer_idx,
            "component":      "mlp" if is_mlp else f"head_{int(comp_id[4:6])}",
            "is_mlp":         is_mlp,
            "attention_type": ("local" if layer_idx % 2 == 0 else "global") if not is_mlp else None,
            "delta_score":    delta,
            "is_crucial":     abs(delta) >= S2_IMPORTANCE_THRESHOLD,
            "role":           role_labels.get(comp_id, "unclassified") if not is_mlp else "mlp",
        }
    save_json(full_output, os.path.join(output_dir, "importance_scores_full.json"))

    stats = {
        "model":              model_name,
        "total_components":   n_total,
        "crucial_components": n_crucial,
        "crucial_fraction":   n_crucial / n_total,
        "threshold":          S2_IMPORTANCE_THRESHOLD,
        "top_10_crucial":     crucial_list[:10],
        "knockout_passed":    knockout_results["validation_passed"],
        "elapsed_s":          time.time() - t_start,
    }
    save_json(stats, os.path.join(output_dir, "stage2_stats.json"))

    logger.info(f"Stage 2 complete in {(time.time()-t_start)/60:.1f}min")
    return {
        "importance_scores":  importance_scores,
        "crucial_components": crucial_ids,
        "role_labels":        role_labels,
        "knockout_results":   knockout_results,
        "stats":              stats,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subspace",  type=int, default=S2_N_SUBSPACE)
    parser.add_argument("--n-score",     type=int, default=S2_N_SCORE)
    parser.add_argument("--batch-size",  type=int, default=S2_BATCH_SIZE)
    args = parser.parse_args()
    run_stage2(n_subspace=args.n_subspace, n_score=args.n_score, batch_size=args.batch_size)
