# Translation Pipeline — Implementation Guide

A 7-stage mechanistic interpretability pipeline for low-resource machine translation.
The pipeline identifies which components of a pretrained decoder-only language model are
causally responsible for translation, then fine-tunes *only those components* using
supervised learning (SFT) followed by preference optimisation (DPO).

The approach works for any language pair and any decoder-only backbone that follows the
standard `model.model.layers[l].self_attn` / `model.model.layers[l].mlp` layout.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Quick Start](#quick-start)
4. [Adapting to a New Language Pair](#adapting-to-a-new-language-pair)
5. [Data Preparation](#data-preparation)
6. [Pipeline Stages](#pipeline-stages)
   - [Stage 1 — Contrastive Dataset Construction](#stage-1--contrastive-dataset-construction)
   - [Stage 2 — Causal Circuit Identification](#stage-2--causal-circuit-identification)
   - [Stage 3 — Sparse Autoencoder Training](#stage-3--sparse-autoencoder-training)
   - [Stage 4 — SAE Feature Scoring](#stage-4--sae-feature-scoring)
   - [Stage 5 — Component Mapping](#stage-5--component-mapping)
   - [Stage 6 — Targeted Supervised Fine-Tuning](#stage-6--targeted-supervised-fine-tuning)
   - [Stage 7 — Targeted DPO](#stage-7--targeted-dpo)
7. [Running the Pipeline](#running-the-pipeline)
8. [Evaluation](#evaluation)
9. [Configuration Reference](#configuration-reference)

---

## Repository Structure

```
.
├── config.py              # All hyperparameters and paths — edit this first
├── data_prep.py           # Dataset preparation (HuggingFace or local CSV)
├── run_pipeline.py        # End-to-end runner (stages 1–7)
├── evaluate.py            # Translation evaluation (BLEU, chrF, TER, BERTScore)
├── utils.py               # Shared utilities (logging, seeding, metrics)
├── requirements.txt
├── data/                  # Auto-created by data_prep.py
│   ├── val.csv            # 1 K validation pairs
│   ├── sft_train.csv      # SFT training pairs
│   └── dpo_train.csv      # DPO training pairs
├── stage1/
│   └── stage1_data.py     # Contrastive dataset construction
├── stage2/
│   └── stage2_sipp.py     # Causal circuit identification
├── stage3/
│   └── stage3_sae.py      # Sparse autoencoder training
├── stage4/
│   └── stage4_scoring.py  # SAE feature scoring and classification
├── stage5/
│   └── stage5_mapping.py  # Component mapping → components.json
├── stage6/
│   └── stage6_sft.py      # Targeted supervised fine-tuning
└── stage7/
    └── stage7_dpo.py      # Targeted direct preference optimisation
```

---

## Requirements

**Python 3.10+**

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.2 | Model training and inference |
| `transformers` | ≥ 4.40 | Model loading and tokenisation |
| `peft` | ≥ 0.10 | LoRA (optional, for Stage 6/7 adapters) |
| `trl` | ≥ 0.8 | DPO trainer (Stage 7) |
| `datasets` | ≥ 2.18 | HuggingFace dataset streaming |
| `sacrebleu` | ≥ 2.4 | BLEU and chrF evaluation |
| `bert-score` | ≥ 0.3.13 | BERTScore evaluation |

A CUDA-capable GPU is strongly recommended. The pipeline has been tested on 40 GB GPUs.
Model activations are cached to disk during Stage 2 to avoid re-running forward passes.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your language pair and model (see next section)
nano config.py

# 3. Prepare data
python data_prep.py

# 4. Run the full pipeline
python run_pipeline.py

# 5. Evaluate
python evaluate.py
```

To resume from a specific stage (e.g., if Stage 3 onwards needs to be re-run):

```bash
python run_pipeline.py --start-stage 3
```

To run only selected stages:

```bash
python run_pipeline.py --stages 6 7
```

---

## Adapting to a New Language Pair

All language-specific settings live in the `LANG_PAIR` block at the top of `config.py`.
Edit this block — nothing else needs to change.

```python
LANG_PAIR: dict = {
    "src_lang":         "Hindi",           # human-readable source language name
    "tgt_lang":         "Urdu",            # human-readable target language name

    "src_col":          "src",             # column name in your CSV files
    "tgt_col":          "tgt",             # column name in your CSV files

    "dir_fwd":          "Src→Tgt",         # label used in logs
    "dir_bwd":          "Tgt→Src",

    # Prompt templates — {source} is replaced at runtime
    "template_fwd":     "Hindi: {source} → Urdu:",
    "template_bwd":     "Urdu: {source} → Hindi:",

    # Tokens that mark translation direction in the prompt
    "indicator_tokens": ["Hindi", "Urdu", "→", ":"],

    # Characters unique to the target script.
    # Used to identify script-selective SAE features (Stage 4).
    # Set to [] if source and target share the same script.
    "tgt_script_chars": list("اآأإؤئبتثجحخدذرزسشصضطظعغفقكلمنهوي"),

    # BCP-47 language code for BERTScore
    "tgt_lang_code":    "ur",

    # HuggingFace dataset (set hf_dataset to None to use a local CSV instead)
    "hf_dataset":       None,
    "hf_config":        None,
    "hf_src_key":       None,
    "hf_tgt_key":       None,
}
```

Also set `MODEL_NAME` in `config.py` to your HuggingFace model identifier and update the
architecture constants (`NUM_LAYERS`, `HIDDEN_DIM`, `MLP_DIM`, `NUM_HEADS`, `KV_HEADS`,
`HEAD_DIM`) to match your backbone.

---

## Data Preparation

Run `data_prep.py` once before starting the pipeline. It produces three CSV files with
standardised `src` / `tgt` columns.

**Option A — HuggingFace dataset**

Set `hf_dataset`, `hf_config`, `hf_src_key`, `hf_tgt_key` in `LANG_PAIR`, then:

```bash
python data_prep.py
```

**Option B — Local CSV file**

Your CSV must have at least two columns (names configurable via `src_col` / `tgt_col`):

```
src,tgt
"source sentence 1","target sentence 1"
"source sentence 2","target sentence 2"
...
```

Then run:

```bash
python data_prep.py --from-csv /path/to/your/data.csv
```

**Default split sizes** (adjust in `config.py`):

| Split | Size | Used by |
|-------|------|---------|
| `val.csv` | 1,000 pairs | Evaluation (all stages) |
| `sft_train.csv` | 30,000 pairs | Stage 6 (SFT) |
| `dpo_train.csv` | 20,000 pairs | Stage 7 (DPO) |

> **Note on DPO data:** `dpo_train.csv` must have three columns — `prompt`, `chosen`
> (reference translation), and `rejected` (a lower-quality translation). The `rejected`
> column must be pre-generated before running Stage 7.

---

## Pipeline Stages

### Stage 1 — Contrastive Dataset Construction

**Script:** `stage1/stage1_data.py`

Builds a contrastive analysis dataset from `sft_train.csv`. For each sentence pair,
it creates:
- **X⁺** — the valid translation prompt (e.g., `"SrcLang: {sentence} → TgtLang:"`)
- **X⁻** — four counterfactual variants that remove the translation intent
  (target nullification, action distortion, semantic obfuscation, paradox insertion)

The output `analysis_contrastive.jsonl` is consumed by Stages 2–4 to probe which
model components respond specifically to the *translation task* rather than to
surface-level text features.

**Output:** `stage1/outputs/analysis_contrastive.jsonl`

---

### Stage 2 — Causal Circuit Identification

**Script:** `stage2/stage2_sipp.py`

Identifies which attention heads and MLP layers are *causally* responsible for
translation. The stage runs two algorithms:

**Algorithm 1 — Task Steering Subspace Identification**
For each layer, computes the low-rank subspace (via SVD) that best separates
activations on X⁺ prompts from X⁻ prompts. This subspace is the "translation
direction" in that layer's representation space.

**Algorithm 2 — Importance Scoring via Path Patching**
For each component (attention head or MLP), patches activations from a source
run into a target run *along the steering subspace only*, then measures the
change in output distribution (KL divergence). A component is marked **crucial**
if `|δ| ≥ 10%` (configurable via `S2_IMPORTANCE_THRESHOLD`).

Crucial components are saved to `stage2_results.json` and drive all downstream stages.

**Output:** `stage2/outputs/stage2_results.json`, cached activations in `stage2/outputs/cache/`

**Key hyperparameters** (in `config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `S2_IMPORTANCE_THRESHOLD` | 0.10 | Min \|δ\| to mark a component crucial |
| `S2_N_SUBSPACE` | 500 | Sentence pairs used for Algorithm 1 |
| `S2_N_SCORE` | 200 | Sentence pairs used for Algorithm 2 |
| `S2_SVD_RANK` | 1 | Rank of the steering subspace |

---

### Stage 3 — Sparse Autoencoder Training

**Script:** `stage3/stage3_sae.py`

Trains one Sparse Autoencoder (SAE) per crucial layer identified in Stage 2.
Each SAE learns a sparse dictionary over the MLP residual activations of its layer,
decomposing them into interpretable features.

**Architecture:**
```
Encoder : Linear(d_model, d_sae) → ReLU    # d_sae = 4 × d_model
Decoder : Linear(d_sae, d_model)            # tied weights (W_dec = W_enc.T)
Loss    : ‖h − ĥ‖² + λ‖z‖₁
```

**Stability measures:**
- Trained with 3 random seeds; best checkpoint (lowest validation loss) is kept
- Sparsity weight λ is reduced automatically if the dead-feature fraction exceeds 20%
- Reconstruction error is flagged if it exceeds 10% of the input variance

**Output:** `stage3/outputs/checkpoints/sae_layer{N}.pt` per crucial layer

**Key hyperparameters** (in `config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SAE_EXPANSION_FACTOR` | 4 | SAE width = `d_model × factor` |
| `SAE_LAMBDA_SPARSITY` | 1e-3 | L1 sparsity weight |
| `SAE_NUM_SEEDS` | 3 | Seeds for ensemble stability |
| `SAE_LEARNING_RATE` | 1e-4 | Adam learning rate |
| `SAE_MAX_EPOCHS` | 50 | Training epochs per seed |

---

### Stage 4 — SAE Feature Scoring

**Script:** `stage4/stage4_scoring.py`

Scores every SAE feature in every crucial layer on three statistics computed over
the contrastive dataset from Stage 1:

| Statistic | Meaning |
|-----------|---------|
| **lift** | Mean activation ratio on translation text vs. permuted baseline |
| **selectivity** | Mean activation on source language minus mean on target language |
| **script\_sel** | Selectivity restricted to target-script-character samples |

Each feature is assigned exactly one class:

| Class | Meaning |
|-------|---------|
| `SHARED` | Active for both source and target language |
| `SRC_SPECIFIC` | Active primarily for source-language input |
| `TGT_SPECIFIC` | Active primarily for target-language input |
| `SCRIPT` | Selective on target-script characters |
| `INDICATOR` | Strong language-direction signal |
| `NOISE` | Low lift and low selectivity — excluded from fine-tuning |

**Output:** `stage4/outputs/feature_scores.json`

---

### Stage 5 — Component Mapping

**Script:** `stage5/stage5_mapping.py`

Merges Stage 2 importance scores and Stage 4 feature classifications into a single
`components.json` that is the sole input to Stages 6 and 7.

Each entry records:
```json
{
  "component_id":           "layer12_mlp",
  "layer_index":            12,
  "is_mlp":                 true,
  "head_index":             null,
  "delta_score":            0.34,
  "role":                   "source",
  "dominant_feature_class": "SRC_SPECIFIC",
  "include_in_finetuning":  true
}
```

**Exclusion rule:** components whose SAE features are all `NOISE` are excluded.
The stage also validates that the selected components account for fewer than 5% of
total model parameters (`SFT_MAX_TRAINABLE_PARAM_RATIO`), ensuring that fine-tuning
remains targeted.

**Output:** `stage5/outputs/components.json`

---

### Stage 6 — Targeted Supervised Fine-Tuning

**Script:** `stage6/stage6_sft.py`

Fine-tunes *only* the components listed in `components.json` on `sft_train.csv`.
All other parameters are frozen throughout.

**Curriculum (three phases):**

| Phase | Data | Description |
|-------|------|-------------|
| 1 | Single-word pairs | Word-level alignment |
| 2 | 2–5 word pairs | Phrase-level alignment |
| 3 | Full sentences | Sentence-level translation |

**Regularisation:**
- KL divergence against the frozen base model (weight = `SFT_KL_WEIGHT = 0.1`)
  prevents catastrophic forgetting outside the targeted components
- Label smoothing = 0.1

Each phase saves a checkpoint; training resumes from the latest checkpoint if
interrupted.

**Output:** `stage6/outputs/checkpoints/sft_final.pt`

**Key hyperparameters** (in `config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SFT_LEARNING_RATE` | 2e-5 | Adam learning rate |
| `SFT_BATCH_SIZE` | 4 | Per-GPU batch size |
| `SFT_GRAD_ACCUMULATION_STEPS` | 8 | Effective batch = 32 |
| `SFT_MAX_EPOCHS_PER_PHASE` | 10 | Max epochs per curriculum phase |
| `SFT_KL_WEIGHT` | 0.1 | KL regularisation weight |
| `SFT_MAX_TRAINABLE_PARAM_RATIO` | 0.05 | Hard cap on trainable parameters |

---

### Stage 7 — Targeted DPO

**Script:** `stage7/stage7_dpo.py`

Applies Direct Preference Optimisation (DPO) to the SFT checkpoint, again training
only the crucial components.

**DPO objective:**
```
L = -E[ log σ( β · (log π_θ(y_w|x) − log π_ref(y_w|x))
              − β · (log π_θ(y_l|x) − log π_ref(y_l|x)) ) ]
```

**Important:** the KL reference model is the *SFT checkpoint*, not the base model.
This preserves the translation gains from Stage 6 while allowing preference learning.

**DPO data format** (`dpo_train.csv`):

| Column | Content |
|--------|---------|
| `prompt` | Source sentence (same format as SFT prompt template) |
| `chosen` | Reference / high-quality translation |
| `rejected` | Lower-quality translation (e.g., output from the SFT model itself or a baseline) |

**Output:** `stage7/outputs/checkpoints/dpo_final.pt`

**Key hyperparameters** (in `config.py`):

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `DPO_BETA` | 0.5 | KL penalty coefficient |
| `DPO_LEARNING_RATE` | 1e-6 | Adam learning rate |
| `DPO_BATCH_SIZE` | 8 | Per-GPU batch size |
| `DPO_MAX_EPOCHS` | 2 | Training epochs |

---

## Running the Pipeline

**Full run from scratch:**
```bash
python data_prep.py
python run_pipeline.py
```

**Resume from a specific stage:**
```bash
python run_pipeline.py --start-stage 4
```

**Run selected stages only:**
```bash
python run_pipeline.py --stages 6 7
```

**Override the random seed:**
```bash
python run_pipeline.py --seed 123
```

**Use a local CSV instead of HuggingFace:**
```bash
python data_prep.py --from-csv /path/to/data.csv
python run_pipeline.py
```

Stage outputs accumulate under `stage{N}/outputs/`. Re-running a stage overwrites
its outputs but does not affect earlier stages.

---

## Evaluation

```bash
python evaluate.py
```

Evaluates the final DPO model in both translation directions on `data/val.csv`.

**Metrics reported:**

| Metric | Description |
|--------|-------------|
| BLEU | n-gram precision (sacreBLEU implementation) |
| chrF++ | Character n-gram F-score |
| TER | Translation edit rate |
| Comet | Cross-lingual Optimized Metric for Evaluation of Translation |

Results are written to `outputs/eval_results.json`.

---

## Configuration Reference

All settings are in `config.py`. The table below covers the most commonly adjusted ones.

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `<your-model-name>` | HuggingFace model identifier |
| `NUM_LAYERS` | 42 | Number of transformer layers in the backbone |
| `HIDDEN_DIM` | 3584 | Model hidden dimension |
| `MLP_DIM` | 14336 | MLP intermediate dimension |
| `N_VAL` | 1,000 | Validation set size |
| `N_SFT` | 30,000 | SFT training set size |
| `N_DPO` | 20,000 | DPO training set size |
| `S2_IMPORTANCE_THRESHOLD` | 0.10 | Min causal importance δ to mark a component crucial |
| `S2_N_SUBSPACE` | 500 | Pairs for steering subspace computation |
| `S2_N_SCORE` | 200 | Pairs for importance scoring |
| `SAE_EXPANSION_FACTOR` | 4 | SAE width multiplier |
| `SAE_LAMBDA_SPARSITY` | 1e-3 | SAE L1 sparsity weight |
| `SFT_LEARNING_RATE` | 2e-5 | SFT optimiser learning rate |
| `SFT_KL_WEIGHT` | 0.1 | KL divergence regularisation weight |
| `SFT_MAX_TRAINABLE_PARAM_RATIO` | 0.05 | Max fraction of parameters that can be trained |
| `DPO_BETA` | 0.5 | DPO KL penalty coefficient |
| `DPO_LEARNING_RATE` | 1e-6 | DPO optimiser learning rate |

---

## Notes

- **Hook paths:** Stage 2 and Stage 6 use `model.model.layers[l].self_attn` and
  `model.model.layers[l].mlp` to register forward hooks. If your backbone uses
  different attribute names, update these paths in `stage2_sipp.py` and
  `stage6_sft.py`.

- **Memory:** Stages 2 and 3 cache activations to disk. Ensure sufficient disk space
  (roughly `n_pairs × n_layers × d_model × 4 bytes`). For 500 pairs, 42 layers, and
  d_model = 3584, this is approximately 1.1 GB.

- **Parallelism:** The pipeline is single-GPU by default. Multi-GPU setups can be
  enabled by wrapping models with `DataParallel` or using HuggingFace `accelerate`
  in the stage scripts.

- **DPO rejected responses:** The quality of DPO training depends directly on the
  quality of the `rejected` column in `dpo_train.csv`. Rejected responses should be
  plausible but worse than the reference — outputs from the SFT model work well for
  this purpose.
