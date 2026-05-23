"""
config.py — Central configuration for the pipeline.

To adapt for a new language pair, edit the LANG_PAIR block below.
All remaining settings (model architecture, training hyperparameters) are
language-pair-agnostic and only need to change when switching base models.
"""

from pathlib import Path

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Language pair — edit this block for any language pair
# ---------------------------------------------------------------------------
LANG_PAIR: dict = {
    # Human-readable language names (used in prompts and logs)
    "src_lang":         "English",
    "tgt_lang":         "French",

    # Column names in your input CSV files
    "src_col":          "src",
    "tgt_col":          "tgt",

    # Direction labels (used as keys throughout the pipeline)
    "dir_fwd":          "Src→Tgt",
    "dir_bwd":          "Tgt→Src",

    # Prompt templates — {source} is replaced with the source sentence
    "template_fwd":     "English: {source} → French:",
    "template_bwd":     "French: {source} → English:",

    # Tokens that signal translation direction in the prompt
    "indicator_tokens": ["English", "French", "→", ":"],

    # Script-specific characters for the target language.
    # Used to detect target-script-selective SAE features (Stage 4).
    # Set to [] if the target language shares script with the source.
    "tgt_script_chars": list("àâäéèêëîïôùûüçœæÀÂÄÉÈÊËÎÏÔÙÛÜÇŒÆ"),

    # BCP-47 code for the target language (used by BERTScore)
    "tgt_lang_code":    "fr",

    # HuggingFace dataset config for data_prep.py (set to None to use a local CSV)
    "hf_dataset":       "wmt14",
    "hf_config":        "fr-en",
    "hf_src_key":       "en",
    "hf_tgt_key":       "fr",
}

# Derived shortcuts (avoids repeated dict lookups in stage files)
DIR_FWD:          str  = LANG_PAIR["dir_fwd"]
DIR_BWD:          str  = LANG_PAIR["dir_bwd"]
TEMPLATE_FWD:     str  = LANG_PAIR["template_fwd"]
TEMPLATE_BWD:     str  = LANG_PAIR["template_bwd"]
INDICATOR_TOKENS: list = LANG_PAIR["indicator_tokens"]
TGT_SCRIPT_CHARS: list = LANG_PAIR["tgt_script_chars"]

# ---------------------------------------------------------------------------
# Paths — all relative to the project root (no hardcoded absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR:     Path = PROJECT_ROOT / "data"

SFT_TRAIN_PATH: str = str(DATA_DIR / "sft_train.csv")
SFT_VAL_PATH:   str = str(DATA_DIR / "val.csv")
DPO_TRAIN_PATH: str = str(DATA_DIR / "dpo_train.csv")

STAGE1_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage1" / "outputs")
STAGE2_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage2" / "outputs")
STAGE3_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage3" / "outputs")
STAGE4_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage4" / "outputs")
STAGE5_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage5" / "outputs")
STAGE6_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage6" / "outputs")
STAGE7_OUTPUT_DIR: str = str(PROJECT_ROOT / "stage7" / "outputs")

# ---------------------------------------------------------------------------
# Contrastive prompt templates (Stage 1)
# X+ = valid translation prompt; X- = counterfactual (translation intent removed)
# ---------------------------------------------------------------------------
COUNTERFACTUAL_TEMPLATES: dict = {
    "target_nullification": "{lang}: {source} → Nothing:",
    "action_distortion":    "Eat the following: {source} → Result:",
    "semantic_obfuscation": "{lang}: {source} → Color:",
    "paradox_insertion":    "{lang}: {source} → Silent Rock:",
}

# ---------------------------------------------------------------------------
# Base model architecture
# Set MODEL_NAME to your HuggingFace model identifier.
# Update NUM_LAYERS / HIDDEN_DIM / MLP_DIM / NUM_HEADS to match your backbone.
# ---------------------------------------------------------------------------
MODEL_NAME: str        = "<your-model-name>"   # e.g. "organization/model-name"
NUM_LAYERS: int        = 42
HIDDEN_DIM: int        = 3584
MLP_DIM: int           = 14336
NUM_HEADS: int         = 16          # query heads
KV_HEADS: int          = 8           # GQA key/value heads
HEAD_DIM: int          = 256
ATTN_LOGIT_CAP: float  = 50.0
FINAL_LOGIT_CAP: float = 30.0

LOCAL_ATTN_LAYERS: list  = list(range(0, NUM_LAYERS, 2))
GLOBAL_ATTN_LAYERS: list = list(range(1, NUM_LAYERS, 2))

# ---------------------------------------------------------------------------
# Dataset split sizes
# ---------------------------------------------------------------------------
N_VAL: int = 1_000
N_SFT: int = 30_000
N_DPO: int = 20_000

# ---------------------------------------------------------------------------
# Stage 2 — causal circuit identification hyperparameters
# ---------------------------------------------------------------------------
S2_IMPORTANCE_THRESHOLD: float  = 0.10   # |δ| ≥ 10% → translation-crucial
S2_RANDOM_ABLATION_BUDGET: int  = 20
S2_SVD_RANK: int                = 1
S2_EPS: float                   = 1e-6
S2_N_SUBSPACE: int              = 500    # pairs for Algorithm 1 (steering subspace)
S2_N_SCORE: int                 = 200    # pairs for Algorithm 2 (importance scoring)
S2_BATCH_SIZE: int              = 4
S2_NULL_EDIT_THRESHOLD: float   = 1e-4
S2_KNOCKOUT_STEPS: int          = 10

# ---------------------------------------------------------------------------
# SAE (Stage 3) — sparse autoencoder training
# ---------------------------------------------------------------------------
SAE_EXPANSION_FACTOR: int         = 4        # d_sae = 4 × d_model
SAE_LAMBDA_SPARSITY: float        = 1e-3
SAE_MAX_DEAD_FEATURE_RATIO: float = 0.20
SAE_NUM_SEEDS: int                = 3
SAE_LEARNING_RATE: float          = 1e-4
SAE_BATCH_SIZE: int               = 256
SAE_MAX_EPOCHS: int               = 50

# ---------------------------------------------------------------------------
# Feature scoring (Stage 4)
# ---------------------------------------------------------------------------
LIFT_DELTA_VALUES: list       = [0.5, 1.0, 2.0, 5.0]
SELECTIVITY_MIN_SAMPLES: int  = 50
SELECTIVITY_HIGH: float       = 0.003
LIFT_HIGH: float              = 0.05
INDICATOR_HIGH: float         = 0.006
SCRIPT_HIGH: float            = 0.004     # threshold for target-script-selective features

FEATURE_CLASSES: list = [
    "SHARED",        # active for both source and target language
    "SRC_SPECIFIC",  # active primarily for source language input
    "TGT_SPECIFIC",  # active primarily for target language input
    "SCRIPT",        # selective on target-script characters
    "INDICATOR",     # tracks translation direction signal
    "NOISE",         # low selectivity + low lift → excluded
]

COMPONENT_ROLES: list = ["source", "indicator", "positional", "shared_vocab"]

# ---------------------------------------------------------------------------
# SFT (Stage 6) — targeted supervised fine-tuning
# ---------------------------------------------------------------------------
SFT_LEARNING_RATE: float          = 2e-5
SFT_BATCH_SIZE: int               = 4
SFT_GRAD_ACCUMULATION_STEPS: int  = 8       # effective batch = 32
SFT_MAX_EPOCHS_PER_PHASE: int     = 10
SFT_LABEL_SMOOTHING: float        = 0.1
SFT_KL_WEIGHT: float              = 0.1
SFT_EARLY_STOPPING_PATIENCE: int  = 3
SFT_MAX_TRAINABLE_PARAM_RATIO: float = 0.05  # < 5% of total parameters

# ---------------------------------------------------------------------------
# DPO (Stage 7) — targeted direct preference optimisation
# ---------------------------------------------------------------------------
DPO_BETA: float                   = 0.5
DPO_LEARNING_RATE: float          = 1e-6
DPO_BATCH_SIZE: int               = 8
DPO_GRAD_ACCUMULATION_STEPS: int  = 4
DPO_MAX_EPOCHS: int               = 1
DPO_EARLY_STOPPING_PATIENCE: int  = 1

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_METRICS: list      = ["bleu", "chrf", "ter", "bertscore"]
BERTSCORE_MODEL: str    = "bert-base-multilingual-cased"
MAX_NEW_TOKENS_EVAL: int = 128
