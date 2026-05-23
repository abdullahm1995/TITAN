"""
run_pipeline.py — End-to-end pipeline runner

Runs all 7 stages in sequence.  Each stage can be skipped with --start-stage
if earlier outputs already exist.

Usage:
  python run_pipeline.py                          # run all stages
  python run_pipeline.py --start-stage 3          # resume from Stage 3
  python run_pipeline.py --stages 1 2             # run only stages 1 and 2
  python run_pipeline.py --from-csv /path/to.csv  # use a local CSV instead of HuggingFace

Adapt the LANG_PAIR block in config.py before running.
"""

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import setup_logging

logger = setup_logging("pipeline", log_file=str(_REPO_ROOT / "outputs" / "pipeline.log"))


def run_stage(stage_num: int, kwargs: dict) -> None:
    t0 = time.time()
    logger.info(f"\n{'='*70}")
    logger.info(f"  Launching Stage {stage_num}")
    logger.info(f"{'='*70}")

    if stage_num == 1:
        from stage1.stage1_data import run_stage1
        run_stage1(**{k: v for k, v in kwargs.items() if k in ("output_dir", "seed")})

    elif stage_num == 2:
        from stage2.stage2_sipp import run_stage2
        run_stage2(**{k: v for k, v in kwargs.items() if k in ("stage1_output_dir", "output_dir", "seed")})

    elif stage_num == 3:
        from stage3.stage3_sae import run_stage3
        run_stage3(**{k: v for k, v in kwargs.items()
                      if k in ("stage2_output_dir", "output_dir", "n_seeds", "max_epochs", "seed")})

    elif stage_num == 4:
        from stage4.stage4_scoring import run_stage4
        run_stage4(**{k: v for k, v in kwargs.items()
                      if k in ("stage1_output_dir", "stage2_output_dir",
                               "stage3_output_dir", "output_dir", "seed")})

    elif stage_num == 5:
        from stage5.stage5_mapping import run_stage5
        run_stage5(**{k: v for k, v in kwargs.items()
                      if k in ("stage2_output_dir", "stage4_output_dir", "output_dir", "seed")})

    elif stage_num == 6:
        from stage6.stage6_sft import run_stage6
        run_stage6(**{k: v for k, v in kwargs.items()
                      if k in ("stage5_output_dir", "output_dir",
                               "sft_csv_path", "val_csv_path", "model_name", "seed")})

    elif stage_num == 7:
        from stage7.stage7_dpo import run_stage7
        run_stage7(**{k: v for k, v in kwargs.items()
                      if k in ("stage5_output_dir", "stage6_output_dir", "output_dir",
                               "dpo_csv_path", "val_csv_path", "model_name", "seed")})

    logger.info(f"  Stage {stage_num} done in {(time.time()-t0)/60:.1f}min")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Resume from this stage (1-7)")
    parser.add_argument("--stages", type=int, nargs="+",
                        help="Run only these specific stages (overrides --start-stage)")
    parser.add_argument("--from-csv", metavar="PATH",
                        help="Use a local CSV file for data prep instead of HuggingFace")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the random seed in config.py")
    args = parser.parse_args()

    from config import (
        RANDOM_SEED,
        STAGE1_OUTPUT_DIR, STAGE2_OUTPUT_DIR, STAGE3_OUTPUT_DIR,
        STAGE4_OUTPUT_DIR, STAGE5_OUTPUT_DIR, STAGE6_OUTPUT_DIR, STAGE7_OUTPUT_DIR,
        MODEL_NAME,
        SFT_TRAIN_PATH, SFT_VAL_PATH, DPO_TRAIN_PATH,
    )

    seed = args.seed if args.seed is not None else RANDOM_SEED

    stage_kwargs = {
        # stage dirs
        "stage1_output_dir": STAGE1_OUTPUT_DIR,
        "stage2_output_dir": STAGE2_OUTPUT_DIR,
        "stage3_output_dir": STAGE3_OUTPUT_DIR,
        "stage4_output_dir": STAGE4_OUTPUT_DIR,
        "stage5_output_dir": STAGE5_OUTPUT_DIR,
        "stage6_output_dir": STAGE6_OUTPUT_DIR,
        # output dirs (same as stage dirs by default)
        "output_dir":        None,  # overwritten per stage below
        # model
        "model_name":        MODEL_NAME,
        # data
        "sft_csv_path":  SFT_TRAIN_PATH,
        "val_csv_path":  SFT_VAL_PATH,
        "dpo_csv_path":  DPO_TRAIN_PATH,
        "seed":          seed,
    }

    if args.stages:
        stages_to_run = sorted(args.stages)
    else:
        stages_to_run = list(range(args.start_stage, 8))

    output_dirs = {
        1: STAGE1_OUTPUT_DIR,
        2: STAGE2_OUTPUT_DIR,
        3: STAGE3_OUTPUT_DIR,
        4: STAGE4_OUTPUT_DIR,
        5: STAGE5_OUTPUT_DIR,
        6: STAGE6_OUTPUT_DIR,
        7: STAGE7_OUTPUT_DIR,
    }

    t_total = time.time()
    for s in stages_to_run:
        if s not in range(1, 8):
            logger.warning(f"Unknown stage {s} — skipping")
            continue
        stage_kwargs["output_dir"] = output_dirs[s]
        run_stage(s, stage_kwargs)

    logger.info(f"\nPipeline complete in {(time.time()-t_total)/60:.1f}min")


if __name__ == "__main__":
    main()
