"""Resume a tier-1 ultralytics training that was killed mid-train, then run the
standard val/test/export pipeline so the output directory ends up identical to a
normal `train_one.py` run.

Reads `<out-dir>/train/weights/last.pt` and calls `model.train(resume=True)` —
ultralytics restores epoch/optimizer/lr from the checkpoint's args.yaml.

Usage:
    python baselines/tier1_ultralytics/resume_train.py \
        --model rtdetr_l \
        --out-dir results_v2/baselines/tier1/rtdetr_l_main_holdout_20260525_004042
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before ultralytics/albumentations import — see resume_test_export.py
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

PROJECT_ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import get_logger, set_seed, setup_cuda_optimization  # noqa: E402
from baselines.tier1_ultralytics.train_one import load_train_cfg  # noqa: E402
from baselines.tier1_ultralytics.trainer import (  # noqa: E402
    UltralyticsTrainer,
    _load_ultralytics_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-yaml", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"out-dir does not exist: {out_dir}")
    last_pt = out_dir / "train" / "weights" / "last.pt"
    if not last_pt.is_file():
        raise SystemExit(f"last.pt not found: {last_pt}")

    if args.data_yaml is None:
        data_yaml = (PROJECT_ROOT / "results_v2" / "baselines" / "tier1"
                     / "ultralytics_data" / "data.yaml")
    else:
        data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml not found: {data_yaml}")

    train_cfg = load_train_cfg(args.model)
    set_seed(int(train_cfg.get("seed", 42)))
    setup_cuda_optimization()

    log_file = out_dir / "resume_train.log"
    logger = get_logger(f"resume_train.{args.model}", log_file=log_file)
    logger.info(f"== Resume train+test+export for {args.model} ==")
    logger.info(f"out_dir : {out_dir}")
    logger.info(f"last.pt : {last_pt}")

    # --- 1) resume training from last.pt ---
    is_rtdetr = "rtdetr" in args.model.lower()
    model = _load_ultralytics_model(str(last_pt), is_rtdetr)
    logger.info("[resume] starting model.train(resume=True)")
    t0 = time.time()
    model.train(resume=True)
    train_h = (time.time() - t0) / 3600
    logger.info(f"[resume] train segment finished in {train_h:.2f}h")

    # After resume, ultralytics writes a fresh best.pt/last.pt; reload best for val/test
    best_pt = out_dir / "train" / "weights" / "best.pt"
    if not best_pt.is_file():
        raise RuntimeError(f"best.pt not produced after resume: {best_pt}")

    # --- 2) val / test / export — reuse the standard trainer ---
    trainer = UltralyticsTrainer(
        model_name=args.model,
        data_yaml=data_yaml,
        output_dir=out_dir,
        train_cfg=train_cfg,
        logger=logger,
    )
    trainer.model = _load_ultralytics_model(str(best_pt), trainer.is_rtdetr)
    trainer._train_run_dir = out_dir / "train"

    val_metrics = trainer.validate(split="val")
    (out_dir / "metrics_val.json").write_text(
        json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"val   : mAP50={val_metrics.get('mAP50'):.4f} "
                f"mAP50_95={val_metrics.get('mAP50_95'):.4f}")

    test_metrics = trainer.test()
    ul = test_metrics["ultralytics"]
    sh = test_metrics["shared"]
    cons = test_metrics["consistency"]
    logger.info(f"test  : ultralytics mAP50={ul.get('mAP50'):.4f} | "
                f"shared mAP50={sh.get('mAP50'):.4f} | "
                f"rel_diff={cons['rel_diff']:.4f}")

    trainer.export_standard_outputs(train_cfg_full=train_cfg)

    total_h = (time.time() - t0) / 3600
    summary = {
        "model": args.model,
        "out_dir": str(out_dir),
        "train_h": round(train_h, 2),
        "total_h": round(total_h, 2),
        "ultralytics_mAP50": ul.get("mAP50"),
        "shared_mAP50": sh.get("mAP50"),
        "consistency_rel_diff": cons["rel_diff"],
    }
    print("\n========== Resume Summary ==========")
    print(json.dumps(summary, indent=2))
    print("====================================")


if __name__ == "__main__":
    main()
