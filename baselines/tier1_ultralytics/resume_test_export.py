"""Resume the test() + export step for a tier-1 ultralytics run whose training
already finished but whose post-training pipeline was interrupted (e.g. machine
restart during shared-metric predict).

Reuses train/weights/best.pt — does NOT re-train.

Usage:
    python baselines/tier1_ultralytics/resume_test_export.py \
        --model rtdetr_l \
        --out-dir results_v2/baselines/tier1/rtdetr_l_main_holdout_20260524_141614
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 必须在 import albumentations / ultralytics 之前设, 避免每个 dataloader worker
# 触发 stderr 升级检查 → PowerShell 5.1 把 stderr 包成 NativeCommandError 累积致死。
# 用直接赋值而非 setdefault, 因为 cmd 的 "set VAR=1 && ..." 会把尾空格也吞进值
# (变成 '1 '), albumentations 严格判等 "1" 不通过, 这里强制覆盖为干净的 "1"。
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
    parser.add_argument("--out-dir", required=True,
                        help="Existing run dir containing train/weights/best.pt")
    parser.add_argument("--data-yaml", default=None,
                        help="Path to data.yaml. Defaults to ultralytics_data/data.yaml.")
    parser.add_argument("--rerun-val", action="store_true",
                        help="Also re-run val_val (default: skip if metrics_val.json exists)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"out-dir does not exist: {out_dir}")
    best_pt = out_dir / "train" / "weights" / "best.pt"
    if not best_pt.is_file():
        raise SystemExit(f"best.pt not found: {best_pt}")

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

    log_file = out_dir / "resume.log"
    logger = get_logger(f"resume.{args.model}", log_file=log_file)
    logger.info(f"== Resume test+export for {args.model} ==")
    logger.info(f"out_dir : {out_dir}")
    logger.info(f"best.pt : {best_pt}")

    trainer = UltralyticsTrainer(
        model_name=args.model,
        data_yaml=data_yaml,
        output_dir=out_dir,
        train_cfg=train_cfg,
        logger=logger,
    )
    # Skip train(); load the already-trained best.pt as self.model.
    trainer.model = _load_ultralytics_model(str(best_pt), trainer.is_rtdetr)
    trainer._train_run_dir = out_dir / "train"

    t0 = time.time()

    # Optionally re-run val_val
    metrics_val_path = out_dir / "metrics_val.json"
    if args.rerun_val or not metrics_val_path.is_file():
        val_metrics = trainer.validate(split="val")
        metrics_val_path.write_text(
            json.dumps(val_metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"val   : mAP50={val_metrics.get('mAP50'):.4f} "
                    f"mAP50_95={val_metrics.get('mAP50_95'):.4f}")
    else:
        logger.info(f"val   : reuse existing {metrics_val_path.name}")

    # Always re-run test (this is the part that got interrupted)
    test_metrics = trainer.test()
    ul = test_metrics["ultralytics"]
    sh = test_metrics["shared"]
    cons = test_metrics["consistency"]
    logger.info(f"test  : ultralytics mAP50={ul.get('mAP50'):.4f} | "
                f"shared mAP50={sh.get('mAP50'):.4f} | "
                f"rel_diff={cons['rel_diff']:.4f}")

    trainer.export_standard_outputs(train_cfg_full=train_cfg)

    total = time.time() - t0
    summary = {
        "model": args.model,
        "out_dir": str(out_dir),
        "total_minutes": round(total / 60, 2),
        "ultralytics_mAP50": ul.get("mAP50"),
        "shared_mAP50": sh.get("mAP50"),
        "consistency_rel_diff": cons["rel_diff"],
    }
    print("\n========== Resume Summary ==========")
    print(json.dumps(summary, indent=2))
    print("====================================")


if __name__ == "__main__":
    main()
