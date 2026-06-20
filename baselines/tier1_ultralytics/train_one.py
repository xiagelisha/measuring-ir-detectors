"""Train a single tier-1 ultralytics baseline.

Usage:
    python baselines/tier1_ultralytics/train_one.py \
        --model yolov8s --split main_holdout --epochs 200
    python baselines/tier1_ultralytics/train_one.py \
        --model yolov8s --quick-verify
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

THIS_DIR = Path(__file__).resolve().parent

from shared import get_logger, set_seed, setup_cuda_optimization  # noqa: E402
from baselines.tier1_ultralytics.data_yaml import build_data_yaml  # noqa: E402
from baselines.tier1_ultralytics.trainer import UltralyticsTrainer  # noqa: E402


VALID_MODELS = ["yolov8s", "yolov11s", "yolov12s", "yolov10s", "rtdetr_l"]


def load_train_cfg(model: str) -> dict:
    with open(THIS_DIR / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(THIS_DIR / "configs" / "model_specific.yaml", "r", encoding="utf-8") as f:
        specific = yaml.safe_load(f)
    if model not in specific:
        raise ValueError(f"model '{model}' not in model_specific.yaml: {list(specific.keys())}")
    cfg.update(specific[model])
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=VALID_MODELS, required=True)
    parser.add_argument("--split", default="main_holdout")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--quick-verify", action="store_true")
    args = parser.parse_args()

    train_cfg = load_train_cfg(args.model)

    suffix = args.output_suffix
    if args.quick_verify:
        train_cfg["epochs"] = 3
        train_cfg["close_mosaic"] = 0  # mosaic warmup doesn't matter for 3 epoch
        if not suffix:
            suffix = "_smoke"
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch is not None:
        train_cfg["batch"] = args.batch

    set_seed(int(train_cfg.get("seed", 42)))
    setup_cuda_optimization()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{args.model}_{args.split}_{timestamp}{suffix}"
    out_dir = PROJECT_ROOT / "results_v2" / "baselines" / "tier1" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = out_dir / "train.log"
    logger = get_logger(f"train.{args.model}", log_file=log_file)
    logger.info(f"== Training {args.model} (split={args.split}, epochs={train_cfg['epochs']}) ==")
    logger.info(f"output_dir: {out_dir}")

    # 1) prepare data
    splits_dir = PROJECT_ROOT / "data" / "splits" / args.split
    data_yaml_path = (PROJECT_ROOT / "results_v2" / "baselines" / "tier1"
                      / "ultralytics_data" / "data.yaml")
    build_data_yaml(splits_dir, data_yaml_path, PROJECT_ROOT)

    # 2) train -> validate -> test -> export
    trainer = UltralyticsTrainer(
        model_name=args.model,
        data_yaml=data_yaml_path,
        output_dir=out_dir,
        train_cfg=train_cfg,
        logger=logger,
    )
    t0 = time.time()
    best = trainer.train()
    duration_train = time.time() - t0
    logger.info(f"train done in {duration_train/60:.2f} min. best={best}")

    val_metrics = trainer.validate(split="val")
    logger.info(f"val   : mAP50={val_metrics.get('mAP50'):.4f} "
                f"mAP50_95={val_metrics.get('mAP50_95'):.4f}")
    # also dump metrics_val.json
    (out_dir / "metrics_val.json").write_text(
        json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
        "split": args.split,
        "epochs": train_cfg["epochs"],
        "out_dir": str(out_dir),
        "total_minutes": round(total / 60, 2),
        "ultralytics_mAP50": ul.get("mAP50"),
        "shared_mAP50": sh.get("mAP50"),
        "consistency_rel_diff": cons["rel_diff"],
    }
    print("\n========== Summary ==========")
    print(json.dumps(summary, indent=2))
    print("=============================")


if __name__ == "__main__":
    main()
