"""Run C1 SSL pretrain — orchestrator entry script (MS2 D6).

Used by the orchestrator step `ms2_d6_run_ssl_pretrain`. Loads
configs/ssl_pretrain.yaml, builds the FramePairDataset + YOLOv12s backbone +
projection head, runs 80-epoch pretraining, saves to
results_v2/methods/stc_yolo/_ms2/.

Usage:
    python methods/m6_stc_yolo/scripts/run_ssl_pretrain.py \
        --config methods/m6_stc_yolo/configs/ssl_pretrain.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str,
                   help="path to ssl_pretrain.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}

    import torch
    from ultralytics import YOLO

    from methods.m6_stc_yolo.data.frame_pair_dataset import FramePairDataset
    from methods.m6_stc_yolo.modules.ssl_pretrain import (
        SSLPretrainer,
        SSLProjectionHead,
        probe_layer_channels,
    )
    from shared import set_seed

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 1. Dataset
    data_cfg = cfg.get("data", {}) or {}
    ds = FramePairDataset(
        allowed_splits=tuple(data_cfg.get("include_splits", ["train", "val"])),
        pair_gap_min=int(data_cfg.get("pair_gap_min", 1)),
        pair_gap_max=int(data_cfg.get("pair_gap_max", 2)),
        image_size=int(cfg.get("imgsz", 640)),
        augment=True,
        seed=seed,
    )
    print(f"[ssl] dataset: {len(ds)} frame pairs (train+val, test excluded)")

    # 2. Model -- load full YOLO and unfreeze
    pretrained_path = _PROJECT_ROOT / "yolo12s.pt"
    yolo = YOLO(str(pretrained_path))
    yolo_model = yolo.model
    for p in yolo_model.parameters():
        p.requires_grad = True

    # 3. Probe backbone channel dim at feature_layer_idx
    ssl_cfg = cfg.get("ssl", {}) or {}
    feature_layer_idx = int(ssl_cfg.get("feature_layer_idx", 4))
    in_dim = probe_layer_channels(
        yolo_model, layer_idx=feature_layer_idx, imgsz=int(cfg.get("imgsz", 640))
    )
    print(f"[ssl] backbone layer {feature_layer_idx} feature channels: {in_dim}")

    # 4. Projection head
    proj_head = SSLProjectionHead(
        in_dim=in_dim,
        hidden_dim=int(ssl_cfg.get("projection_hidden", 256)),
        out_dim=int(ssl_cfg.get("projection_dim", 128)),
    )

    # 5. Pretrainer
    pretrain_cfg = {
        "epochs": int(cfg.get("epochs", 80)),
        "batch": int(cfg.get("batch", 32)),
        "lr0": float(cfg.get("lr0", 1e-4)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "workers": int(cfg.get("workers", 4)),
        "temperature": float(ssl_cfg.get("temperature", 0.2)),
        "num_tokens": ssl_cfg.get("num_tokens", None),
        "feature_layer_idx": feature_layer_idx,
        "device": ("cuda:0" if torch.cuda.is_available() else "cpu"),
        "output_dir": str(_PROJECT_ROOT / "results_v2" / "methods"
                          / "stc_yolo" / "_ms2"),
    }
    pretrainer = SSLPretrainer(yolo_model, proj_head, ds, pretrain_cfg)
    backbone_path = pretrainer.fit()
    print(f"[ssl] DONE. backbone state_dict -> {backbone_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
