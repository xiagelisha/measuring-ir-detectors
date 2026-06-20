"""STC-YOLO single-seed training entry.

Wires up the orchestrator-driven config flow:
  - `--data-yaml PATH`   : explicit ultralytics data yaml (advanced)
  - default              : auto-build via baselines.tier1_ultralytics.data_yaml
                           from data/splits/main_holdout/{train,val,test}.txt
                           into results_v2/orchestrator/runs/_data_yaml/data.yaml
                           (idempotent — already-copied images are skipped)
  - `--override k=v ...` : dotted-path config override (repeatable). Examples:
        --override epochs=10
        --override stc.enable_colormap_inversion=true
        --override infer.imgsz=1056
  - `--inherit-config from-state` : read
        results_v2/orchestrator/state.json::config_updates
    and translate to the same kind of overrides. Used by orchestrator to
    propagate gate decisions (c4_in_main, ssl_pretrained_backbone path, ...)
    into the next training run.

Currently the model is vanilla YOLOv12s; modules come online with MS1 D2/D3
(C4 colormap), MS2 D7 (SSL), MS3 D10d/D11 (MT / FiLM).

Test-time evaluation uses cfg.infer.* (mirrors paper3 tuned recipe:
imgsz=1056, augment=True).

Output:
  results_v2/methods/stc_yolo/<run-name>/ultralytics_test_metrics.json
  (the file orchestrator/run_5seed.py aggregates)

Usage:
  python methods/m6_stc_yolo/train_one.py \
      --config methods/m6_stc_yolo/configs/default.yaml \
      --seed 42 --run-name ms1_d4a_baseline_seed42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --- Helpers (unit-tested in tests/test_skeleton.py) -----------------------

def _coerce_value(v: str):
    """Coerce a string from CLI into bool / int / float / None / str.

    Examples:
        "true"  -> True       "false" -> False
        "null"  -> None       "none"  -> None
        "42"    -> 42         "1e-3"  -> 0.001
        "foo"   -> "foo"      "[a,b]" -> [a,b]   (only JSON-style for lists)
    """
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in {"true", "yes"}:
        return True
    if s.lower() in {"false", "no"}:
        return False
    if s.lower() in {"null", "none", ""}:
        return None
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    try:
        i = int(s)
        return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    """Walk a dotted key path in cfg and set the leaf to value, creating dict
    nodes as needed.

    >>> c = {"stc": {"enable_x": False}}
    >>> _set_dotted(c, "stc.enable_x", True); c
    {'stc': {'enable_x': True}}
    >>> _set_dotted(c, "stc.new_branch.flag", 42); c["stc"]["new_branch"]
    {'flag': 42}
    """
    parts = dotted.split(".")
    if not parts:
        raise ValueError(f"empty dotted key: {dotted!r}")
    node = cfg
    for k in parts[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[parts[-1]] = value


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `key=value` strings to cfg in-place (returns cfg for chaining)."""
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--override must be key=value, got {ov!r}")
        key, raw_val = ov.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce_value(raw_val))
    return cfg


# Map orchestrator's config_updates KEYS -> dotted YAML paths in our config.
# Values from state.json are coerced through _coerce_value first (so bool /
# int / float live as native types in cfg). Anything not in this map is
# silently ignored (not all decisions translate to a config override).
ORCHESTRATOR_KEY_MAP = {
    # MS1 gate
    "c4_in_main":          "stc.enable_colormap_inversion",
    "c4_in_ablation":      None,   # informational only; handled at ablation step
    "c4_dropped":          None,   # implies enable_colormap_inversion stays False
    # MS2 gate
    "c1_in_main":          None,   # combined with ssl_pretrained_backbone
    "c1_warmup_only":      None,
    "c1_dropped":          None,
    # generic
    "venue_fallback":      None,
}


def _apply_state_inheritance(cfg: dict, state_path: Path) -> list[str]:
    """Read orchestrator state.json::config_updates, translate, apply.

    Returns the list of (dotted_key, value) entries that were actually applied
    (for journal/debug). Anything in `config_updates` that isn't in
    ORCHESTRATOR_KEY_MAP is logged but ignored.
    """
    if not state_path.is_file():
        return []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    updates = state.get("config_updates", {}) or {}
    applied: list[tuple[str, object]] = []
    for k, v in updates.items():
        dotted = ORCHESTRATOR_KEY_MAP.get(k)
        if dotted is None:
            continue
        _set_dotted(cfg, dotted, v)
        applied.append((dotted, v))
    # Also: if SSL pretrained backbone was produced by MS2 D6, wire it.
    # The orchestrator writes nothing for this directly; we look at
    # step_results.ms2_d6_run_ssl_pretrain for a backbone_path key.
    ssl_step = (state.get("step_results", {})
                .get("ms2_d6_run_ssl_pretrain", {}))
    backbone_path = ssl_step.get("backbone_path")
    # Wire SSL backbone whenever D6 produced one, UNLESS the D8 gate explicitly
    # dropped C1 (c1_dropped). This is required so D8 -- the experiment that
    # *tests* SSL -- actually loads it: at D8 run time the gate output
    # (c1_in_main) does not exist yet, so gating on c1_in_main would silently
    # run D8 with no SSL init and invalidate the gate. Downstream steps (D12)
    # see c1_dropped after a FAIL and correctly skip it.
    if updates.get("c1_dropped"):
        # Gate dropped C1: force SSL off even if a config (e.g.
        # semi_supervised.yaml) hardcodes a backbone path. State overrides yaml.
        _set_dotted(cfg, "stc.ssl_pretrained_backbone", None)
        applied.append(("stc.ssl_pretrained_backbone", None))
    elif backbone_path:
        _set_dotted(cfg, "stc.ssl_pretrained_backbone", str(backbone_path))
        applied.append(("stc.ssl_pretrained_backbone", str(backbone_path)))
    return applied


def _resolve_data_yaml(
    args, project_root: Path, logger=None
) -> Path:
    """If --data-yaml provided use it; else auto-build from main_holdout splits."""
    if args.data_yaml:
        p = Path(args.data_yaml)
        if not p.is_file():
            raise FileNotFoundError(f"--data-yaml not found: {p}")
        return p
    from baselines.tier1_ultralytics.data_yaml import build_data_yaml
    splits_dir = project_root / "data" / "splits" / args.split
    if not splits_dir.is_dir():
        raise FileNotFoundError(
            f"split dir not found: {splits_dir}. "
            f"Available splits: {[p.name for p in (project_root / 'data' / 'splits').glob('*')]}"
        )
    out_yaml = (project_root / "results_v2" / "orchestrator"
                / "runs" / "_data_yaml" / "data.yaml")
    if not out_yaml.is_file():
        if logger:
            logger.info(f"auto-building data.yaml from {splits_dir}")
        # build_data_yaml materializes images/labels into the canonical
        # ultralytics tree under results_v2/baselines/tier1/ultralytics_data/
        build_data_yaml(splits_dir, out_yaml, project_root)
    return out_yaml


# --- CLI ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="path to training yaml (configs/default.yaml etc.)")
    p.add_argument("--seed", type=int, default=None,
                   help="override cfg.seed")
    p.add_argument("--run-name", type=str, default=None,
                   help="run subdir under results_v2/methods/stc_yolo/")
    p.add_argument("--results-dir", type=str, default=None,
                   help="explicit output dir; defaults to "
                        "results_v2/methods/stc_yolo/<run-name>")
    p.add_argument("--data-yaml", type=str, default=None,
                   help="ultralytics data yaml; if None, auto-build from "
                        "data/splits/<split>/")
    p.add_argument("--split", type=str, default="main_holdout",
                   help="split name under data/splits/ (default main_holdout)")
    p.add_argument("--override", action="append", default=[],
                   help="dotted-path cfg override key=value (repeatable)")
    p.add_argument("--inherit-config", type=str, default=None,
                   choices=[None, "from-state"],
                   help="if 'from-state', apply orchestrator state.json"
                        "::config_updates as overrides before --override")
    p.add_argument("--shared-check", action="store_true",
                   help="run per-image predict + shared.compute_detection_"
                        "metrics for dual mAP cross-check (slow, optional)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Load + override cfg
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    inherited: list[tuple[str, object]] = []
    if args.inherit_config == "from-state":
        state_path = (_PROJECT_ROOT / "results_v2" / "orchestrator"
                      / "state.json")
        inherited = _apply_state_inheritance(cfg, state_path)
    _apply_overrides(cfg, args.override)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 42))

    # 2. Output dir
    run_name = args.run_name or f"stc_seed{seed}"
    results_dir = (Path(args.results_dir) if args.results_dir
                   else (_PROJECT_ROOT / "results_v2" / "methods"
                         / "stc_yolo" / run_name))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — keeps `--help` cheap.
    from shared import set_seed, get_logger
    from methods.m6_stc_yolo.model import build_stc_yolo

    set_seed(seed)
    logger = get_logger("stc.train", log_file=str(results_dir / "train.log"))
    logger.info(f"run_name={run_name}  seed={seed}  results_dir={results_dir}")
    logger.info(f"config: {args.config}")
    if inherited:
        for k, v in inherited:
            logger.info(f"  inherited override: {k} = {v}")
    if args.override:
        for ov in args.override:
            logger.info(f"  cli override: {ov}")

    # 3. Resolve data yaml
    data_yaml = _resolve_data_yaml(args, _PROJECT_ROOT, logger=logger)
    logger.info(f"data_yaml: {data_yaml}")

    # 4. Build model (vanilla v12s for MS1 baseline; flags raise NotImpl
    #    until MS1 D2/D3/D3b and MS2 D7 / MS3 D10d are implemented).
    stc_cfg = cfg.get("stc", {}) or {}
    enable_c4 = bool(stc_cfg.get("enable_colormap_inversion", False))
    ssl_backbone = stc_cfg.get("ssl_pretrained_backbone")
    enable_film = bool(stc_cfg.get("enable_session_film", False))
    enable_mt = bool(stc_cfg.get("enable_mean_teacher", False))
    yolo = build_stc_yolo(
        enable_colormap_inversion=enable_c4,
        ssl_pretrained_backbone=ssl_backbone,
        enable_session_film=enable_film,
        enable_mean_teacher=enable_mt,
    )

    # 5. ultralytics train.  Strip our config-only keys.
    train_kwargs = {
        k: v for k, v in cfg.items()
        if k not in {"stc", "infer", "model", "mean_teacher", "loss_reweight",
                     "inherit", "phase"}
    }
    train_kwargs["data"] = str(data_yaml)
    train_kwargs["project"] = str(results_dir)
    train_kwargs["name"] = "train"
    train_kwargs["exist_ok"] = True
    # When C4 and/or SSL is on, we must pass a custom trainer subclass that
    # re-applies them after ultralytics' internal rebuild (paper3 N4 root
    # cause: rebuild silently drops the C4 swap; intersect_dicts likewise
    # drops the SSL backbone -- esp. the C4 1-ch stem mismatch).
    if enable_c4 or ssl_backbone or enable_film or enable_mt:
        from methods.m6_stc_yolo.model import make_stc_trainer
        mt_keys = ("mt_ema_momentum", "mt_pseudo_conf_threshold",
                   "mt_consistency_weight_max", "mt_consistency_rampup_epochs",
                   "mt_neighbor_gap_max")
        mt_cfg = {k: stc_cfg[k] for k in mt_keys if k in stc_cfg}
        # teacher_warmup_epochs lives in the top-level mean_teacher: section.
        mt_cfg["teacher_warmup_epochs"] = int(
            (cfg.get("mean_teacher", {}) or {}).get("teacher_warmup_epochs", 5))
        train_kwargs["trainer"] = make_stc_trainer(
            enable_colormap_inversion=enable_c4,
            ssl_pretrained_backbone=ssl_backbone,
            enable_session_film=enable_film,
            enable_mean_teacher=enable_mt,
            mt_cfg=mt_cfg,
        )
        logger.info(
            f"[train] using _STCDetectionTrainer (c4={enable_c4}, "
            f"ssl={'yes' if ssl_backbone else 'no'}, film={enable_film}, "
            f"mt={enable_mt}) — survives rebuild")
    logger.info(f"starting yolo.train(...) imgsz={train_kwargs.get('imgsz')} "
                f"batch={train_kwargs.get('batch')} epochs={train_kwargs.get('epochs')}")
    yolo.train(**train_kwargs)
    logger.info("train done")

    # 6. Reload best.pt for evaluation (use the saved checkpoint, not the
    #    current in-memory state). With C4 enabled, the swap is idempotent
    #    inside build_stc_yolo, so re-build from best.pt preserves C4.
    best_pt = results_dir / "train" / "weights" / "best.pt"
    if not best_pt.is_file():
        logger.warning(f"best.pt missing at {best_pt}; falling back to current")
        yolo_eval = yolo
    elif enable_c4 or enable_film:
        # Rebuild with the same STC wiring (idempotent swaps/injection). FiLM
        # must be re-injected so the saved film_gen weights load + the wrapped
        # forward_head is restored for eval.
        yolo_eval = build_stc_yolo(
            pretrained=best_pt,
            enable_colormap_inversion=enable_c4,
            enable_session_film=enable_film,
        )
    else:
        from ultralytics import YOLO
        yolo_eval = YOLO(str(best_pt))

    # 7. Test eval at infer config (paper3 tuned recipe: imgsz=1056 augment=True).
    infer_cfg = cfg.get("infer", {}) or {}
    val_kwargs = {
        "data": str(data_yaml),
        "split": "test",
        "imgsz": int(infer_cfg.get("imgsz", train_kwargs.get("imgsz", 800))),
        "augment": bool(infer_cfg.get("augment", False)),
        "conf": float(infer_cfg.get("conf", 0.001)),
        "iou": float(infer_cfg.get("iou", 0.6)),
        "project": str(results_dir),
        "name": "val_test",
        "exist_ok": True,
        "plots": False,
        "verbose": False,
    }
    if enable_film:
        # FiLM-aware validator threads per-image metadata onto the head so the
        # reported test metric actually uses FiLM (not the identity fallback).
        from methods.m6_stc_yolo.model import make_film_validator
        val_kwargs["validator"] = make_film_validator()
    logger.info(f"starting yolo.val test: imgsz={val_kwargs['imgsz']} "
                f"augment={val_kwargs['augment']} film={enable_film}")
    res = yolo_eval.val(**val_kwargs)

    # 8. Extract metrics in orchestrator-compatible JSON.
    ap50 = res.box.ap50 if hasattr(res.box, "ap50") else []
    # Full per-class AP50 (works for any dataset/#classes, e.g. HIT-UAV's 5),
    # mapped name->AP50 via ap_class_index. Non-breaking: Water/Oil kept below.
    _names = getattr(res, "names", {}) or {}
    _apidx = getattr(res.box, "ap_class_index", list(range(len(ap50))))
    per_class_ap50 = {str(_names.get(int(c), int(c))): float(ap50[i])
                      for i, c in enumerate(_apidx)} if len(ap50) else {}
    out = {
        "mAP50": float(res.box.map50),
        "mAP50_95": float(res.box.map),
        "per_class_ap50": per_class_ap50,
        "Water_AP50": float(ap50[0]) if len(ap50) > 0 else None,
        "Oil_AP50": float(ap50[1]) if len(ap50) > 1 else None,
        "precision": float(res.box.mp),
        "recall": float(res.box.mr),
        "seed": seed,
        "run_name": run_name,
        "config_path": str(args.config),
        "data_yaml": str(data_yaml),
        "infer_imgsz": val_kwargs["imgsz"],
        "infer_augment": val_kwargs["augment"],
        "overrides_applied": args.override + [f"{k}={v}" for k, v in inherited],
        "stc_config": stc_cfg,
    }
    metrics_path = results_dir / "ultralytics_test_metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(
        f"test mAP50={out['mAP50']:.4f}  mAP50:95={out['mAP50_95']:.4f}  "
        f"P={out['precision']:.4f}  R={out['recall']:.4f}  "
        f"Oil_AP50={out['Oil_AP50']}"
    )
    logger.info(f"wrote {metrics_path}")

    # 9. Optional dual-mAP shared cross-check.
    if args.shared_check:
        try:
            from shared import compute_detection_metrics
            # Per-image predict + cross-check. Heavy — only for headline runs.
            # Implementation mirrors tg_yolo/train_one.py::predict_for_shared
            # but kept minimal here.
            import numpy as np
            import cv2
            data_info = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            data_root = Path(data_info["path"])
            test_img_dir = data_root / data_info["test"]
            test_lbl_dir = data_root / data_info["test"].replace("images", "labels")
            preds, gts = [], []
            paths = sorted(p for p in test_img_dir.iterdir()
                           if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
            for i, ip in enumerate(paths):
                r = yolo_eval.predict(source=str(ip),
                                      imgsz=val_kwargs["imgsz"],
                                      conf=val_kwargs["conf"],
                                      iou=val_kwargs["iou"],
                                      augment=val_kwargs["augment"],
                                      device=0, verbose=False,
                                      save=False, stream=False)[0]
                try:
                    H, W = int(r.orig_shape[0]), int(r.orig_shape[1])
                except Exception:
                    im = cv2.imread(str(ip)); H, W = im.shape[:2]
                if r.boxes is None or len(r.boxes) == 0:
                    boxes = np.zeros((0, 4), dtype=np.float32)
                    scores = np.zeros((0,), dtype=np.float32)
                    labels = np.zeros((0,), dtype=np.int64)
                else:
                    boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
                    labels = r.boxes.cls.detach().cpu().numpy().astype(np.int64)
                preds.append({"image_id": i, "image_path": str(ip),
                              "boxes": boxes, "scores": scores, "labels": labels})
                # parse YOLO label -> abs xyxy
                lbl = test_lbl_dir / f"{ip.stem}.txt"
                gboxes, glabels = [], []
                if lbl.is_file():
                    for raw in lbl.read_text(encoding="utf-8").splitlines():
                        parts = raw.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(float(parts[0]))
                        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                        gboxes.append([(cx - bw / 2) * W, (cy - bh / 2) * H,
                                       (cx + bw / 2) * W, (cy + bh / 2) * H])
                        glabels.append(cls)
                gts.append({
                    "image_id": i,
                    "boxes": np.asarray(gboxes, dtype=np.float32).reshape(-1, 4),
                    "labels": np.asarray(glabels, dtype=np.int64).reshape(-1),
                    "orig_size": (H, W),
                })
            shared_m = compute_detection_metrics(preds, gts, num_classes=2,
                                                 class_names=("Water", "Oil"))
            rel_diff = (abs(out["mAP50"] - shared_m["mAP50"])
                        / max(abs(out["mAP50"]), 1e-6))
            shared_out = {
                "ultralytics_mAP50": out["mAP50"],
                "shared_mAP50": shared_m["mAP50"],
                "abs_diff": abs(out["mAP50"] - shared_m["mAP50"]),
                "rel_diff": rel_diff,
                "agree_within_5pct": bool(rel_diff < 0.05),
                "shared_full": shared_m,
            }
            (results_dir / "shared_test_metrics.json").write_text(
                json.dumps(shared_out, indent=2), encoding="utf-8")
            logger.info(f"shared mAP50={shared_m['mAP50']:.4f} "
                        f"rel_diff={rel_diff:.4f}")
        except Exception as e:
            logger.warning(f"shared-check failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
