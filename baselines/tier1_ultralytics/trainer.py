"""UltralyticsTrainer: 训练 / 验证 / 测试 / 导出标准输出。

输出目录结构 (output_dir 下):
  weights/{best.pt, last.pt}
  training_log.csv                # 复制自 ultralytics results.csv (列名保持原样, 已通用)
  metrics_val.json
  metrics_test.json               # 含 ultralytics + shared 双 mAP + consistency
  predictions_test.json
  confusion_matrix.png
  pr_curve.png
  f1_curve.png
  complexity.json
  config_snapshot.yaml
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared import (  # noqa: E402
    compute_detection_metrics,
    save_metrics_json,
    measure_complexity,
)


def _is_rtdetr(model_name: str) -> bool:
    return "rtdetr" in model_name.lower() or "rt-detr" in model_name.lower()


def _load_ultralytics_model(model_weight: str, is_rtdetr: bool):
    if is_rtdetr:
        from ultralytics import RTDETR
        return RTDETR(model_weight)
    from ultralytics import YOLO
    return YOLO(model_weight)


def _read_yolo_label_file(p: Path, W: int, H: int):
    if not p.is_file():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes: List[List[float]] = []
    labels: List[int] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        x1 = (cx - w / 2.0) * W
        y1 = (cy - h / 2.0) * H
        x2 = (cx + w / 2.0) * W
        y2 = (cy + h / 2.0) * H
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class UltralyticsTrainer:
    def __init__(
        self,
        model_name: str,
        data_yaml: Path,
        output_dir: Path,
        train_cfg: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_name = model_name
        self.data_yaml = Path(data_yaml)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = dict(train_cfg)
        self.logger = logger or logging.getLogger("UltralyticsTrainer")
        self.is_rtdetr = _is_rtdetr(model_name)
        self.model_weight = self.cfg.pop("model")
        self._train_run_dir: Optional[Path] = None
        self.model = None

    # ------------------------------------------------------------------ train
    def train(self) -> Path:
        self.logger.info(f"[train] loading model: {self.model_weight}")
        self.model = _load_ultralytics_model(self.model_weight, self.is_rtdetr)

        train_kwargs = dict(self.cfg)
        train_kwargs.update(
            data=str(self.data_yaml),
            project=str(self.output_dir),
            name="train",
            exist_ok=True,
        )
        # RT-DETR does not accept some YOLO-only augment knobs in some versions;
        # we still pass them — ultralytics ignores unknowns silently.
        self.logger.info(f"[train] kwargs: epochs={train_kwargs.get('epochs')} "
                         f"batch={train_kwargs.get('batch')} imgsz={train_kwargs.get('imgsz')}")
        self.model.train(**train_kwargs)

        self._train_run_dir = self.output_dir / "train"
        best = self._train_run_dir / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError(f"best.pt not produced: {best}")
        self.logger.info(f"[train] done. best={best}")
        return best

    # --------------------------------------------------------------- validate
    def validate(self, split: str = "val") -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("must call train() first or set self.model")
        self.logger.info(f"[val] split={split}")
        results = self.model.val(data=str(self.data_yaml), split=split,
                                 project=str(self.output_dir),
                                 name=f"val_{split}", exist_ok=True,
                                 plots=True, verbose=False)
        return self._extract_ultralytics_metrics(results)

    # ------------------------------------------------------------------ test
    def test(self) -> Dict[str, Any]:
        """跑 test set: ultralytics mAP + shared mAP + consistency。

        断点续跑: 若 output_dir/metrics_test_ultralytics.json 已存在, 跳过 step 1
        (val_test 那 17s); 若 predictions_test.json 已存在, 跳过 step 2 (predict)。
        """
        if self.model is None:
            raise RuntimeError("must call train() first")

        # 1) ultralytics 自带 mAP — 若已 cache 直接复用
        ul_cache = self.output_dir / "metrics_test_ultralytics.json"
        if ul_cache.is_file():
            ul_metrics = json.loads(ul_cache.read_text(encoding="utf-8"))
            self.logger.info(f"[test] reuse cached ultralytics metrics: {ul_cache.name}")
        else:
            self.logger.info("[test] running model.val on test split")
            ul_results = self.model.val(data=str(self.data_yaml), split="test",
                                        project=str(self.output_dir),
                                        name="val_test", exist_ok=True,
                                        plots=True, verbose=False)
            ul_metrics = self._extract_ultralytics_metrics(ul_results)
            ul_cache.write_text(
                json.dumps(ul_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # 2) shared mAP: 自己 predict + compute_detection_metrics
        self.logger.info("[test] running predict on test images for shared metrics")
        preds, gts = self._predict_test_for_shared()
        shared_metrics = compute_detection_metrics(preds, gts, num_classes=2,
                                                   class_names=("Water", "Oil"))

        # 3) consistency
        ul_map50 = ul_metrics.get("mAP50", 0.0) or 0.0
        sh_map50 = shared_metrics.get("mAP50", 0.0) or 0.0
        denom = max(abs(ul_map50), 1e-6)
        diff_rel = abs(ul_map50 - sh_map50) / denom
        consistency = {
            "ultralytics_mAP50": ul_map50,
            "shared_mAP50": sh_map50,
            "abs_diff": abs(ul_map50 - sh_map50),
            "rel_diff": diff_rel,
            "agree_within_5pct": bool(diff_rel < 0.05),
        }

        metrics_test = {
            "ultralytics": ul_metrics,
            "shared": shared_metrics,
            "consistency": consistency,
            # convenience top-level (for orchestrator fallback)
            "mAP50": ul_map50,
            "mAP50_95": ul_metrics.get("mAP50_95", 0.0),
        }
        # save predictions_test.json
        pred_dump = [
            {
                "image_id": int(p["image_id"]),
                "image_path": p["image_path"],
                "boxes": p["boxes"].tolist(),
                "scores": p["scores"].tolist(),
                "labels": p["labels"].tolist(),
            }
            for p in preds
        ]
        (self.output_dir / "predictions_test.json").write_text(
            json.dumps(pred_dump, ensure_ascii=False), encoding="utf-8"
        )
        save_metrics_json(metrics_test, self.output_dir / "metrics_test.json")
        return metrics_test

    # ------------------------------------------------------------------ export
    def export_standard_outputs(self, train_cfg_full: Dict[str, Any]) -> None:
        """复制 weights / plots, 写 complexity, 写 config_snapshot."""
        run_dir = self._train_run_dir
        if run_dir is None or not run_dir.is_dir():
            run_dir = self.output_dir / "train"

        # weights
        src_w = run_dir / "weights"
        dst_w = self.output_dir / "weights"
        dst_w.mkdir(parents=True, exist_ok=True)
        for fname in ("best.pt", "last.pt"):
            src = src_w / fname
            if src.is_file():
                shutil.copyfile(src, dst_w / fname)

        # training_log.csv
        src_csv = run_dir / "results.csv"
        if src_csv.is_file():
            shutil.copyfile(src_csv, self.output_dir / "training_log.csv")

        # plots: ultralytics 8.x writes BoxPR_curve.png / BoxF1_curve.png in train/val dirs.
        # Older naming PR_curve.png / F1_curve.png also supported as fallback.
        plot_map = {
            "confusion_matrix.png": ["confusion_matrix.png"],
            "pr_curve.png": ["BoxPR_curve.png", "PR_curve.png"],
            "f1_curve.png": ["BoxF1_curve.png", "F1_curve.png"],
        }
        search_bases = [
            self.output_dir / "val_test",   # prefer test-time plots
            self.output_dir / "val_val",
            run_dir,
            self.output_dir,
        ]
        for dst_name, candidates in plot_map.items():
            dst = self.output_dir / dst_name
            if dst.is_file():
                continue
            found = False
            for base in search_bases:
                if found:
                    break
                for fname in candidates:
                    src = base / fname
                    if src.is_file():
                        shutil.copyfile(src, dst)
                        found = True
                        break

        # complexity.json — load best.pt and measure
        try:
            best = dst_w / "best.pt"
            if best.is_file():
                model = _load_ultralytics_model(str(best), self.is_rtdetr)
                inner = model.model  # nn.Module
                comp = measure_complexity(inner, input_shape=(1, 3, 640, 640),
                                          device="cuda", warmup=5, n_iter=20)
                comp["model_name"] = self.model_name
                (self.output_dir / "complexity.json").write_text(
                    json.dumps(comp, indent=2), encoding="utf-8"
                )
        except Exception as e:
            self.logger.warning(f"[export] complexity measure failed: {e}")
            (self.output_dir / "complexity.json").write_text(
                json.dumps({"error": str(e), "model_name": self.model_name}, indent=2),
                encoding="utf-8",
            )

        # config_snapshot.yaml
        snap = {"model_name": self.model_name, "data_yaml": str(self.data_yaml),
                "train_cfg": train_cfg_full}
        (self.output_dir / "config_snapshot.yaml").write_text(
            yaml.safe_dump(snap, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    # ============================================================ internals
    def _extract_ultralytics_metrics(self, results) -> Dict[str, Any]:
        """从 ultralytics val 返回对象提取 mAP / per-class AP / P / R / F1。"""
        out: Dict[str, Any] = {}
        try:
            box = results.box
            out["mAP50"] = float(box.map50)
            out["mAP50_95"] = float(box.map)
            # per-class
            class_names = ["Water", "Oil"]
            per_class = {}
            try:
                ap50 = box.ap50  # (nc,)
                ap = box.ap      # (nc, 10)
                for i, name in enumerate(class_names):
                    per_class[name] = {
                        "AP50": float(ap50[i]) if i < len(ap50) else float("nan"),
                        "AP50_95": float(ap[i].mean()) if i < len(ap) else float("nan"),
                    }
            except Exception:
                pass
            out["per_class"] = per_class
            try:
                out["P"] = float(box.mp)
                out["R"] = float(box.mr)
                p, r = out["P"], out["R"]
                out["F1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            except Exception:
                pass
        except Exception as e:
            out["error"] = f"extract failed: {e}"
        return out

    def _predict_test_for_shared(self):
        """对 test 集所有图跑 predict (基于内存目录), 拼成 shared.compute 输入格式."""
        # parse data.yaml to get test images dir
        info = yaml.safe_load(self.data_yaml.read_text(encoding="utf-8"))
        data_root = Path(info["path"])
        test_img_dir = data_root / info["test"]
        test_lbl_dir = data_root / info["test"].replace("images", "labels")
        if not test_img_dir.is_dir():
            raise FileNotFoundError(f"test images dir not found: {test_img_dir}")

        img_paths = sorted(
            [p for p in test_img_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        )
        self.logger.info(f"[test] predicting {len(img_paths)} test images (per-image)")

        # 实测 ultralytics `model.predict(source=list, stream=True)` 在 RTDETR + list
        # source 下不真正流式: 第一次 next() 前在内部把所有图跑完并累积 Results, 内存
        # 涨到 18GB+ page file 耗尽。改成逐图独立 predict, 每张算完立即转 numpy + del,
        # 内存稳定在几百 MB。imgsz/conf/iou 等参数固定, ultralytics 内部会缓存预处理。
        imgsz = int(self.cfg.get("imgsz", 640))
        device = self.cfg.get("device", 0)

        preds: List[Dict[str, Any]] = []
        gts: List[Dict[str, Any]] = []
        log_every = max(1, len(img_paths) // 10)
        for i, img_path in enumerate(img_paths):
            r_list = self.model.predict(
                source=str(img_path),
                imgsz=imgsz, conf=0.001, iou=0.6,
                device=device, verbose=False, save=False, stream=False,
            )
            r = r_list[0]
            try:
                H, W = int(r.orig_shape[0]), int(r.orig_shape[1])
            except Exception:
                import cv2
                im = cv2.imread(str(img_path))
                H, W = im.shape[:2]

            b = r.boxes
            if b is None or len(b) == 0:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                boxes = b.xyxy.detach().cpu().numpy().astype(np.float32)
                scores = b.conf.detach().cpu().numpy().astype(np.float32)
                labels = b.cls.detach().cpu().numpy().astype(np.int64)
            preds.append({
                "image_id": i, "image_path": str(img_path),
                "boxes": boxes, "scores": scores, "labels": labels,
            })

            lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
            gb, gl = _read_yolo_label_file(lbl_path, W, H)
            gts.append({
                "image_id": i, "boxes": gb, "labels": gl,
                "orig_size": (H, W),
            })

            del r, r_list
            if (i + 1) % log_every == 0 or (i + 1) == len(img_paths):
                self.logger.info(f"[test] predict progress: {i + 1}/{len(img_paths)}")

        return preds, gts
