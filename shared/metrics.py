"""检测指标 (mAP/AP per class/P/R/F1), 基于 faster_coco_eval。

设计:
  * 用 COCO 字典内存构造 cocoGt, 再用 loadRes() 包预测形成 cocoDt。
  * P/R/F1: 取每类各置信度阈值下的 TP/FP/FN, 在 conf>=conf_thresh & IoU>=0.5 下统计;
    返回的是 *summary* 的 P/R/F1 (按 score 阈值 0.5, 经典操作), 非 COCO 的 mAP recall。
  * 空 gt 或空 pred 不报错, 直接返回 0 / nan-safe。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster

ArrayLike = Union[np.ndarray, torch.Tensor, list]


def _to_np(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    out = boxes.copy().astype(np.float64)
    out[:, 2] = out[:, 2] - out[:, 0]
    out[:, 3] = out[:, 3] - out[:, 1]
    return out


def _build_coco_gt(gts: Sequence[Dict[str, Any]], num_classes: int, class_names: Sequence[str]) -> COCO:
    """从 gts 列表 (dict per image) 构造内存 COCO 对象。"""
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1
    for g in gts:
        image_id = int(g["image_id"])
        boxes_xyxy = _to_np(g["boxes"]).reshape(-1, 4).astype(np.float64)
        labels = _to_np(g["labels"]).astype(np.int64).reshape(-1)
        H, W = g.get("orig_size", (1024, 1024))
        images.append({"id": image_id, "height": int(H), "width": int(W),
                       "file_name": f"img_{image_id}.jpg"})
        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)
        for box, cls in zip(boxes_xywh, labels):
            x, y, w, h = [float(v) for v in box]
            area = max(0.0, w * h)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls) + 1,  # COCO category ids are 1-based
                "bbox": [x, y, w, h],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(class_names)]
    dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    return coco


def _pred_to_coco_results(preds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in preds:
        image_id = int(p["image_id"])
        boxes_xyxy = _to_np(p["boxes"]).reshape(-1, 4).astype(np.float64)
        scores = _to_np(p["scores"]).astype(np.float64).reshape(-1)
        labels = _to_np(p["labels"]).astype(np.int64).reshape(-1)
        boxes_xywh = _xyxy_to_xywh(boxes_xyxy)
        for box, sc, cls in zip(boxes_xywh, scores, labels):
            x, y, w, h = [float(v) for v in box]
            out.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, w, h],
                "score": float(sc),
            })
    return out


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4), return IoU (N,M)。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _compute_prf1(preds: Sequence[Dict[str, Any]],
                  gts: Sequence[Dict[str, Any]],
                  iou_thr: float = 0.5,
                  conf_thr: float = 0.5) -> Dict[str, float]:
    """简单的 P/R/F1: 同类 IoU>=iou_thr 配对, conf>=conf_thr 后再算。"""
    tp = fp = fn = 0
    gt_by_img = {int(g["image_id"]): g for g in gts}
    pred_by_img = {int(p["image_id"]): p for p in preds}
    all_ids = set(gt_by_img.keys()) | set(pred_by_img.keys())
    for iid in all_ids:
        g = gt_by_img.get(iid, {"boxes": np.zeros((0, 4)), "labels": np.zeros((0,), dtype=np.int64)})
        p = pred_by_img.get(iid, {"boxes": np.zeros((0, 4)), "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=np.int64)})
        gb = _to_np(g["boxes"]).reshape(-1, 4)
        gl = _to_np(g["labels"]).astype(np.int64).reshape(-1)
        pb = _to_np(p["boxes"]).reshape(-1, 4)
        ps = _to_np(p["scores"]).astype(np.float64).reshape(-1)
        pl = _to_np(p["labels"]).astype(np.int64).reshape(-1)
        keep = ps >= conf_thr
        pb, ps, pl = pb[keep], ps[keep], pl[keep]
        if gb.shape[0] == 0:
            fp += pb.shape[0]
            continue
        if pb.shape[0] == 0:
            fn += gb.shape[0]
            continue
        ious = _box_iou_xyxy(pb, gb)
        # 同类掩码
        same_cls = pl[:, None] == gl[None, :]
        ious = np.where(same_cls, ious, 0.0)
        matched_gt = set()
        order = np.argsort(-ps)
        for pi in order:
            best = -1
            best_iou = iou_thr
            for gi in range(gb.shape[0]):
                if gi in matched_gt:
                    continue
                if ious[pi, gi] >= best_iou:
                    best_iou = ious[pi, gi]
                    best = gi
            if best >= 0:
                tp += 1
                matched_gt.add(best)
            else:
                fp += 1
        fn += gb.shape[0] - len(matched_gt)
    p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
    return {"P": float(p_val), "R": float(r_val), "F1": float(f1),
            "TP": int(tp), "FP": int(fp), "FN": int(fn)}


def compute_detection_metrics(
    preds: Sequence[Dict[str, Any]],
    gts: Sequence[Dict[str, Any]],
    num_classes: int = 2,
    class_names: Sequence[str] = ("Water", "Oil"),
) -> Dict[str, Any]:
    """COCO mAP + 简单 P/R/F1。"""
    coco_gt = _build_coco_gt(gts, num_classes, class_names)
    pred_anns = _pred_to_coco_results(preds)

    if not pred_anns:
        # 空预测: COCO mAP 直接 0
        zeros = {"mAP50": 0.0, "mAP50_95": 0.0,
                 "mAP_small": 0.0, "mAP_medium": 0.0, "mAP_large": 0.0,
                 "per_class": {name: {"AP50": 0.0, "AP50_95": 0.0} for name in class_names},
                 **_compute_prf1(preds, gts)}
        return zeros

    coco_dt = coco_gt.loadRes(pred_anns)
    ev = COCOeval_faster(coco_gt, coco_dt, iouType="bbox", extra_calc=True)
    ev.params.imgIds = sorted([int(g["image_id"]) for g in gts])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = ev.stats
    out: Dict[str, Any] = {
        "mAP50_95": float(s[0]),
        "mAP50": float(s[1]),
        "mAP_small": float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large": float(s[5]),
    }

    # per-class AP: precision shape (T, R, K, A, M)
    precision = ev.eval["precision"]  # (T, R, K, A, M)
    per_class: Dict[str, Dict[str, float]] = {}
    iou_thrs = ev.params.iouThrs
    idx50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    for k, name in enumerate(class_names):
        ap50 = precision[idx50, :, k, 0, -1]
        ap50 = ap50[ap50 > -1]
        ap5095 = precision[:, :, k, 0, -1]
        ap5095 = ap5095[ap5095 > -1]
        per_class[name] = {
            "AP50": float(ap50.mean()) if ap50.size > 0 else 0.0,
            "AP50_95": float(ap5095.mean()) if ap5095.size > 0 else 0.0,
        }
    out["per_class"] = per_class

    out.update(_compute_prf1(preds, gts))
    return out


def save_metrics_json(metrics: Dict[str, Any], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    np.random.seed(0)

    # 合成 5 张图, 每图 1-3 个 box (随机), 类别 0/1
    gts: List[Dict[str, Any]] = []
    for iid in range(5):
        n = int(np.random.randint(1, 4))
        boxes = []
        for _ in range(n):
            x1 = float(np.random.uniform(10, 400))
            y1 = float(np.random.uniform(10, 400))
            w = float(np.random.uniform(30, 100))
            h = float(np.random.uniform(30, 100))
            boxes.append([x1, y1, x1 + w, y1 + h])
        labels = np.random.randint(0, 2, size=n).tolist()
        gts.append({"image_id": iid, "boxes": np.array(boxes, dtype=np.float32),
                    "labels": np.array(labels, dtype=np.int64), "orig_size": (480, 640)})

    # sanity 1: gt 作为 pred
    preds_perfect = [{"image_id": g["image_id"], "boxes": g["boxes"].copy(),
                      "scores": np.ones((len(g["labels"]),), dtype=np.float32),
                      "labels": g["labels"].copy()} for g in gts]
    m1 = compute_detection_metrics(preds_perfect, gts)
    print(f"[metrics] sanity1 perfect: mAP50={m1['mAP50']:.4f} mAP50_95={m1['mAP50_95']:.4f} "
          f"F1={m1['F1']:.4f}")
    assert m1["mAP50"] > 0.99, m1["mAP50"]

    # sanity 2: 空预测
    preds_empty = [{"image_id": g["image_id"], "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,)), "labels": np.zeros((0,), dtype=np.int64)} for g in gts]
    m2 = compute_detection_metrics(preds_empty, gts)
    print(f"[metrics] sanity2 empty:   mAP50={m2['mAP50']:.4f} F1={m2['F1']:.4f}")
    assert m2["mAP50"] == 0.0

    # sanity 3: 加微小扰动
    preds_jitter = []
    for g in gts:
        b = g["boxes"].copy() + np.random.uniform(-2, 2, size=g["boxes"].shape).astype(np.float32)
        preds_jitter.append({"image_id": g["image_id"], "boxes": b,
                             "scores": np.ones((len(g["labels"]),), dtype=np.float32) * 0.9,
                             "labels": g["labels"].copy()})
    m3 = compute_detection_metrics(preds_jitter, gts)
    print(f"[metrics] sanity3 jitter:  mAP50={m3['mAP50']:.4f} F1={m3['F1']:.4f}")
    assert m3["mAP50"] > 0.8, m3["mAP50"]

    out_path = save_metrics_json(m3, "results_v2/sanity_check/_metrics_jitter.json")
    print(f"[metrics] saved {out_path}")
    print("[metrics] OK")
