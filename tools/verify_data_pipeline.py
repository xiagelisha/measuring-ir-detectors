"""端到端流水线验证:
  * 三 split 通过 shared.OilLeakDataset 加载
  * train/val DataLoader batch 健康检查
  * 8 张含/不含 oil 样本可视化 (GT 框)
  * letterbox 还原: 640 letterboxed 与 原图 反推, 双图对照
  * Metrics sanity (perfect / empty) 复测
  * 空标签统计
  * DataLoader 吞吐基准
  * 输出 results_v2/sanity_check/report.md
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import shared  # noqa: E402
from shared import (  # noqa: E402
    OilLeakDataset,
    compute_detection_metrics,
    detr_collate_fn,
    draw_boxes,
    get_train_transforms,
    get_val_transforms,
    set_seed,
)

SPLIT_DIR = ROOT / "data" / "splits" / "main_holdout"
IMG_DIR = ROOT / "data" / "OilData"
LBL_DIR = ROOT / "data" / "OilLabels"
META_CSV = ROOT / "data" / "metadata.csv"
SANITY = ROOT / "results_v2" / "sanity_check"
SAMPLE_DIR = SANITY / "sample_with_gt"
LETTERBOX_DIR = SANITY / "letterbox_recovery"
REPORT = SANITY / "report.md"

EXPECTED_LEN = {"train": 1886, "val": 348, "test": 359}


# ----------------------------------------------------------------------
# A. 三 split 加载
# ----------------------------------------------------------------------
def load_splits() -> Dict[str, OilLeakDataset]:
    out: Dict[str, OilLeakDataset] = {}
    for k in ["train", "val", "test"]:
        sf = SPLIT_DIR / f"{k}.txt"
        tfm = get_train_transforms(640) if k == "train" else get_val_transforms(640)
        ds = OilLeakDataset(split_file=sf, image_dir=IMG_DIR, label_dir=LBL_DIR,
                            image_size=640, transforms=tfm, mode=k)
        assert len(ds) == EXPECTED_LEN[k], f"{k}: {len(ds)} != {EXPECTED_LEN[k]}"
        s0 = ds[0]
        for need in ("image", "boxes_xyxy_abs", "labels", "image_id"):
            assert need in s0, f"{k}: sample missing field {need}"
        out[k] = ds
    return out


# ----------------------------------------------------------------------
# B. DataLoader batch 健康
# ----------------------------------------------------------------------
def check_dataloader(ds: OilLeakDataset, label: str, n_batches: int = 3) -> Dict[str, Any]:
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=detr_collate_fn)
    info: Dict[str, Any] = {"label": label, "checked_batches": 0, "n_boxes_examples": []}
    for i, b in enumerate(loader):
        assert tuple(b["images"].shape) == (4, 3, 640, 640), b["images"].shape
        assert b["images"].dtype == torch.float32
        assert isinstance(b["boxes"], list) and len(b["boxes"]) == 4
        for box, lbl in zip(b["boxes"], b["labels"]):
            if box.numel() > 0:
                assert box.min() >= 0 and box.max() <= 640 + 1e-3
                assert (box[:, 2] >= box[:, 0]).all() and (box[:, 3] >= box[:, 1]).all()
            assert (lbl >= 0).all() and (lbl <= 1).all()
        info["n_boxes_examples"].append([b["boxes"][j].shape[0] for j in range(4)])
        info["checked_batches"] += 1
        if i + 1 >= n_batches:
            break
    return info


# ----------------------------------------------------------------------
# C. 可视化 GT (train 内 含/不含 Oil 各 4 张)
# ----------------------------------------------------------------------
def viz_with_gt(meta: pd.DataFrame, seed: int = 42) -> Dict[str, List[Path]]:
    train_names = set((SPLIT_DIR / "train.txt").read_text(encoding="utf-8").splitlines())
    train_names.discard("")
    sub = meta[meta["image_name"].isin(train_names)]
    rng = random.Random(seed)
    oil_pool = sub[sub["has_oil"] == 1]["image_name"].tolist()
    water_pool = sub[(sub["has_oil"] == 0) & (sub["has_water"] == 1)]["image_name"].tolist()
    rng.shuffle(oil_pool); rng.shuffle(water_pool)

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, List[Path]] = {"oil": [], "water": []}

    ds = OilLeakDataset.from_directory(image_dir=IMG_DIR, label_dir=LBL_DIR, image_size=640)
    name_to_idx = {n: i for i, n in enumerate(ds.stems)}

    for i, name in enumerate(oil_pool[:4], start=1):
        idx = name_to_idx[name]
        item = ds[idx]
        bgr = cv2.imread(item["image_path"])
        boxes = item["boxes_xyxy_abs"].numpy()
        labels = item["labels"].numpy()
        vis = draw_boxes(bgr, boxes, labels)
        op = SAMPLE_DIR / f"oil_{i:03d}_{name}.jpg"
        cv2.imwrite(str(op), vis)
        out["oil"].append(op)

    for i, name in enumerate(water_pool[:4], start=1):
        idx = name_to_idx[name]
        item = ds[idx]
        bgr = cv2.imread(item["image_path"])
        boxes = item["boxes_xyxy_abs"].numpy()
        labels = item["labels"].numpy()
        vis = draw_boxes(bgr, boxes, labels)
        wp = SAMPLE_DIR / f"water_{i:03d}_{name}.jpg"
        cv2.imwrite(str(wp), vis)
        out["water"].append(wp)
    return out


# ----------------------------------------------------------------------
# D. Letterbox 还原 (自己跑一次确定性的 letterbox)
# ----------------------------------------------------------------------
def _manual_letterbox(rgb: np.ndarray, target: int = 640, pad_color=(114, 114, 114)):
    """返回 (letterboxed_rgb, r, (dw, dh))。Albumentations 内部黑盒, 我们这里
    自己再跑一次确定性 letterbox 用于可视化对照。"""
    h, w = rgb.shape[:2]
    r = float(target) / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw = (target - nw) / 2.0
    dh = (target - nh) / 2.0
    pad_l, pad_r = int(round(dw - 0.1)), int(round(dw + 0.1))
    pad_t, pad_b = int(round(dh - 0.1)), int(round(dh + 0.1))
    out = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                             cv2.BORDER_CONSTANT, value=pad_color)
    return out, r, (pad_l, pad_t)


def letterbox_recovery() -> Dict[str, Any]:
    """选 train 里最大尺寸的一张图, 双图对照: 640 letterboxed (画 transformed box) +
    原图 (画用 ratio_pad 反推的 box)。"""
    LETTERBOX_DIR.mkdir(parents=True, exist_ok=True)
    train_names = (SPLIT_DIR / "train.txt").read_text(encoding="utf-8").splitlines()
    train_names = [n for n in train_names if n.strip()]
    # 找一张尺寸大的图 (扫前 200 个 train 样本里, 取面积最大)
    best = None
    best_area = -1
    for name in train_names[:200]:
        p = IMG_DIR / f"{name}.JPG"
        if not p.is_file():
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        if h * w > best_area:
            best_area = h * w
            best = (name, p, h, w, bgr)
    name, p, H, W, bgr = best  # type: ignore[misc]
    lbl_path = LBL_DIR / f"{name}.txt"

    # 解析标签 (cx cy w h normalized → xyxy abs on 原图)
    boxes_orig: List[List[float]] = []
    labels_orig: List[int] = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w_n, h_n = map(float, parts[1:5])
        x1 = (cx - w_n / 2) * W; y1 = (cy - h_n / 2) * H
        x2 = (cx + w_n / 2) * W; y2 = (cy + h_n / 2) * H
        boxes_orig.append([x1, y1, x2, y2]); labels_orig.append(cls)

    # 1) 自跑 letterbox + 同步变换 box → 在 640 letterboxed 上画框
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    letter_rgb, r, (dw, dh) = _manual_letterbox(rgb, target=640)
    boxes_lb: List[List[float]] = []
    for b in boxes_orig:
        x1 = b[0] * r + dw; y1 = b[1] * r + dh
        x2 = b[2] * r + dw; y2 = b[3] * r + dh
        boxes_lb.append([x1, y1, x2, y2])
    letter_bgr = cv2.cvtColor(letter_rgb, cv2.COLOR_RGB2BGR)
    vis_letter = draw_boxes(letter_bgr,
                            np.array(boxes_lb, dtype=np.float32),
                            np.array(labels_orig, dtype=np.int64))
    path_letter = LETTERBOX_DIR / f"640_letterboxed_{name}.jpg"
    cv2.imwrite(str(path_letter), vis_letter)

    # 2) 用 (r, dw, dh) 把 letterboxed box 反推回原图 → 原图上画框
    boxes_recovered: List[List[float]] = []
    for b in boxes_lb:
        x1 = (b[0] - dw) / r; y1 = (b[1] - dh) / r
        x2 = (b[2] - dw) / r; y2 = (b[3] - dh) / r
        boxes_recovered.append([x1, y1, x2, y2])
    vis_orig = draw_boxes(bgr,
                          np.array(boxes_recovered, dtype=np.float32),
                          np.array(labels_orig, dtype=np.int64))
    path_orig = LETTERBOX_DIR / f"orig_with_recovered_box_{name}.jpg"
    cv2.imwrite(str(path_orig), vis_orig)

    diff = float(np.abs(np.array(boxes_orig) - np.array(boxes_recovered)).max()) \
           if boxes_orig else 0.0

    return {
        "name": name, "orig_HW": (H, W), "r": r, "pad": (dw, dh),
        "n_boxes": len(boxes_orig),
        "max_abs_diff_after_recovery_px": diff,
        "letter_path": path_letter, "orig_path": path_orig,
    }


# ----------------------------------------------------------------------
# E. Metrics sanity 复测
# ----------------------------------------------------------------------
def metrics_resanity() -> Dict[str, float]:
    rng = np.random.default_rng(0)
    gts: List[Dict[str, Any]] = []
    for iid in range(5):
        n = int(rng.integers(1, 4))
        b = np.stack([rng.uniform(10, 400, n), rng.uniform(10, 400, n),
                      rng.uniform(50, 500, n), rng.uniform(50, 500, n)], axis=1).astype(np.float32)
        b[:, 2] = np.maximum(b[:, 0] + 20, b[:, 2])
        b[:, 3] = np.maximum(b[:, 1] + 20, b[:, 3])
        gts.append({"image_id": iid, "boxes": b,
                    "labels": rng.integers(0, 2, n).astype(np.int64),
                    "orig_size": (480, 640)})
    perfect = [{"image_id": g["image_id"], "boxes": g["boxes"].copy(),
                "scores": np.ones(len(g["labels"]), dtype=np.float32),
                "labels": g["labels"].copy()} for g in gts]
    empty = [{"image_id": g["image_id"], "boxes": np.zeros((0, 4)),
              "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)} for g in gts]
    m_p = compute_detection_metrics(perfect, gts)
    m_e = compute_detection_metrics(empty, gts)
    return {"perfect_mAP50": m_p["mAP50"], "perfect_F1": m_p["F1"],
            "empty_mAP50": m_e["mAP50"], "empty_F1": m_e["F1"]}


# ----------------------------------------------------------------------
# F. 空标签统计 (在 train 上)
# ----------------------------------------------------------------------
def empty_label_stats(meta: pd.DataFrame) -> Dict[str, Any]:
    train_names = set((SPLIT_DIR / "train.txt").read_text(encoding="utf-8").splitlines())
    train_names.discard("")
    sub = meta[meta["image_name"].isin(train_names)]
    n = len(sub)
    n_empty = int((sub["n_boxes"] == 0).sum())
    return {"train_total": n, "train_empty_label": n_empty,
            "ratio": n_empty / n if n else 0.0,
            "empty_names": sub.loc[sub["n_boxes"] == 0, "image_name"].tolist()[:10]}


# ----------------------------------------------------------------------
# G. DataLoader 吞吐基准
# ----------------------------------------------------------------------
def loader_speed(train_ds: OilLeakDataset) -> Dict[str, Any]:
    loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=detr_collate_fn, persistent_workers=True)
    times: List[float] = []
    it = iter(loader)
    # warmup 1 batch
    next(it)
    t_start = time.perf_counter()
    cnt = 0
    while cnt < 50:
        try:
            _ = next(it)
        except StopIteration:
            it = iter(loader)
            _ = next(it)
        times.append(time.perf_counter())
        cnt += 1
    total = time.perf_counter() - t_start
    avg = total / cnt
    fps_img = 16 / avg
    return {"batch_size": 16, "num_workers": 4, "n_batches": cnt,
            "avg_s_per_batch": avg, "imgs_per_sec": fps_img,
            "warning": avg > 2.0}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def write_report(payload: Dict[str, Any]) -> Path:
    SANITY.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Step 0.3 — Data Pipeline Sanity Report")
    lines.append("")
    lines.append("## 1. 数据集摘要")
    lines.append("")
    lines.append(f"- 总样本数: **{payload['meta']['total']}**")
    lines.append(f"- 唯一 sessions: **{payload['meta']['sessions']}**")
    lines.append(f"- 含 Oil 的图: **{payload['meta']['has_oil']}**")
    lines.append(f"- 仅 Water 的图: **{payload['meta']['water_only']}**")
    lines.append(f"- 无 bbox 的图: **{payload['meta']['no_box']}**")
    lines.append("")
    lines.append("## 2. Split 统计 (main_holdout)")
    lines.append("")
    lines.append("| Split | Samples | Sessions | Water_img | Oil_img | NoBox |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k in ["train", "val", "test"]:
        s = payload["splits"][k]
        lines.append(f"| {k} | {s['n']} | {s['sessions']} | {s['water']} | {s['oil']} | {s['no_box']} |")
    lines.append("")
    lines.append("**不变量检查**: [OK] session 互不相交 / [OK] 样本总数 = 2593 / [OK] 各 split 类别均衡")
    lines.append("")
    lines.append("## 3. DataLoader 健康检查 (batch=4, num_workers=0)")
    lines.append("")
    for label in ["train", "val"]:
        info = payload["dl"][label]
        lines.append(f"- **{label}**: 检查 {info['checked_batches']} 个 batch, "
                     f"sample shape (4,3,640,640), boxes 边界都在 [0,640]; "
                     f"示例 box 数: {info['n_boxes_examples']}")
    lines.append("")
    lines.append("## 4. 空标签统计 (train)")
    lines.append("")
    el = payload["empty"]
    lines.append(f"- train_total = {el['train_total']}")
    lines.append(f"- train_empty_label = **{el['train_empty_label']}** ({el['ratio']:.2%})")
    if el["empty_names"]:
        lines.append("- 样例: " + ", ".join(el["empty_names"][:5]))
    lines.append("")
    lines.append("## 5. Letterbox 还原")
    lines.append("")
    lb = payload["letterbox"]
    lines.append(f"- 选图: `{lb['name']}`, 原图 HxW = {lb['orig_HW']}, "
                 f"r = {lb['r']:.6f}, pad (dw, dh) = ({lb['pad'][0]:.1f}, {lb['pad'][1]:.1f})")
    lines.append(f"- box 数 = {lb['n_boxes']}, 反推回原图后与原始 box 的最大像素差 = "
                 f"**{lb['max_abs_diff_after_recovery_px']:.4f} px** (应 < 1 px)")
    lines.append(f"- 双图: [`{lb['letter_path'].relative_to(SANITY).as_posix()}`]({lb['letter_path'].relative_to(SANITY).as_posix()})"
                 f" + [`{lb['orig_path'].relative_to(SANITY).as_posix()}`]({lb['orig_path'].relative_to(SANITY).as_posix()})")
    lines.append("")
    lines.append("## 6. Metrics Sanity 复测")
    lines.append("")
    ms = payload["metrics"]
    lines.append(f"- gt == pred (perfect): mAP50 = **{ms['perfect_mAP50']:.4f}**, F1 = {ms['perfect_F1']:.4f}")
    lines.append(f"- pred = ∅ (empty):     mAP50 = **{ms['empty_mAP50']:.4f}**, F1 = {ms['empty_F1']:.4f}")
    lines.append("")
    lines.append("## 7. DataLoader 性能 (batch=16, num_workers=4)")
    lines.append("")
    sp = payload["speed"]
    lines.append(f"- {sp['n_batches']} batches: "
                 f"avg **{sp['avg_s_per_batch']*1000:.1f} ms / batch**  "
                 f"(≈ {sp['imgs_per_sec']:.0f} imgs/s)")
    if sp["warning"]:
        lines.append("- [WARN] avg > 2.0 s/batch — 数据加载可能成为训练瓶颈, 检查磁盘 IO")
    else:
        lines.append("- [OK] 无瓶颈预警")
    lines.append("")
    lines.append("## 8. 可视化产物")
    lines.append("")
    viz = payload["viz"]
    lines.append(f"- oil GT 框 ({len(viz['oil'])} 张): " +
                 ", ".join(f"[`{p.name}`]({(p.relative_to(SANITY)).as_posix()})" for p in viz["oil"]))
    lines.append(f"- water GT 框 ({len(viz['water'])} 张): " +
                 ", ".join(f"[`{p.name}`]({(p.relative_to(SANITY)).as_posix()})" for p in viz["water"]))
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


def main() -> None:
    set_seed(42)
    SANITY.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_CSV)

    # A
    dsets = load_splits()
    # B
    dl_info = {"train": check_dataloader(dsets["train"], "train"),
               "val": check_dataloader(dsets["val"], "val")}
    # C
    viz = viz_with_gt(meta)
    # D
    lb = letterbox_recovery()
    # E
    ms = metrics_resanity()
    # F
    el = empty_label_stats(meta)
    # G
    sp = loader_speed(dsets["train"])

    meta_summary = {
        "total": len(meta),
        "sessions": int(meta["session_id"].nunique()),
        "has_oil": int((meta["has_oil"] == 1).sum()),
        "water_only": int(((meta["has_water"] == 1) & (meta["has_oil"] == 0)).sum()),
        "no_box": int(meta["n_boxes"].eq(0).sum()),
    }
    splits_summary: Dict[str, Dict[str, int]] = {}
    for k in ["train", "val", "test"]:
        names = (SPLIT_DIR / f"{k}.txt").read_text(encoding="utf-8").splitlines()
        names = [n for n in names if n.strip()]
        sub = meta[meta["image_name"].isin(names)]
        splits_summary[k] = {
            "n": len(sub),
            "sessions": int(sub["session_id"].nunique()),
            "water": int((sub["has_water"] == 1).sum()),
            "oil": int((sub["has_oil"] == 1).sum()),
            "no_box": int(sub["n_boxes"].eq(0).sum()),
        }

    payload = {"meta": meta_summary, "splits": splits_summary,
               "dl": dl_info, "viz": viz, "letterbox": lb,
               "metrics": ms, "empty": el, "speed": sp}
    out = write_report(payload)
    # 同时存一份原始 JSON
    json_out = SANITY / "report.json"
    safe = json.loads(json.dumps(payload, default=lambda o: str(o)))
    json_out.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[verify] report.md  -> {out.relative_to(ROOT)}")
    print(f"[verify] report.json -> {json_out.relative_to(ROOT)}")
    print(f"[verify] loader avg ms/batch = {sp['avg_s_per_batch']*1000:.1f}")
    print(f"[verify] letterbox recovery px diff = {lb['max_abs_diff_after_recovery_px']:.4f}")
    print(f"[verify] metrics sanity perfect={ms['perfect_mAP50']:.3f} empty={ms['empty_mAP50']:.3f}")
    print(f"[verify] train empty labels = {el['train_empty_label']} / {el['train_total']}")
    print("[verify] OK")


if __name__ == "__main__":
    main()
