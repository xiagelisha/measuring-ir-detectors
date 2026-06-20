"""可视化: 画框、抽样保存、PR / F1 / 混淆矩阵 (matplotlib)。"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # 避免后台运行时缺 GUI 报错
import matplotlib.pyplot as plt
import numpy as np

# Water=蓝, Oil=红 (BGR)
_COLOR_MAP = {0: (255, 96, 32), 1: (32, 32, 220)}
_DEFAULT_NAMES = ("Water", "Oil")


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: Optional[np.ndarray] = None,
    class_names: Sequence[str] = _DEFAULT_NAMES,
    thickness: int = 2,
) -> np.ndarray:
    """在 BGR 图上画 xyxy box, 返回新图 (不修改原 in-place)。"""
    out = image.copy()
    if boxes is None or len(boxes) == 0:
        return out
    boxes = np.asarray(boxes).reshape(-1, 4)
    labels = np.asarray(labels).reshape(-1).astype(int)
    for i, (b, l) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        color = _COLOR_MAP.get(int(l), (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        name = class_names[l] if 0 <= l < len(class_names) else f"cls{l}"
        if scores is not None:
            text = f"{name} {float(scores[i]):.2f}"
        else:
            text = f"{name}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 4, y_top + th + 4), color, -1)
        cv2.putText(out, text, (x1 + 2, y_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def visualize_samples(
    dataset,
    out_dir: Union[str, Path],
    n: int = 16,
    class_names: Sequence[str] = _DEFAULT_NAMES,
    seed: int = 0,
) -> List[Path]:
    """从 dataset 抽 n 个样本 (尽量含 oil 与 water 各半), 画框存为 jpg。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    half = n // 2
    has_oil, no_oil = [], []
    # 不能扫全部 2593 张 (慢); 抽样扫描确认
    candidates = list(range(len(dataset)))
    rng.shuffle(candidates)
    for i in candidates:
        if len(has_oil) >= half and len(no_oil) >= (n - half):
            break
        item = dataset[i]
        lbls = item["labels"].tolist() if hasattr(item["labels"], "tolist") else list(item["labels"])
        if any(int(l) == 1 for l in lbls):
            if len(has_oil) < half:
                has_oil.append(i)
        else:
            if len(no_oil) < (n - half):
                no_oil.append(i)
    pick_ids = (has_oil + no_oil)[:n]
    if len(pick_ids) < n:
        # 不足 n, 补满
        for i in candidates:
            if i in pick_ids:
                continue
            pick_ids.append(i)
            if len(pick_ids) >= n:
                break

    written: List[Path] = []
    for i, idx in enumerate(pick_ids):
        item = dataset[idx]
        img_path = Path(item["image_path"])
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        boxes = item["boxes_xyxy_abs"].cpu().numpy() if hasattr(item["boxes_xyxy_abs"], "cpu") else np.asarray(item["boxes_xyxy_abs"])
        labels = item["labels"].cpu().numpy() if hasattr(item["labels"], "cpu") else np.asarray(item["labels"])
        vis = draw_boxes(bgr, boxes, labels, scores=None, class_names=class_names)
        out_path = out_dir / f"sample_{i:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis)
        written.append(out_path)
    return written


def plot_pr_curve(
    recall: np.ndarray, precision: np.ndarray,
    out_path: Union[str, Path], title: str = "PR Curve", labels: Optional[Sequence[str]] = None,
) -> Path:
    """绘制 PR 曲线; recall/precision 可为 (N,) 或 (C, N) 多类别。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    recall = np.asarray(recall)
    precision = np.asarray(precision)
    if recall.ndim == 1:
        ax.plot(recall, precision, label=labels[0] if labels else "PR")
    else:
        for i in range(recall.shape[0]):
            ax.plot(recall[i], precision[i], label=(labels[i] if labels and i < len(labels) else f"cls{i}"))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_f1_curve(
    conf: np.ndarray, f1: np.ndarray, out_path: Union[str, Path],
    title: str = "F1 vs Confidence",
) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    ax.plot(conf, f1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("F1"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray, class_names: Sequence[str], out_path: Union[str, Path],
    title: str = "Confusion Matrix", normalize: bool = False,
) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Pred"); ax.set_ylabel("GT"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared.dataset import OilLeakDataset
    ds = OilLeakDataset.from_directory(
        image_dir=Path("data/OilData"),
        label_dir=Path("data/OilLabels"),
        image_size=640,
    )
    out_dir = Path("results_v2/sanity_check/visualizer")
    paths = visualize_samples(ds, out_dir, n=8)
    print(f"[visualizer] saved {len(paths)} sample images to {out_dir}")

    # PR / F1 / confusion 测试
    rec = np.linspace(0, 1, 20)
    pre = np.exp(-rec * 1.5)
    f1c = np.linspace(0.3, 0.95, 20)
    cm = np.array([[100, 5], [10, 80]])
    plot_pr_curve(rec, pre, out_dir / "_pr.png", labels=["all"])
    plot_f1_curve(np.linspace(0, 1, 20), f1c, out_dir / "_f1.png")
    plot_confusion_matrix(cm, ["Water", "Oil"], out_dir / "_cm.png")
    print(f"[visualizer] plotted PR / F1 / CM into {out_dir}")
    print("[visualizer] OK")
