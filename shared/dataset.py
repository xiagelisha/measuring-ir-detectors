"""OilLeakDataset — 红外漏油检测的通用数据集。

设计要点:
  * YOLO 归一化标签 (class cx cy w h) → 内部统一转 xyxy 绝对坐标。
  * 支持两种构造方式:
      A. 传入 split_file (train/val/test.txt, 每行 basename 或完整路径)
      B. from_directory(image_dir, label_dir): splits 还没生成前的回退路径
  * 不在 __getitem__ 内做 letterbox — 让 transforms (Albumentations) 处理,
    Dataset 层只把原图 + 绝对坐标 boxes 喂给 transforms。
  * 若 transforms is None: 返回原始 RGB 图 (HWC uint8 -> CHW float, 0-1) +
    原图 (H, W) 上的绝对坐标 boxes, ratio_pad=(1.0, (0,0))。
  * 空标签 (.txt 存在但 0 行 / 全空白): 返回 N=0 tensor; 标签文件缺失: 抛错。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _read_split(split_file: Path) -> List[str]:
    """读取 split 文件 (basename 或完整路径), 返回 stem 列表。"""
    text = split_file.read_text(encoding="utf-8").splitlines()
    stems: List[str] = []
    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 兼容: 完整路径 / 相对路径 / 仅 basename
        p = Path(line)
        stems.append(p.stem)
    return stems


def _list_image_stems(image_dir: Path) -> List[str]:
    """扫描 image_dir 下所有支持的图像扩展, 返回 stem 列表 (排序)。"""
    out: List[str] = []
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            out.append(p.stem)
    return out


def _find_image_path(image_dir: Path, stem: str) -> Path:
    """根据 stem 在 image_dir 找匹配的图 (大小写不敏感, 多扩展名)。"""
    for ext in _IMG_EXTS:
        for cand in (image_dir / f"{stem}{ext}", image_dir / f"{stem}{ext.upper()}"):
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"Image not found for stem '{stem}' in {image_dir}")


def _load_yolo_label(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """读取 YOLO 标签 -> (boxes_xyxy_abs (N,4) float32, labels (N,) int64)。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file missing: {label_path}")
    boxes: List[List[float]] = []
    labels: List[int] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]))
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(cls)
    if not boxes:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return (np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.int64))


class OilLeakDataset(Dataset):
    """红外漏油检测数据集 (YOLO label 格式)."""

    def __init__(
        self,
        split_file: Optional[Union[str, Path]],
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        image_size: int = 640,
        transforms: Optional[Callable] = None,
        mode: str = "train",
        stems: Optional[List[str]] = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.image_size = int(image_size)
        self.transforms = transforms
        self.mode = mode

        if stems is not None:
            self.stems = list(stems)
        else:
            if split_file is None:
                raise ValueError("Either split_file or stems must be provided")
            self.split_file = Path(split_file)
            self.stems = _read_split(self.split_file)

        if not self.stems:
            raise ValueError("Dataset is empty (no stems)")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {self.image_dir}")
        if not self.label_dir.is_dir():
            raise FileNotFoundError(f"label_dir not found: {self.label_dir}")

    @classmethod
    def from_directory(
        cls,
        image_dir: Union[str, Path],
        label_dir: Union[str, Path],
        image_size: int = 640,
        transforms: Optional[Callable] = None,
        mode: str = "train",
    ) -> "OilLeakDataset":
        """回退构造: 无 splits 时, 直接用 image_dir 下全部图像。"""
        image_dir = Path(image_dir)
        stems = _list_image_stems(image_dir)
        return cls(split_file=None, image_dir=image_dir, label_dir=label_dir,
                   image_size=image_size, transforms=transforms, mode=mode, stems=stems)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.stems[idx]
        img_path = _find_image_path(self.image_dir, stem)
        label_path = self.label_dir / f"{stem}.txt"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cv2.imread failed: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        boxes_xyxy, labels = _load_yolo_label(label_path, w, h)

        if self.transforms is not None:
            out = self.transforms(image=rgb,
                                  bboxes=boxes_xyxy.tolist(),
                                  labels=labels.tolist())
            img_t = out["image"]
            new_boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
            new_labels = np.asarray(out["labels"], dtype=np.int64).reshape(-1)
            boxes_t = torch.from_numpy(new_boxes)
            labels_t = torch.from_numpy(new_labels)
            # transforms 内 letterbox 后, 我们无法可靠反推 r/dw/dh; 留给 transforms 扩展
            ratio_pad = (1.0, (0.0, 0.0))
        else:
            # 无 transforms 时, 原图 HWC uint8 -> CHW float [0,1]
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            boxes_t = torch.from_numpy(boxes_xyxy)
            labels_t = torch.from_numpy(labels)
            ratio_pad = (1.0, (0.0, 0.0))

        return {
            "image": img_t,
            "boxes_xyxy_abs": boxes_t,
            "labels": labels_t,
            "image_id": int(idx),
            "orig_size": (int(h), int(w)),
            "image_path": str(img_path),
            "ratio_pad": ratio_pad,
        }


if __name__ == "__main__":
    ds = OilLeakDataset.from_directory(
        image_dir=Path("data/OilData"),
        label_dir=Path("data/OilLabels"),
        image_size=640,
    )
    print(f"[dataset] len = {len(ds)}")
    s0 = ds[0]
    print(f"[dataset] sample 0 image shape = {tuple(s0['image'].shape)}, "
          f"boxes = {tuple(s0['boxes_xyxy_abs'].shape)}, labels = {tuple(s0['labels'].shape)}, "
          f"orig_size = {s0['orig_size']}")
    print(f"[dataset] sample 0 path = {Path(s0['image_path']).name}")

    # 找含 oil (label=1) 和 纯水 (无 box 或 仅 label=0) 的样本各一个
    oil_idx = water_idx = None
    for i in range(min(100, len(ds))):
        item = ds[i]
        lbls = item["labels"].tolist()
        if oil_idx is None and any(l == 1 for l in lbls):
            oil_idx = i
        if water_idx is None and (len(lbls) == 0 or all(l == 0 for l in lbls)):
            water_idx = i
        if oil_idx is not None and water_idx is not None:
            break
    print(f"[dataset] found oil sample idx={oil_idx} "
          f"(boxes={ds[oil_idx]['boxes_xyxy_abs'].shape[0] if oil_idx is not None else 'N/A'})")
    print(f"[dataset] found pure-water sample idx={water_idx} "
          f"(boxes={ds[water_idx]['boxes_xyxy_abs'].shape[0] if water_idx is not None else 'N/A'})")

    # 验证前 50 样本: boxes 都在原图范围内
    bad = 0
    for i in range(min(50, len(ds))):
        item = ds[i]
        H, W = item["orig_size"]
        b = item["boxes_xyxy_abs"]
        if b.numel() == 0:
            continue
        if not (b[:, [0, 2]].min() >= 0 and b[:, [1, 3]].min() >= 0
                and b[:, [0, 2]].max() <= W + 1e-3 and b[:, [1, 3]].max() <= H + 1e-3):
            bad += 1
    print(f"[dataset] bbox bound check on 50 samples: bad = {bad}")
    assert bad == 0
    print("[dataset] OK")
