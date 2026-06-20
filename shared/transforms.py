"""Albumentations 训练/推理增强管线。

设计要点:
  * letterbox 由 LongestMaxSize + PadIfNeeded(灰色 114,114,114) 实现, 与 ultralytics 默认一致。
  * 不放 Mosaic / MixUp / RandomCrop (ultralytics 自己会做);
    不放 ColorJitter (红外色调单调, 强色调增强会破坏物理意义)。
  * bbox 格式 = pascal_voc (xyxy 绝对坐标), 与 OilLeakDataset.__getitem__ 输出一致。
  * min_visibility=0.3 防止 padding/crop 把目标切到几乎不可见。
"""
from __future__ import annotations

from typing import Callable

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD_GRAY = (114, 114, 114)


def get_train_transforms(image_size: int = 640) -> Callable:
    """训练增强: letterbox + 翻转 + 亮度/对比度 + 弱模糊噪声 + ImageNet 归一化。"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=_PAD_GRAY,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(image_size: int = 640) -> Callable:
    """推理/验证: 仅 letterbox + 归一化, 无随机增强。"""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=_PAD_GRAY,
            ),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_visibility=0.0,
        ),
    )


if __name__ == "__main__":
    import numpy as np
    from pathlib import Path

    img_path = next(p for p in Path("data/OilData").iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg"))
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    fake_boxes = [[w * 0.2, h * 0.2, w * 0.5, h * 0.6],
                  [w * 0.55, h * 0.1, w * 0.95, h * 0.4]]
    fake_labels = [0, 1]

    train = get_train_transforms(image_size=640)
    out = train(image=rgb, bboxes=fake_boxes, labels=fake_labels)
    img_t = out["image"]
    new_boxes = np.asarray(out["bboxes"], dtype=np.float32).reshape(-1, 4)
    new_labels = out["labels"]
    print(f"[transforms] input  HxW = {h}x{w}, boxes = 2")
    print(f"[transforms] output image shape = {tuple(img_t.shape)}, "
          f"dtype = {img_t.dtype}")
    print(f"[transforms] output boxes = {new_boxes.tolist()}")
    print(f"[transforms] output labels = {list(new_labels)}")
    assert tuple(img_t.shape) == (3, 640, 640), img_t.shape
    if new_boxes.size > 0:
        assert new_boxes.min() >= 0 and new_boxes.max() <= 640 + 1e-3
        assert (new_boxes[:, 2] > new_boxes[:, 0]).all()
        assert (new_boxes[:, 3] > new_boxes[:, 1]).all()

    val = get_val_transforms(image_size=640)
    out2 = val(image=rgb, bboxes=fake_boxes, labels=fake_labels)
    assert tuple(out2["image"].shape) == (3, 640, 640)
    print("[transforms] OK")
