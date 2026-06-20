"""变长 box batching — DETR 风格 (images stack, boxes/labels list)。"""
from __future__ import annotations

from typing import Any, Dict, List

import torch


def detr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 dataset 样本列表拼成 batch。

    输入: list of OilLeakDataset 返回的 dict。
    输出: {
        'images':     (B, 3, H, W)        — stack 的张量
        'boxes':      List[Tensor(N_i,4)] — 每张图自带 box 数量
        'labels':     List[Tensor(N_i,)]
        'image_ids':  Tensor (B,) long
        'orig_sizes': List[Tuple[int,int]]
        'image_paths':List[str]
        'ratio_pads': List[Tuple[float, Tuple[float,float]]]
    }
    """
    if not batch:
        raise ValueError("Empty batch")

    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes_xyxy_abs"].float() for b in batch]
    labels = [b["labels"].long() for b in batch]
    image_ids = torch.tensor([int(b["image_id"]) for b in batch], dtype=torch.long)
    orig_sizes = [b["orig_size"] for b in batch]
    image_paths = [b["image_path"] for b in batch]
    ratio_pads = [b["ratio_pad"] for b in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
        "image_paths": image_paths,
        "ratio_pads": ratio_pads,
    }


if __name__ == "__main__":
    # 4 个样本: N = 0, 1, 3, 5
    def mk(n: int, idx: int) -> Dict[str, Any]:
        return {
            "image": torch.randn(3, 640, 640),
            "boxes_xyxy_abs": torch.rand(n, 4) * 640.0,
            "labels": torch.randint(0, 2, (n,)),
            "image_id": idx,
            "orig_size": (480, 640),
            "image_path": f"fake_{idx}.jpg",
            "ratio_pad": (1.0, (0.0, 0.0)),
        }

    batch = [mk(0, 0), mk(1, 1), mk(3, 2), mk(5, 3)]
    out = detr_collate_fn(batch)
    print(f"[collate] images shape = {tuple(out['images'].shape)}")
    print(f"[collate] box counts   = {[b.shape[0] for b in out['boxes']]}")
    print(f"[collate] label counts = {[l.shape[0] for l in out['labels']]}")
    print(f"[collate] image_ids    = {out['image_ids'].tolist()}")
    assert tuple(out["images"].shape) == (4, 3, 640, 640)
    assert [b.shape[0] for b in out["boxes"]] == [0, 1, 3, 5]
    assert out["image_ids"].dtype == torch.long
    print("[collate] OK")
