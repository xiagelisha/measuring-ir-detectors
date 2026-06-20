"""STC-YOLO data — Frame pair dataset for C1 SSL pretrain (MS2 D5).

Samples (img_t, img_t+k) pairs from intra-session continuous segments, with
k in [pair_gap_min, pair_gap_max]. Strict exclusion of `test` split frames to
avoid pretrain-to-test leakage (paper3 main_holdout test = 359 imgs reserved).

Reuses the segment iteration logic from session_metadata.py (segment-gap=1
default = "true consecutive frames only"), filters by allowed_splits, then
enumerates (t, t+k) pairs whose BOTH endpoints lie in allowed splits.

Output (__getitem__):
    img_t:  Tensor[3, image_size, image_size] in [0, 1]
    img_tk: Tensor[3, image_size, image_size] in [0, 1]
    k:      int, the actual gap that was sampled
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .session_metadata import (
    DEFAULT_METADATA_CSV,
    build_session_segments,
    load_metadata,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE_DIR = _PROJECT_ROOT / "data" / "OilData"
DEFAULT_SPLIT_DIR = _PROJECT_ROOT / "data" / "splits" / "main_holdout"


def _read_split_names(txt_path: Path) -> list[str]:
    """Read a split file (one image stem per line, # for comments)."""
    out: list[str] = []
    for raw in txt_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s).stem)
    return out


class FramePairDataset(Dataset):
    """(img_t, img_t+k) pair sampler within intra-session continuous segments.

    Args:
        metadata_csv: path to data/metadata.csv (default = DEFAULT_METADATA_CSV)
        image_dir: path to OilData (default = DEFAULT_IMAGE_DIR)
        split_dir: path to data/splits/main_holdout/ (default = DEFAULT_SPLIT_DIR)
        allowed_splits: subset of {'train','val','test'}; pairs are restricted
            to image_names appearing in these splits. Default ('train','val')
            -- this excludes the 359-image test split entirely to prevent any
            pretrain-to-test leakage.
        pair_gap_min, pair_gap_max: range of k for (t, t+k) sampling
        image_size: square resize target (default 640)
        augment: random h-flip (applied to both halves of the pair consistently)
        seed: RNG seed for augmentation
    """

    def __init__(
        self,
        metadata_csv: str | Path | None = None,
        image_dir: str | Path | None = None,
        split_dir: str | Path | None = None,
        allowed_splits: tuple[str, ...] = ("train", "val"),
        pair_gap_min: int = 1,
        pair_gap_max: int = 2,
        image_size: int = 640,
        augment: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        metadata_csv = (Path(metadata_csv) if metadata_csv is not None
                        else DEFAULT_METADATA_CSV)
        self.image_dir = (Path(image_dir) if image_dir is not None
                          else DEFAULT_IMAGE_DIR)
        split_dir = (Path(split_dir) if split_dir is not None
                     else DEFAULT_SPLIT_DIR)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.pair_gap_min = int(pair_gap_min)
        self.pair_gap_max = int(pair_gap_max)
        if self.pair_gap_min < 1 or self.pair_gap_max < self.pair_gap_min:
            raise ValueError(
                f"invalid pair gap range: [{self.pair_gap_min}, "
                f"{self.pair_gap_max}]"
            )
        self._rng = random.Random(int(seed))

        # Build allowed names from split files (strict subset)
        allowed: set[str] = set()
        for split in allowed_splits:
            txt = split_dir / f"{split}.txt"
            if not txt.is_file():
                raise FileNotFoundError(f"split file missing: {txt}")
            allowed.update(_read_split_names(txt))
        if not allowed:
            raise ValueError(
                f"no allowed names from splits {allowed_splits} at {split_dir}"
            )
        self.allowed_names = allowed

        # Enumerate (t, t+k) pairs within each segment, filter to allowed
        metas = load_metadata(metadata_csv)
        segments = build_session_segments(metas, max_gap=1)
        pairs: list[tuple[str, str, int]] = []
        for sid in sorted(segments.keys()):
            for seg in segments[sid]:
                for i in range(len(seg)):
                    n_t = seg[i]
                    if n_t not in allowed:
                        continue
                    for k in range(self.pair_gap_min, self.pair_gap_max + 1):
                        j = i + k
                        if j >= len(seg):
                            break
                        n_tk = seg[j]
                        if n_tk not in allowed:
                            continue
                        pairs.append((n_t, n_tk, k))
        if not pairs:
            raise ValueError(
                "no valid frame pairs after filtering; check split files"
            )
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, name: str) -> np.ndarray:
        """Load by stem, try common extensions; returns RGB uint8 (H, W, 3)."""
        for ext in (".JPG", ".jpg", ".jpeg", ".png", ".bmp"):
            p = self.image_dir / f"{name}{ext}"
            if p.is_file():
                bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise IOError(f"cv2.imread failed: {p}")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"image not found for stem '{name}' in {self.image_dir}")

    def _augment_pair(
        self, img_t: np.ndarray, img_tk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the SAME augmentation to both pair members (preserves
        spatial correspondence between positives)."""
        if not self.augment:
            return img_t, img_tk
        if self._rng.random() < 0.5:
            img_t = img_t[:, ::-1, :].copy()
            img_tk = img_tk[:, ::-1, :].copy()
        return img_t, img_tk

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """Stretch resize to square image_size (no letterbox; SSL doesn't
        need preserved aspect ratio)."""
        return cv2.resize(
            img, (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        n_t, n_tk, k = self.pairs[idx]
        img_t = self._load_image(n_t)
        img_tk = self._load_image(n_tk)
        img_t, img_tk = self._augment_pair(img_t, img_tk)
        img_t = self._resize(img_t)
        img_tk = self._resize(img_tk)
        t_t = torch.from_numpy(img_t).permute(2, 0, 1).contiguous().float() / 255.0
        t_tk = torch.from_numpy(img_tk).permute(2, 0, 1).contiguous().float() / 255.0
        return t_t, t_tk, int(k)


if __name__ == "__main__":
    ds = FramePairDataset(image_size=320)
    print(f"FramePairDataset: {len(ds)} pairs from allowed splits "
          f"(train + val, test excluded)")
    img_t, img_tk, k = ds[0]
    print(f"sample 0: img_t {tuple(img_t.shape)} img_tk {tuple(img_tk.shape)} k={k}")
    # Gap distribution
    from collections import Counter
    gaps = Counter(p[2] for p in ds.pairs)
    print(f"gap distribution: {dict(gaps)}")
