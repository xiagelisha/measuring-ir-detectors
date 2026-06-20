"""STC-YOLO data — Unlabeled neighbor sampler for C2 Mean-Teacher (MS3).

For each labeled frame f_t in the supervised train batch, samples ONE
unlabeled neighbor f_{t+δ} from the SAME train session and within frame_idx
distance δ in {1, ..., mt_neighbor_gap_max}.

Critical constraint: f_{t+δ} must come from the train split (image_name not
in val/test). val/test session frames are excluded from sampling to prevent
leakage. Because candidates are filtered to `train_image_names` up front, a
sampled neighbor is *always* a train frame regardless of the metadata index.

Temporal proximity is enforced directly via |Δframe_idx| <= max_gap on the
same session_id: frame_idx is the true capture index, so a small gap means the
frames are genuinely adjacent in time (no need for explicit segment objects).
"""
from __future__ import annotations

from pathlib import Path

import torch

from .session_metadata import DEFAULT_METADATA_CSV, load_metadata


class UnlabeledNeighborSampler:
    """Maps a labeled image_name -> 1 sampled unlabeled neighbor image_name.

    Args:
        train_image_names: set of image_names (stems) in the TRAIN split. Both
            the query frames and the sampled neighbors are confined to this set.
        metadata_csv: path to metadata.csv (defaults to the dataset's).
        max_gap: max |Δframe_idx| within the same session for a neighbor.
        seed: RNG seed for reproducible uniform sampling.
    """

    def __init__(
        self,
        train_image_names: set[str],
        metadata_csv: str | Path | None = None,
        max_gap: int = 3,
        seed: int = 42,
    ) -> None:
        self.train_image_names = set(train_image_names)
        self.max_gap = int(max_gap)
        self.seed = int(seed)
        self._rng = torch.Generator().manual_seed(self.seed)

        metas = load_metadata(metadata_csv or DEFAULT_METADATA_CSV)
        # Per-frame (session, idx) lookup and per-session sorted train frames.
        self._frame_info: dict[str, tuple[str, int]] = {}
        self._by_session: dict[str, list[tuple[int, str]]] = {}
        for name in self.train_image_names:
            m = metas.get(name)
            if m is None:
                continue  # name without metadata can't be neighbored
            self._frame_info[name] = (m.session_id, m.frame_idx)
            self._by_session.setdefault(m.session_id, []).append((m.frame_idx, name))
        for sid in self._by_session:
            self._by_session[sid].sort()

    def candidates(self, labeled_image_name: str) -> list[tuple[str, int]]:
        """Return [(neighbor_name, gap), ...] within max_gap, same session,
        train-only, excluding the frame itself. Empty if none."""
        info = self._frame_info.get(labeled_image_name)
        if info is None:
            return []
        sid, idx = info
        out: list[tuple[str, int]] = []
        for fidx, name in self._by_session.get(sid, []):
            gap = abs(fidx - idx)
            if 0 < gap <= self.max_gap and name != labeled_image_name:
                out.append((name, gap))
        return out

    def sample(self, labeled_image_name: str) -> tuple[str, int]:
        """Return (neighbor_image_name, gap). Raises if no valid neighbor."""
        cands = self.candidates(labeled_image_name)
        if not cands:
            raise ValueError(
                f"no train-split neighbor within gap {self.max_gap} for "
                f"{labeled_image_name!r}")
        j = int(torch.randint(len(cands), (1,), generator=self._rng).item())
        return cands[j]
