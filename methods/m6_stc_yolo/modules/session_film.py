"""STC-YOLO C3 — Session-FiLM detect head (MS3).

FiLM-modulates YOLO detect head's cls branch with affine parameters (gamma,
beta) generated from per-frame session metadata (temperature, location,
condition, day_night). Lets the head adapt its box / class prior to the
deployment environment without changing the backbone.

Why this is *NOT* paper3 N4 (TG-Spatial-Attention which failed with alpha->0):
  - N4 conditioned attention on the *image's own* Sobel magnitude — the
    backbone could already compute that, so the gate carried no new info and
    alpha collapsed to zero.
  - C3 conditions on EXTERNAL metadata (T_amb, location, ...) that is NOT
    derivable from the image. The information is fresh, so FiLM has something
    real to do. And FiLM is an affine y = gamma*x + beta (not a sigmoid gate),
    so even if gamma -> 1 and beta -> 0, the head reduces gracefully to the
    vanilla baseline rather than degrading below it.

Contract:
    metadata vector m_t in R^10 (encoded by data.session_metadata)
    -> MLP(10 -> 64 -> 128) -> split -> (gamma, beta) each in R^128
    -> feat' = gamma * feat + beta applied to cls branch features at P3/P4/P5

Implementation status: MS3 — stub.
"""
from __future__ import annotations

import torch
import torch.nn as nn


METADATA_DIM_DEFAULT = 8  # see data.session_metadata.encode_metadata


class SessionFiLMGenerator(nn.Module):
    """MLP mapping a session-metadata vector -> per-FPN-level (gamma, beta).

    A shared trunk (metadata_dim -> hidden) feeds one small head per level that
    emits 2*C values, split into (gamma, beta). The per-level head's final layer
    is ZERO-initialised so at start gamma==1 and beta==0 — i.e. FiLM begins as
    the identity and the detect head reduces *exactly* to the vanilla baseline.
    It can only help from there (the C3 non-degradation argument; contrast with
    paper3 N4 spatial-attention which collapsed below baseline).
    Metadata excludes the day_night field.

    Args:
        metadata_dim: input vector dim (8 from encode_metadata).
        hidden_dim: shared trunk width.
        feature_dim: per-level channel count C. Either an int (same C for every
            level) or a dict level->C (real YOLOv12s head channels differ).
        inject_levels: FPN levels to generate params for.
    """

    def __init__(
        self,
        metadata_dim: int = METADATA_DIM_DEFAULT,
        hidden_dim: int = 64,
        feature_dim: int | dict[str, int] = 128,
        inject_levels: tuple[str, ...] = ("P3", "P4", "P5"),
    ) -> None:
        super().__init__()
        self.metadata_dim = int(metadata_dim)
        self.inject_levels = tuple(inject_levels)
        if isinstance(feature_dim, dict):
            self.level_dims = {k: int(v) for k, v in feature_dim.items()}
        else:
            self.level_dims = {lv: int(feature_dim) for lv in self.inject_levels}

        self.trunk = nn.Sequential(
            nn.Linear(self.metadata_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleDict()
        for lv in self.inject_levels:
            c = self.level_dims[lv]
            head = nn.Linear(hidden_dim, 2 * c)
            nn.init.zeros_(head.weight)   # identity init: gamma=1, beta=0
            nn.init.zeros_(head.bias)
            self.heads[lv] = head

    def forward(
        self, metadata: torch.Tensor
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """metadata: (B, metadata_dim) -> {level: (gamma (B,C), beta (B,C))}."""
        if metadata.dim() != 2 or metadata.shape[1] != self.metadata_dim:
            raise ValueError(
                f"expected (B, {self.metadata_dim}); got {tuple(metadata.shape)}")
        h = self.trunk(metadata)
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for lv in self.inject_levels:
            raw = self.heads[lv](h)                 # (B, 2C)
            delta_gamma, beta = raw.chunk(2, dim=1)
            gamma = 1.0 + delta_gamma               # identity at init (delta=0)
            out[lv] = (gamma, beta)
        return out


def apply_film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """FiLM affine: feat' = gamma * feat + beta with broadcasting.

    feat:  (B, C, H, W)
    gamma: (B, C)
    beta:  (B, C)
    """
    if feat.dim() != 4:
        raise ValueError(f"expected feat (B,C,H,W); got {tuple(feat.shape)}")
    if gamma.shape != beta.shape or gamma.shape[:2] != feat.shape[:2]:
        raise ValueError(
            f"gamma/beta {tuple(gamma.shape)} incompatible with feat "
            f"{tuple(feat.shape)}")
    return gamma[:, :, None, None] * feat + beta[:, :, None, None]
