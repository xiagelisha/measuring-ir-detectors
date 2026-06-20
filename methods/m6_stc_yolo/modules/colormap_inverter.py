"""STC-YOLO C4 sub-module 1 — Colormap inverter.

Maps IR pseudo-color RGB to a 1-channel temperature surrogate T_hat in [0, 1].

D1b finding (2026-05-29):
    - Dataset uses a SINGLE LUT (inferno-LIKE but not matplotlib's inferno)
      across all 22 sessions, with 99.5% per-image consistency.
    - An empirical LUT fit directly from data has held-out MSE 0.000568
      (~JPG noise floor), 18x tighter than matplotlib inferno (0.0105).
    - => `mode='empirical'` is the recommended path; analytic_matplotlib
      and grayscale_collapse paths are kept for A15 ablation only.
    - Empirical LUT is saved at
      results_v2/methods/stc_yolo/_ms1/empirical_lut.npz
      (keys: lut_rgb of shape (K, 3) float32; t of shape (K,) float32).

Why this is *NOT* paper3 N6 (IR intensity calibration):
  - N6 was a *post-hoc confidence calibrator* operating on RGB intensity
    averaged inside predicted boxes. It changed only the conf score, not the
    backbone input. Bhattacharyya distance 0.071 showed the per-box mean
    intensity is uninformative for Oil-vs-Water -- N6 failed.
  - This module is an *input-space transform*: it inverts the camera's
    colormap LUT to recover the underlying scalar temperature field, before
    the backbone sees anything. We are not asking "is high intensity Oil?";
    we are removing the 3-channel pseudo-color redundancy that paper3
    silently inherited from ImageNet RGB pipelines.

Contract:
    forward(rgb: Tensor[B, 3, H, W] in [0, 1])
        -> t_hat: Tensor[B, 1, H, W] in [0, 1]

    (D1b confirmed single LUT, so num_colormaps = 0 in conv1ch_from_rgb.py.)

Supported modes:
  - 'empirical'           : load empirical_lut.npz (default, main path)
  - 'analytic_matplotlib' : build a LUT from matplotlib's named colormap
                            (A15 ablation: empirical vs textbook reference)
  - 'grayscale_collapse'  : Y = 0.299R + 0.587G + 0.114B (A15 ablation:
                            empirical vs naive grayscale)
  - 'mlp'                 : NOT IMPLEMENTED -- D1b finding makes it unnecessary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SUPPORTED_MODES = ("empirical", "analytic_matplotlib", "grayscale_collapse", "mlp")
SUPPORTED_MATPLOTLIB_LUTS = (
    "grayscale_pseudo", "afmhot", "hot", "inferno", "plasma",
    "viridis", "jet", "rainbow", "gist_heat",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EMPIRICAL_LUT_PATH = (
    _PROJECT_ROOT / "results_v2" / "methods" / "stc_yolo" / "_ms1" / "empirical_lut.npz"
)

# Chunk size for the per-pixel NN lookup. Default 256k pixels:
#   cdist output buffer = 256k * 256 floats * 4 bytes = ~256 MB per chunk
#   on 16+ GB GPU this is comfortable; reduce only for smaller GPUs.
# For batch=8, imgsz=800 (5.12M pixels), this gives ~20 chunks per forward
# vs 160 chunks at the old 32k -- roughly 5-10x speedup over the v1 D4b run
# (which took 38 min/seed vs D4a's 13 min). Tradeoff: more transient VRAM.
DEFAULT_CHUNK_SIZE = 262144

# ITU-R BT.601 luminance coefficients for grayscale collapse mode.
_GRAY_W = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


class ColormapInverter(nn.Module):
    """RGB -> T_hat colormap inverter.

    Args:
        mode: one of SUPPORTED_MODES. Default 'empirical' is the D1b-recommended
            path; the matplotlib / grayscale modes exist for the A15 ablation.
        empirical_lut_path: path to D1b's `empirical_lut.npz`. None -> default.
            Used when mode='empirical'.
        matplotlib_colormap: name in SUPPORTED_MATPLOTLIB_LUTS. Used when
            mode='analytic_matplotlib'.
        mlp_ckpt: path to a saved 3->1 MLP checkpoint. NOT IMPLEMENTED.
        chunk_size: pixel chunk size for the NN lookup (memory tradeoff).

    Buffers (registered non-persistent so the state_dict stays compact):
        lut_rgb: (K, 3) float32, K=256 by convention.
        t:       (K,)  float32, normalised parameter in [0, 1].
    Absent when mode='grayscale_collapse' (no LUT needed).
    """

    def __init__(
        self,
        mode: str = "empirical",
        empirical_lut_path: str | Path | None = None,
        matplotlib_colormap: str = "inferno",
        mlp_ckpt: str | Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}, got {mode!r}")
        if mode == "analytic_matplotlib" and matplotlib_colormap not in SUPPORTED_MATPLOTLIB_LUTS:
            raise ValueError(
                f"unsupported matplotlib colormap {matplotlib_colormap!r}; "
                f"expected one of {SUPPORTED_MATPLOTLIB_LUTS}"
            )
        if mode == "mlp":
            raise NotImplementedError(
                "mode='mlp' is intentionally not implemented. D1b found the "
                "empirical LUT achieves MSE 0.000568 (JPG noise floor), "
                "making the calibration MLP unnecessary. Use mode='empirical'."
            )

        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.matplotlib_colormap = matplotlib_colormap
        self.mlp_ckpt = Path(mlp_ckpt) if mlp_ckpt is not None else None

        if mode == "empirical":
            path = (Path(empirical_lut_path) if empirical_lut_path is not None
                    else DEFAULT_EMPIRICAL_LUT_PATH)
            if not path.is_file():
                raise FileNotFoundError(
                    f"empirical LUT not found at {path}. Run "
                    "methods/m6_stc_yolo/analysis/colormap_analysis.py first."
                )
            self.empirical_lut_path = path
            data = np.load(str(path))
            if "lut_rgb" not in data or "t" not in data:
                raise ValueError(
                    f"{path} missing required keys 'lut_rgb' and 't'; "
                    f"got {list(data.keys())}"
                )
            lut_rgb = np.asarray(data["lut_rgb"], dtype=np.float32)
            t = np.asarray(data["t"], dtype=np.float32)
            if lut_rgb.ndim != 2 or lut_rgb.shape[1] != 3:
                raise ValueError(f"lut_rgb has shape {lut_rgb.shape}; expected (K, 3)")
            if t.ndim != 1 or t.shape[0] != lut_rgb.shape[0]:
                raise ValueError(
                    f"t has shape {t.shape}; expected ({lut_rgb.shape[0]},)"
                )
            self.register_buffer("lut_rgb",
                                 torch.from_numpy(lut_rgb).float(),
                                 persistent=False)
            self.register_buffer("t",
                                 torch.from_numpy(t).float(),
                                 persistent=False)

        elif mode == "analytic_matplotlib":
            import matplotlib as mpl  # lazy import
            cmap = mpl.colormaps.get_cmap(matplotlib_colormap)
            n = 256
            t_np = np.linspace(0.0, 1.0, n).astype(np.float32)
            lut_np = cmap(t_np)[:, :3].astype(np.float32)
            self.register_buffer("lut_rgb",
                                 torch.from_numpy(lut_np).float(),
                                 persistent=False)
            self.register_buffer("t",
                                 torch.from_numpy(t_np).float(),
                                 persistent=False)

        elif mode == "grayscale_collapse":
            # No LUT needed; just a constant coefficient buffer.
            self.register_buffer("_gray_w", _GRAY_W.clone(), persistent=False)

    # -------- forward ---------------------------------------------------------

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """RGB (B, 3, H, W) in [0, 1] -> T_hat (B, 1, H, W) in [0, 1]."""
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"expected RGB tensor (B, 3, H, W), got {tuple(rgb.shape)}"
            )

        if self.mode == "grayscale_collapse":
            # (B, 3, H, W) * (1, 3, 1, 1) -> sum over ch -> (B, 1, H, W).
            w = self._gray_w.to(rgb.device).to(rgb.dtype)
            return (rgb * w).sum(dim=1, keepdim=True).clamp_(0.0, 1.0)

        # NN LUT lookup path (empirical / analytic_matplotlib).
        B, _, H, W = rgb.shape
        N = B * H * W
        # (B, 3, H, W) -> (B, H, W, 3) -> (N, 3)
        pixels = rgb.permute(0, 2, 3, 1).reshape(N, 3).contiguous()
        lut = self.lut_rgb.to(rgb.device).to(rgb.dtype)
        t_table = self.t.to(rgb.device).to(rgb.dtype)

        t_hat = torch.empty(N, dtype=rgb.dtype, device=rgb.device)
        chunk = self.chunk_size
        for i in range(0, N, chunk):
            seg = pixels[i:i + chunk]                       # (m, 3)
            # cdist returns (m, K); argmin -> (m,)
            d = torch.cdist(seg.unsqueeze(0), lut.unsqueeze(0)).squeeze(0)
            idx = d.argmin(dim=1)
            t_hat[i:i + chunk] = t_table[idx]

        return t_hat.view(B, H, W, 1).permute(0, 3, 1, 2).contiguous()
