"""D2 smoke test — verify ColormapInverter on a real IR image.

Tests all 3 modes (empirical, analytic_matplotlib inferno, grayscale_collapse)
end-to-end on one real image. Produces a 4-panel comparison figure for the
paper's C4 ablation figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods.m6_stc_yolo.modules.colormap_inverter import ColormapInverter


def main() -> int:
    img_path = _PROJECT_ROOT / "data" / "OilData" / "Oil_30_1_1_1_00000009.JPG"
    arr = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8).astype(np.float32) / 255.0
    rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    print(f"input shape: {tuple(rgb.shape)}  range: [{float(rgb.min()):.3f}, {float(rgb.max()):.3f}]")

    for kwargs in [
        {"mode": "empirical"},
        {"mode": "analytic_matplotlib", "matplotlib_colormap": "inferno"},
        {"mode": "grayscale_collapse"},
    ]:
        inv = ColormapInverter(**kwargs)
        t_hat = inv(rgb)
        print(f"  {kwargs['mode']:>22}: shape={tuple(t_hat.shape)}  "
              f"range=[{float(t_hat.min()):.3f}, {float(t_hat.max()):.3f}]  "
              f"mean={float(t_hat.mean()):.3f}")

    inv_emp = ColormapInverter(mode="empirical")
    inv_mpl = ColormapInverter(mode="analytic_matplotlib", matplotlib_colormap="inferno")
    inv_gs = ColormapInverter(mode="grayscale_collapse")
    t_emp = inv_emp(rgb)
    t_mpl = inv_mpl(rgb)
    t_gs = inv_gs(rgb)
    diff_emp_mpl = (t_emp - t_mpl).abs()
    diff_emp_gs = (t_emp - t_gs).abs()
    print()
    print(f"emp vs mpl-inferno: mean diff {float(diff_emp_mpl.mean()):.4f}, "
          f"max diff {float(diff_emp_mpl.max()):.4f}")
    print(f"emp vs grayscale  : mean diff {float(diff_emp_gs.mean()):.4f}, "
          f"max diff {float(diff_emp_gs.max()):.4f}")
    print("(per D1b, empirical and mpl-inferno should differ measurably; "
          "grayscale should differ even more.)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(arr)
    axes[0].set_title("Original IR pseudo-color (RGB)")
    axes[0].axis("off")
    axes[1].imshow(t_emp[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"T_hat (empirical, D1b)  mean={float(t_emp.mean()):.3f}")
    axes[1].axis("off")
    axes[2].imshow(t_mpl[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"T_hat (mpl inferno)  mean={float(t_mpl.mean()):.3f}")
    axes[2].axis("off")
    axes[3].imshow(t_gs[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"T_hat (grayscale Y)  mean={float(t_gs.mean()):.3f}")
    axes[3].axis("off")
    plt.tight_layout()
    out_path = (_PROJECT_ROOT / "results_v2" / "methods" / "stc_yolo"
                / "_ms1" / "d2_smoke_test.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nsaved smoke visualization: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
