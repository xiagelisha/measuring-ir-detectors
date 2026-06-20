"""STC-YOLO MS1 D1b — Colormap analysis on the 2593-image IR dataset.

Determines whether the IR dataset uses:
  (a) grayscale_pseudo  — R = G = B, no chromatic information (FLIR "white-hot"
                          or "black-hot" preset)
  (b) afmhot / iron     — thermal iron-bow (red→orange→yellow→white)
  (c) hot               — matplotlib hot (black→red→yellow→white)
  (d) inferno / plasma / viridis / jet / rainbow / gist_heat — other presets
  (e) mixed             — different images use different LUTs

Decision rule:
  best_mse < 1e-4 → exact  → analytic_inversion
  best_mse < 1e-3 → good   → analytic_inversion
  best_mse < 5e-3 → approx → analytic_inversion_with_fallback (5-shot calib)
  else            → poor   → mlp_5shot

Output:
  results_v2/methods/stc_yolo/_ms1/colormap_analysis.json
  results_v2/methods/stc_yolo/_ms1/colormap_analysis.png

Usage:
  conda activate oilleak
  python methods/m6_stc_yolo/analysis/colormap_analysis.py [--n-images N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from methods.m6_stc_yolo.data.session_metadata import load_metadata  # noqa: E402


CANDIDATE_NAMES = [
    "grayscale_pseudo",   # R = G = B (FLIR white-hot / black-hot)
    "afmhot",             # close to FLIR "ironbow"
    "hot",                # mpl hot
    "inferno",
    "plasma",
    "viridis",
    "jet",
    "rainbow",
    "gist_heat",
]

DEFAULT_N_IMAGES = 200
DEFAULT_N_PIXELS = 1000
RNG_SEED = 42


def build_lut(name: str, n: int = 256) -> np.ndarray:
    """Return (n, 3) reference LUT in [0, 1] (float64)."""
    if name == "grayscale_pseudo":
        t = np.linspace(0.0, 1.0, n, dtype=np.float64)
        return np.stack([t, t, t], axis=1)
    import matplotlib as mpl
    cmap = mpl.colormaps.get_cmap(name)
    t = np.linspace(0.0, 1.0, n)
    return cmap(t)[:, :3].astype(np.float64)


def stratified_image_sample(metas, n: int, seed: int) -> list[str]:
    """Sample image_names uniformly across the 22 sessions."""
    by_session: dict[str, list[str]] = defaultdict(list)
    for m in metas.values():
        by_session[m.session_id].append(m.image_name)
    rng = np.random.RandomState(seed)
    selections: list[str] = []
    sessions = sorted(by_session.keys())
    per_session = max(1, n // len(sessions))
    for sid in sessions:
        names = sorted(by_session[sid])
        idx = rng.choice(len(names), size=min(per_session, len(names)), replace=False)
        selections.extend(names[i] for i in idx)
    rng.shuffle(selections)
    return selections[:n]


def sample_image_pixels(
    img_path: Path, n: int, rng: np.random.RandomState
) -> np.ndarray:
    """Return (n, 3) pixel triplets in [0, 1] (float64)."""
    img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    h, w = img.shape[:2]
    ys = rng.randint(0, h, size=n)
    xs = rng.randint(0, w, size=n)
    px = img[ys, xs, :].astype(np.float64) / 255.0
    return px


def lut_fit_per_pixel(
    pixels: np.ndarray, lut: np.ndarray, chunk: int = 5000
) -> np.ndarray:
    """Per-pixel min squared distance to nearest LUT entry. Returns (N,) float64."""
    out = np.empty(len(pixels), dtype=np.float64)
    for i in range(0, len(pixels), chunk):
        chunk_px = pixels[i:i + chunk]
        diff = chunk_px[:, None, :] - lut[None, :, :]
        dist_sq = (diff ** 2).sum(axis=-1)
        out[i:i + chunk] = dist_sq.min(axis=1)
    return out


def _classify_quality(best_mse: float) -> tuple[str, str]:
    if best_mse < 1e-4:
        return "exact", "analytic_inversion"
    if best_mse < 1e-3:
        return "good", "analytic_inversion"
    if best_mse < 5e-3:
        return "approximate", "analytic_inversion_with_fallback"
    return "poor", "mlp_5shot"


def fit_empirical_lut(pixels: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Fit empirical 1D LUT directly from the dataset RGB cloud.

    Idea: when the dataset is colormap-encoded, all pixels lie on a 1D curve
    in RGB-cube. We use the luminance Y = 0.299R + 0.587G + 0.114B as a
    monotonic proxy for the colormap parameter t in [0, 1], bin pixels by Y,
    and average within each bin to get a robust LUT entry.

    This dispenses with the matplotlib-reference assumption entirely — we
    learn the actual camera LUT from the data itself.

    Args:
        pixels: (N, 3) in [0, 1]
        n_bins: 256 by default (one LUT entry per bin)
    Returns:
        lut: (n_bins, 3) float64, monotonic in t along axis 0.
    """
    Y = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    sort_idx = np.argsort(Y, kind="stable")
    sorted_px = pixels[sort_idx]
    n = len(sorted_px)
    edges = np.linspace(0, n, n_bins + 1, dtype=np.int64)
    lut = np.empty((n_bins, 3), dtype=np.float64)
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi <= lo:
            lut[i] = sorted_px[max(0, lo - 1)]
            continue
        lut[i] = sorted_px[lo:hi].mean(axis=0)
    return lut


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=DEFAULT_N_IMAGES)
    p.add_argument("--n-pixels", type=int, default=DEFAULT_N_PIXELS)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: results_v2/methods/stc_yolo/_ms1/")
    p.add_argument("--no-figure", action="store_true",
                   help="Skip png generation (faster)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metas = load_metadata()
    image_names = stratified_image_sample(metas, args.n_images, RNG_SEED)
    n_sessions = len(set(metas[n].session_id for n in image_names))
    print(f"sampled {len(image_names)} images across {n_sessions} sessions")

    luts = {name: build_lut(name) for name in CANDIDATE_NAMES}

    rng = np.random.RandomState(RNG_SEED + 1)
    all_pixels: list[np.ndarray] = []
    per_image_diagonal: list[float] = []
    per_image_best_cmap: list[str] = []
    per_image_best_mse: list[float] = []
    per_session_best_cmap: dict[str, Counter] = defaultdict(Counter)

    for k, name in enumerate(image_names):
        img_path = _PROJECT_ROOT / "data" / "OilData" / f"{name}.JPG"
        if not img_path.is_file():
            print(f"  warn: missing {img_path}")
            continue
        px = sample_image_pixels(img_path, args.n_pixels, rng)
        all_pixels.append(px)
        # Per-image "diagonalness" of RGB cube: 0 = perfect grayscale (R=G=B).
        diag = float(np.mean(np.abs(px[:, 0] - px[:, 1]) +
                              np.abs(px[:, 1] - px[:, 2]) +
                              np.abs(px[:, 0] - px[:, 2])))
        per_image_diagonal.append(diag)
        # Per-image best colormap (cheap MSE over this image's pixels).
        per_image_mse = {n: float(lut_fit_per_pixel(px, lut).mean())
                          for n, lut in luts.items()}
        best_for_img = min(per_image_mse, key=per_image_mse.get)
        per_image_best_cmap.append(best_for_img)
        per_image_best_mse.append(per_image_mse[best_for_img])
        per_session_best_cmap[metas[name].session_id][best_for_img] += 1
        if (k + 1) % 50 == 0:
            print(f"  processed {k+1}/{len(image_names)}")

    pixels = np.concatenate(all_pixels, axis=0)
    print(f"total pixels: {pixels.shape[0]}")

    # Dataset-level MSE per candidate.
    print("computing dataset-level MSE per candidate...")
    dataset_mse: dict[str, float] = {}
    for name, lut in luts.items():
        dataset_mse[name] = float(lut_fit_per_pixel(pixels, lut).mean())
    ranked = sorted(dataset_mse.items(), key=lambda kv: kv[1])
    best_name, best_mse = ranked[0]
    second_name, second_mse = ranked[1]
    fit_quality, recommendation = _classify_quality(best_mse)

    # ---- Empirical LUT fit (D1b extension) ----
    # Split pixels 50/50 by image-index for fit/eval to avoid in-sample MSE.
    print("fitting empirical LUT from dataset (50/50 image-stratified split)...")
    n_fit_imgs = len(all_pixels) // 2
    fit_pixels = np.concatenate(all_pixels[:n_fit_imgs], axis=0)
    eval_pixels = np.concatenate(all_pixels[n_fit_imgs:], axis=0)
    empirical_lut = fit_empirical_lut(fit_pixels, n_bins=256)
    empirical_mse_train = float(lut_fit_per_pixel(fit_pixels, empirical_lut).mean())
    empirical_mse_held = float(lut_fit_per_pixel(eval_pixels, empirical_lut).mean())
    print(f"  empirical LUT MSE: train={empirical_mse_train:.6f}  "
          f"held-out={empirical_mse_held:.6f}")
    # Generalization sanity: train/held MSE should be close → no overfit.
    empirical_generalization_ratio = (
        empirical_mse_held / max(empirical_mse_train, 1e-12)
    )

    # Image-level consistency.
    img_counter = Counter(per_image_best_cmap)
    consistency = img_counter[best_name] / len(per_image_best_cmap)
    # Session-level: which sessions disagree with the dataset best?
    session_consensus = {
        sid: c.most_common(1)[0][0]
        for sid, c in per_session_best_cmap.items()
    }
    sessions_agreeing = sum(1 for v in session_consensus.values() if v == best_name)

    # Re-decide using empirical LUT result. If empirical LUT held-out MSE is
    # very small (close to JPG noise floor) and generalization is tight, then
    # the dataset IS single-LUT and we should just analytic-invert the
    # empirical LUT — bypassing matplotlib reference and MLP fallback both.
    if empirical_mse_held < 5e-4 and empirical_generalization_ratio < 1.5:
        final_recommendation = "empirical_analytic_inversion"
        final_quality = "single_LUT_exact_via_empirical"
    elif empirical_mse_held < 2e-3 and empirical_generalization_ratio < 2.0:
        final_recommendation = "empirical_analytic_inversion_with_residual_mlp"
        final_quality = "single_LUT_good_via_empirical"
    else:
        # Empirical fit didn't tighten enough → genuine multi-LUT or noisy
        final_recommendation = recommendation
        final_quality = fit_quality

    out = {
        "n_images_analyzed": int(len(image_names)),
        "n_pixels_per_image": int(args.n_pixels),
        "n_total_pixels": int(pixels.shape[0]),
        "candidate_colormaps": CANDIDATE_NAMES,
        "dataset_level_fit_mse": dataset_mse,
        "ranking": [{"name": k, "mse": v} for k, v in ranked],
        "decision": {
            "best_matplotlib_colormap": best_name,
            "best_matplotlib_mse": float(best_mse),
            "second_best_colormap": second_name,
            "second_mse": float(second_mse),
            "gap_to_second": float(second_mse - best_mse),
            "matplotlib_only_quality": fit_quality,
            "matplotlib_only_recommendation": recommendation,
            "empirical_lut_mse_train": empirical_mse_train,
            "empirical_lut_mse_held": empirical_mse_held,
            "empirical_generalization_ratio": float(empirical_generalization_ratio),
            "final_quality": final_quality,
            "final_recommendation": final_recommendation,
        },
        "per_image_summary": {
            "best_colormap_vote": dict(img_counter.most_common()),
            "consistency_pct": float(consistency * 100),
            "mean_per_image_diagonal_l1": float(np.mean(per_image_diagonal)),
            "max_per_image_diagonal_l1": float(np.max(per_image_diagonal)),
            "median_per_image_diagonal_l1": float(np.median(per_image_diagonal)),
            "mean_per_image_best_mse": float(np.mean(per_image_best_mse)),
        },
        "per_session_consensus": {
            sid: {
                "winner": session_consensus[sid],
                "vote": dict(per_session_best_cmap[sid].most_common()),
            }
            for sid in sorted(session_consensus.keys())
        },
        "n_sessions_agreeing_with_dataset_best": int(sessions_agreeing),
        "n_sessions_total": int(len(session_consensus)),
        "sampling_protocol": {
            "rng_seed_image_select": int(RNG_SEED),
            "rng_seed_pixel_sample": int(RNG_SEED + 1),
            "sessions_covered": int(n_sessions),
            "stratification": "uniform per session",
        },
    }

    # Save JSON.
    snap_dir = (Path(args.output_dir) if args.output_dir else
                _PROJECT_ROOT / "results_v2" / "methods" / "stc_yolo" / "_ms1")
    snap_dir.mkdir(parents=True, exist_ok=True)
    json_path = snap_dir / "colormap_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # Save the empirical LUT as npz for downstream C4 consumption.
    lut_path = snap_dir / "empirical_lut.npz"
    np.savez(
        lut_path,
        lut_rgb=empirical_lut.astype(np.float32),
        t=np.linspace(0.0, 1.0, empirical_lut.shape[0], dtype=np.float32),
    )
    print(f"\nwrote {json_path}")
    print(f"wrote {lut_path}")
    print(f"=" * 60)
    print(f"MATPLOTLIB-LUT BEST: {best_name}  mse={best_mse:.6f}  ({fit_quality})")
    print(f"  gap to 2nd ({second_name}): {second_mse - best_mse:.6f}")
    print(f"EMPIRICAL LUT: train={empirical_mse_train:.6f}  "
          f"held={empirical_mse_held:.6f}  "
          f"gen-ratio={empirical_generalization_ratio:.3f}")
    print(f"FINAL: quality={final_quality}")
    print(f"       recommendation={final_recommendation}")
    print(f"       per-image consistency: {consistency*100:.1f}%")
    print(f"       sessions agreeing: {sessions_agreeing}/{len(session_consensus)}")
    print(f"       mean |R-G|+|G-B|+|R-B|: {np.mean(per_image_diagonal):.4f}")
    print(f"=" * 60)

    # Save figure.
    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # (a) Bar chart of MSE per candidate.
            names = [r["name"] for r in out["ranking"]]
            mses = [r["mse"] for r in out["ranking"]]
            colors = ["tab:green" if n == best_name else "tab:gray" for n in names]
            axes[0].barh(names, mses, color=colors)
            axes[0].set_xlabel("Mean squared distance to nearest LUT entry")
            axes[0].set_title(f"Colormap fit MSE on {pixels.shape[0]} pixels  (lower = better)")
            axes[0].invert_yaxis()
            for i, v in enumerate(mses):
                axes[0].text(v, i, f"  {v:.4f}", va="center", fontsize=8)
            # (b) RGB cloud (R vs G colored by B) + best LUT + empirical LUT.
            n_plot = min(5000, pixels.shape[0])
            sub_idx = np.random.RandomState(0).choice(
                pixels.shape[0], n_plot, replace=False)
            sub = pixels[sub_idx]
            scat = axes[1].scatter(sub[:, 0], sub[:, 1], c=sub[:, 2],
                                    cmap="gray", s=4, alpha=0.5, vmin=0, vmax=1)
            best_lut = luts[best_name]
            axes[1].plot(best_lut[:, 0], best_lut[:, 1], "r--", lw=2,
                          label=f"matplotlib {best_name}  (mse={best_mse:.4f})")
            axes[1].plot(empirical_lut[:, 0], empirical_lut[:, 1], "lime", lw=2,
                          label=f"empirical LUT  (held-mse={empirical_mse_held:.4f})")
            axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
            axes[1].set_xlabel("R"); axes[1].set_ylabel("G")
            axes[1].set_title("Dataset RGB cloud (point color = B)  + LUT fits")
            axes[1].legend(loc="upper left")
            plt.colorbar(scat, ax=axes[1], label="B value")
            plt.tight_layout()
            png_path = snap_dir / "colormap_analysis.png"
            plt.savefig(png_path, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {png_path}")
        except Exception as e:
            print(f"warn: figure generation failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
