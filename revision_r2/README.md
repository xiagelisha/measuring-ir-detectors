# Source data and statistical analysis

Code, measurements and reproducible analyses for infrared detector evaluation.

The main estimand is the difference of ten-run mean mAP50 (SSL minus baseline).
Twenty existing checkpoints were re-evaluated; all 60 original aggregate and class
scores were reproduced exactly. The original 95 training-run records are retained.

## Reanalysis

From the repository root, with Python 3.11 or later:

```bash
python -m pip install -r revision_r2/requirements.txt
python revision_r2/code/analyse_uncertainty.py --bundle revision_r2 --replicates 2000
python revision_r2/code/reanalyse_statistics.py --bundle revision_r2
```

No GPU, detector weights, images or Ultralytics installation are needed for these
two commands. Use a working copy: the bootstrap command rewrites its derived
outputs in `analysis/`. The second command verifies the saved statistical summaries
and writes `analysis/recomputed_statistics.json`.

The 2,000-replicate image-bootstrap caches are included. To rebuild them from the
per-image detection statistics, move `analysis/bootstrap_*.npy` aside before running
the first command. The nine group-count vectors and all joint intervals are
recomputed on every run. AP is recomputed from pooled detections rather than
averaged over per-image AP. Random seed: 20261014.

## Files

| Location | Contents |
|---|---|
| `evaluation/` | Twenty NPZ files of detection statistics; checkpoint SHA-256, software versions, inference settings and original/reproduced scores |
| `metadata.csv` | Image-to-filename-group mapping and recorded metadata |
| `analysis/per_run_test_uncertainty.csv` | Per-run point values and image/group intervals |
| `analysis/per_group_ap.csv` | Every represented-class group/seed AP |
| `analysis/uncertainty_summary.json` | Shared-image and joint seed/test intervals |
| `analysis/additional_statistics.json` | Matching correlations, Welch intervals, HIT-UAV covariance, condition counts and saved SSL-state audit |
| `analysis/group_allocation.csv` | Complete group/class allocation checked against annotations |
| `analysis/bootstrap_distributions.npz` | Full resampling distributions |
| `original_source_data/` | Unchanged source measurements and statistical summaries from the preceding dataset |
| `code/` | Portable bootstrap and run-level statistical reanalysis |
| `provenance/` | Original checkpoint-capture, audit and figure-generation scripts |
| `figures/` | Six analysis figures and three supporting figures |
| `SHA256.json` | SHA-256 checksums of all supplied files except this manifest |

In each evaluation NPZ, `names` identifies images; `pred_counts` and `target_counts`
index the concatenated arrays; `tp` is the detection-match matrix at IoU
0.50:0.05:0.95; `conf` and `pred_cls` describe detections, and `target_cls` describes
ground-truth objects. Class 0 is Water and class 1 is Oil. Validator order is retained.

## Figure and table map

- Figure 1 and Table 1: `analysis/group_allocation.csv`, `metadata.csv`, original split summaries.
- Figures 2–4 and Table 2: `original_source_data/data.json`, `stats.json` and `diagnostics.json`.
- Figure 5: `analysis/per_run_test_uncertainty.csv`, `per_group_ap.csv` and `uncertainty_summary.json`.
- Figure 6: assumed paired effect sizes 0.2, 0.4 and 0.8, and the pilot effect in `stats.json`; exact noncentral-t power calculations.
- Supplementary Figures S1–S3: original run records, pretraining history, distribution diagnostics and historical comparisons.
- The supplementary uncertainty, correlation and covariance results are in `analysis/`.

## Scope and provenance

Image-only intervals condition on the 20 fitted detectors. Joint intervals also
resample downstream training runs, holding the training data and SSL checkpoint
fixed. The same image resample is shared across models. Independent seed resampling
is the primary analysis, with nominal matching retained as a sensitivity analysis.
Group resampling draws two groups per class from the four observed test groups,
giving nine distinct count vectors. All reported bootstrap intervals are percentile intervals.

C3/C4 and interaction records are historical reporting examples. Original comparison
families and measurements are preserved. The full-precision stored values underpin
the rounded values in the analysis summaries.

Scripts in `provenance/` document the original project execution. They require the
original `paper4` layout, model checkpoints and (for plotting) its dependencies and
source files. They are not the standalone entry points above. Model checkpoints
are not duplicated in this analysis folder. The two portable commands reproduce
the new statistical calculations from the supplied measurements.

## Archive and licence

This analysis package is archived in Zenodo version 2.0.0, DOI
[10.5281/zenodo.23430799](https://doi.org/10.5281/zenodo.23430799), alongside the
original datasets and supporting code. See README_v2.txt in the archive for package
selection. The preceding version DOI 10.5281/zenodo.23073467 remains available.
Project code is MIT-licensed; study data retain CC BY 4.0. Third-party materials
retain the terms documented in the root repository and preceding archive.
