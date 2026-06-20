# Reproducibility — Measuring Detector Improvements Reliably on Small IR Datasets

Code and configs to reproduce every result in the paper. All experiments use the
`main_holdout` split of the substation IR dataset (2,593 images; 1,886 train /
348 val / 359 test). The test split is excluded from all SSL pretraining.

## Environment
- Python 3.11, PyTorch (CUDA), `ultralytics` (YOLOv12s), `scipy`, `matplotlib`.
- Conda env used: `oilleak`. Single GPU (results obtained on an RTX 5080, 16 GB).

## Layout
- `methods/m6_stc_yolo/` — model factory, configs, training entry points.
  - `configs/default.yaml` — tuned baseline recipe (AdamW, lr 1e-3, 30 ep, imgsz 800, mixup 0.1; infer imgsz 1056 + TTA).
  - `configs/ssl_pretrain.yaml` — C1 SSL pretrain (cross-frame dense InfoNCE, gap k≤2, 1024 tokens, batch 16).
  - `scripts/run_ssl_pretrain.py` — SSL backbone pretraining (resumable; per-epoch `last.pt`).
  - `train_one.py` — single-seed train + test eval; `--override key=value` for STC switches.
- `tools/paper4_run_5seed.py` — multi-seed driver (writes `5seed_summary.json` per run).
- `results_v2/methods/stc_yolo/_paper/` — frozen results, figures, tables (see `RESULTS_FROZEN.md`).

## Step 1 — SSL backbone (C1)
```
python methods/m6_stc_yolo/scripts/run_ssl_pretrain.py \
    --config methods/m6_stc_yolo/configs/ssl_pretrain.yaml
# -> results_v2/methods/stc_yolo/_ms2/pretrained_backbone_stc.pt  (final loss 2.576)
```

## Step 2 — multi-seed runs (seeds 42-51 for n=10)
`$BB = results_v2/methods/stc_yolo/_ms2/pretrained_backbone_stc.pt`. Run each with
`--seeds 42,43,44,45,46` then `47,48,49,50,51`.
```
# baseline (vanilla, tuned)        : (no stc overrides)
# SSL (C1)                         : --override stc.ssl_pretrained_backbone=$BB --override stc.enable_colormap_inversion=false
# SSL+C4                           : ... ssl_pretrained_backbone=$BB --override stc.enable_colormap_inversion=true
# SSL+C3                           : ... ssl_pretrained_backbone=$BB --override stc.enable_session_film=true
# SSL+C2                           : ... ssl_pretrained_backbone=$BB --override stc.enable_mean_teacher=true
# vanilla+C3                       : --override stc.enable_session_film=true
# vanilla+C2                       : --override stc.enable_mean_teacher=true
# C4 alone                         : --override stc.enable_colormap_inversion=true
# gap6 ablation                    : re-run Step 1 with pair_gap_max=6, then SSL-only run
```
Each invocation: `python tools/paper4_run_5seed.py --config methods/m6_stc_yolo/configs/default.yaml --run-name <name> --seeds <list> <overrides>`.
Per-seed test metrics: `results_v2/methods/stc_yolo/<run_name>_seed<k>/ultralytics_test_metrics.json`.

## Step 3 — statistics & figures
- `RESULTS_FROZEN.json` — all configs' n / mean / std / per-seed (re-derived from the run summaries; single source of truth).
- `significance_n5.json`, `significance_n10.json` — paired t-test / Wilcoxon / Cohen's d (mAP50, Oil AP50; SSL vs baseline and each enhancement vs SSL).
- `figures_ieee/` — the 6 paper figures (PDF+PNG) regenerated from `RESULTS_FROZEN.json`.

## Claim → artifact map
| Paper claim | Artifact |
|---|---|
| SSL +0.023 (n=5) → +0.008 (n=10, p=0.69) | `significance_n5/n10.json`, Fig 2 |
| Seed lottery: 64% gain / 36% loss over 252 draws | Fig 6 (computed from frozen per-seed diffs) |
| Oil +0.035 / Water −0.018 reallocation | `final_2x2_and_reallocation.json`, Table II, Fig 5 |
| C3 −0.072 (p=0.045), C2 −0.063 (p=0.039); both hurt vanilla too | `significance_n10.json`, Table III |
| gap6 → 0.590, 3× variance | `RESULTS_FROZEN.json` (`SSL gap6`) |

## Notes
- SSL pretraining is transductive (uses all non-test frames, label-free); a session-disjoint LOSO CV is not included (see paper Limitations).
- RNG nondeterminism is controlled by the per-run seed (`set_seed`); minor hardware/library drift may shift the third decimal.
