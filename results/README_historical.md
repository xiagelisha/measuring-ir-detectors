> Historical documentation retained from the preceding GitHub version. For the current analysis and interpretation, see [revision_r2](../revision_r2/README.md).

# result_data_paper4 — Experimental Data Supporting the Paper

All numbers in the paper's tables and figures derive from the JSON files here.
Each `runs/<config>/` holds the multi-seed summary (`5seed_summary.json`) and the
per-seed test metrics (`seed<k>_test_metrics.json`, incl. `per_class_ap50`).
Protocol: test mAP50, tuned inference (imgsz 1056 + TTA for the oil dataset),
paired by seed. Test split excluded from all SSL pretraining.

## `runs/` — per-configuration results
| folder | paper label | seeds (n) | role |
|---|---|---|---|
| `ms1_d4a_baseline` | Baseline (vanilla YOLOv12s, tuned) | 42–46 | baseline (n=5 half) |
| `baseline_ext_n10` | Baseline (extension) | 47–51 | baseline (n=10 with above) |
| `ms2_d8b_ssl_only_diag` | + SSL pretrain (C1) | 42–46 | SSL-only (n=5 half) |
| `ssl_only_ext_n10` | + SSL pretrain (extension) | 47–51 | SSL-only (n=10 with above) |
| `ms1_d4b_c4_enabled` | C4 alone (colormap-inversion, vanilla) | 42–46 | n=5 |
| `ms2_d8_supervised_ft_5seed` | SSL + C4 | 42–46 | SSL+C4 (n=5 half) |
| `ssl_c4_ext_n10` | SSL + C4 (extension) | 47–51 | SSL+C4 (n=10 with above) |
| `ssl_c3_5seed` | SSL + C3 (session-FiLM) | 42–46 | n=5 half |
| `ssl_c3_ext_n10` | SSL + C3 (extension) | 47–51 | n=10 with above |
| `ssl_c2_5seed` | SSL + C2 (Mean-Teacher) | 42–46 | n=5 half |
| `ssl_c2_ext_n10` | SSL + C2 (extension) | 47–51 | n=10 with above |
| `vanilla_c3_5seed` | vanilla + C3 (no SSL) | 42–46 | falsifies SSL-fragility (Table 3) |
| `vanilla_c2_5seed` | vanilla + C2 (no SSL) | 42–46 | falsifies SSL-fragility (Table 3) |
| `ssl_gap6_5seed` | SSL, wider pretext gap k≤6 | 42–46 | pretext-hardness ablation |
| `hituav_baseline_n10` | HIT-UAV baseline (2nd dataset) | 42–51 | replication (Sec. HIT-UAV, Fig 7) |

Note: `ms1_d4a_baseline`, `ms1_d4b_c4_enabled`, `ms2_d8_supervised_ft_5seed` per-seed
values are also embedded in `orchestrator_state.json`.

## `analysis/` — derived results (source of the paper's stated numbers)
- `RESULTS_FROZEN.json` — all oil-dataset configs: n / mean / std / per-seed (single source of truth).
- `RESULTS_FROZEN_hituav.json` — HIT-UAV aggregate + per-class + minority/majority std ratio (10.4×).
- `significance_n5.json` / `significance_n10.json` — paired t-test / Wilcoxon / Cohen's d (SSL vs baseline; enhancements vs SSL), at 5 and 10 seeds.
- `final_2x2_and_reallocation.json` — fragility verdict + Oil/Water class reallocation.
- `main_results_table.json` — compiled main results table.

## `ssl_pretrain/`
- `ssl_pretrain_summary.json` — C1 SSL pretraining config + 80-epoch loss history (Fig 1).

## Paper artifact → data map
- Table 1 (n5 vs n10) ← `analysis/RESULTS_FROZEN.json` + `significance_n5/n10.json`
- Table 2 (per-class reallocation) ← `analysis/final_2x2_and_reallocation.json`
- Table 3 (vanilla vs SSL backbone) ← `runs/vanilla_c*`, `runs/ssl_c*`, `RESULTS_FROZEN.json`
- Fig 1 (SSL loss) ← `ssl_pretrain/ssl_pretrain_summary.json`
- Fig 2/3/6 (evaporation / per-seed / seed-lottery) ← baseline + SSL-only per-seed
- Fig 4 (main bar n=10) ← `RESULTS_FROZEN.json`
- Fig 5 (reallocation) ← `final_2x2_and_reallocation.json`
- Fig 7 (HIT-UAV per-class) ← `runs/hituav_baseline_n10` + `RESULTS_FROZEN_hituav.json`
- Fig 8 (power curve) ← computed from the effect sizes in `significance_n10.json`

Heavy artifacts NOT included here (kept in original run dirs): model checkpoints
(`best.pt`), training curves (`results.csv`), ultralytics plots.
