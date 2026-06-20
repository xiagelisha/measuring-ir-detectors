# STC-YOLO — Semi-supervised Temporal-Consistency YOLO (Paper 4)

> Base: YOLOv12s.
> Status: MS1 skeleton (2026-05-29).

## 4+1 Innovations

| ID | Module file | Milestone | Status |
|---|---|---|---|
| C1 | `modules/ssl_pretrain.py` — Cross-Frame Token Consistency SSL pretrain | MS2 | stub |
| C2 | `modules/mean_teacher.py` — Temporal Mean-Teacher semi-supervised FT | MS3 | stub |
| C3 | `modules/session_film.py` — Session-FiLM detect head | MS3 | stub |
| C4 | `modules/colormap_inverter.py` + `modules/conv1ch_from_rgb.py` — colormap inversion input layer | MS1 | stub (this MS) |
| C5 | `modules/temporal_voting.py` (optional) — test-time 1-frame voting | MS4 | not yet |

## What this method explicitly does **NOT** do (vs paper3)

Paper3 already explored these mechanisms — STC-YOLO must not overlap:

- (excluded) Frozen Sobel thermal gradient (TG-Adapter, N5)
- (excluded) Multi-scale thermal gradient (N2)
- (excluded) Oil-Aware Focal Loss (N1)
- (excluded) L2-SP knowledge-preserving FT (N3)
- (excluded) Mid-level spatial attention on Sobel magnitude (N4)
- (excluded) IR intensity post-hoc calibration (N6)
- (excluded) Custom multi-scale WBF TTA (N7)
- (excluded) Input-level 3D conv on stacked frames (m1 TempoSpread paused)

The mechanism distinction is enforced in module docstrings.

## Directory layout

```
m6_stc_yolo/
├── README.md                            # this file
├── __init__.py
├── model.py                             # build_stc_yolo(...) factory
├── train_one.py                         # single-seed training entry
├── configs/
│   ├── default.yaml                     # base config (paper3 tuned recipe baseline)
│   ├── ssl_pretrain.yaml                # C1 SSL pretrain config (MS2)
│   └── semi_supervised.yaml             # C2+C3 MT+FiLM fine-tune config (MS3)
├── modules/
│   ├── __init__.py
│   ├── colormap_inverter.py             # C4 (D2)
│   ├── conv1ch_from_rgb.py              # C4 (D3) — 1-ch first layer + ch-mean transfer
│   ├── ssl_pretrain.py                  # C1 (MS2)
│   ├── mean_teacher.py                  # C2 (MS3)
│   └── session_film.py                  # C3 (MS3)
├── data/
│   ├── __init__.py
│   ├── session_metadata.py              # functional — metadata.csv loader for C3
│   ├── frame_pair_dataset.py            # C1 (MS2) — (t, t+k) pair sampler
│   └── unlabeled_neighbor_sampler.py    # C2 (MS3) — train-session-only gap=1 sampler
└── tests/
    ├── __init__.py
    └── test_skeleton.py                 # import sanity + metadata loader
```

## How to run (MS1)

```powershell
# Sanity test the skeleton imports
conda activate oilleak
python -m pytest methods\m6_stc_yolo\tests\test_skeleton.py -v

# Baseline reproduce (D4 — vanilla YOLOv12s tuned recipe)
python methods\m6_stc_yolo\train_one.py --config methods\m6_stc_yolo\configs\default.yaml --seed 42
```

## Engineering discipline (沿用 paper3)

1. 每模块独立 unit test 才进下一步
2. 30-epoch quick-verify (mAP > 0.10) 才进 200 epoch full training
3. 每 milestone 通过 → snapshot 到 `results_v2/methods/stc_yolo/_msN/`
4. baselines/ 不动 — STC-YOLO 在 methods/ 下独立
5. 共享接口走 `shared.OilLeakDataset` / `shared.compute_detection_metrics`
