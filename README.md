# Measuring Detector Improvements Reliably on Small Infrared Datasets

Code, results, and reproduction scripts for the paper *"Measuring Detector
Improvements Reliably on Small Infrared Datasets: A Cautionary Study of
Self-Supervised and Architectural Enhancements"* (under review).

The study evaluates five reasonable enhancements to a tuned YOLOv12s detector on
small infrared (IR) data and uses the campaign to expose two measurement
pitfalls — **seed under-sampling** and **aggregate masking** — then distills a
power-aware, class-stratified reporting protocol. This repository releases the
full pipeline so the comparisons can be reproduced and extended.

This repository contains everything needed to verify every claim in the paper:
the **full oil dataset**, the frozen per-seed results behind every table and
figure, and the complete training/evaluation/analysis pipeline.

## Repository layout
```
methods/m6_stc_yolo/        STC-YOLO implementation
  model.py                  YOLOv12s + optional C1–C4 modules
  train_one.py              single-run train/eval entry point
  modules/                  ssl_pretrain (C1), colormap_inverter (C4),
                            session_film (C3), mean_teacher (C2), conv1ch_from_rgb
  data/                     frame-pair dataset, session metadata, neighbour sampler
  scripts/run_ssl_pretrain.py   cross-frame SSL pretraining
  configs/                  default / ssl_pretrain / semi_supervised yaml
  tests/                    pytest suite
baselines/tier1_ultralytics/  tuned YOLOv12s baseline + data-yaml builder
shared/                     seeding, metrics, logging, dataset utilities
tools/                      paper4_orchestrator.py (decision tree), paper4_run_5seed.py,
                            split / metadata / verification scripts
data/                       full oil dataset + HIT-UAV (see data/README.md)
results/                    all experimental data behind the paper's tables & figures
doc/                        reproducibility notes + paper figures
```

## Installation
```bash
conda create -n oilleak python=3.11 -y
conda activate oilleak
pip install -r requirements.txt
# Install a torch build matching your CUDA (the paper used torch 2.x + CUDA 12.8).
```

## Reproducing the results
The numbers in every table and figure are already in `results/` (see
`results/README.md` for the artifact→file map). To re-run experiments:
```bash
# from the repository root (so package imports resolve)
# 1) cross-frame SSL pretraining (C1)
python -m methods.m6_stc_yolo.scripts.run_ssl_pretrain --config methods/m6_stc_yolo/configs/ssl_pretrain.yaml
# 2) a single seeded train+eval (set the config to choose C1–C4)
python -m methods.m6_stc_yolo.train_one --config methods/m6_stc_yolo/configs/default.yaml --seed 42
# 3) the multi-seed campaign / decision tree
python tools/paper4_run_5seed.py --help
python tools/paper4_orchestrator.py status
```
Run `pytest methods/m6_stc_yolo/tests` for the unit suite.

## Data
- **Oil dataset (public, full).** 2,593 substation IR frames (1280x720) with
  YOLO-format labels, per-image metadata, and the fixed `main_holdout` split,
  under `data/oil/` and `data/splits/`. Released under **CC BY 4.0**
  (`data/oil/LICENSE`).
- **HIT-UAV (public).** Included under `data/HIT_UAV/` under its original
  CC BY 4.0 license (see `data/HIT_UAV/ATTRIBUTION.md`).

See `data/README.md` for the column guide (including a label-leakage warning
about the `day_night` metadata column) and dataset statistics.

## License
- **Code:** MIT License (see `LICENSE`).
- **Oil dataset:** CC BY 4.0 (`data/oil/LICENSE`).
- **HIT-UAV:** original dataset license, CC BY 4.0 (`data/HIT_UAV/`).

## Citation
```bibtex
@misc{tan_measuring_ir,
  author = {Tan, Wenbi and others},
  title  = {Measuring Detector Improvements Reliably on Small Infrared Datasets:
            A Cautionary Study of Self-Supervised and Architectural Enhancements},
  year   = {2026},
  note   = {Under review}
}
```
