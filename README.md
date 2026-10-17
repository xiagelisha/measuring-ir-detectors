# A case study of training and test uncertainty in infrared transformer leak detection

Code and source data accompanying the revised manuscript (under review).

## October 2026 revision

Start with **[revision_r2/README.md](revision_r2/README.md)** for the current analyses,
source data and six main figures. The primary comparison uses ten downstream runs
per initialisation on a fixed split of 2,593 transformer infrared images. The revision
re-evaluates the 20 saved primary checkpoints and separates test-sample uncertainty
from variation across training runs. No new detector training was performed.

The mean SSL–baseline mAP50 difference is +0.008456. Shared-image resampling with
the fitted models held fixed gives a positive interval; adding training variation
gives a 95% interval of −0.03231 to 0.04707. Class-level summaries, filename-group
resampling and HIT-UAV baseline repetitions show how aggregation and sampling
choices affect the interpretation of detector comparisons.

## Reproduce the revision analysis

Requires Python 3.11 or later. From the repository root:

```bash
python -m pip install -r revision_r2/requirements.txt
python revision_r2/code/analyse_uncertainty.py --bundle revision_r2 --replicates 2000
python revision_r2/code/reanalyse_statistics.py --bundle revision_r2
```

These commands use the supplied measurements and per-image detection statistics;
they do not require a GPU, model weights or Ultralytics. The bootstrap command
checks the point estimates against all 60 recorded primary scores. It uses the
supplied per-run caches for the 2,000 image resamples, and recomputes group and
joint intervals. Move the `analysis/bootstrap_*.npy` caches aside to recompute
the image resamples from detections. The statistics command independently
recomputes the contrasts and covariance summaries and checks the saved values.

## Repository layout

- `revision_r2/`: current revision measurements, analysis code, figures and checksums.
- `methods/m6_stc_yolo/`: original detector implementation and configurations.
- `baselines/`, `shared/`, `tools/`: original training and evaluation utilities.
- `data/`: transformer images, annotations, fixed splits and HIT-UAV export.
- `results/`, `doc/figures/`: retained historical results and figures.
- [doc/README_reproducibility.md](doc/README_reproducibility.md): current file map and execution notes.

The original 95 training-run measurements are preserved. C3/C4 and interaction
summaries serve as historical reporting examples in the Supplementary Information.
The primary uncertainty analysis holds the training data and SSL checkpoint fixed.
The four test filename groups describe the recorded test contexts.

## Data and licences

The transformer dataset contains 2,593 stored pseudo-colour infrared images at
640 × 480 pixels, collected mainly during controlled oil-leakage simulations.
Water labels denote applied water used as a non-oil liquid target. Equipment-location
codes identify the component imaged. The fixed split has 1,886 training, 348
validation and 359 test images. SSL uses training and validation images with their
annotations ignored. See [data documentation](data/README.md).

Project code retains the MIT licence. Transformer images and annotations and the
redistributed HIT-UAV export retain CC BY 4.0, with attribution. Third-party software
and upstream model-derived materials retain their own terms.

## Authors and support

Wenbi Tan¹˒², Tzer Hwai Gilbert Thio², Fei Lu Siaw², Shuai Cui¹, Zilong Su¹ and Yule Xia²˒³*.

1. School of Intelligent Manufacturing, Yunnan Technology and Business University, Kunming, 651700, China.
2. Center for Sustainability in Advanced Electrical and Electronic Systems (CSAEES), Faculty of Engineering, Built Environment and Information Technology, SEGi University, Petaling Jaya 47810, Malaysia.
3. School of Mechanical and Electrical Engineering, Hunan Biological and Electromechanical Polytechnic, Changsha 410127, China.

*Correspondence: xiayule@hnbemc.edu.cn.

This research was funded by the Scientific Research Fund of the Education Department of Hunan Province, China (Nos. 25B1073, 24C1020), and the Scientific Research and Social Service Team Project of Hunan Biological and Electromechanical Polytechnic (No. 25TD05).

## Citation and archive versions

The October 2026 revision is archived as **version 2.0.0** at
[10.5281/zenodo.23430799](https://doi.org/10.5281/zenodo.23430799).
Use this version DOI when citing the code and source data for the revised study;
machine-readable metadata are in [CITATION.cff](CITATION.cff).

The archive contains the original complete research package, the current revision
analyses and figures, and `README_v2.txt`, which explains how to use both packages.
The revision code and scientific data correspond to commit
`80f24e6556677a121559a388add2b3cbfc1e8dd8`; archive-status wording and checksum
metadata were adapted for the Zenodo deposit.

The **preceding version 1.0.0** remains available at
[10.5281/zenodo.23073467](https://doi.org/10.5281/zenodo.23073467), under the earlier
title “Seed variability and class averaging can obscure detector improvements on
small infrared datasets”. Cite that version only for the preceding release.
