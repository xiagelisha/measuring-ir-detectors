# Reproducing the October 2026 revision

Current manuscript: **A case study of training and test uncertainty in infrared transformer leak detection**.

Use [revision_r2/README.md](../revision_r2/README.md) for the runnable commands,
file definitions and current figure/table map. That folder contains original
run-level measurements, per-image statistics for the 20 primary detectors,
new uncertainty analyses and the current six main and three supplementary figures.

The portable analysis commands require Python, NumPy and SciPy. They verify the
point estimates and recompute the reported intervals and statistical comparisons
without retraining. Image-bootstrap caches can be rebuilt from the supplied
detection statistics. The original capture and plotting scripts are retained as
provenance with their project-layout requirements stated explicitly.

Historical `results/` and `doc/figures/` files retain earlier numbering and
interpretations. Use the current map in `revision_r2/README.md` for this manuscript.

The preceding complete archive, including datasets and supporting training code,
is [Zenodo version 1.0.0](https://doi.org/10.5281/zenodo.23073467). It uses the earlier
title “Seed variability and class averaging can obscure detector improvements on
small infrared datasets”. The new GitHub revision folder is awaiting a new Zenodo
version; the old DOI continues to identify the preceding archive.

The original training utilities and checkpoint-capture scripts have additional
dependencies and expect the recorded project layout and model weights. The archive
contains a saved SSL checkpoint, but does not include every detector checkpoint.
The portable statistical reanalysis is independent of those training dependencies.
