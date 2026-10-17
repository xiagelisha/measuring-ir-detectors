# Reproducing the detector analyses

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
interpretations. Use the current map in `revision_r2/README.md` for these analyses.

The datasets, supporting training code and statistical analysis package are
archived at https://doi.org/10.5281/zenodo.23430799 (version 2.0.0).
The preceding archive remains at https://doi.org/10.5281/zenodo.23073467.

The original training utilities and checkpoint-capture scripts have additional
dependencies and expect the recorded project layout and model weights. The archive
contains a saved SSL checkpoint, but does not include every detector checkpoint.
The portable statistical reanalysis is independent of those training dependencies.
