# Seed variability and class averaging can obscure detector improvements on small infrared datasets

Research code and historical results accompanying the revised manuscript of the same title.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23073467.svg)](https://doi.org/10.5281/zenodo.23073467)

## Revised manuscript materials

The **complete version 1.0.0 archive** is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.23073467). Download `measuring-ir-detectors-v1.0.0.zip` from that record for the revised figures, source data, 95 saved detector-training run records, datasets and supporting code.

This GitHub repository retains the earlier code and results layout. Its historical figures and summaries should not be confused with the revised manuscript materials in the DOI archive. The archive README and `doc/REPRODUCIBILITY.md` describe its contents and limitations.

In the extracted Zenodo archive, start with:
- `paper/source_data/per_seed_results.csv` and `data.json`: saved run-level measurements.
- `paper/source_data/stats.json` and `diagnostics.json`: existing statistical summaries.
- `paper/figures/`: revised main and supplementary figures.
- `doc/REPRODUCIBILITY.md`: file map, recorded environment and optional execution instructions.
- `ARCHIVE_MANIFEST.csv`: file checksums and sizes.

Inspecting the saved evidence does not require retraining. No training, inference, statistical reanalysis or figure generation was performed during archive assembly or this documentation update.

## Study scope

The main comparison uses ten matched downstream seeds on one fixed transformer-data split and one SSL checkpoint. Five-seed subsets describe sensitivity within the observed runs. HIT-UAV contains baseline repetitions only. Aggregate and class-level effects should be interpreted with their confidence intervals and multiplicity qualifications.

C3 uses the archived metadata encoding, including its limitations. C4 uses an empirical colour-map lookup table calibrated using all 22 filename-defined groups, including detector-test groups; it is an exploratory comparison under that protocol. These qualifications do not change the recorded experimental results.

## Repository layout

- `methods/m6_stc_yolo/`: method implementation and configurations.
- `baselines/`, `shared/`, `tools/`: training utilities and historical experiment drivers.
- `data/`: dataset exports, metadata and split lists.
- `results/`, `doc/`: historical results and documentation.

For the revised archive's layout and execution instructions, use the documentation inside the Zenodo ZIP. Historical scripts may refer to paths from the original research computer; exact retraining across environments is not asserted.

## Data and licences

The transformer dataset contains 2,593 stored pseudo-colour infrared images at **640 × 480 pixels**, collected mainly during controlled oil-leakage simulations with a FOTRIC 626C camera. Equipment-location identifiers describe the component imaged, not separate substations. Filename groups do not establish independent physical recording sessions.

Project code retains its **MIT licence**. Transformer images and annotations and the redistributed HIT-UAV export retain **CC BY 4.0**, with attribution. Third-party software and upstream model-derived materials remain subject to their own terms. See [data documentation](data/README.md) and the licence files.

## Citation

Please cite the archived code and source data using this version-specific DOI. It identifies the research archive, not a published journal article.

> Tan, W., Thio, T. H. G., Siaw, F. L., Cui, S., Su, Z. & Xia, Y. (2026). Code and source data for “Seed variability and class averaging can obscure detector improvements on small infrared datasets”. Zenodo. https://doi.org/10.5281/zenodo.23073467

Machine-readable citation metadata is provided in `CITATION.cff`.
