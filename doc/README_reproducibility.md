# Reproducibility and revised archive

For the revised manuscript **Seed variability and class averaging can obscure detector improvements on small infrared datasets**, use [Zenodo version 1.0.0](https://doi.org/10.5281/zenodo.23073467).

The GitHub results directory and historical experiment drivers retain earlier project material. Revised figure numbering, statistical summaries and source-data files are supplied in the archive; the old claim-to-figure map is superseded.

## Inspecting the saved evidence

After extracting `measuring-ir-detectors-v1.0.0.zip`, consult:
- `paper/source_data/per_seed_results.csv` and `data.json` for the 95 saved run records.
- `paper/source_data/stats.json` and `diagnostics.json` for existing summaries.
- `paper/figures/` for the revised main and supplementary figures.
- `doc/REPRODUCIBILITY.md` for the recorded environment, detailed file map, optional execution instructions and limitations.
- `provenance/` and `ARCHIVE_MANIFEST.csv` for source provenance and checksums.

Reading these files does not require running research programs. No training, inference, statistical reanalysis or plotting was performed during archive assembly or this documentation update.

## Interpretation and execution limits

The main comparison uses ten matched downstream seeds on one fixed transformer-data split and one SSL checkpoint. SSL used training and validation images, not detector-test images. HIT-UAV includes baseline repetitions only.

C3 clips the temperature encoding, so recorded temperatures 50 and 70 share an encoding; equipment code 5 produces an all-zero location vector. C4 calibration sampled all 22 filename-defined groups, including detector-test groups, and must be interpreted as exploratory under that protocol.

The archive contains a saved SSL checkpoint, but not the upstream detector checkpoint, all trained detector checkpoints or the gap-six pretraining checkpoint. Historical scripts can reference paths from the original research computer. Dependencies are not a complete locked environment, and exact retraining is not asserted. The archive documentation explains these limits before any optional execution.
