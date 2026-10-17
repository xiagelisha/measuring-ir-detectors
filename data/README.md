# Data

The datasets and supporting code are archived in [Zenodo version 2.0.0](https://doi.org/10.5281/zenodo.23430799), alongside the [statistical analysis package](../revision_r2/README.md). The original images, annotations and split lists are preserved. The preceding archive is available at https://doi.org/10.5281/zenodo.23073467.

## Transformer infrared images

There are 2,593 stored images at 640 × 480 pixels and matching YOLO annotations: Water = 0 and Oil = 1. Rechecking every annotation confirms 17,654 Water boxes and 7,803 Oil boxes. Water denotes applied water used as a non-oil liquid target. The full group/class audit is in `revision_r2/analysis/group_allocation.csv` at the repository root. The fixed split contains 1,886 training, 348 validation and 359 test images across 14, four and four filename-defined groups.

Images were collected mainly in controlled transformer oil-leakage simulations using a FOTRIC 626C camera at a laboratory at Kunming University, China, in October 2025. Stored image dimensions are not a camera-sensor specification. Location codes identify the equipment component viewed: insulator, tank cover, tank side, radiator or drain valve. They do not identify separate substations. Physical-session independence cannot be established from filename groups alone.

The GitHub release uses `oil/images/`, `oil/labels/` and `oil/metadata.csv`. The Zenodo archive restores `data/OilData/`, `data/OilLabels/` and `data/metadata.csv` for the archived code. The original numerical metadata and fixed split lists have not been changed.

### Metadata and partition

- `image_name` is an image stem; image and annotation paths identify files.
- `session_id` is a filename-defined operational grouping, not a verified independent recording session.
- `location` encodes the imaged equipment component.
- `day_night` is a legacy class-related field, not a time-of-day annotation; it is excluded from C3. Label-derived fields are for auditing, not model inputs.
- `splits/main_holdout/` contains the fixed partition. SSL used training and validation images, but not detector-test images.

The team has confirmed redistribution rights for its images and annotations. The dataset is released under CC BY 4.0; see `oil/LICENSE`.

## HIT-UAV

`HIT_UAV/` contains the Roboflow version 2 export used in the study: 2,024 training, 578 validation and 296 test images, with Bicycle, Car, DontCare, OtherVehicle and Person labels. Images were resized to 640 × 640. Four-class summaries exclude DontCare only when averaging saved AP values; they are not a four-class retraining or ignore-region evaluation.

Retained attribution and export documentation are in `HIT_UAV/ATTRIBUTION.md`, `README.dataset.txt` and `README.roboflow.txt`. The export retains CC BY 4.0. Please also cite the original dataset:

Suo, J. et al. HIT-UAV: A high-altitude infrared thermal dataset for unmanned aerial vehicle-based object detection. *Scientific Data* **10**, 227 (2023). https://doi.org/10.1038/s41597-023-02066-6
