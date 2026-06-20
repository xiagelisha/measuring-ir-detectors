# Data

## Oil dataset — FULL RELEASE (CC BY 4.0; public upon publication)

The complete substation IR oil-leakage dataset used in the paper:

```
oil/
  images/        2,593 IR frames, 1280x720 JPG
  labels/        2,593 matching YOLO-format label files (class cx cy w h, normalised)
  metadata.csv   one row per image (see column guide below)
  LICENSE        CC BY 4.0 (dataset only; code is MIT-licensed at the repo root)
```

- **Classes**: `0 = Water` (condensation / water-like thermal patterns, 15,041 boxes),
  `1 = Oil` (active leak, 6,199 boxes); 2.43:1 majority-class skew. `Classes.txt`
  (this directory) lists the class order.
- **Acquisition**: 22 intra-session sequences captured at multiple substations under
  varying ambient temperature and humidity. Most objects are small (majority of
  boxes below 64x64 px on the 1280x720 frame).
- **File naming**: `Oil_S_R_P_c_F.JPG`. The 5th segment `c` encodes the
  expert-assigned ground-truth class of the scene (`1`/`2` = Water; `3`/`3x`/`4` =
  Oil); `S` identifies the session and `F` the frame index.

### `metadata.csv` column guide
| column | meaning |
|---|---|
| `image_name`, `image_path`, `label_path` | file identifiers |
| `session_id`, `frame_idx` | capture session (22 total) and frame order within it |
| `temperature`, `location`, `condition` | session-level acquisition metadata |
| `day_night` | **misnomer — do not treat as time-of-day.** This column mirrors the filename `c` segment, i.e. it is a class encoding, not an acquisition condition. Never feed it (or any column derived from it) to a model as an input feature: it leaks the label. |
| `n_boxes`, `classes`, `has_oil`, `has_water`, `n_water`, `n_oil` | per-image box statistics derived from the labels |

### Split
`splits/main_holdout/` lists the fixed train/val/test partition used for every
experiment in the paper: 1,886 / 348 / 359 images (`train.txt`, `val.txt`,
`test.txt` name the image files). The test split is excluded from all SSL
pretraining.

### License & citation
The oil dataset is released under **CC BY 4.0** (`oil/LICENSE`). If you use it,
please cite the paper (see the repository root README for the reference).

## HIT-UAV — PUBLIC (full, included under CC BY 4.0)
`HIT_UAV/` is the public high-altitude infrared UAV detection benchmark
(train/valid/test + `data.yaml`), included to make the second-dataset replication
reproducible out of the box.

It is redistributed under **CC BY 4.0**, with full attribution and a record of the
modifications made, in **`HIT_UAV/ATTRIBUTION.md`** (see also the retained
`README.dataset.txt` and `README.roboflow.txt`). Please cite the original authors:

> Suo, J. et al. *HIT-UAV: A high-altitude infrared thermal dataset for unmanned
> aerial vehicle-based object detection.* Scientific Data **10**, 227 (2023).
> DOI: 10.1038/s41597-023-02066-6

If you prefer, obtain the dataset from the original source
(<https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset>) and point
`data.yaml` at your local copy instead.

`Classes.txt` lists the oil-dataset class order: line 1 = `Water` (0),
line 2 = `Oil` (1).
