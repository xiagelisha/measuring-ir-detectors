# tier1_ultralytics — Stage 1 baselines

Ultralytics-based detectors trained on the oil-leak infrared dataset.

## Models

| Key         | Weight       | Notes                                |
| ----------- | ------------ | ------------------------------------ |
| `yolov8s`   | yolov8s.pt   | Anchor-based, 2023 baseline          |
| `yolov11s`  | yolo11s.pt   | 2024 update with C3k2/PSA            |
| `yolov12s`  | yolo12s.pt   | 2025 attention-centric variant       |
| `yolov10s`  | yolov10s.pt  | NMS-free / consistent dual labels    |
| `rtdetr_l`  | rtdetr-l.pt  | RT-DETR Large (transformer detector) |

## Run a single baseline

```powershell
cd C:\twb\oil_leak\Paper3\paper3-10
python baselines\tier1_ultralytics\train_one.py --model yolov8s --split main_holdout --epochs 200
```

3-epoch smoke test (no real training, sanity only):

```powershell
python baselines\tier1_ultralytics\train_one.py --model yolov8s --quick-verify
```

## Run all 5 baselines (overnight)

```powershell
python baselines\tier1_ultralytics\run_all_baselines.py
```

Use `.\baselines\tier1_ultralytics\check.ps1` to inspect progress / final report.

## Outputs per run

`results_v2/baselines/tier1/<model>_<split>_<timestamp>/`

- `weights/{best.pt,last.pt}`
- `training_log.csv` (renamed from ultralytics `results.csv`)
- `metrics_val.json`, `metrics_test.json` (ultralytics + shared dual mAP + consistency)
- `predictions_test.json`
- `confusion_matrix.png`, `pr_curve.png`, `f1_curve.png`
- `complexity.json` (params / GFLOPs / FPS / VRAM)
- `config_snapshot.yaml`, `train.log`
