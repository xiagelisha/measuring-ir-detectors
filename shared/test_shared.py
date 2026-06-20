"""shared/ 集成测试 — 10 项, 全部通过才能进入 Step 0.3.

无 pytest 时降级到内置 runner (oilleak 环境未装 pytest)。
跑法:   python shared/test_shared.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

# 让脚本既能被 pytest 收集, 也能 `python shared/test_shared.py` 直接跑
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import shared
from shared import (
    OilLeakDataset, set_seed, get_device, setup_cuda_optimization, print_gpu_info,
    get_logger, load_config, save_config_snapshot,
    get_train_transforms, get_val_transforms,
    detr_collate_fn, compute_detection_metrics, save_metrics_json,
    draw_boxes, visualize_samples, measure_complexity,
)

DATA_IMG = ROOT / "data" / "OilData"
DATA_LBL = ROOT / "data" / "OilLabels"
SANITY = ROOT / "results_v2" / "sanity_check" / "test"


# ---------- 1. seed -------------------------------------------------------
def test_01_seed_reproducibility() -> None:
    set_seed(42); a = torch.randn(3)
    set_seed(42); b = torch.randn(3)
    assert torch.equal(a, b), "set_seed not reproducible"
    set_seed(43); c = torch.randn(3)
    assert not torch.equal(a, c), "different seeds should differ"


# ---------- 2. device -----------------------------------------------------
def test_02_device_setup() -> None:
    assert torch.cuda.is_available(), "需要 CUDA"
    dev = get_device()
    assert dev.type == "cuda"
    setup_cuda_optimization()
    assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.benchmark
    print_gpu_info()


# ---------- 3. logger -----------------------------------------------------
def test_03_logger_basic() -> None:
    log_file = SANITY / "logs" / "test_logger.log"
    log = get_logger("shared.tests.logger", log_file=log_file, level="INFO")
    log.info("info-line"); log.warning("warn-line"); log.error("error-line")
    assert log_file.exists() and log_file.stat().st_size > 0


# ---------- 4. config -----------------------------------------------------
def test_04_config_load() -> None:
    cfg = load_config(ROOT / "shared" / "configs" / "global.yaml")
    assert cfg.classes.num_classes == 2
    assert list(cfg.classes.names) == ["Water", "Oil"]
    save_config_snapshot(cfg, SANITY / "config_snapshot")
    assert (SANITY / "config_snapshot" / "config.yaml").exists()


# ---------- 5. dataset ----------------------------------------------------
def test_05_dataset_from_directory() -> None:
    ds = OilLeakDataset.from_directory(
        image_dir=DATA_IMG, label_dir=DATA_LBL, image_size=640,
    )
    assert len(ds) == 2593, f"expected 2593, got {len(ds)}"
    s = ds[0]
    assert "image" in s and "boxes_xyxy_abs" in s and "labels" in s
    assert s["image"].ndim == 3
    assert s["boxes_xyxy_abs"].dtype == torch.float32
    assert s["labels"].dtype == torch.int64
    H, W = s["orig_size"]
    if s["boxes_xyxy_abs"].numel() > 0:
        b = s["boxes_xyxy_abs"]
        assert b[:, 0].min() >= 0 and b[:, 1].min() >= 0
        assert b[:, 2].max() <= W + 1e-3 and b[:, 3].max() <= H + 1e-3


# ---------- 6. transforms -------------------------------------------------
def test_06_transforms() -> None:
    img_path = next(p for p in DATA_IMG.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    bgr = cv2.imread(str(img_path)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    boxes = [[w * 0.1, h * 0.1, w * 0.4, h * 0.5],
             [w * 0.5, h * 0.2, w * 0.9, h * 0.7]]
    labels = [0, 1]
    train_t = get_train_transforms(640)
    set_seed(0)
    out = train_t(image=rgb, bboxes=boxes, labels=labels)
    assert tuple(out["image"].shape) == (3, 640, 640)
    nb = np.asarray(out["bboxes"]).reshape(-1, 4)
    if nb.size > 0:
        assert nb.min() >= 0 and nb.max() <= 640 + 1e-3
    val_t = get_val_transforms(640)
    out2 = val_t(image=rgb, bboxes=boxes, labels=labels)
    assert tuple(out2["image"].shape) == (3, 640, 640)


# ---------- 7. collate ----------------------------------------------------
def test_07_collate() -> None:
    def mk(n: int, idx: int):
        return {
            "image": torch.randn(3, 640, 640),
            "boxes_xyxy_abs": torch.rand(n, 4) * 640.0,
            "labels": torch.randint(0, 2, (n,)),
            "image_id": idx,
            "orig_size": (480, 640),
            "image_path": f"fake_{idx}.jpg",
            "ratio_pad": (1.0, (0.0, 0.0)),
        }
    batch = [mk(0, 0), mk(1, 1), mk(3, 2), mk(5, 3)]
    out = detr_collate_fn(batch)
    assert tuple(out["images"].shape) == (4, 3, 640, 640)
    assert [b.shape[0] for b in out["boxes"]] == [0, 1, 3, 5]
    assert out["image_ids"].dtype == torch.long


# ---------- 8. metrics ----------------------------------------------------
def test_08_metrics_sanity() -> None:
    rng = np.random.default_rng(0)
    gts = []
    for iid in range(5):
        n = int(rng.integers(1, 4))
        b = np.stack([rng.uniform(10, 400, n), rng.uniform(10, 400, n),
                      rng.uniform(50, 500, n), rng.uniform(50, 500, n)], axis=1).astype(np.float32)
        b[:, 2] = np.maximum(b[:, 0] + 20, b[:, 2])
        b[:, 3] = np.maximum(b[:, 1] + 20, b[:, 3])
        labels = rng.integers(0, 2, size=n).astype(np.int64)
        gts.append({"image_id": iid, "boxes": b, "labels": labels, "orig_size": (480, 640)})

    perfect = [{"image_id": g["image_id"], "boxes": g["boxes"].copy(),
                "scores": np.ones(len(g["labels"]), dtype=np.float32),
                "labels": g["labels"].copy()} for g in gts]
    m1 = compute_detection_metrics(perfect, gts)
    assert m1["mAP50"] > 0.99, m1["mAP50"]
    assert m1["F1"] > 0.99

    empty = [{"image_id": g["image_id"], "boxes": np.zeros((0, 4)),
              "scores": np.zeros(0), "labels": np.zeros(0, dtype=np.int64)} for g in gts]
    m2 = compute_detection_metrics(empty, gts)
    assert m2["mAP50"] == 0.0
    assert m2["F1"] == 0.0

    save_metrics_json(m1, SANITY / "metrics_perfect.json")


# ---------- 9. visualizer -------------------------------------------------
def test_09_visualizer() -> None:
    ds = OilLeakDataset.from_directory(
        image_dir=DATA_IMG, label_dir=DATA_LBL, image_size=640,
    )
    out_dir = SANITY / "visualizer"
    paths = visualize_samples(ds, out_dir, n=4, seed=1)
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


# ---------- 10. complexity ------------------------------------------------
def test_10_complexity_simple_model() -> None:
    import torch.nn as nn
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
    out = measure_complexity(model, input_shape=(1, 3, 640, 640), device="cuda",
                             warmup=3, n_iter=10)
    assert out["params_M"] > 0
    assert out["gflops"] > 0
    assert out["latency_ms"] > 0
    assert out["fps"] > 0
    assert out["vram_MB"] > 0


# ---------- runner --------------------------------------------------------
TESTS = [
    test_01_seed_reproducibility,
    test_02_device_setup,
    test_03_logger_basic,
    test_04_config_load,
    test_05_dataset_from_directory,
    test_06_transforms,
    test_07_collate,
    test_08_metrics_sanity,
    test_09_visualizer,
    test_10_complexity_simple_model,
]


def _run_builtin() -> int:
    passed = failed = 0
    for t in TESTS:
        name = t.__name__
        try:
            t()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            failed += 1
            break  # 任一失败立即停止
    print(f"\n{passed}/{len(TESTS)} PASSED, {failed} FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    SANITY.mkdir(parents=True, exist_ok=True)
    try:
        import pytest  # type: ignore
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        sys.exit(_run_builtin())
