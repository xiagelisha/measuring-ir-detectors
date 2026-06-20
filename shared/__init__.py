"""shared/ — 共享基础设施 (Step 0.2)."""
from .seed import set_seed
from .device import get_device, setup_cuda_optimization, print_gpu_info
from .logger import get_logger
from .config_utils import load_config, save_config_snapshot, merge_configs
from .dataset import OilLeakDataset
from .transforms import get_train_transforms, get_val_transforms
from .collate import detr_collate_fn
from .metrics import compute_detection_metrics, save_metrics_json
from .visualizer import (
    draw_boxes,
    visualize_samples,
    plot_pr_curve,
    plot_f1_curve,
    plot_confusion_matrix,
)
from .complexity import measure_complexity

__all__ = [
    "set_seed",
    "get_device", "setup_cuda_optimization", "print_gpu_info",
    "get_logger",
    "load_config", "save_config_snapshot", "merge_configs",
    "OilLeakDataset",
    "get_train_transforms", "get_val_transforms",
    "detr_collate_fn",
    "compute_detection_metrics", "save_metrics_json",
    "draw_boxes", "visualize_samples",
    "plot_pr_curve", "plot_f1_curve", "plot_confusion_matrix",
    "measure_complexity",
]
