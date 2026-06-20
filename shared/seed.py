"""统一的随机种子设置 — 保证 Python / NumPy / PyTorch (CPU & CUDA) 复现性。"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置全部随机源, 用于实验复现。

    注意:
      - 不调用 torch.use_deterministic_algorithms(True): 会与 AMP / 某些 cuDNN
        kernel 冲突, 在工程级训练里得不偿失。
      - cudnn.benchmark 关闭, deterministic 打开, 牺牲一点速度换稳定性。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    print("a =", a.tolist())
    print("b =", b.tolist())
    assert torch.equal(a, b), "set_seed not reproducible!"
    print("[seed] reproducibility OK")
