"""设备与 CUDA 性能开关。"""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """返回首选设备 (cuda:0 优先, 否则 cpu)。"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def setup_cuda_optimization() -> None:
    """启用 TF32 + cuDNN benchmark, 用于训练加速。

    与 set_seed() 的 deterministic 设置可能冲突 — 训练流程里, 一般
    先 set_seed() 求复现, 再 setup_cuda_optimization() 求速度时, 此函数
    会把 cudnn.benchmark 切回 True (覆盖 deterministic 的选择)。
    """
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def print_gpu_info() -> None:
    """打印当前 GPU 摘要。无 GPU 时打印提示。"""
    if not torch.cuda.is_available():
        print("[device] CUDA not available; running on CPU.")
        return
    idx = 0
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    vram_gb = props.total_memory / (1024 ** 3)
    cuda_ver = torch.version.cuda
    print(f"[device] GPU {idx}: {name}")
    print(f"[device]   SM capability: sm_{cap[0]}{cap[1]} (compute {cap[0]}.{cap[1]})")
    print(f"[device]   VRAM: {vram_gb:.2f} GiB")
    print(f"[device]   CUDA runtime: {cuda_ver}")
    print(f"[device]   torch: {torch.__version__}")


if __name__ == "__main__":
    print_gpu_info()
    setup_cuda_optimization()
    dev = get_device()
    print(f"[device] selected: {dev}")
    print(f"[device] TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"[device] cudnn benchmark: {torch.backends.cudnn.benchmark}")
