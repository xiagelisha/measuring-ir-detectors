"""模型复杂度: 参数 / FLOPs / 延迟 / FPS / 显存峰值。"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _profile_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> float:
    """用 thop 测 FLOPs (单次前向), 返回 G FLOPs (浮点运算次数 / 1e9)。"""
    try:
        from thop import profile  # type: ignore
    except Exception:
        return float("nan")
    model = model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_shape, device=device)
    macs, _ = profile(model, inputs=(x,), verbose=False)
    # MACs ≈ FLOPs / 2 (常见约定: 1 MAC = 2 FLOPs). thop 返 MACs.
    return float(macs * 2) / 1e9


def measure_complexity(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    device: str = "cuda",
    warmup: int = 10,
    n_iter: int = 50,
) -> Dict[str, float]:
    """测 params / flops / fps / latency / vram。

    Returns:
        dict 含 params_M, gflops, fps, latency_ms, vram_MB
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev).eval()

    params = _count_params(model)
    gflops = _profile_flops(model, input_shape)

    x = torch.randn(*input_shape, device=dev)

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if use_cuda:
            torch.cuda.synchronize(dev)

        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            timings = []
            for _ in range(n_iter):
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize(dev)
                timings.append(starter.elapsed_time(ender))  # ms
            latency_ms = float(sum(timings) / len(timings))
            vram_mb = float(torch.cuda.max_memory_allocated(dev) / (1024 ** 2))
        else:
            import time
            timings = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                _ = model(x)
                timings.append((time.perf_counter() - t0) * 1000.0)
            latency_ms = float(sum(timings) / len(timings))
            vram_mb = float("nan")

    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return {
        "params_M": float(params) / 1e6,
        "gflops": gflops,
        "fps": fps,
        "latency_ms": latency_ms,
        "vram_MB": vram_mb,
    }


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    out = measure_complexity(model, input_shape=(1, 3, 640, 640), device="cuda")
    print(f"[complexity] params_M   = {out['params_M']:.4f} M")
    print(f"[complexity] gflops     = {out['gflops']:.4f} G")
    print(f"[complexity] latency_ms = {out['latency_ms']:.4f} ms")
    print(f"[complexity] fps        = {out['fps']:.1f}")
    print(f"[complexity] vram_MB    = {out['vram_MB']:.2f} MB")
    assert out["params_M"] > 0
    assert out["latency_ms"] > 0
    print("[complexity] OK")
