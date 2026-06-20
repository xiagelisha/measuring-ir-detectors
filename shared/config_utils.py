"""基于 OmegaConf 的轻量配置工具: 加载、合并、快照保存。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Union

from omegaconf import DictConfig, OmegaConf

ConfigLike = Union[DictConfig, dict]


def load_config(
    yaml_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """加载 YAML 配置并应用命令行风格 overrides。

    Args:
        yaml_path: YAML 文件路径
        overrides: 形如 ["paths.results_dir=/tmp/out", "seed=7"] 的覆盖项
    Returns:
        DictConfig (已 resolve interpolation 之前的形态; 调用方按需 resolve)
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    cfg = OmegaConf.load(yaml_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Top-level config must be a mapping, got {type(cfg).__name__}")
    if overrides:
        cfg_override = OmegaConf.from_dotlist(list(overrides))
        cfg = OmegaConf.merge(cfg, cfg_override)
    return cfg  # type: ignore[return-value]


def merge_configs(*cfgs: ConfigLike) -> DictConfig:
    """按顺序合并多个配置 (后者覆盖前者)。"""
    if not cfgs:
        raise ValueError("merge_configs requires at least one config")
    base = OmegaConf.create(cfgs[0]) if not isinstance(cfgs[0], DictConfig) else cfgs[0]
    for c in cfgs[1:]:
        c2 = OmegaConf.create(c) if not isinstance(c, DictConfig) else c
        base = OmegaConf.merge(base, c2)
    return base  # type: ignore[return-value]


def save_config_snapshot(cfg: ConfigLike, out_dir: Union[str, Path]) -> Path:
    """把 cfg 保存到 out_dir/config.yaml (resolved, UTF-8 无 BOM)。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.yaml"
    cfg_obj = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    text = OmegaConf.to_yaml(cfg_obj, resolve=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    cfg_path = Path("shared/configs/global.yaml")
    cfg = load_config(cfg_path)
    assert cfg.classes.num_classes == 2, f"num_classes != 2: {cfg.classes.num_classes}"
    names = list(cfg.classes.names)
    assert names == ["Water", "Oil"], f"names != [Water, Oil]: {names}"
    snap = save_config_snapshot(cfg, "results_v2/sanity_check/_config_snapshot")
    print(f"[config] num_classes = {cfg.classes.num_classes}")
    print(f"[config] names       = {names}")
    print(f"[config] image_dir   = {cfg.paths.image_dir}")
    print(f"[config] snapshot    = {snap}  ({snap.stat().st_size} bytes)")

    cfg2 = load_config(cfg_path, overrides=["seed=7", "image.size=512"])
    assert cfg2.seed == 7
    assert cfg2.image.size == 512
    print(f"[config] override seed -> {cfg2.seed}, image.size -> {cfg2.image.size}")
    print("[config] OK")
