"""解析文件名 + 标签 → data/metadata.csv (跨算法元数据中心)。

文件名规律 (实地扫描得出, 2593/2593 全部匹配):
  Oil_<T>_<a>_<b>_<c>_<frame>.JPG
    T : 温度       30 / 50 / 70   (°C)
    a : 位置编码   1 / 2 / 3 / 4 / 5
    b : 工况编码   1 / 2
    c : 场景/昼夜  1 / 2 / 3 / 3x / 4   ("3x" 是 "3" 的变体)
    frame : 8 位帧号 (该 session 内时间序, 不连续无所谓)

22 个唯一 (T, a, b, c) = 22 个 session, 与规划文档一致。
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "OilData"
LBL_DIR = ROOT / "data" / "OilLabels"
OUT_CSV = ROOT / "data" / "metadata.csv"

_NAME_RE = re.compile(r"^Oil_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([0-9]+)$")


def parse_filename(stem: str) -> Optional[Dict[str, Any]]:
    """从文件名 stem 解析结构化字段; 不匹配返回 None。"""
    m = _NAME_RE.match(stem)
    if not m:
        return None
    T, a, b, c, frame = m.groups()
    return {
        "temperature": int(T),
        "location": a,        # 保留字符串便于 '3x' 之类的兼容
        "condition": b,
        "day_night": c,
        "frame_idx": int(frame),
        "session_id": f"T{T}_a{a}_b{b}_c{c}",
    }


def parse_label(label_path: Path) -> Dict[str, Any]:
    """解析 YOLO 标签 → 计数信息。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"label missing: {label_path}")
    n_water = n_oil = 0
    classes_seen: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        classes_seen.add(cls)
        if cls == 0:
            n_water += 1
        elif cls == 1:
            n_oil += 1
    n_boxes = n_water + n_oil
    return {
        "n_boxes": int(n_boxes),
        "classes": sorted(classes_seen),
        "has_water": int(n_water > 0),
        "has_oil": int(n_oil > 0),
        "n_water": int(n_water),
        "n_oil": int(n_oil),
    }


def build_metadata() -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    imgs = sorted(IMG_DIR.glob("*.JPG"))
    if not imgs:
        # 兼容大小写
        imgs = sorted(p for p in IMG_DIR.iterdir()
                      if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    for p in imgs:
        meta_name = parse_filename(p.stem)
        if meta_name is None:
            raise ValueError(f"Filename does not match expected pattern: {p.name}")
        label_path = LBL_DIR / f"{p.stem}.txt"
        meta_label = parse_label(label_path)
        rows.append({
            "image_name": p.stem,
            "image_path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "label_path": str(label_path.relative_to(ROOT)).replace("\\", "/"),
            **meta_name,
            **meta_label,
        })
    df = pd.DataFrame(rows)
    cols = [
        "image_name", "image_path", "label_path",
        "session_id", "frame_idx",
        "temperature", "location", "condition", "day_night",
        "n_boxes", "classes", "has_oil", "has_water", "n_water", "n_oil",
    ]
    df = df[cols].sort_values(["session_id", "frame_idx"]).reset_index(drop=True)
    return df


def _print_stats(df: pd.DataFrame) -> None:
    print(f"Total samples         : {len(df)}")
    sess = df.groupby("session_id").size()
    print(f"Unique sessions       : {sess.size}")
    print(f"Per-session samples   : min={sess.min()}, max={sess.max()}, "
          f"mean={sess.mean():.1f}, median={sess.median():.1f}")
    n_oil_img = int((df["has_oil"] == 1).sum())
    n_water_only = int(((df["has_water"] == 1) & (df["has_oil"] == 0)).sum())
    n_nothing = int(((df["has_water"] == 0) & (df["has_oil"] == 0)).sum())
    print(f"Class dist (image)    : has_oil={n_oil_img}, water_only={n_water_only}, "
          f"no_box={n_nothing}, total={n_oil_img + n_water_only + n_nothing}")

    box_counts = df["n_boxes"]
    bins = Counter()
    for n in box_counts:
        if n == 0:
            bins["0"] += 1
        elif n == 1:
            bins["1"] += 1
        elif n == 2:
            bins["2"] += 1
        else:
            bins["3+"] += 1
    print(f"Bbox-count dist       : 0={bins['0']}, 1={bins['1']}, "
          f"2={bins['2']}, 3+={bins['3+']}")
    print(f"Total box objects     : water={int(df['n_water'].sum())}, "
          f"oil={int(df['n_oil'].sum())}, "
          f"all={int(df['n_boxes'].sum())}")
    print()
    print("Sessions detail (sorted by sample count desc):")
    by_sess = (df.groupby("session_id")
                 .agg(n_samples=("image_name", "size"),
                      has_oil=("has_oil", "sum"),
                      has_water=("has_water", "sum"))
                 .sort_values("n_samples", ascending=False))
    for sid, row in by_sess.iterrows():
        print(f"  {sid:>16}  n={int(row.n_samples):>4}  "
              f"oil_img={int(row.has_oil):>4}  water_img={int(row.has_water):>4}")


def main() -> None:
    df = build_metadata()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[metadata] wrote {OUT_CSV} ({OUT_CSV.stat().st_size} bytes)")
    print()
    _print_stats(df)


if __name__ == "__main__":
    main()
