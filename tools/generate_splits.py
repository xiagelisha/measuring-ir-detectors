"""按整 session 划分 train/val/test (70/15/15), 类别均衡, seed=42。

关键观察 (parse_metadata 后得出):
  * 22 个 session, 每个 session 是 *单类* —
        11 个 oil-bearing (该 session 内全部图至少含 1 个 Oil bbox)
        11 个 water-only (该 session 内全部图为 Water 或无 box)
  * 因此类别均衡 = 把 oil 和 water 两个 session 池分别 70/15/15 划分。
  * 用 LPT (Largest Processing Time) 贪心: 每池按样本数降序, 依次把
    session 放进 "当前样本数 / 目标样本数" 最小的 split。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META_CSV = ROOT / "data" / "metadata.csv"
SPLIT_DIR = ROOT / "data" / "splits" / "main_holdout"

SEED = 42
TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _assign_pool(sessions: List[Tuple[str, int]], totals: int,
                 rng: random.Random) -> Dict[str, List[str]]:
    """LPT 贪心: 按样本数降序, 同样 size 时随机打乱。"""
    # 同 size 用稳定随机扰动 (rng) 决定先后
    sess = sorted(sessions, key=lambda x: (-x[1], rng.random()))
    targets_n = {k: TARGETS[k] * totals for k in TARGETS}
    cur = {k: 0 for k in TARGETS}
    bucket: Dict[str, List[str]] = {k: [] for k in TARGETS}
    for sid, n in sess:
        # 选择 (cur+n)/target 之后最小的 split
        choice = min(TARGETS, key=lambda k: (cur[k] + n) / max(targets_n[k], 1e-6))
        bucket[choice].append(sid)
        cur[choice] += n
    return bucket


def make_splits(df: pd.DataFrame, seed: int = SEED) -> Dict[str, List[str]]:
    rng = random.Random(seed)

    # 每个 session 的类别归属 (oil-bearing or water-only)
    sess_info = (df.groupby("session_id")
                   .agg(n=("image_name", "size"),
                        has_oil_img=("has_oil", "sum"))
                   .reset_index())
    oil_pool, water_pool = [], []
    for _, r in sess_info.iterrows():
        sid, n = r["session_id"], int(r["n"])
        if int(r["has_oil_img"]) > 0:
            oil_pool.append((sid, n))
        else:
            water_pool.append((sid, n))

    oil_total = sum(n for _, n in oil_pool)
    water_total = sum(n for _, n in water_pool)
    oil_bucket = _assign_pool(oil_pool, oil_total, rng)
    water_bucket = _assign_pool(water_pool, water_total, rng)

    splits = {k: sorted(oil_bucket[k] + water_bucket[k]) for k in TARGETS}
    return splits


def write_split_files(splits: Dict[str, List[str]], df: pd.DataFrame) -> Dict[str, Path]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for k, session_ids in splits.items():
        sids = set(session_ids)
        names = sorted(df.loc[df["session_id"].isin(sids), "image_name"].tolist())
        p = SPLIT_DIR / f"{k}.txt"
        p.write_text("\n".join(names) + "\n", encoding="utf-8")
        out[k] = p
    return out


def _print_table(splits: Dict[str, List[str]], df: pd.DataFrame) -> None:
    print()
    print(f"{'Split':<8}{'Samples':>10}{'Sessions':>10}{'Water_img':>12}{'Oil_img':>10}{'NoBox':>8}")
    total_s = total_w = total_o = total_n = 0
    for k in ["train", "val", "test"]:
        sids = set(splits[k])
        sub = df[df["session_id"].isin(sids)]
        n = len(sub)
        nw = int((sub["has_water"] == 1).sum())
        no = int((sub["has_oil"] == 1).sum())
        nb = int(((sub["has_water"] == 0) & (sub["has_oil"] == 0)).sum())
        print(f"{k:<8}{n:>10}{len(sids):>10}{nw:>12}{no:>10}{nb:>8}")
        total_s += n; total_w += nw; total_o += no; total_n += nb
    print(f"{'Total':<8}{total_s:>10}{sum(len(s) for s in splits.values()):>10}"
          f"{total_w:>12}{total_o:>10}{total_n:>8}")
    print()
    for k in ["train", "val", "test"]:
        print(f"  {k} sessions ({len(splits[k])}): {splits[k]}")


def _assert_invariants(splits: Dict[str, List[str]], df: pd.DataFrame) -> None:
    names_total = {k: set() for k in splits}
    for k, sids in splits.items():
        names_total[k] = set(df.loc[df["session_id"].isin(set(sids)), "image_name"])
    total = sum(len(v) for v in names_total.values())
    overlap_names = (names_total["train"] & names_total["val"]) | \
                    (names_total["train"] & names_total["test"]) | \
                    (names_total["val"] & names_total["test"])
    assert total == len(df), f"sample count mismatch: {total} != {len(df)}"
    assert not overlap_names, f"sample overlap across splits: {len(overlap_names)}"

    s_train, s_val, s_test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert s_train.isdisjoint(s_val), "train/val share sessions"
    assert s_train.isdisjoint(s_test), "train/test share sessions"
    assert s_val.isdisjoint(s_test), "val/test share sessions"

    for k in ["train", "val", "test"]:
        sub = df[df["session_id"].isin(set(splits[k]))]
        n_oil = int((sub["has_oil"] == 1).sum())
        n_water = int((sub["has_water"] == 1).sum())
        assert n_oil > 0, f"{k}: no oil samples"
        assert n_water > 0, f"{k}: no water samples"

    # 比例 ±5% 检查
    n = len(df)
    for k, t in TARGETS.items():
        got = len(df[df["session_id"].isin(set(splits[k]))]) / n
        assert abs(got - t) <= 0.05, f"{k} ratio {got:.3f} drifts > 0.05 from {t}"


def main() -> None:
    if not META_CSV.is_file():
        raise FileNotFoundError(f"Run parse_metadata.py first; metadata.csv not found at {META_CSV}")
    df = pd.read_csv(META_CSV)
    splits = make_splits(df, seed=SEED)
    _print_table(splits, df)
    _assert_invariants(splits, df)
    out = write_split_files(splits, df)
    print()
    for k, p in out.items():
        n_lines = sum(1 for _ in p.open(encoding="utf-8"))
        print(f"[splits] {p.relative_to(ROOT)}  -> {n_lines} lines")
    print("[splits] OK")


if __name__ == "__main__":
    main()
