"""STC-YOLO data helper — Session metadata loader.

Loads `data/metadata.csv` and produces per-image metadata vectors for C3
(Session-FiLM). Also exposes per-session continuous-segment information for
C1 (frame pair sampling) and C2 (unlabeled neighbor sampling).

CSV columns (verified 2026-05-29):
    image_name, image_path, label_path, session_id, frame_idx, temperature,
    location, condition, day_night, n_boxes, classes, has_oil, has_water,
    n_water, n_oil

IMPORTANT DATA FINDING (2026-05-29, MS1 skeleton):
    The `day_night` column is a misnomer carried over from CSV generation.
    Its values are the file-name token `c` from the `Oil_S_R_P_c_F` schema
    (per JRTIP §3.1: `c=1` or `2` -> Water; `c=3`, `3x`, `4` -> Oil), e.g.
    rows like `Oil_50_1_1_3x_00000001` carry day_night="3x".

    Consequences:
      - day_night is NOT day-vs-night information; it leaks the per-image
        class encoding from the filename.
      - Using it as a FiLM input would be label leakage.
    Therefore the encoding below DROPS day_night and uses only
    {temperature, location, condition} as clean metadata.

Encoding (8-dim vector, post-fix):
    [temperature_norm,                # continuous, normalized to [0, 1]
     location_oh (4),                 # one-hot of 4 location categories
     condition_oh (3)]                # one-hot of 3 condition categories
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_CSV = _PROJECT_ROOT / "data" / "metadata.csv"

# Post-fix dim: temperature(1) + location_oh(4) + condition_oh(3) = 8.
# day_night column is intentionally dropped from FiLM input — see module
# docstring "IMPORTANT DATA FINDING".
METADATA_DIM = 8


class ImageMeta(NamedTuple):
    image_name: str
    session_id: str
    frame_idx: int
    temperature: float
    location: int
    condition: int
    day_night_raw: str       # raw string (kept for analysis; not used by FiLM)
    n_oil: int
    n_water: int


def load_metadata(csv_path: str | Path = DEFAULT_METADATA_CSV) -> dict[str, ImageMeta]:
    """Read metadata.csv into a dict keyed by image_name (stem without ext).

    Returns 2593 entries. Raises FileNotFoundError if the csv is absent.
    Note: `day_night` is parsed as raw string (e.g. "1", "2", "3x", "4") —
    its values are actually the filename `c` token (class encoding), not
    day/night info. See module docstring.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"metadata.csv not found at {csv_path}")
    out: dict[str, ImageMeta] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["image_name"].strip()
            out[name] = ImageMeta(
                image_name=name,
                session_id=row["session_id"].strip(),
                frame_idx=int(row["frame_idx"]),
                temperature=float(row["temperature"]),
                location=int(row["location"]),
                condition=int(row["condition"]),
                day_night_raw=row["day_night"].strip(),
                n_oil=int(row["n_oil"]),
                n_water=int(row["n_water"]),
            )
    return out


def encode_metadata(
    meta: ImageMeta,
    temperature_min: float = -10.0,
    temperature_max: float = 50.0,
    n_locations: int = 4,
    n_conditions: int = 3,
) -> torch.Tensor:
    """Encode one ImageMeta -> 8-dim tensor.

    Layout: [T_norm, loc_oh(4), cond_oh(3)]

    day_night is intentionally dropped — see module docstring. The 8-dim
    vector contains only clean metadata that is independent of the per-image
    class label.

    Args:
        meta: a single ImageMeta
        temperature_min/max: ambient T normalization range; defaults cover
            our 22-session dataset (verified: T in {30, 50, ...} celsius).
        n_locations / n_conditions: cardinality of the categorical fields.

    Returns:
        Tensor of shape (METADATA_DIM,) on CPU. Caller moves to device.
    """
    t_norm = (meta.temperature - temperature_min) / (temperature_max - temperature_min)
    t_norm = max(0.0, min(1.0, t_norm))

    loc_oh = [0.0] * n_locations
    if 1 <= meta.location <= n_locations:
        loc_oh[meta.location - 1] = 1.0

    cond_oh = [0.0] * n_conditions
    if 1 <= meta.condition <= n_conditions:
        cond_oh[meta.condition - 1] = 1.0

    vec = [t_norm] + loc_oh + cond_oh
    assert len(vec) == METADATA_DIM, f"vec dim {len(vec)} != {METADATA_DIM}"
    return torch.tensor(vec, dtype=torch.float32)


def build_session_segments(
    metas: dict[str, ImageMeta], max_gap: int = 1
) -> dict[str, list[list[str]]]:
    """Group image_names into continuous-frame segments per session.

    A new segment starts whenever frame_idx jumps by more than `max_gap`
    within the same session_id, or when session_id changes.

    Args:
        metas: output of load_metadata()
        max_gap: maximum allowed frame_idx jump within one segment

    Returns:
        dict session_id -> list of segments, each segment a list of image_names
        in ascending frame_idx order.
    """
    by_session: dict[str, list[ImageMeta]] = {}
    for m in metas.values():
        by_session.setdefault(m.session_id, []).append(m)

    out: dict[str, list[list[str]]] = {}
    for sid, frames in by_session.items():
        frames_sorted = sorted(frames, key=lambda m: m.frame_idx)
        segments: list[list[str]] = []
        current: list[str] = []
        prev_idx: int | None = None
        for m in frames_sorted:
            if prev_idx is None or (m.frame_idx - prev_idx) > max_gap:
                if current:
                    segments.append(current)
                current = [m.image_name]
            else:
                current.append(m.image_name)
            prev_idx = m.frame_idx
        if current:
            segments.append(current)
        out[sid] = segments
    return out


if __name__ == "__main__":
    # Quick sanity demo when run directly.
    metas = load_metadata()
    print(f"loaded {len(metas)} entries")
    sample = next(iter(metas.values()))
    print(f"sample: {sample}")
    vec = encode_metadata(sample)
    print(f"encoded vector ({METADATA_DIM}-dim): {vec.tolist()}")
    segs = build_session_segments(metas, max_gap=1)
    print(f"sessions: {len(segs)}, segments: {sum(len(s) for s in segs.values())}")
    longest = max((len(seg) for s in segs.values() for seg in s), default=0)
    print(f"longest segment: {longest} frames")
    # day_night column distribution (proves it is the filename c token, not day/night).
    from collections import Counter
    dn_counts = Counter(m.day_night_raw for m in metas.values())
    print(f"day_night_raw distribution: {dict(dn_counts)}")
