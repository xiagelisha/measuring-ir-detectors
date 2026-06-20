"""Cross-check metadata.csv label statistics against the actual YOLO .txt files.

For every row in data/metadata.csv, re-parse the corresponding label file and
compare n_boxes / n_water / n_oil / classes / has_oil / has_water with the CSV
values. Class convention (from Classes.txt): 0=Water, 1=Oil.
"""
from __future__ import annotations

import ast
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
CSV_PATH = ROOT / "data" / "metadata.csv"

CLS_WATER, CLS_OIL = 0, 1


def parse_label(p: Path):
    n_water = n_oil = 0
    other = 0
    classes = set()
    n_boxes = 0
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        n_boxes += 1
        classes.add(cls)
        if cls == CLS_WATER:
            n_water += 1
        elif cls == CLS_OIL:
            n_oil += 1
        else:
            other += 1
    return n_boxes, n_water, n_oil, sorted(classes), other


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    missing = []
    mismatches = defaultdict(list)
    total_other = 0
    for r in rows:
        lbl = ROOT / r["label_path"].replace("/", "\\")
        parsed = parse_label(lbl)
        if parsed is None:
            missing.append(r["image_name"])
            continue
        n_box, n_w, n_o, cls_list, other = parsed
        total_other += other

        meta = {
            "n_boxes":   int(r["n_boxes"]),
            "n_water":   int(r["n_water"]),
            "n_oil":     int(r["n_oil"]),
            "has_oil":   int(r["has_oil"]),
            "has_water": int(r["has_water"]),
            "classes":   ast.literal_eval(r["classes"]) if r["classes"] else [],
        }
        actual = {
            "n_boxes":   n_box,
            "n_water":   n_w,
            "n_oil":     n_o,
            "has_oil":   int(n_o > 0),
            "has_water": int(n_w > 0),
            "classes":   cls_list,
        }
        for k in actual:
            if meta[k] != actual[k]:
                mismatches[k].append((r["image_name"], meta[k], actual[k]))

    print(f"rows checked   : {len(rows)}")
    print(f"missing labels : {len(missing)}")
    print(f"non-{{0,1}} cls : {total_other}")
    print()
    print("--- mismatches per field ---")
    for k in ("n_boxes", "n_water", "n_oil", "has_oil", "has_water", "classes"):
        ms = mismatches.get(k, [])
        print(f"  {k:9s}: {len(ms)}")
        for name, meta_v, real_v in ms[:5]:
            print(f"      {name}  meta={meta_v}  actual={real_v}")
    if missing:
        print(f"\nmissing label files (first 10): {missing[:10]}")


if __name__ == "__main__":
    main()
