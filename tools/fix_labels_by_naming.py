"""Enforce the naming-rule class convention on data/OilLabels/*.txt.

Filename: Oil_<T>_<a>_<b>_<c>_<frame>.txt
  c ∈ {1, 2}      → all boxes must be class 0 (Water)
  c ∈ {3, 3x, 4}  → all boxes must be class 1 (Oil)

Behaviour:
  --dry-run  : list mismatches only, no writes
  (default)  : rewrite offending files in-place, keeping bbox coordinates,
               only flipping the class id. Reports per-file change counts.

Skips: Classes.txt, labelfilelist.txt (non-label files).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
LBL_DIR = ROOT / "data" / "OilLabels"

CLS_WATER, CLS_OIL = 0, 1
WATER_C = {"1", "2"}
OIL_C   = {"3", "3x", "4"}

_NAME_RE = re.compile(r"^Oil_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([0-9]+)$")
SKIP_STEMS = {"Classes", "labelfilelist"}


def expected_class(stem: str) -> int | None:
    m = _NAME_RE.match(stem)
    if not m:
        return None
    c = m.group(4)
    if c in WATER_C:
        return CLS_WATER
    if c in OIL_C:
        return CLS_OIL
    return None


def process_file(p: Path, expected: int, dry_run: bool) -> tuple[int, int]:
    """Return (n_boxes, n_changed)."""
    lines = p.read_text(encoding="utf-8").splitlines()
    new_lines: list[str] = []
    n_boxes = 0
    n_changed = 0
    for raw in lines:
        s = raw.rstrip()
        if not s.strip():
            new_lines.append(raw)
            continue
        parts = s.split()
        if len(parts) < 5:
            new_lines.append(raw)
            continue
        n_boxes += 1
        try:
            cls = int(float(parts[0]))
        except ValueError:
            new_lines.append(raw)
            continue
        if cls != expected:
            n_changed += 1
            parts[0] = str(expected)
            # preserve any trailing whitespace shape on a best-effort basis
            new_lines.append(" ".join(parts))
        else:
            new_lines.append(raw)

    if n_changed and not dry_run:
        # ultralytics yolo labels: LF-terminated, no trailing newline required, but keep one
        p.write_text("\n".join(new_lines) + ("\n" if lines and lines[-1] == "" else ""),
                     encoding="utf-8")
    return n_boxes, n_changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = [p for p in LBL_DIR.glob("*.txt") if p.stem not in SKIP_STEMS]
    total_files = len(files)
    skipped_unmatched = 0
    bad_files: list[tuple[str, int, int, int]] = []  # (stem, expected, n_boxes, n_changed)
    flip_counter: Counter[str] = Counter()
    grand_changed = 0

    for p in sorted(files):
        exp = expected_class(p.stem)
        if exp is None:
            skipped_unmatched += 1
            continue
        n_boxes, n_changed = process_file(p, exp, args.dry_run)
        if n_changed:
            bad_files.append((p.stem, exp, n_boxes, n_changed))
            grand_changed += n_changed
            flip_counter[f"to_{('Water' if exp == 0 else 'Oil')}"] += n_changed

    mode = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"== {mode} ==")
    print(f"label files scanned     : {total_files}")
    print(f"skipped (name unmatched): {skipped_unmatched}")
    print(f"files with wrong class  : {len(bad_files)}")
    print(f"box flips (total)       : {grand_changed}  "
          f"({dict(flip_counter)})")
    if bad_files:
        print("\nfirst 30 offending files (stem, expected_cls, n_boxes, n_changed):")
        for stem, exp, nb, nc in bad_files[:30]:
            print(f"  {stem}  exp={exp}  boxes={nb}  changed={nc}")
        if len(bad_files) > 30:
            print(f"  ... +{len(bad_files) - 30} more")


if __name__ == "__main__":
    main()
