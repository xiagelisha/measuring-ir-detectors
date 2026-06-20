"""Paper 4 5-seed runner — invoked by paper4_orchestrator for EXPERIMENT steps.

For each seed in [42, 43, 44, 45, 46]:
  - Calls methods/m6_stc_yolo/train_one.py with seed override
  - Collects ultralytics_test_metrics.json from each run
Aggregates into:
  results_v2/orchestrator/runs/<run-name>/5seed_summary.json
  {
    "config_path": ...,
    "seeds": [42, ...],
    "per_seed": [ {seed: 42, mAP50: ..., ...}, ... ],
    "mAP50_5seed_mean": float, "mAP50_5seed_std": float,
    "mAP50_5seed_min": float, "mAP50_5seed_max": float,
    "recall_5seed_mean": float, "recall_5seed_std": float,
    "Oil_AP50_5seed_mean": float, ...
    "n_completed": int  (out of 5)
  }

Decisions are made downstream by the orchestrator; this script only computes.

Usage:
    python tools/paper4_run_5seed.py --config <yaml> --run-name <name>
                                     [--override key=value ...]
                                     [--inherit-config from-state]
                                     [--data-yaml <path>]
                                     [--seeds 42,43,44,45,46]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--run-name", required=True, type=str)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46",
                   help="comma-separated seed list")
    p.add_argument("--override", action="append", default=[],
                   help="key=value override; can be passed multiple times")
    p.add_argument("--inherit-config", type=str, default=None,
                   help="'from-state' to load orchestrator's config_updates")
    p.add_argument("--data-yaml", type=str, default=None,
                   help="ultralytics data yaml. If None, the train_one.py will "
                        "raise — wire this up in MS1 D0 before D4a runs.")
    p.add_argument("--results-dir", type=str, default=None,
                   help="override default: results_v2/orchestrator/runs/<run-name>")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.results_dir:
        out_dir = Path(args.results_dir)
    else:
        out_dir = (PROJECT_ROOT / "results_v2" / "orchestrator" /
                   "runs" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []
    n_completed = 0

    for seed in seeds:
        run_name_seed = f"{args.run_name}_seed{seed}"
        cmd: list[str] = [
            "python", "methods/m6_stc_yolo/train_one.py",
            "--config", args.config,
            "--seed", str(seed),
            "--run-name", run_name_seed,
        ]
        if args.data_yaml:
            cmd.extend(["--data-yaml", args.data_yaml])
        # NOTE: --override and --inherit-config are not yet plumbed through
        # train_one.py. When MS1 D4a is wired this script should write a tmp
        # yaml that merges the base config + overrides + config_updates, and
        # pass that via --config. For now, we forward overrides as a marker
        # so the failure is loud rather than silent.
        for ov in args.override:
            cmd.extend(["--override", ov])
        if args.inherit_config:
            cmd.extend(["--inherit-config", args.inherit_config])

        seed_log = out_dir / f"seed{seed}.stdout.log"
        seed_err = out_dir / f"seed{seed}.stderr.log"
        print(f"[seed {seed}] running: {' '.join(cmd)}")
        with open(seed_log, "w", encoding="utf-8") as so, \
             open(seed_err, "w", encoding="utf-8") as se:
            proc = subprocess.run(
                cmd, cwd=PROJECT_ROOT, stdout=so, stderr=se,
                text=True, encoding="utf-8")
        if proc.returncode != 0:
            print(f"[seed {seed}] FAILED (returncode {proc.returncode})")
            per_seed.append({"seed": seed, "status": "failed",
                             "returncode": proc.returncode})
            continue

        # train_one.py writes ultralytics_test_metrics.json at
        # results_v2/methods/stc_yolo/<run_name_seed>/ultralytics_test_metrics.json
        metrics_path = (PROJECT_ROOT / "results_v2" / "methods" / "stc_yolo"
                        / run_name_seed / "ultralytics_test_metrics.json")
        if not metrics_path.exists():
            print(f"[seed {seed}] WARN: metrics not found at {metrics_path}")
            per_seed.append({"seed": seed, "status": "no_metrics"})
            continue

        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[seed {seed}] WARN: failed to parse metrics: {e}")
            per_seed.append({"seed": seed, "status": "parse_error"})
            continue

        per_seed.append({
            "seed": seed,
            "status": "completed",
            "mAP50": m.get("mAP50"),
            "mAP50_95": m.get("mAP50_95"),
            "Water_AP50": m.get("Water_AP50"),
            "Oil_AP50": m.get("Oil_AP50"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
        })
        n_completed += 1

    # Aggregate
    def _vals(key: str) -> list[float]:
        return [d[key] for d in per_seed
                if d.get("status") == "completed" and d.get(key) is not None]

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    summary = {
        "config_path": args.config,
        "run_name": args.run_name,
        "seeds": seeds,
        "n_completed": n_completed,
        "n_total": len(seeds),
        "per_seed": per_seed,
    }
    for key in ["mAP50", "mAP50_95", "Water_AP50", "Oil_AP50",
                "precision", "recall"]:
        s = _stats(_vals(key))
        suffix = "5seed"
        if key in {"precision", "recall"}:
            summary[f"{key}_{suffix}_mean"] = s["mean"]
            summary[f"{key}_{suffix}_std"] = s["std"]
        else:
            summary[f"{key}_{suffix}_mean"] = s["mean"]
            summary[f"{key}_{suffix}_std"] = s["std"]
            summary[f"{key}_{suffix}_min"] = s["min"]
            summary[f"{key}_{suffix}_max"] = s["max"]

    out_json = out_dir / "5seed_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    print(f"n_completed: {n_completed}/{len(seeds)}")
    if summary.get("mAP50_5seed_mean") is not None:
        print(f"mAP50 5-seed: mean={summary['mAP50_5seed_mean']:.4f}  "
              f"std={summary['mAP50_5seed_std']:.4f}")
        print(f"Recall 5-seed: mean={summary['recall_5seed_mean']:.4f}")
    return 0 if n_completed >= 3 else 1  # require at least 3 of 5 seeds


if __name__ == "__main__":
    sys.exit(main())
