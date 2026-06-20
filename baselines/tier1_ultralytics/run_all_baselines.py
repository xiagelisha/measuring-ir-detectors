#!/usr/bin/env python
"""Stage 1 overnight orchestrator: train 5 tier-1 baselines sequentially + aggregate.

Independence: any baseline failing (timeout/crash/low mAP) does not block the rest.
Updates STATUS.json after every baseline so the user can monitor progress.
Total hard wall-clock: TOTAL_TIMEOUT_H hours.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"C:\twb\oil_leak\Paper3\paper3-10")
TIER1_DIR = PROJECT_ROOT / "results_v2" / "baselines" / "tier1"
STATUS_FILE = TIER1_DIR / "STATUS.json"
LOG_FILE = TIER1_DIR / "orchestrator.log"
FINAL_REPORT = TIER1_DIR / "FINAL_REPORT.md"

# YOLO 系列在前 (稳定快速), RT-DETR 最后 (慢且 DETR 收敛敏感)
BASELINES = [
    {"model": "yolov8s",  "timeout_h": 3.0},
    {"model": "yolov11s", "timeout_h": 3.0},
    {"model": "yolov12s", "timeout_h": 3.0},
    {"model": "yolov10s", "timeout_h": 3.0},
    {"model": "rtdetr_l", "timeout_h": 4.0},
]
TOTAL_TIMEOUT_H = 14.0  # 整体硬上限 (留出时间给聚合)


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def update_status(status_dict: dict) -> None:
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_dict, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        log(f"WARN: failed to write STATUS.json: {e}")


def find_latest_output(model_name: str) -> Path | None:
    """Find the latest non-smoke output dir for this baseline."""
    pattern = f"{model_name}_main_holdout_*"
    candidates = [
        d for d in TIER1_DIR.glob(pattern)
        if d.is_dir() and "_smoke" not in d.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def train_baseline(spec: dict) -> dict:
    model = spec["model"]
    timeout_s = int(spec["timeout_h"] * 3600)
    log(f"==== Starting {model} (epochs=200, timeout={spec['timeout_h']}h) ====")

    start = time.time()
    try:
        # NOTE: subprocess inherits the same Python interpreter and cwd.
        # We do NOT capture output to avoid huge memory; train_one.py writes its own train.log.
        run_log_path = TIER1_DIR / f"_orchestrator_{model}.log"
        with open(run_log_path, "w", encoding="utf-8", errors="replace") as run_log:
            proc = subprocess.run(
                [sys.executable, "baselines/tier1_ultralytics/train_one.py",
                 "--model", model, "--split", "main_holdout", "--epochs", "200"],
                cwd=str(PROJECT_ROOT),
                timeout=timeout_s,
                stdout=run_log,
                stderr=subprocess.STDOUT,
            )
        duration_h = (time.time() - start) / 3600

        if proc.returncode != 0:
            tail = ""
            try:
                with open(run_log_path, "r", encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-40:])
            except Exception:
                pass
            return {"model": model, "status": "CRASHED",
                    "duration_h": round(duration_h, 2),
                    "returncode": proc.returncode,
                    "log_tail": tail[-2000:]}

        out_dir = find_latest_output(model)
        if out_dir is None:
            return {"model": model, "status": "NO_OUTPUT",
                    "duration_h": round(duration_h, 2)}

        metrics_path = out_dir / "metrics_test.json"
        if not metrics_path.exists():
            return {"model": model, "status": "NO_METRICS",
                    "duration_h": round(duration_h, 2),
                    "out_dir": str(out_dir)}

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Prefer shared mAP (公平); fall back to ultralytics (top-level) if missing.
        sh = metrics.get("shared", {}) or {}
        ul = metrics.get("ultralytics", {}) or {}
        mAP50 = sh.get("mAP50") or ul.get("mAP50") or metrics.get("mAP50") or 0.0
        mAP5095 = sh.get("mAP50_95") or ul.get("mAP50_95") or metrics.get("mAP50_95") or 0.0

        status_str = "OK" if mAP50 >= 0.10 else "LOW_MAP"

        return {
            "model": model,
            "status": status_str,
            "duration_h": round(duration_h, 2),
            "mAP50": round(float(mAP50), 4),
            "mAP50_95": round(float(mAP5095), 4),
            "out_dir": str(out_dir),
        }

    except subprocess.TimeoutExpired:
        duration_h = (time.time() - start) / 3600
        return {"model": model, "status": "TIMEOUT",
                "duration_h": round(duration_h, 2)}

    except Exception as e:
        duration_h = (time.time() - start) / 3600
        return {"model": model, "status": "EXCEPTION",
                "duration_h": round(duration_h, 2),
                "error": str(e),
                "traceback": traceback.format_exc()[-2000:]}


def write_final_report(status: dict, total_h: float, agg_status: str) -> None:
    ok_count = sum(1 for c in status["completed"] if c.get("status") == "OK")
    lines = []
    lines.append("# Stage 1 Final Report\n")
    lines.append(f"- Started:   {status.get('started_at')}\n")
    lines.append(f"- Completed: {status.get('completed_at')}\n")
    lines.append(f"- Total:     {total_h:.2f} h\n")
    lines.append(f"- OK / Total: {ok_count} / {len(BASELINES)}\n")
    lines.append(f"- Aggregation: {agg_status}\n\n")
    lines.append("## Per-baseline results\n\n")
    lines.append("| Model | Status | mAP@.5 | mAP@.5:.95 | Duration |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for c in status["completed"]:
        lines.append(
            f"| {c.get('model')} | {c.get('status')} "
            f"| {c.get('mAP50','N/A')} | {c.get('mAP50_95','N/A')} "
            f"| {c.get('duration_h','N/A')}h |\n"
        )
    lines.append("\n## Output directories\n\n")
    for c in status["completed"]:
        if c.get("out_dir"):
            lines.append(f"- {c['model']}: `{c['out_dir']}`\n")
    lines.append("\n## Next steps\n\n")
    if ok_count >= 3:
        lines.append("- At least 3 baselines OK. Tier-1 phase done.\n")
        lines.append("- See `results_v2/paper_materials/tables/tier1_comparison.csv` for the full table.\n")
        lines.append("- Proceed to Step 2 (Tier-2 baselines: Faster R-CNN, DETR, Deformable DETR).\n")
    else:
        lines.append("- Less than 3 baselines OK. Debug failed runs before proceeding.\n")
        lines.append("- Check `results_v2/baselines/tier1/_orchestrator_<model>.log` for each model.\n")

    FINAL_REPORT.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    overall_start = time.time()
    status = {
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "phase": "starting",
        "current": None,
        "completed": [],
        "pending": [b["model"] for b in BASELINES],
        "elapsed_h": 0.0,
        "total_timeout_h": TOTAL_TIMEOUT_H,
    }
    update_status(status)
    log(f"Orchestrator started. PID={os.getpid()}  total_timeout={TOTAL_TIMEOUT_H}h")
    log(f"Baselines: {[b['model'] for b in BASELINES]}")

    for i, spec in enumerate(BASELINES):
        model = spec["model"]
        elapsed_h = (time.time() - overall_start) / 3600

        # Skip if not enough time left
        if elapsed_h > TOTAL_TIMEOUT_H - 0.25:  # need 15min slack for aggregation
            log(f"Skipping {model}: time exhausted (elapsed={elapsed_h:.2f}h)")
            result = {"model": model, "status": "SKIPPED_TIMEOUT",
                      "duration_h": 0, "elapsed_h_at_start": round(elapsed_h, 2)}
            status["completed"].append(result)
            if model in status["pending"]:
                status["pending"].remove(model)
            update_status(status)
            continue

        status["phase"] = f"training ({i+1}/{len(BASELINES)})"
        status["current"] = model
        status["elapsed_h"] = round(elapsed_h, 2)
        update_status(status)

        result = train_baseline(spec)
        log(f"==== {model} finished: status={result.get('status')} "
            f"mAP50={result.get('mAP50','N/A')} duration={result.get('duration_h','N/A')}h ====")

        status["completed"].append(result)
        if model in status["pending"]:
            status["pending"].remove(model)
        status["current"] = None
        status["elapsed_h"] = round((time.time() - overall_start) / 3600, 2)
        update_status(status)

    # ============================================================ Aggregate
    status["phase"] = "aggregating"
    update_status(status)
    log("==== All baselines finished. Running aggregator ====")

    agg_status = "skipped"
    agg_script = PROJECT_ROOT / "tools" / "aggregate_tier1.py"
    if agg_script.is_file():
        agg_log = TIER1_DIR / "_orchestrator_aggregate.log"
        try:
            with open(agg_log, "w", encoding="utf-8", errors="replace") as f:
                rc = subprocess.run(
                    [sys.executable, str(agg_script)],
                    cwd=str(PROJECT_ROOT),
                    timeout=900,
                    stdout=f, stderr=subprocess.STDOUT,
                ).returncode
            agg_status = "ok" if rc == 0 else f"failed (rc={rc})"
        except subprocess.TimeoutExpired:
            agg_status = "timeout"
        except Exception as e:
            agg_status = f"exception: {e}"
    else:
        log(f"aggregator not found: {agg_script}")
        agg_status = "missing"

    total_h = (time.time() - overall_start) / 3600
    status["phase"] = "complete"
    status["elapsed_h"] = round(total_h, 2)
    status["completed_at"] = datetime.now().isoformat()
    status["aggregation"] = agg_status
    update_status(status)

    try:
        write_final_report(status, total_h, agg_status)
        log(f"Final report: {FINAL_REPORT}")
    except Exception as e:
        log(f"WARN: failed to write FINAL_REPORT.md: {e}")

    ok = sum(1 for c in status["completed"] if c.get("status") == "OK")
    log(f"DONE. {ok}/{len(BASELINES)} OK. Total={total_h:.2f}h. Aggregation={agg_status}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n[FATAL] {traceback.format_exc()}\n")
        except Exception:
            pass
        raise
