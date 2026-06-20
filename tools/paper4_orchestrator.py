"""Paper 4 STC-YOLO decision-tree orchestrator.

Drives the paper4 milestones toward the Tier-2 metric thresholds
(mAP50_mean >= 0.64, Recall >= 0.55, mAP50_std <= 0.025) by:
  - running experiment steps as subprocesses
  - parsing their metrics output JSON
  - applying gate-decision functions
  - routing to the next step (or to a human gate)

State + journal persist across sessions:
  - results_v2/orchestrator/state.json     (current snapshot)
  - results_v2/orchestrator/journal.jsonl  (append-only event log)

CLI:
    python tools/paper4_orchestrator.py status      # current state, next step
    python tools/paper4_orchestrator.py dry-run     # show what `next` would run
    python tools/paper4_orchestrator.py next        # run one step, then stop
    python tools/paper4_orchestrator.py auto        # run until BLOCKED / target
    python tools/paper4_orchestrator.py auto --max-hours 8
    python tools/paper4_orchestrator.py reset --yes # clear state (DEBUG)
    python tools/paper4_orchestrator.py mark <step_id> completed [--metrics PATH]
                                                    # manual override
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


# ============================================================================
# Globals
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "results_v2" / "orchestrator"

TIER2 = {"mAP50_mean": 0.64, "recall_mean": 0.55, "mAP50_std_max": 0.025}
TIER3 = {"mAP50_mean": 0.62, "recall_mean": 0.50, "mAP50_std_max": 0.035}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Types
# ============================================================================

class StepKind(str, Enum):
    CODE = "CODE"           # requires manual implementation; surfaces BLOCKED
    EXPERIMENT = "EXPERIMENT"  # subprocess
    ANALYSIS = "ANALYSIS"   # subprocess (same path as EXPERIMENT)
    HUMAN = "HUMAN"         # human approval / judgment gate


class StepStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclasses.dataclass
class Decision:
    next_step_id: Optional[str] = None
    config_updates: dict = dataclasses.field(default_factory=dict)
    target_met: bool = False
    note: str = ""


@dataclasses.dataclass
class Step:
    id: str
    phase: str
    kind: StepKind
    description: str
    deps: list[str] = dataclasses.field(default_factory=list)
    # CODE specifics
    code_target_file: Optional[str] = None
    code_completion_check: Optional[str] = None
    # EXPERIMENT/ANALYSIS specifics
    cmd: Optional[list[str]] = None
    metrics_path: Optional[str] = None  # rel-to-project-root JSON output
    # Decision rule (called after step completion with parsed metrics + state)
    decision_rule: Optional[Callable[[dict, dict], Decision]] = None
    # Free-form notes
    notes: str = ""


# ============================================================================
# Decision rules
# ============================================================================

def decide_ms1_d4b_c4_gate(result: dict, state: dict) -> Decision:
    """C4 gate: did colormap inversion preserve or improve mAP?

    Pass: mAP50_mean >= base - 0.005
    Marginal: 0.55 <= mAP50_mean < base - 0.005
    Fail: mAP50_mean < 0.55
    Anomaly: outside [0.40, 0.70]
    """
    mean = result.get("mAP50_5seed_mean")
    if mean is None:
        return Decision(note="metrics missing; manual review needed")
    if mean < 0.40 or mean > 0.70:
        return Decision(next_step_id=None,
                        note=f"ANOMALY mAP50={mean:.3f}; environment / data check (HUMAN)")
    base = (state.get("step_results", {}).get("ms1_d4a_baseline", {})
            .get("mAP50_5seed_mean", 0.602))
    if mean >= base - 0.005:
        return Decision(
            next_step_id="ms2_d5_impl_ssl_pipeline",
            config_updates={"c4_in_main": True},
            note=f"C4 PASS: mean={mean:.4f} >= base({base:.4f}) - 0.005",
        )
    if mean >= 0.55:
        return Decision(
            next_step_id="ms2_d5_impl_ssl_pipeline",
            config_updates={"c4_in_main": False, "c4_in_ablation": True},
            note=f"C4 MARGINAL: mean={mean:.4f}; ablation-only",
        )
    return Decision(
        next_step_id="ms2_d5_impl_ssl_pipeline",
        config_updates={"c4_in_main": False, "c4_in_ablation": False,
                        "c4_dropped": True},
        note=f"C4 FAIL: mean={mean:.4f} < 0.55; dropped from paper",
    )


def decide_ms2_d8_ssl_gate(result: dict, state: dict) -> Decision:
    """SSL gate: did SSL pretrain improve mean OR cut variance?"""
    mean = result.get("mAP50_5seed_mean")
    std = result.get("mAP50_5seed_std", 0.043)
    if mean is None:
        return Decision(note="metrics missing")
    prev = (state.get("step_results", {}).get("ms1_d4b_c4_enabled", {})
            or state.get("step_results", {}).get("ms1_d4a_baseline", {}))
    prev_mean = prev.get("mAP50_5seed_mean", 0.602)
    prev_std = prev.get("mAP50_5seed_std", 0.043)
    delta = mean - prev_mean
    std_ratio = std / max(prev_std, 1e-6)
    if delta >= 0.008 or std_ratio <= 0.70:
        return Decision(
            next_step_id="ms3_d10_impl_ema_teacher",
            config_updates={"c1_in_main": True},
            note=f"C1 PASS: Δmean={delta:+.4f}, std×{std_ratio:.2f}",
        )
    if delta >= -0.005:
        return Decision(
            next_step_id="ms3_d10_impl_ema_teacher",
            config_updates={"c1_in_main": False, "c1_warmup_only": True},
            note=f"C1 MARGINAL: Δmean={delta:+.4f}; warm-up only",
        )
    return Decision(
        next_step_id="ms3_d10_impl_ema_teacher",
        config_updates={"c1_in_main": False, "c1_dropped": True},
        note=f"C1 FAIL: Δmean={delta:+.4f}; dropped",
    )


def decide_ms3_d12_full_stc_gate(result: dict, state: dict) -> Decision:
    """Final headline gate: did full STC-YOLO hit Tier-2 / Tier-3 target?"""
    mean = result.get("mAP50_5seed_mean")
    std = result.get("mAP50_5seed_std")
    recall = result.get("recall_5seed_mean")
    if mean is None or recall is None:
        return Decision(note="metrics missing")
    if (mean >= TIER2["mAP50_mean"] and recall >= TIER2["recall_mean"]
            and (std is None or std <= TIER2["mAP50_std_max"])):
        return Decision(
            next_step_id="ms4_d15_voting_ablation",
            target_met=True,
            note=f"TIER-2 MET: mAP50={mean:.4f}, R={recall:.4f}, std={std}",
        )
    if (mean >= TIER3["mAP50_mean"] and recall >= TIER3["recall_mean"]
            and (std is None or std <= TIER3["mAP50_std_max"])):
        return Decision(
            next_step_id="ms4_d15_voting_ablation",
            config_updates={"venue_fallback": "tier3"},
            note=f"TIER-3 MET: mAP50={mean:.4f}, R={recall:.4f}, std={std}",
        )
    return Decision(
        next_step_id="ms3_fallback_recovery",
        note=f"BELOW BOTH TIERS: mAP50={mean:.4f}, R={recall:.4f}; HUMAN recovery",
    )


# ============================================================================
# Tree definition
# ============================================================================

def build_tree() -> dict[str, Step]:
    steps = [
        # ---- MS1 prefix (already done before orchestrator built) ----
        Step(id="ms1_d1_skeleton", phase="MS1", kind=StepKind.CODE,
             description="m6_stc_yolo skeleton (20 files, tests 10/10 ok)",
             code_target_file="methods/m6_stc_yolo/",
             notes="Pre-completed 2026-05-29."),
        Step(id="ms1_d1b_colormap_analysis", phase="MS1",
             kind=StepKind.EXPERIMENT,
             description="Colormap analysis: empirical LUT MSE 0.000568 (single LUT)",
             deps=["ms1_d1_skeleton"],
             cmd=["python", "methods/m6_stc_yolo/analysis/colormap_analysis.py"],
             metrics_path="results_v2/methods/stc_yolo/_ms1/colormap_analysis.json",
             notes="Pre-completed 2026-05-29."),
        # ---- MS1 active ----
        Step(id="ms1_d4a_baseline", phase="MS1", kind=StepKind.EXPERIMENT,
             description="Vanilla YOLOv12s tuned recipe, 5-seed baseline",
             deps=["ms1_d1b_colormap_analysis"],
             cmd=["python", "tools/paper4_run_5seed.py",
                  "--config", "methods/m6_stc_yolo/configs/default.yaml",
                  "--run-name", "ms1_d4a_baseline"],
             metrics_path="results_v2/orchestrator/runs/ms1_d4a_baseline/5seed_summary.json",
             notes="Validates paper3 0.602±0.043 reproduces; no C4."),
        Step(id="ms1_d2_impl_colormap_inverter", phase="MS1", kind=StepKind.CODE,
             description="Implement ColormapInverter.forward (mode='empirical')",
             deps=["ms1_d1b_colormap_analysis"],
             code_target_file="methods/m6_stc_yolo/modules/colormap_inverter.py",
             code_completion_check="ColormapInverter().forward(torch.rand(1,3,8,8)) returns (1,1,8,8) tensor in [0,1]",
             notes="Load empirical_lut.npz, register lut_rgb + t as buffers, batched NN lookup."),
        Step(id="ms1_d3_impl_conv1ch", phase="MS1", kind=StepKind.CODE,
             description="Implement Conv1chFromRGB.build_from_pretrained_conv",
             deps=["ms1_d2_impl_colormap_inverter"],
             code_target_file="methods/m6_stc_yolo/modules/conv1ch_from_rgb.py",
             code_completion_check="Conv1chFromRGB.build_from_pretrained_conv(orig_conv) returns a Conv1chFromRGB instance; forward(rand(1,1,640,640)) returns (1,32,320,320)",
             notes="Subclass ultralytics Conv (survive deepcopy); RGB-mean weight migration; copy BN eps/momentum."),
        Step(id="ms1_d3b_integrate_c4", phase="MS1", kind=StepKind.CODE,
             description="Wire C4 into build_stc_yolo + DetectionTrainer subclass",
             deps=["ms1_d3_impl_conv1ch"],
             code_target_file="methods/m6_stc_yolo/model.py",
             code_completion_check="build_stc_yolo(enable_colormap_inversion=True) returns YOLO with model.model[0] = Conv1chFromRGB; make_stc_trainer() provided",
             notes="Mirror tg_yolo's make_tg_trainer trick: override get_model() to re-swap after ultralytics rebuild."),
        Step(id="ms1_d4b_c4_enabled", phase="MS1", kind=StepKind.EXPERIMENT,
             description="C4-enabled YOLOv12s, 5-seed (gate)",
             deps=["ms1_d3b_integrate_c4", "ms1_d4a_baseline"],
             cmd=["python", "tools/paper4_run_5seed.py",
                  "--config", "methods/m6_stc_yolo/configs/default.yaml",
                  "--override", "stc.enable_colormap_inversion=true",
                  "--run-name", "ms1_d4b_c4_enabled"],
             metrics_path="results_v2/orchestrator/runs/ms1_d4b_c4_enabled/5seed_summary.json",
             decision_rule=decide_ms1_d4b_c4_gate),
        # ---- MS2 ----
        Step(id="ms2_d5_impl_ssl_pipeline", phase="MS2", kind=StepKind.CODE,
             description="Implement FramePairDataset + CrossFrameTokenConsistencyLoss + SSLPretrainer.fit",
             deps=["ms1_d4b_c4_enabled"],
             code_target_file="methods/m6_stc_yolo/modules/ssl_pretrain.py",
             code_completion_check="SSLPretrainer(backbone=..., cfg=...).fit() runs 1 step without NotImpl",
             notes="DenseCL-style InfoNCE on P3 tokens, in-batch negatives; sample (t,t+k) pairs same-session."),
        Step(id="ms2_d6_run_ssl_pretrain", phase="MS2", kind=StepKind.EXPERIMENT,
             description="80-ep SSL pretrain -> pretrained_backbone_stc.pt (~12h)",
             deps=["ms2_d5_impl_ssl_pipeline"],
             cmd=["python", "methods/m6_stc_yolo/scripts/run_ssl_pretrain.py",
                  "--config", "methods/m6_stc_yolo/configs/ssl_pretrain.yaml"],
             metrics_path="results_v2/methods/stc_yolo/_ms2/ssl_pretrain_summary.json",
             notes="Run helper script TBD."),
        Step(id="ms2_d7_load_ssl_in_build", phase="MS2", kind=StepKind.CODE,
             description="Wire ssl_pretrained_backbone in build_stc_yolo",
             deps=["ms2_d6_run_ssl_pretrain"],
             code_target_file="methods/m6_stc_yolo/model.py",
             code_completion_check="build_stc_yolo(ssl_pretrained_backbone=path) replaces backbone state_dict"),
        Step(id="ms2_d8_supervised_ft_5seed", phase="MS2", kind=StepKind.EXPERIMENT,
             description="SSL-init + supervised FT (inherits C4 decision), 5-seed",
             deps=["ms2_d7_load_ssl_in_build"],
             cmd=["python", "tools/paper4_run_5seed.py",
                  "--config", "methods/m6_stc_yolo/configs/default.yaml",
                  "--inherit-config", "from-state",
                  "--run-name", "ms2_d8_supervised_ft_5seed"],
             metrics_path="results_v2/orchestrator/runs/ms2_d8_supervised_ft_5seed/5seed_summary.json",
             decision_rule=decide_ms2_d8_ssl_gate),
        # ---- MS3 ----
        Step(id="ms3_d10_impl_ema_teacher", phase="MS3", kind=StepKind.CODE,
             description="EMATeacher + TemporalConsistencyLoss + UnlabeledNeighborSampler",
             deps=["ms2_d8_supervised_ft_5seed"],
             code_target_file="methods/m6_stc_yolo/modules/mean_teacher.py",
             code_completion_check="EMATeacher.predict_pseudo(rand_frames, 0.5) returns List[Tensor]"),
        Step(id="ms3_d10d_impl_mt_trainer", phase="MS3", kind=StepKind.CODE,
             description="DetectionTrainer subclass: MT consistency loss in training step",
             deps=["ms3_d10_impl_ema_teacher"],
             code_target_file="methods/m6_stc_yolo/model.py",
             code_completion_check="make_stc_trainer(enable_mean_teacher=True) returns subclass"),
        Step(id="ms3_d11_impl_session_film", phase="MS3", kind=StepKind.CODE,
             description="SessionFiLMGenerator + apply_film + head injection",
             deps=["ms2_d8_supervised_ft_5seed"],  # parallel with d10d
             code_target_file="methods/m6_stc_yolo/modules/session_film.py",
             code_completion_check="SessionFiLMGenerator(8)(rand(2,8)) returns dict P3/P4/P5 -> (gamma, beta)"),
        Step(id="ms3_d12_full_stc_5seed", phase="MS3", kind=StepKind.EXPERIMENT,
             description="Full STC-YOLO (C1+C2+C3+C4 as configured), 5-seed — HEADLINE",
             deps=["ms3_d10d_impl_mt_trainer", "ms3_d11_impl_session_film"],
             cmd=["python", "tools/paper4_run_5seed.py",
                  "--config", "methods/m6_stc_yolo/configs/semi_supervised.yaml",
                  "--inherit-config", "from-state",
                  "--run-name", "ms3_d12_full_stc_5seed"],
             metrics_path="results_v2/orchestrator/runs/ms3_d12_full_stc_5seed/5seed_summary.json",
             decision_rule=decide_ms3_d12_full_stc_gate),
        Step(id="ms3_fallback_recovery", phase="MS3", kind=StepKind.HUMAN,
             description="Below both tiers - recovery: ext training | alt MT lambda | drop module | abort",
             deps=["ms3_d12_full_stc_5seed"],
             notes="Reached only when ms3 gate returns no target met."),
        # ---- MS4 ----
        Step(id="ms4_d15_voting_ablation", phase="MS4", kind=StepKind.EXPERIMENT,
             description="C5 1-frame voting ablation",
             deps=["ms3_d12_full_stc_5seed"],
             cmd=["python", "methods/m6_stc_yolo/scripts/run_voting_ablation.py"],
             metrics_path="results_v2/methods/stc_yolo/_ms4/voting_ablation.json"),
        Step(id="ms4_d15b_full_ablation", phase="MS4", kind=StepKind.EXPERIMENT,
             description="Full ablation table A0-A17",
             deps=["ms3_d12_full_stc_5seed"],
             cmd=["python", "tools/paper4_run_full_ablation.py"],
             metrics_path="results_v2/methods/stc_yolo/_ms4/full_ablation.json"),
        Step(id="ms4_d16_session_disjoint", phase="MS4", kind=StepKind.EXPERIMENT,
             description="22-fold leave-one-session-out cross-validation",
             deps=["ms3_d12_full_stc_5seed"],
             cmd=["python", "tools/paper4_run_loso.py"],
             metrics_path="results_v2/methods/stc_yolo/_ms4/loso_22fold.json"),
        Step(id="ms4_d17_paper_figures", phase="MS4", kind=StepKind.ANALYSIS,
             description="Generate Figs 1-4 + LaTeX tables",
             deps=["ms4_d15_voting_ablation", "ms4_d15b_full_ablation",
                   "ms4_d16_session_disjoint"],
             cmd=["python", "tools/paper4_generate_figures.py"],
             metrics_path="results_v2/methods/stc_yolo/_ms4/figures/manifest.json"),
        Step(id="ms4_d18_manuscript", phase="MS4", kind=StepKind.CODE,
             description="Draft manuscript.tex (sn-jnl, mirrors paper3v2 layout)",
             deps=["ms4_d17_paper_figures"],
             code_target_file="doc/paper4/manuscript.tex",
             code_completion_check="manuscript.tex exists; has Abstract/Intro/Related/Method/Experiments/Results/Conclusion"),
        Step(id="ms4_submit", phase="MS4", kind=StepKind.HUMAN,
             description="Submit to Tier-2 venue (or Tier-3 fallback if venue_fallback set)",
             deps=["ms4_d18_manuscript"],
             notes="HUMAN GATE."),
    ]
    return {s.id: s for s in steps}


# ============================================================================
# Runtime
# ============================================================================

# Steps already completed before the orchestrator existed (set at init).
PRE_COMPLETED = {
    "ms1_d1_skeleton": {
        "n_files": 20,
        "tests_passed": "10/10",
    },
    "ms1_d1b_colormap_analysis": {
        "best_colormap": "inferno",
        "empirical_lut_mse_train": 0.000651,
        "empirical_lut_mse_held": 0.000568,
        "consistency_pct": 99.49,
        "recommendation": "empirical_analytic_inversion_with_residual_mlp",
    },
}


class Runtime:
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.state_dir = STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "runs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.state_dir / "state.json"
        self.journal_path = self.state_dir / "journal.jsonl"
        self.tree = build_tree()
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return self._init_state()

    def _init_state(self) -> dict:
        step_status = {sid: StepStatus.PENDING.value for sid in self.tree}
        step_results: dict = {}
        for sid, results in PRE_COMPLETED.items():
            if sid in step_status:
                step_status[sid] = StepStatus.COMPLETED.value
                step_results[sid] = results
        state = {
            "started_at": now_iso(),
            "last_updated": now_iso(),
            "target_met": False,
            "venue_fallback": None,
            "step_status": step_status,
            "step_results": step_results,
            "config_updates": {},
        }
        self._write_state(state)
        self._journal({"event": "init", "pre_completed": list(PRE_COMPLETED.keys())},
                      state=state)
        return state

    def _write_state(self, state: dict | None = None) -> None:
        s = state or self.state
        s["last_updated"] = now_iso()
        self.state_path.write_text(
            json.dumps(s, indent=2, default=str), encoding="utf-8")

    def _journal(self, event: dict, state: dict | None = None) -> None:
        event = {"ts": now_iso(), **event}
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

    # ----------------------------------- helpers -----------------------------
    def status_of(self, sid: str) -> str:
        return self.state["step_status"].get(sid, StepStatus.PENDING.value)

    def set_status(self, sid: str, status: StepStatus) -> None:
        self.state["step_status"][sid] = status.value
        self._write_state()

    def _deps_satisfied(self, step: Step) -> bool:
        ok_states = {StepStatus.COMPLETED.value, StepStatus.SKIPPED.value}
        return all(self.status_of(d) in ok_states for d in step.deps)

    def next_runnable_step(self) -> Optional[Step]:
        # Honor the decision rule's last next_step_id if present in config_updates
        for step in self.tree.values():
            if self.status_of(step.id) != StepStatus.PENDING.value:
                continue
            if not self._deps_satisfied(step):
                continue
            return step
        return None

    # ----------------------------------- exec --------------------------------
    def execute(self, step: Step) -> bool:
        """Returns True on completed, False on blocked/failed."""
        if step.kind in (StepKind.CODE, StepKind.HUMAN):
            self.set_status(step.id, StepStatus.BLOCKED)
            self._journal({
                "step": step.id, "event": "blocked",
                "kind": step.kind.value,
                "description": step.description,
                "target_file": step.code_target_file,
                "completion_check": step.code_completion_check,
                "notes": step.notes,
            })
            return False
        return self._run_subprocess(step)

    def _run_subprocess(self, step: Step) -> bool:
        run_dir = self.state_dir / "runs" / step.id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_p = run_dir / "stdout.log"
        stderr_p = run_dir / "stderr.log"
        self.set_status(step.id, StepStatus.RUNNING)
        self._journal({"step": step.id, "event": "running", "cmd": step.cmd})
        start = time.time()
        try:
            with open(stdout_p, "w", encoding="utf-8") as so, \
                 open(stderr_p, "w", encoding="utf-8") as se:
                proc = subprocess.run(
                    step.cmd, cwd=self.project_root,
                    stdout=so, stderr=se, text=True, encoding="utf-8")
            elapsed = time.time() - start
        except Exception as e:
            self.set_status(step.id, StepStatus.FAILED)
            self._journal({"step": step.id, "event": "exception", "error": str(e)})
            return False
        if proc.returncode != 0:
            self.set_status(step.id, StepStatus.FAILED)
            self._journal({"step": step.id, "event": "failed",
                           "returncode": proc.returncode,
                           "elapsed_s": elapsed,
                           "stderr_tail": stderr_p.read_text(encoding="utf-8",
                                                              errors="replace")[-2000:]})
            return False
        # parse metrics
        metrics: dict = {}
        if step.metrics_path:
            mpath = self.project_root / step.metrics_path
            if mpath.exists():
                try:
                    metrics = json.loads(mpath.read_text(encoding="utf-8"))
                except Exception as e:
                    self.set_status(step.id, StepStatus.FAILED)
                    self._journal({"step": step.id, "event": "metrics_parse_fail",
                                   "path": str(mpath), "error": str(e)})
                    return False
            else:
                self.set_status(step.id, StepStatus.FAILED)
                self._journal({"step": step.id, "event": "metrics_missing",
                               "path": str(mpath)})
                return False
        self.state["step_results"][step.id] = metrics
        # decision rule
        if step.decision_rule is not None:
            decision = step.decision_rule(metrics, self.state)
            self.state["config_updates"].update(decision.config_updates)
            if decision.target_met:
                self.state["target_met"] = True
            if "venue_fallback" in decision.config_updates:
                self.state["venue_fallback"] = decision.config_updates["venue_fallback"]
            self._journal({"step": step.id, "event": "decision",
                           "next": decision.next_step_id,
                           "note": decision.note,
                           "config_updates": decision.config_updates,
                           "target_met": decision.target_met})
        self.set_status(step.id, StepStatus.COMPLETED)
        self._journal({"step": step.id, "event": "completed",
                       "elapsed_s": elapsed, "metrics": metrics})
        return True

    # ----------------------------------- status ------------------------------
    def status_report(self) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("Paper 4 STC-YOLO Orchestrator -- Status")
        lines.append("=" * 78)
        lines.append(f"started_at:     {self.state['started_at']}")
        lines.append(f"last_updated:   {self.state['last_updated']}")
        lines.append(f"target_met:     {self.state['target_met']}")
        lines.append(f"venue_fallback: {self.state['venue_fallback']}")
        if self.state["config_updates"]:
            lines.append("config_updates:")
            for k, v in self.state["config_updates"].items():
                lines.append(f"  - {k} = {v}")
        for phase in ["MS1", "MS2", "MS3", "MS4"]:
            lines.append("")
            lines.append(f"[{phase}]")
            for step in self.tree.values():
                if step.phase != phase:
                    continue
                st = self.status_of(step.id)
                glyph = {
                    "completed": "[OK ]", "pending": "[ . ]", "running": "[>>>]",
                    "blocked": "[BLK]", "failed": "[FAIL]", "skipped": "[SKP]",
                }.get(st, "[ ? ]")
                kind_short = step.kind.value[:4].ljust(4)
                lines.append(f"  {glyph} {kind_short} {step.id:<32} {step.description[:48]}")
        nxt = self.next_runnable_step()
        lines.append("")
        lines.append("-" * 78)
        if nxt is None:
            lines.append("NO RUNNABLE STEP (all blocked, complete, or failed)")
        else:
            lines.append(f"NEXT RUNNABLE: {nxt.id}  ({nxt.kind.value})")
            lines.append(f"  description: {nxt.description}")
            if nxt.kind == StepKind.CODE:
                lines.append(f"  target file: {nxt.code_target_file}")
                lines.append(f"  completion check: {nxt.code_completion_check}")
            elif nxt.kind == StepKind.HUMAN:
                lines.append(f"  notes: {nxt.notes}")
            elif nxt.cmd:
                cmd_str = " ".join(nxt.cmd)
                if len(cmd_str) > 70:
                    cmd_str = cmd_str[:67] + "..."
                lines.append(f"  cmd: {cmd_str}")
                if nxt.metrics_path:
                    lines.append(f"  metrics path: {nxt.metrics_path}")
        return "\n".join(lines)


# ============================================================================
# CLI commands
# ============================================================================

def cmd_status(rt: Runtime, args) -> int:
    print(rt.status_report())
    return 0


def cmd_dry_run(rt: Runtime, args) -> int:
    nxt = rt.next_runnable_step()
    if nxt is None:
        print("No runnable step.")
        return 0
    print(f"Would execute: {nxt.id}")
    print(f"  kind: {nxt.kind.value}")
    print(f"  description: {nxt.description}")
    if nxt.cmd:
        print(f"  cmd: {' '.join(nxt.cmd)}")
    if nxt.metrics_path:
        print(f"  metrics_path: {nxt.metrics_path}")
    if nxt.decision_rule:
        print(f"  decision_rule: {nxt.decision_rule.__name__}")
    return 0


def cmd_next(rt: Runtime, args) -> int:
    nxt = rt.next_runnable_step()
    if nxt is None:
        print("No runnable step.")
        return 0
    print(f"Executing: {nxt.id} ({nxt.kind.value})")
    ok = rt.execute(nxt)
    print("DONE" if ok else "BLOCKED / FAILED — see journal.jsonl")
    return 0 if ok else 1


def cmd_auto(rt: Runtime, args) -> int:
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    while True:
        if rt.state.get("target_met"):
            print("TARGET MET — stopping auto.")
            return 0
        nxt = rt.next_runnable_step()
        if nxt is None:
            print("No runnable step (all blocked or complete).")
            return 0
        if nxt.kind in (StepKind.CODE, StepKind.HUMAN):
            print(f"Auto stopped: next step {nxt.id} is {nxt.kind.value} — manual.")
            rt.execute(nxt)  # mark blocked + journal
            return 0
        if deadline and time.time() > deadline:
            print(f"Time budget exhausted ({args.max_hours} h).")
            return 0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] running {nxt.id}")
        ok = rt.execute(nxt)
        if not ok:
            print(f"Step {nxt.id} failed; halting auto.")
            return 1


def cmd_reset(rt: Runtime, args) -> int:
    if not args.yes:
        print("Refusing without --yes.")
        return 1
    if rt.state_path.exists():
        rt.state_path.unlink()
    if rt.journal_path.exists():
        rt.journal_path.unlink()
    print("State + journal cleared.")
    return 0


def cmd_mark(rt: Runtime, args) -> int:
    sid = args.step_id
    if sid not in rt.tree:
        print(f"Unknown step: {sid}")
        return 1
    if args.new_status not in [s.value for s in StepStatus]:
        print(f"Unknown status: {args.new_status}")
        return 1
    rt.set_status(sid, StepStatus(args.new_status))
    if args.metrics:
        rt.state["step_results"][sid] = json.loads(Path(args.metrics).read_text())
        rt._write_state()
    rt._journal({"step": sid, "event": "manual_mark",
                 "new_status": args.new_status,
                 "metrics_injected": bool(args.metrics)})
    print(f"Marked {sid} → {args.new_status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paper 4 STC-YOLO orchestrator")
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("status")
    sub.add_parser("dry-run")
    sub.add_parser("next")
    auto = sub.add_parser("auto")
    auto.add_argument("--max-hours", type=float, default=None)
    reset = sub.add_parser("reset")
    reset.add_argument("--yes", action="store_true")
    mark = sub.add_parser("mark")
    mark.add_argument("step_id")
    mark.add_argument("new_status",
                      choices=[s.value for s in StepStatus])
    mark.add_argument("--metrics", type=str, default=None)
    args = p.parse_args(argv)
    rt = Runtime()
    dispatch = {
        None: cmd_status, "status": cmd_status, "dry-run": cmd_dry_run,
        "next": cmd_next, "auto": cmd_auto, "reset": cmd_reset, "mark": cmd_mark,
    }
    return dispatch[args.cmd](rt, args)


if __name__ == "__main__":
    sys.exit(main())
