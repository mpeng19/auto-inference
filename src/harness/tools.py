"""Tools an agent can call from its shell, and that a person can call too.

An agent's feedback loop is otherwise one bit every 25-60 minutes: it writes a
diff and eventually learns a price. Everything here exists to put *some*
signal in front of that, cheaply.

    harness tool recall "I am about to raise chunked_prefill_size"
    harness tool preflight --workspace agents/a01
    harness tool roofline --context 20583 --batch 12

`recall` matters most. The fleet's memory is injected into the prompt once per
attempt, but an agent that gets a surprising result mid-task should be able to
*ask* rather than wait for the next injection.

`preflight` is the cheap half of an evaluation. `ast.parse` already runs before
a stack is built, but a NameError costs six GPU-minutes to discover, so this
also runs ruff's undefined-name checks -- which need no SGLang install, unlike
actually importing the module.

`roofline` is the arithmetic an agent would otherwise guess at: given a context
and a batch, what step time and cost per output token should follow, both from
first principles and from what this stack actually measures. The gap between
those two is the optimisation headroom (`docs/methodology.md` §8.3).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
from dataclasses import dataclass

# Measured on the 1xH100 baseline at the marketplace's 20,583-token context.
# See docs/methodology.md 8.3 -- these are what the stack does, not what it
# should do, and the ratio between them is the point.
MEASURED_FIXED_MS = 10.44
MEASURED_SLOPE_MS = 1.585
MEASURED_CONTEXT = 20583


@dataclass(frozen=True)
class Roofline:
    context: int
    batch: int
    n_gpu: int
    weights_gb: float
    per_seq_gb: float
    roofline_fixed_ms: float
    roofline_slope_ms: float
    roofline_step_ms: float
    measured_step_ms: float
    f_weights: float
    f_kv: float
    gpu_s_per_output_token: float
    usd_per_m_output: float

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def roofline(context: int = MEASURED_CONTEXT, batch: int = 12,
             model: str = "Qwen/Qwen3.8-27B-FP8", gpu: str = "H100",
             n_gpu: int = 1, rate_per_gpu_hour: float = 3.00,
             utilisation: float = 0.50) -> Roofline:
    """Predicted decode step time and cost, from first principles and measured.

    Decode is bandwidth-bound: every step reads the whole weight tensor once
    plus each running sequence's KV. So `step = (W + B x per_seq) / bandwidth`
    is a floor, and what the stack achieves against it is the headroom.
    """
    from simulator.specs import HARDWARE, MODELS

    m, hw = MODELS[model], HARDWARE[gpu]
    weights = m.active_params * 1.0                    # FP8: one byte a parameter
    per_seq = m.bytes_per_seq(context, 2.0)            # KV is bf16 even at FP8
    bw = hw.hbm_bandwidth * n_gpu
    rf_fixed = weights / bw * 1e3
    rf_slope = per_seq / bw * 1e3
    rf_step = rf_fixed + rf_slope * batch

    # The measured affine model, scaled: the slope is per-sequence state, so it
    # tracks context; the fixed term is the weight read and does not.
    scale = per_seq / MODELS[model].bytes_per_seq(MEASURED_CONTEXT, 2.0)
    meas_step = (MEASURED_FIXED_MS + MEASURED_SLOPE_MS * scale * batch) / n_gpu

    gpu_s = (meas_step / 1e3) * n_gpu / max(batch, 1)
    usd = gpu_s * rate_per_gpu_hour / 3600 / utilisation * 1e6
    return Roofline(
        context=context, batch=batch, n_gpu=n_gpu,
        weights_gb=round(weights / 1e9, 2), per_seq_gb=round(per_seq / 1e9, 4),
        roofline_fixed_ms=round(rf_fixed, 3), roofline_slope_ms=round(rf_slope, 4),
        roofline_step_ms=round(rf_step, 3), measured_step_ms=round(meas_step, 3),
        f_weights=round(rf_fixed / (MEASURED_FIXED_MS / n_gpu), 3),
        f_kv=round(rf_slope / (MEASURED_SLOPE_MS * scale / n_gpu), 3),
        gpu_s_per_output_token=round(gpu_s, 6), usd_per_m_output=round(usd, 3))


def preflight(workspace: str | pathlib.Path, source=None) -> dict:
    """Everything checkable without a GPU. Returns a report, never raises.

    Ruff rather than an import: checking `srt/managers/scheduler.py` by
    importing it needs SGLang and a CUDA build present, which an agent's
    machine does not have. Ruff's F-rules catch undefined names and bad
    imports statically, which is the failure that actually costs a sweep.
    """
    from harness.agent.workspace import Workspace

    # `source` is injectable so this is testable without the real 1,600-module
    # SGLang tree; the CLI never passes one.
    ws = Workspace(workspace, source=source)
    ok, why = ws.check()
    report = {"ok": ok, "reason": why, "touched": list(ws.touched()),
              "diff_lines": len(ws.diff().splitlines()), "lint": []}
    if not report["touched"]:
        return report

    ruff = _ruff()
    if ruff is None:
        # Degrade rather than fail: `ast.parse` has already run, and an agent's
        # machine is not guaranteed to have ruff. Say so, so a clean report is
        # not mistaken for a thorough one.
        report["lint_skipped"] = "ruff not found; only the parse check ran"
        return report

    paths = [str(ws.candidates / r) for r in report["touched"]]
    try:
        r = _run_ruff(ruff, paths)
    except OSError as e:
        # The binary vanished between resolution and use. Report it; crashing
        # here would take down an agent for a diagnostic it did not ask for.
        report["lint_skipped"] = f"could not run ruff: {e}"
        return report

    # 0 = clean, 1 = findings, anything else = ruff itself failed. That last
    # case must not read as "clean": selecting a rule ruff had removed made
    # this exit 2 and report nothing, which is the exact failure preflight
    # exists to catch.
    if r.returncode not in (0, 1):
        report["lint_skipped"] = (
            f"ruff exited {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}")
        return report
    findings = [ln for ln in r.stdout.splitlines() if ln.strip()
                and not ln.startswith("Found")]
    report["lint"] = findings
    if any("F821" in f for f in findings):
        report["ok"] = False
        report["reason"] = "undefined name; this would fail at import on the GPU"
    return report


def _run_ruff(ruff: list[str], paths: list[str]) -> subprocess.CompletedProcess:
    # Undefined names and bad imports: the ones that fail at import time on the
    # GPU box. Syntax is already covered by `ast.parse`, and style is not our
    # business here.
    return subprocess.run(
        [*ruff, "check", "--no-cache", "--isolated",
         "--select", "F821,F811,F401", "--output-format", "concise", *paths],
        capture_output=True, text=True)


def _ruff() -> list[str] | None:
    """Find ruff: beside this interpreter first, then PATH, then as a module.

    The venv's `bin` is not on PATH inside a subprocess spawned from a test, so
    `shutil.which` alone silently skipped every lint check.
    """
    import shutil
    import sys

    beside = pathlib.Path(sys.executable).parent / "ruff"
    if beside.is_file():
        return [str(beside)]
    found = shutil.which("ruff")
    if found:
        return [found]
    probe = subprocess.run([sys.executable, "-m", "ruff", "--version"],
                           capture_output=True, text=True)
    return [sys.executable, "-m", "ruff"] if probe.returncode == 0 else None


def recall(intent: str, root: str | pathlib.Path | None = None, k: int = 8,
           agent_id: str = "") -> dict:
    """Ask the fleet's memory what is already known about what you are doing."""
    from harness.contracts import Recall
    from harness.memory import SqliteMemory

    db = _find_memory(root)
    if db is None:
        return {"found": False, "reason": "no memory database found",
                "brief": "", "hits": []}
    brief = SqliteMemory(db).recall(Recall(intent=intent, k=k, agent_id=agent_id))
    return {"found": True, "db": str(db), "brief": brief.text,
            "est_tokens": brief.est_tokens,
            "hits": [{"id": h.experiment.id, "verdict": h.experiment.verdict,
                      "hypothesis": h.experiment.hypothesis,
                      "summary": h.experiment.summary, "why": h.why,
                      "score": h.score} for h in brief.hits]}


def _find_memory(root: str | pathlib.Path | None) -> pathlib.Path | None:
    if root:
        p = pathlib.Path(root)
        for c in (p, p / "memory.db"):
            if c.is_file():
                return c
    for base in (pathlib.Path.cwd() / "agents", pathlib.Path.home() / ".auto-inference"):
        if base.is_dir():
            found = sorted(base.rglob("memory.db"),
                           key=lambda f: -f.stat().st_mtime)
            if found:
                return found[0]
    return None


def main(action: str, args) -> int:
    """Dispatch for `harness tool <action>`. Prints JSON when asked."""
    if action == "roofline":
        r = roofline(context=args.context, batch=args.batch, model=args.model,
                     gpu=args.gpu, n_gpu=args.n_gpu)
        if args.json:
            print(json.dumps(r.as_dict(), indent=1))
            return 0
        print(f"{args.model} on {args.n_gpu}x{args.gpu}, context {r.context:,}, "
              f"batch {r.batch}")
        print(f"  weights {r.weights_gb} GB   per-sequence state {r.per_seq_gb} GB")
        print(f"  roofline step  {r.roofline_fixed_ms} ms + "
              f"{r.roofline_slope_ms} ms x batch = {r.roofline_step_ms} ms")
        print(f"  measured step  {r.measured_step_ms} ms"
              f"   ({r.f_weights:.0%} of roofline on the weight read,"
              f" {r.f_kv:.0%} on the KV read)")
        print(f"  -> {r.gpu_s_per_output_token:.2e} GPU-s per output token"
              f" = ${r.usd_per_m_output:.2f}/M")
        print("\n  The KV term is the one a TPOT SLO turns into money: it is "
              "per-sequence,\n  so no batch size amortises it away.")
        return 0

    if action == "preflight":
        rep = preflight(args.workspace)
        if args.json:
            print(json.dumps(rep, indent=1))
            return 0 if rep["ok"] else 1
        print(("OK" if rep["ok"] else "BLOCKED") + f": {rep['reason'] or 'ready to evaluate'}")
        print(f"  {len(rep['touched'])} file(s) changed, {rep['diff_lines']} diff lines")
        for f in rep["touched"]:
            print(f"    {f}")
        for ln in rep["lint"]:
            print(f"  ! {ln}")
        return 0 if rep["ok"] else 1

    rep = recall(args.intent, root=args.root or None, k=args.k)
    if args.json:
        print(json.dumps(rep, indent=1))
        return 0
    if not rep["found"]:
        print(rep["reason"])
        return 1
    print(rep["brief"])
    return 0
