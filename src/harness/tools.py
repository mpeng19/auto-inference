"""Tools an agent can call from its shell, and that a person can call too.

An agent's feedback loop is otherwise one bit every 25-60 minutes: it writes a
diff and eventually learns a price. Everything here exists to put *some*
signal in front of that, cheaply.

    harness tool recall "I am about to raise chunked_prefill_size"
    harness tool preflight --workspace agents/a01
    harness tool roofline --context 20583 --batch 12
    harness tool gpu-run bench.py --workspace agents/a01
    harness tool equivalence --workspace agents/a01

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

`gpu_run` and `equivalence` are the two that make kernel work possible at all.
Everything above is free and everything below a sweep was, until now, a sweep:
17-35 minutes and a price. A Triton kernel needs to be asked whether it
compiles, how fast it is, and whether it still computes the same numbers, and
none of those questions has anything to do with a price. Both rent an H100 for
minutes and cost real money -- `cost_usd` comes back with the answer.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import subprocess
import sys
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


# ── the GPU workbench: minutes of H100 instead of a sweep ────────────────

def _workspace_stack(workspace: str | pathlib.Path | None, source=None):
    """(stack, artifact root, note) for a workspace, or stock if there is none.

    A workspace with **no edits is stock**, not an error. `Workspace.stack()`
    refuses that case because for a sweep it would re-measure the baseline, but
    here it is the thing an agent should do first: find out what stock's kernel
    costs before writing a replacement for it. The note says which was run, so
    a result is never silently about the wrong code.

    A workspace that does not **parse** is refused. That is preflight's whole
    argument -- a GPU is an expensive place to discover a syntax error.
    """
    from simulator import InferenceStack

    if not workspace:
        return InferenceStack.stock(), pathlib.Path.cwd(), "no workspace: stock sglang"
    from harness.agent.workspace import Workspace

    # `source` is injectable for the same reason `preflight`'s is: so this is
    # testable without the real 1,600-module SGLang tree. The CLI never passes one.
    ws = Workspace(workspace, source=source)
    if not ws.touched():
        return (InferenceStack.stock(), ws.root,
                "workspace has no changes: ran stock sglang")
    ok, why = ws.check()
    if not ok:
        raise ValueError(why)
    return ws.stack(), ws.root, ""


def gpu_run(script_path: str | pathlib.Path,
            workspace: str | pathlib.Path | None = None,
            timeout_s: int = 600, files: dict[str, str] | None = None,
            source=None) -> dict:
    """Run one python script on an H100, against this workspace's stack.

    The inner loop kernel work needs. The script runs in a fresh container with
    the candidate files written over the installed sglang, its working
    directory to itself, and `files` (path -> text) beside it for helpers.
    Whatever it printed comes back, along with what it cost.

    **The candidate's files travel; its `serving.json` `env` does not.** The
    runner applies `stack.env` to a sweep's server and the workbench script
    gets the container's own environment, so a change hidden behind an
    environment variable runs here as stock -- on 2026-09-02 that cost an agent
    a workbench run and an equivalence run to discover. Export the variable in
    the script itself.

    Returns a report rather than raising, like `preflight`: an agent in a loop
    needs the reason as data, and a missing script or an unparseable workspace
    should not cost a GPU to discover.
    """
    import asyncio

    p = pathlib.Path(script_path)
    if not p.is_file():
        return {"ok": False, "error":
                f"no script at {p}. gpu_run runs a python file on the GPU; "
                "write the file first, then give this its path."}
    try:
        stack, root, note = _workspace_stack(workspace, source)
    except ValueError as e:
        return {"ok": False, "error": f"workspace is not runnable: {e}"}

    from simulator import Simulator

    sim = Simulator(root_dir=root, stack=stack)
    rec = asyncio.run(sim.workbench(p.read_text(), files=files,
                                    timeout_s=timeout_s))
    rec["script"] = str(p)
    rec["stack_digest"] = stack.digest
    if note:
        rec["note"] = note
    return rec


NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
)

NCU_SCRIPT = r"""
import csv, io, json, os, subprocess, sys
metrics = os.environ.get("NCU_METRICS", "")
kernel = os.environ.get("NCU_KERNEL", "")
count = os.environ.get("NCU_LAUNCH_COUNT", "20")
cmd = ["ncu", "--clock-control", "none", "--csv", "--launch-count", count,
       "--metrics", metrics]
if kernel:
    cmd += ["--kernel-name", "regex:" + kernel]
cmd += [sys.executable, "target.py"]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
sys.stderr.write(r.stderr[-4000:])
lines = [ln for ln in r.stdout.splitlines() if ln.startswith('"')]
rows = list(csv.DictReader(io.StringIO("\n".join(lines)))) if lines else []
out = {}
for row in rows:
    k = row.get("Kernel Name", "?")
    m = row.get("Metric Name", "")
    v = row.get("Metric Value", "").replace(",", "")
    try:
        v = float(v)
    except ValueError:
        pass
    out.setdefault(k, {"launches": 0})[m] = v
    if m == "gpu__time_duration.sum":
        out[k]["launches"] += 1
print("NCU_JSON " + json.dumps({"rc": r.returncode, "kernels": out}))
"""


def ncu(script_path: str | pathlib.Path, workspace: str | pathlib.Path | None = None,
        kernel: str = "", metrics: tuple[str, ...] = NCU_METRICS, launch_count: int = 20,
        timeout_s: int = 1800, source=None) -> dict:
    """Hardware counters per kernel, from Nsight Compute, on an H100.

    tracedb says which kernel takes the step and for how long; this says
    *why*: achieved DRAM and SM throughput as a percent of peak, warp
    occupancy, L2 hit rate. It is the instrument that turns "the KV read
    runs at 22-28% of bandwidth" from an inference into a measurement.

    The script is run under `ncu --clock-control none`: the container may
    not lock GPU clocks, and the first attempt with locking failed on
    exactly that. Each profiled kernel replays ~9 passes, so profile a
    decode step or a micro-benchmark, not a sweep, and narrow `kernel` to a
    regex when you can. Returns per-kernel metrics as data; stdout/stderr
    from the workbench are kept for the cases the parser cannot read.
    """
    import asyncio

    p = pathlib.Path(script_path)
    if not p.is_file():
        return {"ok": False, "error": f"no script at {p}. ncu profiles a python file; "
                                      "write the file first, then give this its path."}
    try:
        stack, root, note = _workspace_stack(workspace, source)
    except ValueError as e:
        return {"ok": False, "error": f"workspace is not runnable: {e}"}

    from simulator import Simulator

    driver = ("import os\n"
              f"os.environ['NCU_METRICS'] = {','.join(metrics)!r}\n"
              f"os.environ['NCU_KERNEL'] = {kernel!r}\n"
              f"os.environ['NCU_LAUNCH_COUNT'] = {str(launch_count)!r}\n"
              + NCU_SCRIPT)
    sim = Simulator(root_dir=root, stack=stack)
    rec = asyncio.run(sim.workbench(driver, files={"target.py": p.read_text()},
                                    timeout_s=timeout_s))
    rec["script"] = str(p)
    rec["stack_digest"] = stack.digest
    if note:
        rec["note"] = note
    for line in (rec.get("stdout") or "").splitlines():
        if line.startswith("NCU_JSON "):
            with contextlib.suppress(json.JSONDecodeError):
                rec["ncu"] = json.loads(line[len("NCU_JSON "):])
    if "ncu" not in rec:
        rec["ok"] = False
        rec["error"] = "ncu produced no parseable output; see stderr"
    return rec


def equivalence(workspace: str | pathlib.Path | None = None,
                timeout_s: int = 1800, min_agreement: float | None = None,
                max_mean_dlogprob: float | None = None, source=None) -> dict:
    """Does this stack still compute what stock computes? Token by token.

    The gate GSM8K is too blunt for. A rewritten attention kernel can move
    every logit and still get the same 36 of 50 answers right; this scores
    ~2,000 teacher-forced positions against a cached stock reference and
    reports how far the distribution moved. See
    `simulator.measure.equivalence` -- including why the thresholds are
    provisional, and why the first run of a model pays for the reference.

    Same caveat as `gpu_run`: this runs on the workbench, so a candidate whose
    new path is gated on an environment variable is scored with the path off.
    Agreement of exactly 1.0000 with |dlogprob| exactly 0.0000 from a change
    that touches numerics is that, not a lossless kernel.
    """
    import asyncio

    try:
        stack, root, note = _workspace_stack(workspace, source)
    except ValueError as e:
        return {"ok": False, "error": f"workspace is not runnable: {e}"}

    from simulator import Simulator
    from simulator.measure import equivalence as eq

    sim = Simulator(root_dir=root, stack=stack)
    rec = asyncio.run(eq.measure(
        sim, timeout_s=timeout_s,
        min_agreement=eq.MIN_AGREEMENT if min_agreement is None else min_agreement,
        max_mean_dlogprob=(eq.MAX_MEAN_DLOGPROB if max_mean_dlogprob is None
                           else max_mean_dlogprob)))
    if note:
        rec["note"] = note
    return rec


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
        # An explicit root is an answer, not a hint: a missing database there
        # must not quietly become whichever fleet last ran in this directory.
        p = pathlib.Path(root)
        for c in (p, p / "memory.db"):
            if c.is_file():
                return c
        return None
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

    if action == "gpu-run":
        rep = gpu_run(args.intent, workspace=args.workspace or None,
                      timeout_s=args.timeout or 600)
        if args.json:
            print(json.dumps(rep, indent=1, default=str))
            return 0 if rep.get("ok") else 1
        if "error" in rep and "exit_code" not in rep:
            print(rep["error"], file=sys.stderr)
            return 2
        print(("OK" if rep.get("ok") else "FAILED")
              + f"   exit {rep.get('exit_code')}   {rep.get('elapsed_s', 0)}s"
              + f" on {rep.get('gpu', '?')}   ${rep.get('cost_usd', 0):.3f}")
        if rep.get("note"):
            print(f"  note: {rep['note']}")
        if rep.get("stdout"):
            print(rep["stdout"].rstrip())
        if rep.get("stderr"):
            print("--- stderr ---", file=sys.stderr)
            print(rep["stderr"].rstrip(), file=sys.stderr)
        print(f"artifacts: {rep.get('dir')}")
        return 0 if rep.get("ok") else 1

    if action == "ncu":
        rep = ncu(args.intent, workspace=args.workspace or None,
                  kernel=getattr(args, "kernel", "") or "",
                  timeout_s=args.timeout or 1800)
        if args.json or "ncu" not in rep:
            print(json.dumps(rep if args.json else {k: rep.get(k) for k in ("ok", "error", "stderr")},
                             indent=1, default=str))
            return 0 if rep.get("ok") else 1
        print(f"ncu on {rep.get('gpu', 'GPU')}  stack {rep['stack_digest']}  ${rep.get('cost_usd', 0):.2f}")
        print(f"{'kernel':<48}{'launches':>9}{'us':>10}{'dram%':>8}{'sm%':>7}{'occ%':>7}{'l2hit%':>8}")
        def g(m, name):
            v = m.get(name)
            return f"{v:.1f}" if isinstance(v, float) else "-"

        for k, m in sorted(rep["ncu"]["kernels"].items(),
                           key=lambda kv: -float(kv[1].get("gpu__time_duration.sum", 0) or 0)):
            print(f"{k[:47]:<48}{m.get('launches', 0):>9}{g(m, 'gpu__time_duration.sum'):>10}"
                  f"{g(m, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'):>8}"
                  f"{g(m, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):>7}"
                  f"{g(m, 'sm__warps_active.avg.pct_of_peak_sustained_active'):>7}"
                  f"{g(m, 'lts__t_sector_hit_rate.pct'):>8}")
        return 0

    if action == "equivalence":
        rep = equivalence(workspace=args.workspace or None,
                          timeout_s=args.timeout or 1800)
        if args.json:
            print(json.dumps(rep, indent=1, default=str))
            return 0 if rep.get("ok") and not rep.get("regressed") else 1
        if not rep.get("ok"):
            print(f"NOT MEASURED: {rep.get('error', 'unknown')}", file=sys.stderr)
            return 2
        print(rep["stack"])
        if rep.get("note"):
            print(f"  note: {rep['note']}")
        print(f"  {rep['summary']}")
        print("  " + ("REGRESSION: " + rep["why"] if rep["regressed"]
                      else "equivalent to stock within the provisional thresholds"))
        print(f"  spent ${rep['cost_usd']:.3f}")
        return 1 if rep["regressed"] else 0

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
