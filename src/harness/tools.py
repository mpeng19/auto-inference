"""Tools an agent can call from its shell, and that a person can call too.

An agent's feedback loop is otherwise one bit every 25-60 minutes: it writes a
diff and eventually learns a price. Everything here exists to put *some*
signal in front of that, cheaply.

    harness tool recall "I am about to raise chunked_prefill_size"
    harness tool preflight --workspace agents/a01
    harness tool roofline --context 20583 --batch 12
    harness tool gpu-run bench.py --workspace agents/a01
    harness tool equivalence --workspace agents/a01
    harness tool ablate --env SGLANG_DISABLE_X=1 --tier screen --workspace agents/a01

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

`gpu_run`, `ncu` and `equivalence` are the GPU workbench, the part that makes
kernel work possible at all. A Triton kernel needs to be asked whether it
compiles, how fast it is, what its counters say, and whether it still computes
the same numbers -- none of which has anything to do with a price, and a
sweep is 17-35 minutes and a price. Each rents an H100 for minutes and costs
real money: `cost_usd` comes back with the answer and lands in the agent's
ledger (`harness.agent.ledger`), so the fleet's spend counts it.

`ablate` is the most expensive and the one that makes a win *publishable*: it
prices the stack twice, as is and with the mechanism switched off through its
env kill switch, and says how much of the measured delta the mechanism
accounts for. A faster stack without that number is a speed-up nobody can
explain (`docs/methodology.md` §5f); `harness results` shows the difference
as the `pub` column.
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
        # A compounding fleet's workspace builds on a base stack (earlier
        # wins, `Workspace.base`); with no edits that base *is* the stack,
        # not stock -- pricing stock here would ablate the wrong thing.
        return (ws.base or InferenceStack.stock(), ws.root,
                f"workspace has no changes: ran {ws.base_name}")
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
    _ledger(root, "gpu-run", rec)
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
    _ledger(root, "ncu", rec)
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
    _ledger(root, "equivalence", rec)
    if note:
        rec["note"] = note
    # Kept on disk under the agent, keyed by stack digest: `results.py` reads
    # the latest one as the experiment's decode-agreement evidence, and the
    # paper cites it. Before this the record lived only in the agent's
    # transcript.
    with contextlib.suppress(OSError, TypeError, ValueError):
        import time

        d = root / "equivalence"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{rec.get('stack_digest') or 'stock'}-{int(time.time())}.json"
        path.write_text(json.dumps(rec, indent=1, default=str))
        rec["path"] = str(path)
    return rec


# The fleet's two tiers, as `harness.daemon.FleetConfig` defaults them:
# levels x seconds per level. Repeated rather than imported because this
# runs from an agent's shell and the daemon module drags in the whole
# orchestration layer; `test_ablate_tiers_match_the_daemon` keeps them equal.
TIERS: dict[str, tuple[tuple[int, ...], float]] = {
    "screen": ((8, 12), 60.0),
    "full": ((4, 8, 12, 16, 24), 120.0),
}
# Run-to-run noise of a sweep (loop.IterativeAgent.NOISE_PCT): the disabled
# stack has to land inside this band of baseline for the mechanism to be the
# explanation, not a bystander.
ABLATION_NOISE_PCT = 3.0


def ablate(workspace: str | pathlib.Path | None, env: dict[str, str],
           tier: str = "screen", baseline: float | None = None,
           source=None) -> dict:
    """Price the stack with and without its mechanism. **Two sweeps of real
    money** -- a screen is ~$0.80 and ~10 minutes, a full tier ~$4.50 and
    ~35 minutes, paid twice -- so run it once, on the diff that won.

    `env` is the kill switch the diff must have: the variables that, set on
    the server, make the new path inert (`SGLANG_DISABLE_ADAPTIVE_CHUNK=1`).
    The workspace's stack is priced as is and with `env` merged into its
    `serving.json` env, at the fleet's `tier` (`TIERS`), both sweeps in
    parallel on two GPUs. The answer is the fraction of the delta against
    baseline that disappears when the mechanism is off:

        explained = (disabled - as_is) / (baseline - as_is)

    and whether the disabled stack is *within noise of baseline*
    (`ABLATION_NOISE_PCT`) -- which is what `results.py` reads as "the
    mechanism explains the delta" and what makes a replicated win
    publishable. A disabled stack still well under baseline means something
    else in the diff is doing the work, and the paper cannot claim the
    mechanism.

    `baseline` is stock's bill at this tier; taken from the fleet's
    `fleet.json` beside the agent directory when not given. Without one the
    two prices and their delta are still reported, but not the explained
    fraction.

    Writes `<agent>/ablations/<n>/ablation.json` (both prices, N*, the delta,
    the env, the stack digests, the cost) with each sweep's own artifacts in
    `as-is/` and `disabled/` beside it, and returns the same record with a
    one-paragraph `verdict`. Returns rather than raises, like the other
    tools: a bad workspace or an unknown tier must not cost a GPU.
    """
    import asyncio
    import time
    from dataclasses import replace

    env = {str(k): str(v) for k, v in (env or {}).items()}
    if not env:
        return {"ok": False, "error": "ablate needs the mechanism's kill switch: "
                                      "--env KEY=VALUE (the variable that makes the "
                                      "new path inert on the server)"}
    if tier not in TIERS:
        return {"ok": False, "error": f"unknown tier {tier!r}; one of {', '.join(TIERS)}"}
    try:
        stack, root, note = _workspace_stack(workspace, source)
    except ValueError as e:
        return {"ok": False, "error": f"workspace is not runnable: {e}"}
    levels, seconds = TIERS[tier]
    if baseline is None:
        baseline = _fleet_baseline(root, tier)
    disabled = replace(stack, env={**stack.env, **env},
                       label=f"{stack.label} [ablation: "
                             + ", ".join(f"{k}={v}" for k, v in sorted(env.items())) + "]")

    from simulator import Simulator

    out = _next_dir(root / "ablations")
    dirs = {"as_is": out / "as-is", "disabled": out / "disabled"}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    sims = {"as_is": Simulator(root_dir=dirs["as_is"], stack=stack,
                               levels=levels, seconds_per_level=seconds),
            "disabled": Simulator(root_dir=dirs["disabled"], stack=disabled,
                                  levels=levels, seconds_per_level=seconds)}

    async def _both():
        # Submitted together, collected together: the point of two sweeps is
        # one answer, and serially they would be an hour apart at full tier.
        ids = await asyncio.gather(*(s.submit_async() for s in sims.values()))
        return await asyncio.gather(*(s.collect(i) for s, i in zip(sims.values(), ids, strict=True)))

    t0 = time.time()
    results = dict(zip(sims, asyncio.run(_both()), strict=True))
    from harness.agent.evaluator import sweep_cost

    arms = {}
    cost = 0.0
    for name, res in results.items():
        c = sweep_cost(res.record)
        cost += c
        arms[name] = {"ok": bool(res.ok), "reason": res.reason or "",
                      "bill_per_1k": res.bill_per_1k, "n_star": res.n_star,
                      "stack_digest": sims[name].stack.digest,
                      "cost_usd": round(c, 4), "dir": str(dirs[name])}
    rec = {"ok": all(a["ok"] for a in arms.values()),
           "tier": tier, "levels": list(levels), "seconds_per_level": seconds,
           "env": env, "stack_digest": stack.digest,
           "disabled_digest": disabled.digest,
           "baseline_bill_per_1k": baseline,
           "as_is": arms["as_is"], "disabled": arms["disabled"],
           "cost_usd": round(cost, 4), "elapsed_s": round(time.time() - t0, 1),
           "gpu": (results["as_is"].record.get("serving") or {}).get("gpu", ""),
           "ts": time.time(), "dir": str(out)}
    rec.update(_ablation_arithmetic(rec))
    rec["verdict"] = _ablation_verdict(rec)
    if note:
        rec["note"] = note
    rec["path"] = str(out / "ablation.json")
    (out / "ablation.json").write_text(json.dumps(rec, indent=1, default=str))
    _ledger(root, "ablate", rec)
    return rec


def _ablation_arithmetic(rec: dict) -> dict:
    """The deltas an ablation record implies. Pure, so `results.py` can
    recompute them for a record written with a different baseline."""
    on, off = rec["as_is"].get("bill_per_1k"), rec["disabled"].get("bill_per_1k")
    base = rec.get("baseline_bill_per_1k")
    out: dict = {"delta_pct": None, "total_pct": None, "explained_pct": None,
                 "disabled_vs_baseline_pct": None, "explains": None}
    if not (isinstance(on, (int, float)) and isinstance(off, (int, float)) and off):
        return out
    out["delta_pct"] = round((on - off) / off * 100, 2)      # the mechanism's own effect
    if isinstance(base, (int, float)) and base:
        out["total_pct"] = round((on - base) / base * 100, 2)
        out["disabled_vs_baseline_pct"] = round((off - base) / base * 100, 2)
        if base != on:
            out["explained_pct"] = round((off - on) / (base - on) * 100, 1)
        out["explains"] = abs(out["disabled_vs_baseline_pct"]) <= ABLATION_NOISE_PCT
    return out


def _ablation_verdict(rec: dict) -> str:
    env = ", ".join(f"{k}={v}" for k, v in sorted(rec["env"].items()))
    head = (f"Ablation at {rec['tier']} tier ({'/'.join(map(str, rec['levels']))} users x "
            f"{rec['seconds_per_level']:.0f}s), mechanism disabled with {env}. ")
    for name, label in (("as_is", "as-is"), ("disabled", "disabled")):
        if not rec[name]["ok"]:
            return (head + f"The {label} sweep did not price: {rec[name]['reason'] or 'no reason'}. "
                    f"Nothing is explained. Cost ${rec['cost_usd']:.2f} of GPU time.")
    on, off = rec["as_is"], rec["disabled"]
    text = (head + f"As is ${on['bill_per_1k']:.2f}/1k (N*={on['n_star']}); disabled "
            f"${off['bill_per_1k']:.2f}/1k (N*={off['n_star']}): switching the mechanism off "
            f"moves the price {-rec['delta_pct']:+.1f}%.")
    if rec.get("total_pct") is None:
        text += (" No baseline was given (fleet.json has none), so the share of the delta "
                 "the mechanism explains is not computed; pass one to get it.")
    else:
        share = ("undefined (the stack prices at baseline)" if rec["explained_pct"] is None
                 else f"{rec['explained_pct']:.0f}%")
        text += (f" Against the baseline ${rec['baseline_bill_per_1k']:.2f}/1k the stack is "
                 f"{rec['total_pct']:+.1f}%; the mechanism accounts for {share} of that "
                 f"{rec['total_pct']:+.1f}% delta. The disabled stack sits "
                 f"{rec['disabled_vs_baseline_pct']:+.1f}% from baseline, "
                 + (f"within the {ABLATION_NOISE_PCT:.0f}% noise floor: the mechanism explains "
                    "the delta." if rec["explains"] else
                    f"outside the {ABLATION_NOISE_PCT:.0f}% noise floor: something other than "
                    "this mechanism is moving the price, and the paper cannot claim it."))
    return text + f" Cost ${rec['cost_usd']:.2f} of GPU time (two sweeps)."


def _fleet_baseline(agent_root: pathlib.Path, tier: str) -> float | None:
    """Stock's bill at `tier` from the fleet's `fleet.json` (the agent
    directory's parent), the way `loop._delta` reads it: the `screen` map
    for a screen, the top-level figure otherwise."""
    for cand in (agent_root.parent / "fleet.json", agent_root / "fleet.json"):
        try:
            base = (json.loads(cand.read_text()) or {}).get("baseline") or {}
        except (OSError, ValueError, AttributeError):
            continue
        if tier == "screen" and isinstance(base.get("screen"), dict):
            base = base["screen"]
        b = base.get("bill_per_1k")
        if isinstance(b, (int, float)):
            return float(b)
    return None


def _next_dir(base: pathlib.Path) -> pathlib.Path:
    n = 0
    while (base / str(n)).exists():
        n += 1
    return base / str(n)


def _parse_env(raw) -> dict[str, str]:
    """`--env KEY=VAL` in whatever shape the CLI hands over: a dict, a list
    of `KEY=VAL`, or one comma-separated string."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    out = {}
    for it in items:
        k, _, v = str(it).partition("=")
        if k.strip():
            out[k.strip()] = v.strip()
    return out


def _ledger(root, tool: str, rec: dict) -> None:
    """Record what a tool call cost where the fleet will find it. Never
    fails the call: the result matters more than the receipt."""
    from harness.agent import ledger

    with contextlib.suppress(Exception):
        ledger.append(root, tool, rec.get("cost_usd") or 0.0,
                      elapsed_s=rec.get("elapsed_s") or 0.0,
                      gpu=str(rec.get("gpu") or ""), where=str(rec.get("dir") or ""))


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
    import signal

    # A tool killed from outside -- the agent's shell timeout, mostly --
    # must take its GPU call down with it. SIGTERM/SIGHUP become
    # KeyboardInterrupt, which `Simulator.workbench` turns into a cancel of
    # the spawned call. Before this a killed gpu-run ran to completion and
    # billed with nobody waiting for the answer.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        with contextlib.suppress(Exception):
            signal.signal(sig, signal.default_int_handler)
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
        if rep.get("lossless") is not None:
            print("  " + ("lossless: greedy decode matches stock's" if rep["lossless"] else
                          "lossy: not a rejection -- the accuracy suites decide; the "
                          "write-up must say so"))
        print(f"  spent ${rep['cost_usd']:.3f}")
        return 1 if rep["regressed"] else 0

    if action == "ablate":
        rep = ablate(args.workspace or None, env=_parse_env(getattr(args, "env", None)),
                     tier=getattr(args, "tier", "") or "screen")
        if args.json:
            print(json.dumps(rep, indent=1, default=str))
            return 0 if rep.get("ok") else 1
        if "error" in rep:
            print(rep["error"], file=sys.stderr)
            return 2
        print(rep["verdict"])
        if rep.get("note"):
            print(f"  note: {rep['note']}")
        print(f"  written: {rep['path']}")
        print(f"  spent ${rep['cost_usd']:.2f} -- two sweeps of real GPU time; the ledger has it")
        return 0 if rep.get("ok") else 1

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
