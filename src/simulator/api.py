"""The public surface: build a Simulator, call `eval()`, get a price curve.

    sim = Simulator(root_dir="runs/baseline")
    result = await sim.eval()
    print(result.summary())

Everything that varies is a field on the object, so a caller configures once
and never passes arguments again. The only required one is `root_dir`, which
must already exist: every artifact the run produces -- the record, the plots,
the curve as JSON -- is written there, so a result is a directory you can hand
to someone rather than a number you have to explain.

Three shapes, same machinery:

    await sim.eval()                 # submit, wait, analyse, write artifacts
    call_id = sim.submit()           # fire and forget; a sweep is 25-60 min
    await sim.collect(call_id)       # pick it up later, maybe another process
    sim.analyse(record)              # no GPU at all: re-score a stored sweep

`analyse` being separable is not tidiness. A sweep stores every percentile and
the raw server counters precisely so that changing the SLO, the cost basis or
the market denominator costs nothing -- which order statistic the frontier is
judged at is a choice we have changed three times, and each change would
otherwise have meant another 25 GPU-minutes.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import time
from dataclasses import dataclass, field, replace

from . import costs
from .config import ServingConfig
from .price import direct as price_direct_mod
from .price.direct import DirectPrice, price_direct
from .price.market import Economics, Market
from .slo import MARKET_SLO, SLO
from .stack import InferenceStack
from .workload.tracelab import MARKET_IN_PER_REQ, MARKET_OUT_PER_REQ

# The deployed Modal app to look functions up in. Read from the environment
# because `runner.modal_runner` reads the *same* variable when it names the app
# at deploy time: a fresh account that deploys under another name would
# otherwise deploy fine and then have every client lookup here ask for
# "auto-inference" and fail. One variable, both sides.
APP_NAME = os.environ.get("SIMULATOR_APP_NAME", "auto-inference")

# How much longer than the script's own timeout the Modal call is allowed to
# take: pulling the image, starting the container, applying the stack. The
# engine load is *not* in here -- it happens inside the script and so comes out
# of `timeout_s`, which is why anything loading an engine needs a generous one
# (3-5 minutes before the first line of the caller's own work runs).
WORKBENCH_OVERHEAD_S = 900


# ── results ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Point:
    """One concurrency level, priced."""
    n_users: int
    meets_slo: bool
    binding: str | None
    n_requests: int
    goodput_rps: float
    batch: float
    hit_rate: float
    # Aggregate delivered output tokens per second per GPU. This is the
    # performance axis: cost per output token is its reciprocal times the
    # rate, so a plot against it is a plot against money.
    output_tps_per_gpu: float
    gpu_s_per_request: float
    effective_in_per_m: float
    out_per_m: float
    bill_per_1k: float
    share_per_node: float
    capacity_per_day: float
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass(frozen=True)
class EvalResult:
    """What an evaluation is worth. `ok=False` means no price, and why."""
    ok: bool
    reason: str = ""
    n_star: int | None = None
    curve: tuple[Point, ...] = ()
    market: Market | None = None
    record: dict = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    stack_digest: str = ""
    # Measured before load. Carried separately from `ok` on purpose: a price
    # was computed *and* accuracy fell are both facts, and collapsing them
    # into one boolean loses the number a caller needs to judge the trade.
    quality: tuple[dict, ...] = ()
    # Where the binding SLO metric crosses its limit, by linear interpolation
    # between the last passing and first failing level, with the bill
    # interpolated to match. Reported beside N*, not instead of it: N* is
    # quantised to the grid, and a level sitting on the SLO line passes or
    # fails on noise -- stock priced $12.2/1k when N=12 held 20 ms mean TPOT
    # and $15.0/1k when it did not. This number moves 3%, not 18%.
    interpolated: dict | None = None

    @property
    def quality_regressed(self) -> bool:
        return any(q.get("regressed") for q in self.quality)

    @property
    def quality_note(self) -> str:
        return "; ".join(q["why"] for q in self.quality if q.get("why"))

    @property
    def best(self) -> Point | None:
        """The priced operating point: the largest level that held the SLO."""
        ok = [p for p in self.curve if p.meets_slo]
        return max(ok, key=lambda p: p.n_users) if ok else None

    # Convenience passthroughs -- a caller should not have to know about Point.
    @property
    def effective_in_per_m(self) -> float | None:
        return self.best.effective_in_per_m if self.best else None

    @property
    def out_per_m(self) -> float | None:
        return self.best.out_per_m if self.best else None

    @property
    def bill_per_1k(self) -> float | None:
        return self.best.bill_per_1k if self.best else None

    @property
    def share_per_node(self) -> float | None:
        return self.best.share_per_node if self.best else None

    def rank(self) -> dict | None:
        """Where the priced point sits on the board, both ways."""
        if not (self.best and self.market):
            return None
        board = self.market.leaderboard(self.best.effective_in_per_m,
                                        self.best.out_per_m)
        us = next(r for r in board if r["us"])
        return {"rank_bill": us["rank_bill"], "rank_eff_in": us["rank_eff_in"],
                "of": len(board), "board": board}

    def as_dict(self) -> dict:
        r = self.rank()
        return {"ok": self.ok, "reason": self.reason, "n_star": self.n_star,
                "interpolated": self.interpolated,
                "quality": list(self.quality),
                "quality_regressed": self.quality_regressed,
                "stack_digest": self.stack_digest,
                "curve": [p.as_dict() for p in self.curve],
                "priced_at": self.best.as_dict() if self.best else None,
                "rank": {k: r[k] for k in ("rank_bill", "rank_eff_in", "of")} if r else None,
                "artifacts": self.artifacts}

    def summary(self) -> str:
        if not self.ok:
            return f"NO PRICE: {self.reason}"
        b, r = self.best, self.rank()
        lines = [
            f"N* = {b.n_users} users   batch {b.batch:.1f}   hit {b.hit_rate:.3f}"
            f"   {b.gpu_s_per_request:.2f} GPU-s per market request",
            f"  effective input  ${b.effective_in_per_m:.4f}/M",
            f"  output           ${b.out_per_m:.4f}/M",
            f"  whole bill       ${b.bill_per_1k:.2f} per 1k requests"
            + (f"   rank {r['rank_bill']}/{r['of']}" if r else ""),
            f"  one node serves  {b.share_per_node:.2%} of the market"
            f"  ({b.capacity_per_day:,.0f} req/day)",
        ]
        if r:
            lines.append(f"  on effective input alone: rank {r['rank_eff_in']}/{r['of']}"
                         "  -- the metric OpenRouter sorts on, not what a buyer pays")
        for q in self.quality:
            if "error" in q:
                lines.append(f"  quality {q['suite']}: NOT MEASURED ({q['error']})")
                continue
            d = f"  ({q['delta_pct']:+.1f} pts)" if q.get("delta_pct") is not None else ""
            lines.append(f"  quality {q['suite']}: {q['accuracy']:.1%}{d}"
                         + ("   REGRESSION" if q.get("regressed") else ""))
        if self.quality_regressed:
            lines.append(f"  !! {self.quality_note}")
        return "\n".join(lines)


# ── the simulator ────────────────────────────────────────────────────────


def interpolate_frontier(pts: list) -> dict | None:
    """The SLO crossing between the last level that held and the first that
    did not, and the bill there. None when the sweep does not bracket it.

    Linear in N for both the metric and the bill. Rough, and honest about
    it: it exists to show how far inside the grid step the frontier sits,
    which is what decides whether a one-level move is a result or noise.
    """
    passing = [p for p in pts if p.meets_slo]
    failing = [p for p in pts if not p.meets_slo and p.n_users > max(
        (q.n_users for q in passing), default=-1)]
    if not passing or not failing:
        return None
    lo = max(passing, key=lambda p: p.n_users)
    hi = min(failing, key=lambda p: p.n_users)
    lo_vals = {c["label"]: c for c in lo.checks if c.get("value") is not None}
    crossings = []
    for c in hi.checks:
        if c.get("ok") or c.get("value") is None or c["label"] not in lo_vals:
            continue
        v0, v1, lim = lo_vals[c["label"]]["value"], c["value"], c["limit"]
        if v1 <= v0:
            continue
        frac = min(1.0, max(0.0, (lim - v0) / (v1 - v0)))
        crossings.append((frac, c["label"]))
    if not crossings:
        return None
    frac, label = min(crossings)
    n = lo.n_users + frac * (hi.n_users - lo.n_users)
    bill = lo.bill_per_1k + frac * (hi.bill_per_1k - lo.bill_per_1k)
    gsr = lo.gpu_s_per_request + frac * (hi.gpu_s_per_request - lo.gpu_s_per_request)
    return {"n_star": round(n, 2), "bill_per_1k": round(bill, 4),
            "gpu_s_per_request": round(gsr, 3), "binding": label,
            "between": [lo.n_users, hi.n_users]}


@dataclass
class Simulator:
    """An inference stack, an environment, and the price it can serve at."""

    root_dir: str | pathlib.Path
    stack: InferenceStack = field(default_factory=InferenceStack.stock)

    # environment
    model: str = "Qwen/Qwen3.8-27B-FP8"
    gpu: str = "H100"
    n_gpu: int = 1
    mem_fraction_static: float = 0.85
    max_running_requests: int = 256          # deliberately non-binding
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0

    # the measurement
    # Multiplicative, because the quantities being measured are: batch, cost
    # per token and throughput all move with the ratio of one level to the
    # next, not the difference. A linear sweep spends most of its GPU time
    # re-measuring the same regime.
    # Bracket the 1xH100 frontier (N* ~ 12 on market traffic) with a step fine
    # enough to register a shift of it. Below 4 users a 120 s level completes
    # ~3 requests and measures nothing.
    levels: tuple[int, ...] = (4, 8, 12, 16, 24)
    seconds_per_level: float = 120.0
    repeats: int = 1
    n_sessions: int = 300
    canaries: bool = True
    # Capture a GPU profile at this concurrency level (0 = none). Profiling
    # perturbs what it measures, so it runs at one level and the price is still
    # taken from N*. `docs/methodology.md` §8.3 is the question it answers:
    # the device timer says the KV read is at ~28% of bandwidth but not which
    # kernel spends it.
    profile_level: int = 0
    profile_steps: int = 20
    # A speed win that costs accuracy is not a win, and the price model cannot
    # see the difference. Run before load on an idle server, so this measures
    # the model rather than the scheduler.
    quality_suites: tuple[str, ...] = ("gsm8k", "longbench", "mmlu")
    quality_n: int = 100
    quality_baseline: dict = field(default_factory=dict)
    quality_tolerance_pp: float = 10.0   # see measure/quality.regressed
    slo: SLO = field(default_factory=lambda: SLO(bounds=MARKET_SLO))

    # ── the cost basis: assumptions, never measurements ──
    # Named separately from the measurement because these are the inputs a
    # result must always be quoted with. `gpu_provider=None` takes the agreed
    # default for the GPU (H100 -> $3.00/hr); name a provider to price against
    # a real quote, or set `rate_per_gpu_hour` to override both.
    gpu_provider: str | None = None
    rate_per_gpu_hour: float | None = None
    allow_retail_rate: bool = False
    # 0.50 is the agreed basis, not a measurement. It sits just under the 53%
    # that this model's own daily volume implies for a single-model deployment
    # sized for peak (`market.utilisation_ceiling`) -- pass that instead to
    # price against measured burstiness rather than the agreed round number.
    # Utilisation is the single largest lever on the answer and cannot be
    # measured from inside the harness, so it is always reported alongside it.
    utilisation: float = 0.50
    market: Market = field(default_factory=Market.load)

    note: str = ""
    allow_stale_stack: bool = False

    def __post_init__(self):
        if self.rate_per_gpu_hour is None:
            self.rate_per_gpu_hour = costs.rate(
                self.gpu, self.gpu_provider, allow_retail=self.allow_retail_rate)
        self.root = pathlib.Path(self.root_dir)
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"root_dir {self.root} does not exist. Create it deliberately: "
                "every artifact of this evaluation is written there, and a run "
                "that invents its own output directory is one nobody finds again.")

    # ── identity ─────────────────────────────────────────────────────────
    @property
    def serving(self) -> ServingConfig:
        return ServingConfig(
            model=self.model, gpu=self.gpu, n_gpu=self.n_gpu,
            tp_size=self.n_gpu, mem_fraction_static=self.mem_fraction_static,
            max_running_requests=self.max_running_requests,
            schedule_policy=self.schedule_policy,
            schedule_conservativeness=self.schedule_conservativeness)

    @property
    def util(self) -> float:
        return self.utilisation

    @property
    def cost_basis(self) -> str:
        """One line naming the rate and where it came from."""
        return costs.describe(self.gpu, self.gpu_provider, self.rate_per_gpu_hour)

    def assumptions(self) -> dict:
        """Everything that is an input rather than a measurement.

        Reported with every result, because none of it can be validated from
        inside the harness and all of it scales the answer.
        """
        return {"rate_per_gpu_hour": self.rate_per_gpu_hour,
                "gpu_provider": self.gpu_provider,
                "cost_basis": self.cost_basis,
                "utilisation": self.util,
                "margin": 0.0,
                "market_as_of": self.market.as_of,
                "market_requests_per_day": self.market.requests_per_day,
                "in_per_request": self.market.in_per_request,
                "out_per_request": self.market.out_per_request}

    def digest(self) -> str:
        """Identifies (stack, environment, measurement). Cache key for a loop."""
        import hashlib
        body = json.dumps({"stack": self.stack.digest,
                           "serving": self.serving.digest(),
                           "levels": list(self.levels),
                           "seconds": self.seconds_per_level,
                           "repeats": self.repeats,
                           "slo": self.slo.as_dict()}, sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()[:12]

    # ── running ──────────────────────────────────────────────────────────
    def _fn(self):
        import modal
        n = max(1, self.n_gpu)
        return modal.Function.from_name(APP_NAME, "sweep").with_options(
            gpu=f"{self.gpu}:{n}" if n > 1 else self.gpu,
            cpu=float(max(16, 4 * n)),
            timeout=60 * 60 * (3 if n > 1 else 2))

    def _args(self) -> tuple:
        from dataclasses import asdict
        return (asdict(self.serving), self.slo.as_dict(), self.stack.as_dict(),
                list(self.levels), self.seconds_per_level, self.repeats,
                MARKET_IN_PER_REQ, MARKET_OUT_PER_REQ, self.n_sessions,
                self.canaries, self.note, self.allow_stale_stack,
                self.profile_level, self.profile_steps,
                tuple(self.quality_suites), self.quality_n,
                dict(self.quality_baseline), self.quality_tolerance_pp)

    def submit(self) -> str:
        """Start the sweep and return immediately. Runs outlive this process.

        `.spawn()` rather than `.remote()`: a sweep is 25-60 minutes and a
        `local_entrypoint`'s in-flight call dies with its client, which
        cancelled a sweep three levels in once.
        """
        return self._record(self._fn().spawn(*self._args()))

    async def submit_async(self) -> str:
        """`submit()` for callers already on an event loop (`eval`, the
        harness). Modal's blocking interface warns when used under one."""
        return self._record(await self._fn().spawn.aio(*self._args()))

    def _record(self, call) -> str:
        (self.root / "call_id").write_text(call.object_id)
        return call.object_id

    async def collect(self, call_id: str, poll_s: float = 20.0,
                      timeout_s: float = 4 * 3600) -> EvalResult:
        """Wait for a submitted sweep, then analyse and write artifacts."""
        import modal
        call = modal.FunctionCall.from_id(call_id)
        deadline = time.time() + timeout_s
        while True:
            try:
                rec = await call.get.aio(timeout=0)
                break
            except (TimeoutError, modal.exception.OutputExpiredError) as e:
                if isinstance(e, modal.exception.OutputExpiredError):
                    raise
                if time.time() > deadline:
                    raise TimeoutError(
                        f"sweep {call_id} still running after "
                        f"{timeout_s/3600:.1f}h") from e
                await asyncio.sleep(poll_s)
        return self.finish(rec)

    async def eval(self) -> EvalResult:
        """Submit, wait, analyse, write artifacts. The one call most callers want."""
        return await self.collect(await self.submit_async())

    # ── the workbench: a GPU without a sweep ─────────────────────────────
    def _workbench_fn(self, timeout_s: int):
        import modal
        # The deployed shape is already H100 + WORKBENCH_VCPU; only the ceiling
        # moves, and it has to clear the script's own timeout with room for
        # everything around it (`WORKBENCH_OVERHEAD_S`), or Modal would kill
        # the container while the script still had time on the clock.
        return modal.Function.from_name(APP_NAME, "workbench").with_options(
            timeout=int(timeout_s) + WORKBENCH_OVERHEAD_S)

    async def workbench(self, script_text: str, files: dict[str, str] | None = None,
                        timeout_s: int = 600) -> dict:
        """Run one script on an H100 against this stack, and keep what it said.

        The inner loop for kernel work. A sweep prices a stack and takes 17-35
        minutes; this asks whether the kernel compiles, whether it is faster,
        whether it still computes the same thing -- none of which need a load
        generator, a market workload or a price.

        `files` are helper text files (path relative to the script's directory
        -> text) written beside the script, which is also the working
        directory, so `import my_helper` resolves.

        Every run gets its own `workbench-<n>/` under `root_dir`, written
        before the call is spawned so a script that hangs still leaves the
        script text and the call id behind for someone to look at.
        """
        d = self._workbench_dir()
        (d / "script.py").write_text(script_text)
        call = await self._workbench_fn(timeout_s).spawn.aio(
            self.stack.as_dict(), script_text, int(timeout_s), dict(files or {}))
        (d / "call_id").write_text(call.object_id)

        deadline = time.time() + timeout_s + WORKBENCH_OVERHEAD_S
        while True:
            try:
                rec = await call.get.aio(timeout=0)
                break
            except TimeoutError as e:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"workbench {call.object_id} still running after "
                        f"{timeout_s + WORKBENCH_OVERHEAD_S}s") from e
                await asyncio.sleep(5.0)

        rec = dict(rec)
        (d / "stdout.txt").write_text(rec.get("stdout") or "")
        (d / "stderr.txt").write_text(rec.get("stderr") or "")
        (d / "result.json").write_text(json.dumps(rec, indent=2, default=str))
        rec["dir"] = str(d)
        rec["artifacts"] = {n: str(d / f) for n, f in
                            (("script", "script.py"), ("stdout", "stdout.txt"),
                             ("stderr", "stderr.txt"), ("result", "result.json"))}
        return rec

    def _workbench_dir(self) -> pathlib.Path:
        """The next free `workbench-<n>/`. Numbered, not stamped: an agent
        iterating on one kernel wants them in the order it ran them."""
        n = 0
        while (self.root / f"workbench-{n}").exists():
            n += 1
        d = self.root / f"workbench-{n}"
        d.mkdir()
        return d

    async def equivalence(self, **kw) -> dict:
        """Is this stack computing the same thing stock computes?

        The sharper half of the quality gate, for kernel work that GSM8K cannot
        resolve. See `measure.equivalence`; runs on the workbench, so it costs
        minutes rather than a sweep.
        """
        from .measure import equivalence as eq
        return await eq.measure(self, **kw)

    # ── analysis, which needs no GPU ─────────────────────────────────────
    def analyse(self, record: dict) -> EvalResult:
        """Turn a sweep record into a priced curve. Pure; re-runnable offline."""
        quality = tuple(record.get("quality") or ())
        if record.get("status") != "ok":
            return EvalResult(ok=False, record=record, quality=quality,
                              stack_digest=record.get("stack_digest", ""),
                              reason=record.get("failure", "sweep did not complete"))
        n_gpu = (record.get("serving") or {}).get("n_gpu", 1)
        m, u = self.market, self.util
        pts, skipped = [], []
        for lv in record.get("levels", []):
            ok, why = price_direct_mod.usable(lv, n_gpu=n_gpu)
            if not ok:
                skipped.append(f"N={lv['n_users']}: {why}")
                continue
            c = lv["server_counters"]
            ext = c["sglang:forward_execution_seconds_total[extend]"]
            dec = c["sglang:forward_execution_seconds_total[decode]"]
            p: DirectPrice = price_direct(
                gpu_seconds_input=ext, gpu_seconds_output=dec,
                input_tokens=lv["prompt_tokens"], output_tokens=lv["output_tokens"],
                cached_tokens=lv["cached_tokens"],
                usd_per_gpu_hour=self.rate_per_gpu_hour,
                utilization=u, margin=0.0)
            gsr = price_direct_mod.gpu_seconds_per_request(
                ext, dec, lv["prompt_tokens"], lv["output_tokens"],
                m.in_per_request, m.out_per_request)
            e = Economics(gpu_s_per_request=gsr, n_gpu=n_gpu,
                          rate_per_gpu_hour=self.rate_per_gpu_hour, utilisation=u)
            v = self.slo.judge(lv)
            pts.append(Point(
                n_users=lv["n_users"], meets_slo=v.ok, binding=v.binding,
                n_requests=(lv.get("ttft_ms") or {}).get("n", 0),
                goodput_rps=lv["goodput_rps"],
                batch=(lv.get("batch", {}).get("running") or {}).get("mean", 0.0),
                hit_rate=lv.get("cache_hit_rate") or 0.0,
                output_tps_per_gpu=(lv["output_tokens"]
                                    / max(lv.get("wall_s") or 0.0, 1e-9)
                                    / max(n_gpu, 1)),
                gpu_s_per_request=gsr,
                effective_in_per_m=p.effective_in_per_m, out_per_m=p.out_per_m,
                bill_per_1k=m.bill_per_1k(p.effective_in_per_m, p.out_per_m),
                share_per_node=e.share_per_node(m),
                capacity_per_day=e.capacity_per_node_per_day(),
                checks=v.checks, warnings=v.warnings))
        if not pts:
            return EvalResult(ok=False, record=record, quality=quality,
                              stack_digest=record.get("stack_digest", ""),
                              reason="; ".join(skipped) or "no priceable levels")
        passing = [p for p in pts if p.meets_slo]
        if not passing:
            return EvalResult(
                ok=False, curve=tuple(pts), market=m, record=record,
                quality=quality, stack_digest=record.get("stack_digest", ""),
                reason=("no level met the SLO -- the sweep starts above the "
                        "frontier; lower the levels"))
        star = max(passing, key=lambda p: p.n_users)
        if star.n_users == max(p.n_users for p in pts):
            skipped.append("every level passed: N* is the top of the sweep, so "
                           "the true frontier is higher and the price lower")
        return EvalResult(ok=True, reason="; ".join(skipped), n_star=star.n_users,
                          curve=tuple(pts), market=m, record=record,
                          quality=quality, interpolated=interpolate_frontier(pts),
                          stack_digest=record.get("stack_digest", ""))

    def finish(self, record: dict) -> EvalResult:
        """Analyse, write every artifact into `root_dir`, return the result."""
        res = self.analyse(record)
        arts = self.write_artifacts(res)
        return replace(res, artifacts=arts)

    def write_artifacts(self, res: EvalResult) -> dict[str, str]:
        from .artifacts import plots, report
        out: dict[str, str] = {}
        (self.root / "sweep.json").write_text(json.dumps(res.record, indent=2, default=str))
        out["sweep"] = str(self.root / "sweep.json")
        (self.root / "result.json").write_text(json.dumps(res.as_dict(), indent=2, default=str))
        out["result"] = str(self.root / "result.json")
        (self.root / "config.json").write_text(json.dumps(self.as_dict(), indent=2, default=str))
        out["config"] = str(self.root / "config.json")
        # The candidate itself, in full. An agent's workspace is reset when
        # it moves on, and the first replicated win of the project (build-2,
        # a01, -27%) survived only as a truncated diff in a trace because
        # nothing else kept the files. This is what `--stack <run dir>`
        # loads, and what a re-measurement or a paper starts from.
        (self.root / "stack.json").write_text(json.dumps(self.stack.as_dict(), indent=1))
        out["stack"] = str(self.root / "stack.json")
        txt = report.render(self, res)
        (self.root / "report.txt").write_text(txt)
        out["report"] = str(self.root / "report.txt")
        try:
            out.update(plots.render_all(self, res, self.root))
        except Exception as e:                       # never lose a run to a plot
            (self.root / "plot-error.txt").write_text(f"{type(e).__name__}: {e}")
            out["plot_error"] = str(self.root / "plot-error.txt")
        return out

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return {"digest": self.digest(), "stack": self.stack.describe(),
                "stack_digest": self.stack.digest, "serving": asdict(self.serving),
                "levels": list(self.levels), "seconds_per_level": self.seconds_per_level,
                "repeats": self.repeats, "n_sessions": self.n_sessions,
                "canaries": self.canaries, "slo": self.slo.as_dict(),
                "assumptions": self.assumptions(),
                "note": self.note}
