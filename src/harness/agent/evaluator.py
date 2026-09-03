"""The bridge to the simulator: one attempt in, one set of metrics out.

Kept behind the `Evaluator` protocol so the loop can be exercised without
renting a GPU. The fake and the real one differ only in what happens between
`stack` going in and `metrics` coming out, which is the property that makes the
whole harness testable.

Failures are classified rather than raised, because the loop treats them
differently: an infrastructure failure is retried unchanged, a rejected
hypothesis is not.

**Every return carries `cost_usd`**, including failures. That field is what
`AgentBudget.max_usd` and `FleetBudget.max_usd_total` are checked against, so
omitting it does not make budgets approximate -- it makes them *inert*, and a
ten-agent fleet runs until wall-clock with no spend control at all. A failed
sweep still rented the GPU, so it still costs.
"""
from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass, field


@dataclass
class SimulatorEvaluator:
    """Runs a real sweep. ~25-60 GPU-minutes and real money per call."""
    n_gpu: int = 1
    gpu: str = "H100"
    model: str = "Qwen/Qwen3.8-27B-FP8"
    levels: tuple[int, ...] = (4, 8, 12, 16, 24)
    seconds_per_level: float = 120.0
    gpu_provider: str | None = None
    utilisation: float = 0.50
    extra: dict = field(default_factory=dict)

    # What the sweep container reserves besides the GPU. Kept here rather
    # than imported from the runner so the evaluator never imports modal.
    vcpu: float = 16.0
    memory_gib: float = 0.0
    # Capture a GPU profile at this level (0 = none) and ingest it under
    # `profiles_root/<stack digest>.sqlite`, which the agent then queries
    # through the tracedb MCP tools. Full tier only: a screen is too short.
    profile_level: int = 0
    profile_steps: int = 20
    profiles_root: str = ""

    def _rate(self, gpu: str, n_gpu: int) -> float:
        from simulator import costs

        return costs.container_rate(gpu, n_gpu, vcpu=self.vcpu,
                                    memory_gib=self.memory_gib)

    def _spend(self, record: dict) -> float:
        """What this sweep actually cost us, in Modal dollars.

        Deliberately the *retail* Modal rate, not the serving cost basis. Those
        are different numbers for different questions: $3.00/GPU-hr is what a
        provider would pay to serve, $3.95 is what we are billed to experiment.
        Confusing them is why `costs.py` marks the retail rows
        `serving_basis=False`. And the GPU is not the whole container: the
        16 vCPUs the load generator needs are billed too (`costs.container_rate`).
        """
        if not record:
            return 0.0
        n_gpu = max(1, (record.get("serving") or {}).get("n_gpu", 1))
        gpu = (record.get("serving") or {}).get("gpu", self.gpu)
        seconds = float(record.get("model_load_s") or 0.0)
        seconds += sum(float(lv.get("wall_s") or 0.0)
                       for lv in record.get("levels") or ())
        for m in record.get("mixes") or ():
            seconds += float(m.get("wall_s") or 0.0)
        return round(seconds * self._rate(gpu, n_gpu) / 3600.0, 4)

    def _ingest_profile(self, record: dict, digest: str, fetch=None) -> str:
        """Bring a captured trace local and load it into a tracedb file.
        Never fails the evaluation: the price stands without the profile."""
        try:
            profs = record.get("profiles") or []
            if not profs or not self.profiles_root:
                return ""
            import base64

            if fetch is None:
                import modal

                from simulator.api import APP_NAME
                fetch = modal.Function.from_name(APP_NAME, "fetch_profile").remote
            got = fetch(profs[-1]["dir"])
            files = got.get("files") or []
            if not files:
                return ""
            root = pathlib.Path(self.profiles_root)
            raw = root / "raw" / (digest or "profile")
            raw.mkdir(parents=True, exist_ok=True)
            trace = None
            for f in files:
                p = raw / f["name"]
                p.write_bytes(base64.b64decode(f["b64"]))
                if p.suffix in (".json", ".gz") or "trace" in p.name:
                    trace = trace or p
            if trace is None:
                return ""
            from ..profile import ingest

            out = ingest(trace, root.parent, name=digest or trace.stem)
            return str(out.get("db") or (root / f"{digest}.sqlite"))
        except Exception as e:
            print(f"profile ingest skipped: {type(e).__name__}: {e}", flush=True)
            return ""

    def evaluate(self, stack, run_dir) -> tuple[bool, dict, str]:
        from simulator import Simulator

        sim = Simulator(root_dir=run_dir, stack=stack, model=self.model,
                        gpu=self.gpu, n_gpu=self.n_gpu, levels=self.levels,
                        seconds_per_level=self.seconds_per_level,
                        gpu_provider=self.gpu_provider,
                        utilisation=self.utilisation,
                        profile_level=self.profile_level, profile_steps=self.profile_steps,
                        **self.extra)
        import time

        t0 = time.time()
        try:
            res = asyncio.run(sim.eval())
        except Exception as e:
            # The sweep never produced a record: the GPU died, the image failed,
            # the server never came up. Worth retrying the same diff. The
            # container ran for about as long as we waited, so bill that
            # rather than zero -- zero let a night of cancelled and orphaned
            # sweeps vanish from the budget.
            est = round((time.time() - t0) * self._rate(self.gpu, self.n_gpu)
                        / 3600.0, 4)
            return False, {"error": f"{type(e).__name__}: {e}",
                           "cost_usd": est, "cost_estimated": True}, "infra"

        spent = self._spend(res.record)
        if not res.ok:
            # A record exists and says no. Retrying is a second bill for the
            # same answer -- unless nothing could be priced at all, which is
            # infrastructure wearing a result's clothes. Either way the GPU was
            # rented, so the cost is real.
            kind = "infra" if "DEVICE_TIMER" in res.reason else "slo"
            return False, {"reason": res.reason, "cost_usd": spent}, kind

        if res.quality_regressed:
            # A faster model that answers worse is not an improvement, and the
            # price model cannot tell the difference. Reject it as a rejected
            # hypothesis, not an infra failure: re-running would reproduce it.
            return False, {"reason": res.quality_note,
                           "quality": list(res.quality),
                           "bill_per_1k": res.bill_per_1k,
                           "cost_usd": spent}, "quality"

        b = res.best
        rank = res.rank() or {}
        profile_db = self._ingest_profile(res.record, getattr(stack, "digest", "")) \
            if self.profile_level else ""
        return True, {
            "profile_db": profile_db,
            "bill_per_1k": b.bill_per_1k,
            # Where this price sits on the OpenRouter board, both ways, and
            # the share one node could serve at it: what a watcher wants to
            # see next to an agent, not just a delta.
            "rank_bill": rank.get("rank_bill"),
            "rank_eff_in": rank.get("rank_eff_in"),
            "rank_of": rank.get("of"),
            "interpolated": res.interpolated,
            "effective_in_per_m": b.effective_in_per_m,
            "out_per_m": b.out_per_m,
            "n_star": b.n_users,
            "batch": b.batch,
            "hit_rate": b.hit_rate,
            "gpu_s_per_request": b.gpu_s_per_request,
            "share_per_node": b.share_per_node,
            "quality": list(res.quality),
            "cost_usd": spent,
            "artifacts": res.artifacts,
        }, ""
