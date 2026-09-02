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

    def _spend(self, record: dict) -> float:
        """What this sweep actually cost us, in Modal dollars.

        Deliberately the *retail* Modal rate, not the serving cost basis. Those
        are different numbers for different questions: $3.00/GPU-hr is what a
        provider would pay to serve, $3.95 is what we are billed to experiment.
        Confusing them is why `costs.py` marks the retail rows
        `serving_basis=False`.
        """
        from simulator import costs

        if not record:
            return 0.0
        n_gpu = max(1, (record.get("serving") or {}).get("n_gpu", 1))
        gpu = (record.get("serving") or {}).get("gpu", self.gpu)
        seconds = float(record.get("model_load_s") or 0.0)
        seconds += sum(float(lv.get("wall_s") or 0.0)
                       for lv in record.get("levels") or ())
        for m in record.get("mixes") or ():
            seconds += float(m.get("wall_s") or 0.0)
        try:
            rate = costs.rate(gpu, "modal", allow_retail=True)
        except KeyError:
            rate = 3.95
        return round(seconds * n_gpu * rate / 3600.0, 4)

    def evaluate(self, stack, run_dir) -> tuple[bool, dict, str]:
        from simulator import Simulator

        sim = Simulator(root_dir=run_dir, stack=stack, model=self.model,
                        gpu=self.gpu, n_gpu=self.n_gpu, levels=self.levels,
                        seconds_per_level=self.seconds_per_level,
                        gpu_provider=self.gpu_provider,
                        utilisation=self.utilisation, **self.extra)
        try:
            res = asyncio.run(sim.eval())
        except Exception as e:
            # The sweep never produced a record: the GPU died, the image failed,
            # the server never came up. Worth retrying the same diff. Cost is
            # unknown here rather than zero -- the container may well have run
            # for twenty minutes before dying.
            return False, {"error": f"{type(e).__name__}: {e}",
                           "cost_usd": 0.0, "cost_unknown": True}, "infra"

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
        return True, {
            "bill_per_1k": b.bill_per_1k,
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
