"""The bridge to the simulator: one attempt in, one set of metrics out.

Kept behind the `Evaluator` protocol so the loop can be exercised without
renting a GPU. The fake and the real one differ only in what happens between
`stack` going in and `metrics` coming out, which is the property that makes the
whole harness testable.

Failures are classified rather than raised, because the loop treats them
differently: an infrastructure failure is retried unchanged, a rejected
hypothesis is not.
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
            # the server never came up. Worth retrying the same diff.
            return False, {"error": f"{type(e).__name__}: {e}"}, "infra"

        if not res.ok:
            # A record exists and says no. Retrying is a second bill for the
            # same answer -- unless nothing could be priced at all, which is
            # infrastructure wearing a result's clothes.
            kind = "infra" if "DEVICE_TIMER" in res.reason else "slo"
            return False, {"reason": res.reason}, kind

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
            "artifacts": res.artifacts,
        }, ""
