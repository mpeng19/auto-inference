"""Sample GPU busy time, because the server will not tell us.

Cost attribution needs GPU-seconds actually spent computing, not wall-clock
seconds. Two attempts to get that from SGLang failed:

  * **Wall time.** Every concurrency level ran a fixed 90s regardless of load,
    so the left-hand side was near-constant while token counts varied 10x. The
    regression could not fit it (r2 = -2.9) yet still produced plausible-looking
    prices, which is the dangerous kind of wrong.
  * **`sglang:forward_execution_seconds_total`.** Defined as a Counter in the
    SGLang source and never emitted, even with
    `--enable-metrics-for-all-schedulers`. It appears to be populated only under
    DP-cooperation mode.

So sample `nvidia-smi` directly and integrate. `utilization.gpu` is the
percentage of the sampling interval during which at least one kernel was
resident -- a coarse proxy for "busy", not for FLOP efficiency. That is the
right notion here: we want the fraction of paid-for GPU time that was doing
work, and an idle GPU must not have its cost charged to the few tokens that
happened to pass through.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Sample:
    t: float
    util_pct: float
    mem_used_mib: float


class GpuWatch:
    """Background sampler. Start before a measurement window, stop after."""

    def __init__(self, interval_s: float = 0.25):
        self.interval_s = interval_s
        self._samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._err: str | None = None

    # ── lifecycle ────────────────────────────────────────────────
    def start(self) -> "GpuWatch":
        self._samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def __enter__(self) -> "GpuWatch":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ── sampling ─────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                # One line per GPU; average across the node.
                utils, mems = [], []
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        utils.append(float(parts[0]))
                        mems.append(float(parts[1]))
                if utils:
                    self._samples.append(
                        Sample(time.time(), sum(utils) / len(utils), sum(mems)))
            except Exception as e:
                self._err = f"{type(e).__name__}: {e}"
            # Keep the cadence steady regardless of how long nvidia-smi took.
            time.sleep(max(0.0, self.interval_s - (time.time() - t0)))

    # ── results ──────────────────────────────────────────────────
    def summary(self, wall_s: float, n_gpu: int = 1) -> dict:
        """Busy fraction over the window, and the GPU-seconds it implies."""
        if not self._samples:
            return {"available": False, "error": self._err or "no samples",
                    "busy_frac": None, "gpu_seconds": None}
        utils = [s.util_pct for s in self._samples]
        busy = sum(utils) / len(utils) / 100.0
        return {
            "available": True,
            "n_samples": len(utils),
            "busy_frac": round(busy, 4),
            "util_mean_pct": round(sum(utils) / len(utils), 2),
            "util_p50_pct": round(sorted(utils)[len(utils) // 2], 2),
            "util_max_pct": round(max(utils), 2),
            "mem_used_max_mib": round(max(s.mem_used_mib for s in self._samples), 1),
            # The quantity cost is attributed against.
            "gpu_seconds": round(wall_s * busy * max(1, n_gpu), 3),
            "wall_gpu_seconds": round(wall_s * max(1, n_gpu), 3),
            "error": self._err,
        }
