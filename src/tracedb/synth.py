"""Generate a realistic synthetic training-loop chrome trace (deterministic) for tests/demos.

20 steps of: dataloader -> fwd ops -> loss -> bwd ops (with allreduce overlapping on a comm
stream) -> optimizer. Injected pathologies: a ~300us gap between attn_out and mlp_in every step,
and a 3ms dataloader stall every 5th step.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def generate(path: str | Path, steps: int = 20, seed: int = 7) -> dict:
    rng = random.Random(seed)
    ev = []
    PID = 1
    CPU, S7, S20 = "cpu", "stream 7", "stream 20"
    for tid, name in [(CPU, "python main thread"), (S7, "CUDA stream 7 (compute)"), (S20, "CUDA stream 20 (comm)")]:
        ev.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": name}})

    def x(name, tid, ts, dur, cat):
        ev.append({"ph": "X", "pid": PID, "tid": tid, "name": name, "cat": cat,
                   "ts": round(ts, 1), "dur": round(dur, 1)})

    t = 1000.0
    fwd = ["embed", "attn_qkv", "attn_core", "attn_out", "mlp_in", "mlp_out", "logits", "loss"]
    bwd = ["loss_bwd", "logits_bwd", "mlp_bwd", "attn_bwd", "embed_bwd"]
    for i in range(steps):
        step_t0 = t
        dl = 800 + rng.uniform(-50, 50) + (3000 if i % 5 == 4 else 0)
        x("dataloader_next", CPU, t, dl, "cpu_op"); t += dl + 20
        for op in fwd:
            cpu_d = 60 + rng.uniform(-10, 10)
            x(op, CPU, t, cpu_d, "cpu_op")
            k_d = 180 + rng.uniform(-20, 20)
            x(f"kernel_{op}", S7, t + 25, k_d, "kernel")
            t += max(cpu_d, 25 + k_d) * 0.55
            if op == "attn_out":
                t += 300 + rng.uniform(-30, 30)   # injected gap before mlp_in
        t += 150
        ar_started = []
        for j, op in enumerate(bwd):
            cpu_d = 80 + rng.uniform(-10, 10)
            x(op, CPU, t, cpu_d, "cpu_op")
            k_d = 260 + rng.uniform(-30, 30)
            x(f"kernel_{op}", S7, t + 25, k_d, "kernel")
            if j >= 1:
                ar_d = 400 + rng.uniform(-40, 40)
                x(f"nccl_allreduce_{j}", S20, t + 60, ar_d, "kernel")
                ar_started.append(t + 60 + ar_d)
            t += max(cpu_d, 25 + k_d) * 0.7
        if ar_started:
            t = max(t, max(ar_started) + 30)
        opt_d = 350 + rng.uniform(-30, 30)
        x("optimizer_step", CPU, t, opt_d, "cpu_op")
        x("kernel_optimizer", S7, t + 30, opt_d * 0.8, "kernel")
        t += opt_d + 100
        ev.append({"ph": "X", "pid": PID, "tid": CPU, "name": f"ProfilerStep#{i}", "cat": "user_annotation",
                   "ts": round(step_t0, 1), "dur": round(t - step_t0, 1)})
        t += 50
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"traceEvents": ev}))
    return {"path": str(path), "events": len(ev), "steps": steps}


if __name__ == "__main__":
    import sys
    print(generate(sys.argv[1] if len(sys.argv) > 1 else "fixtures/synth_trace.json"))
