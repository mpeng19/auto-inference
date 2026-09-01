"""Find why `sglang:forward_execution_seconds_total` is declared but never emitted.

It is the metric that would make cost attribution direct: actual forward-pass
GPU time, split by phase, instead of a regression over workload mixes.
"""
import modal

from autoinf.modal_app import image

app = modal.App("auto-inference-probe-metric", image=image)


@app.function(cpu=2, timeout=600)
def find() -> dict:
    import pathlib
    import re
    import subprocess

    import sglang
    root = pathlib.Path(sglang.__file__).parent
    out = {"version": str(getattr(sglang, "__version__", "?")), "root": str(root)}

    hits = subprocess.run(
        ["grep", "-rn", "forward_execution", str(root)],
        capture_output=True, text=True).stdout.strip().splitlines()
    out["grep"] = hits[:40]

    # Where is the counter declared, and is anything calling its observe/inc?
    decl, uses = [], []
    for h in hits:
        if "=" in h and ("Counter" in h or "Histogram" in h or "Gauge" in h):
            decl.append(h)
        elif ".inc(" in h or ".observe(" in h or ".labels(" in h:
            uses.append(h)
    out["declared"] = decl
    out["emitted"] = uses

    def region(rel, lo, hi):
        f = root / rel
        if not f.exists(): return [f"MISSING {rel}"]
        ls = f.read_text().splitlines()
        return [f"{i+1:>5}| {ls[i]}" for i in range(max(0,lo-1), min(len(ls),hi))]

    out["flag"] = subprocess.run(
        ["grep", "-rn", "ENABLE_METRICS_DEVICE_TIMER", str(root)],
        capture_output=True, text=True).stdout.strip().splitlines()
    out["timer"] = subprocess.run(
        ["grep", "-rn", "class DeviceTimer", "-A", "40", str(root)],
        capture_output=True, text=True).stdout.strip().splitlines()[:45]
    out["reporter"] = region("srt/managers/scheduler_components/metrics_reporter.py", 140, 200)
    out["collector"] = region("srt/observability/metrics_collector.py", 1276, 1300)
    return out


@app.local_entrypoint()
def main():
    import json
    r = find.remote()
    print(f"sglang {r['version']} at {r['root']}\n")
    print(f"--- all mentions of 'forward_execution' ({len(r['grep'])}) ---")
    for h in r["grep"]:
        print("  " + h.replace(r["root"], "…"))
    print(f"\n--- declared as a metric ({len(r['declared'])}) ---")
    for h in r["declared"]: print("  " + h.replace(r["root"], "…"))
    print(f"\n--- anything actually emitting it ({len(r['emitted'])}) ---")
    for h in r["emitted"]: print("  " + h.replace(r["root"], "…"))
    print("\n--- ENABLE_METRICS_DEVICE_TIMER ---")
    for l in r["flag"]: print("  " + l.replace(r["root"], "…"))
    print("\n--- DeviceTimer ---")
    for l in r["timer"][:40]: print("  " + l.replace(r["root"], "…"))
    print("\n--- metrics_reporter.py, the caller ---")
    for l in r["reporter"]: print(l)
    print("\n--- metrics_collector.py, increment_forward_execution_seconds ---")
    for l in r["collector"]: print(l)
