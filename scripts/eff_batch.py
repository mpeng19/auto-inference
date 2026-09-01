import modal, json
app = modal.App("eff-batch")
vol = modal.Volume.from_name("auto-inference-results")

@app.function(volumes={"/results": vol}, timeout=600)
def get(name): 
    return json.load(open(f"/results/runs/{name}"))

@app.local_entrypoint()
def main(name: str):
    import numpy as np
    r = get.remote(name)
    print(f"{'N':>4}{'sampled b':>11}{'p50 TPOT':>10}{'out/dec s':>11}"
          f"{'b_eff':>8}{'GPU-s/tok':>11}")
    print("-"*56)
    B, T = [], []
    for lv in r["levels"]:
        c = lv["server_counters"]
        dec = c["sglang:forward_execution_seconds_total[decode]"]
        out = lv["output_tokens"]
        tpot = lv["tpot_ms"]["p50"] / 1e3
        rate = out / dec                     # tokens/s of decode GPU time
        beff = rate * tpot                   # = (batch/step) * step
        sb = (lv["batch"]["running"] or {}).get("mean", 0)
        print(f"{lv['n_users']:>4}{sb:>11.1f}{tpot*1e3:>10.1f}{rate:>11.1f}"
              f"{beff:>8.2f}{dec/out:>11.3e}")
        B.append(beff); T.append(tpot*1e3)
    B, T = np.array(B), np.array(T)
    A = np.vstack([np.ones_like(B), B]).T
    (c0, c1), *_ = np.linalg.lstsq(A, T, rcond=None)
    p = A @ np.array([c0, c1])
    r2 = 1 - ((T-p)**2).sum()/((T-T.mean())**2).sum()
    print(f"\nstep = {c0:.2f} ms + {c1:.3f} ms x b_eff    r2 {r2:.4f}")
    print("  residuals:", np.round(T-p, 2))
