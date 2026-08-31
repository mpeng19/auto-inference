"""Fire a long GPU run that outlives this terminal, then collect it later.

`modal run --detach` keeps the *app* alive, but a `local_entrypoint` calling
`.remote()` holds the function call open against the local client -- lose the
client and the call is cancelled mid-flight. That is exactly what killed a
frontier sweep six minutes in, three levels deep.

A sweep is 15-60 minutes and the research loop will run many unattended, so the
call must not depend on anything local staying up. `.spawn()` returns
immediately with a handle; the work runs to completion on Modal and writes its
result to the results Volume regardless of what happens here.

    uv run python scripts/launch.py frontier --levels 1,2,4,8,16,32,64,128
    uv run python scripts/launch.py collect <call_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import modal

APP = "auto-inference"


def _fn(name: str):
    return modal.Function.from_name(APP, name)


def cmd_frontier(a) -> int:
    import dataclasses

    from autoinf.config import SLO, ServingConfig

    sc = ServingConfig()
    if a.model:
        sc = dataclasses.replace(sc, model=a.model)
    if a.gpu:
        sc = dataclasses.replace(sc, gpu=a.gpu, n_gpu=a.n_gpu,
                                 tp_size=a.n_gpu)
    sl = SLO(ttft_ms=a.ttft_ms, tpot_ms=a.tpot_ms)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    # ServingConfig.n_gpu only sets SGLang's --tp-size; it does not allocate
    # anything. The deployed frontier is declared with one L40S, so asking for
    # TP=8 without this produced "CUDA error: invalid device ordinal" 48
    # seconds in. Resources have to be overridden at call time.
    n = max(1, sc.n_gpu)
    fn = _fn("frontier").with_options(
        gpu=f"{sc.gpu}:{n}" if n > 1 else sc.gpu,
        cpu=float(max(16, 4 * n)),
        timeout=60 * 60 * (3 if n > 1 else 2),
    )
    call = fn.spawn(asdict(sc), levels, asdict(sl),
                    a.seconds, a.repeats, a.note, a.trace_scale)
    print(f"spawned  call_id={call.object_id}")
    print(f"  model   {sc.model.split('/')[-1]} on {sc.n_gpu}x{sc.gpu}")
    print(f"  levels  {levels}  x {a.seconds:.0f}s  x {a.repeats} repeat(s)")
    print(f"  SLOs    TTFT p99 < {a.ttft_ms:.0f}ms, TPOT p99 < {a.tpot_ms:.0f}ms")
    print(f"  traffic {'TraceLab replay at %gx' % a.trace_scale if a.trace_scale > 0 else 'synthetic conversations'}")
    print(f"\ncollect:  uv run python scripts/launch.py collect {call.object_id}")
    return 0


def report(rec: dict) -> None:
    """Frontier curve -> N* -> cost attribution -> effective price -> rank."""
    from autoinf.modal_app import MARKET_QWEN38_27B
    from autoinf.pricing import (Observation, attribute_saturated, conditioning,
                                 effective_in, fmt_prices, prices,
                                 rank_vs_market, usable)

    if rec.get("status") != "ok" or not rec.get("levels"):
        print(json.dumps({k: rec.get(k) for k in ("status", "failure")}, indent=2))
        return

    print(f"\n{'users':>6}{'goodput':>10}{'thruput':>10}{'SLO%':>7}"
          f"{'TTFT p99':>10}{'TPOT p99':>10}{'hit':>7}{'':>7}")
    print("-" * 67)
    for l in rec["levels"]:
        print(f"{l['n_users']:>6}{l['goodput_rps']:>10.2f}{l['throughput_rps']:>10.2f}"
              f"{l['good_frac'] * 100:>7.1f}{(l['ttft_p99_ms'] or 0):>10.0f}"
              f"{(l['tpot_p99_ms'] or 0):>10.1f}{(l['cache_hit_rate'] or 0):>7.2f}"
              f"{'  OK' if l['meets_slo'] else '  MISS':>7}")

    by_n: dict[int, list] = {}
    for l in rec["levels"]:
        by_n.setdefault(l["n_users"], []).append(l)
    # N* must hold in EVERY repeat: near saturation one passing run is luck.
    ok = [n for n, ls in by_n.items() if all(x["meets_slo"] for x in ls)]
    if not ok:
        print("\nno level met the SLOs")
        return
    n_star = max(ok)
    best = max(by_n[n_star], key=lambda l: l["goodput_rps"])
    print(f"\nN* = {n_star} concurrent users   goodput {best['goodput_rps']:.2f} rps"
          f"   cache hit {best['cache_hit_rate']:.2f}")

    # ONLY the saturated phase-B mixes. The concurrency levels must not be
    # mixed in: they run below saturation, where the GPU idles and wall time
    # overstates work, and they vary scale rather than composition. Including
    # them is what produced r2 = -36.7.
    obs = [Observation(m["mix"], m["uncached_tokens"], m["cached_tokens"],
                       m["output_tokens"], m["gpu_seconds"])
           for m in rec.get("mixes", []) if m["output_tokens"] > 0]
    if len(obs) < 3:
        print("\nno saturated mixes recorded; cannot attribute cost")
        return

    cond, attr = conditioning(obs), attribute_saturated(obs)
    print(f"\nGPU-seconds/token  uncached_in {attr.per_uncached_in:.3e}"
          f"  cached_in {attr.per_cached_in:.3e}  out {attr.per_out:.3e}")
    print(f"  fit {attr.r2} (1.0 = perfect, worst residual {attr.residual})"
          f"   condition {cond['condition_number']}"
          f"   {'ok' if cond['well_conditioned'] else 'ILL-CONDITIONED'}")
    if attr.cache_discount is not None:
        print(f"  cache discount {attr.cache_discount:.3f}   (market prices ~0.10)")

    good, why = usable(attr, cond)
    if not good:
        print(f"\n  ATTRIBUTION REJECTED: {why}")
        print("  No price is reported: a bad fit still produces plausible "
              "numbers, which is worse than reporting nothing.")
        return

    p = prices(attr, basis="nebius-h100-committed",
               n_gpu=rec["serving"].get("n_gpu", 1), utilization=0.6, margin=0.25)
    f = fmt_prices(p)
    print(f"\n{p['basis']} @ ${p['usd_per_gpu_hour']}/GPU-hr, util 60%, margin 25%")
    print(f"  in ${f['price_in_per_mtok']}/M   cached ${f['price_cached_in_per_mtok']}/M"
          f"   out ${f['price_out_per_mtok']}/M")

    # Hit rates chosen from reality: 0.70 is a typical OpenRouter provider,
    # 0.82 the best of them, 0.956 what TraceLab shows Anthropic/OpenAI achieve
    # on coding-agent traffic.
    print(f"\n{'hit':>7}{'eff in $/M':>13}{'rank':>10}")
    print("-" * 32)
    for h in (0.70, 0.82, 0.956):
        e = effective_in(p["price_in_per_mtok"], p["price_cached_in_per_mtok"], h)
        r = rank_vs_market(e, MARKET_QWEN38_27B, h)
        print(f"{h:>7.3f}{e:>13.4f}{r['rank']:>6}/{r['of']:<3}")

    if rec["serving"].get("model", "") != "Qwen/Qwen3.8-27B":
        print(f"\n  !! RANK IS NOT A COMPETITIVE RESULT: measured "
              f"{rec['serving'].get('model','?').split('/')[-1]}, but the market "
              f"table is Qwen3.8-27B. A smaller model serves cheaper for reasons "
              f"that have nothing to do with the serving stack.")


def cmd_collect(a) -> int:
    call = modal.FunctionCall.from_id(a.call_id)
    try:
        rec = call.get(timeout=a.timeout)
    except TimeoutError:
        print("still running; re-run collect later")
        return 1
    print(f"status {rec.get('status')}   saved {rec.get('result_path')}")
    report(rec)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frontier")
    f.add_argument("--levels", default="1,2,4,8,16,32,64,128")
    f.add_argument("--seconds", type=float, default=90.0)
    f.add_argument("--repeats", type=int, default=1)
    f.add_argument("--ttft-ms", dest="ttft_ms", type=float, default=1000.0)
    f.add_argument("--tpot-ms", dest="tpot_ms", type=float, default=50.0)
    f.add_argument("--note", default="frontier")
    f.add_argument("--model", default="", help="override the model to serve")
    f.add_argument("--gpu", default="", help="override GPU type, e.g. H100")
    f.add_argument("--n-gpu", dest="n_gpu", type=int, default=1)
    f.add_argument("--trace-scale", dest="trace_scale", type=float, default=0.0,
                   help="replay real TraceLab agent sessions, scaled by this "
                        "factor (0 = synthetic conversations)")
    f.set_defaults(fn=cmd_frontier)

    c = sub.add_parser("collect")
    c.add_argument("call_id")
    c.add_argument("--timeout", type=float, default=10.0)
    c.set_defaults(fn=cmd_collect)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
