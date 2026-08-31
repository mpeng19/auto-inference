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
    from autoinf.config import SLO, ServingConfig

    sc = ServingConfig()
    sl = SLO(ttft_ms=a.ttft_ms, tpot_ms=a.tpot_ms)
    levels = [int(x) for x in a.levels.split(",") if x.strip()]
    call = _fn("frontier").spawn(asdict(sc), levels, asdict(sl),
                                 a.seconds, a.repeats, a.note)
    print(f"spawned  call_id={call.object_id}")
    print(f"  model   {sc.model.split('/')[-1]} on {sc.n_gpu}x{sc.gpu}")
    print(f"  levels  {levels}  x {a.seconds:.0f}s  x {a.repeats} repeat(s)")
    print(f"  SLOs    TTFT p99 < {a.ttft_ms:.0f}ms, TPOT p99 < {a.tpot_ms:.0f}ms")
    print(f"\ncollect:  uv run python scripts/launch.py collect {call.object_id}")
    return 0


def report(rec: dict) -> None:
    """Frontier curve -> N* -> cost attribution -> effective price -> rank."""
    from autoinf.modal_app import MARKET_QWEN38_27B
    from autoinf.pricing import (Observation, attribute, conditioning,
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

    # Attribution uses the phase-B *mixes* first. The concurrency levels vary
    # scale but not ratio, so on their own they are near-collinear and the three
    # per-token costs come out arbitrary however good the r2 looks.
    obs = [Observation(m["mix"], m["uncached_tokens"], m["cached_tokens"],
                       m["output_tokens"], m["gpu_seconds"])
           for m in rec.get("mixes", []) if m["output_tokens"] > 0]
    obs += [Observation(f"N{l['n_users']}r{l['repeat']}", l["uncached_tokens"],
                        l["cached_tokens"], l["output_tokens"], l["gpu_seconds"])
            for l in rec["levels"] if l["output_tokens"] > 0]
    if len(obs) < 3:
        print("\nnot enough observations for cost attribution")
        return

    cond, attr = conditioning(obs), attribute(obs)
    print(f"\nGPU-seconds/token  uncached_in {attr.per_uncached_in:.3e}"
          f"  cached_in {attr.per_cached_in:.3e}  out {attr.per_out:.3e}")
    print(f"  r2 {attr.r2}   condition {cond['condition_number']}"
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

    print(f"\n{'hit':>6}{'eff in $/M':>13}{'rank':>10}")
    print("-" * 31)
    for h in (0.5, 0.9, 0.95):
        e = effective_in(p["price_in_per_mtok"], p["price_cached_in_per_mtok"], h)
        r = rank_vs_market(e, MARKET_QWEN38_27B, h)
        print(f"{h:>6.2f}{e:>13.4f}{r['rank']:>6}/{r['of']:<3}")


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
    f.set_defaults(fn=cmd_frontier)

    c = sub.add_parser("collect")
    c.add_argument("call_id")
    c.add_argument("--timeout", type=float, default=10.0)
    c.set_defaults(fn=cmd_collect)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
