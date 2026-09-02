"""The plain-text report written beside the figures.

Deliberately says both ranks. On the 1xH100 baseline we are 1st of 12 on
effective input price and 9th of 12 on the bill a buyer actually pays, because
input is 5% of that bill. Quoting either number alone is a way to be
accidentally dishonest, so the report always carries both.
"""
from __future__ import annotations


def render(sim, res) -> str:
    L: list[str] = []
    add = L.append
    add("=" * 72)
    add(f"  {sim.model}   {sim.n_gpu}x{sim.gpu}   stack: {sim.stack.describe()}")
    add(f"  {sim.cost_basis}, utilisation {sim.util:.0%}, break-even")
    add(f"  SLO: {sim.slo.describe()}")
    add(f"  eval digest {sim.digest()}   stack digest {sim.stack.digest}")
    if sim.note:
        add(f"  note: {sim.note}")
    add("=" * 72)

    if not res.curve:
        add(f"\nNO RESULT: {res.reason}")
        return "\n".join(L) + "\n"

    add("")
    add(f"{'N':>5}{'reqs':>6}{'goodput':>9}{'batch':>7}{'hit':>7}"
        f"{'GPU-s/req':>11}{'eff-in $/M':>12}{'out $/M':>9}"
        f"{'$/1k':>8}{'share':>8}  SLO")
    add("-" * 96)
    for p in res.curve:
        add(f"{p.n_users:>5}{p.n_requests:>6}{p.goodput_rps:>9.2f}{p.batch:>7.1f}"
            f"{p.hit_rate:>7.3f}{p.gpu_s_per_request:>11.2f}"
            f"{p.effective_in_per_m:>12.4f}{p.out_per_m:>9.3f}"
            f"{p.bill_per_1k:>8.2f}{p.share_per_node:>8.2%}"
            + ("  ok" if p.meets_slo else f"  MISS <- {p.binding}"))

    if not res.ok:
        add(f"\nNO PRICE: {res.reason}")
        return "\n".join(L) + "\n"

    add("")
    add(res.summary())
    if res.interpolated:
        i = res.interpolated
        add(f"  frontier by interpolation: N*~{i['n_star']:.1f}  ${i['bill_per_1k']:.2f}/1k"
            f"  ({i['binding']} crosses between N={i['between'][0]} and {i['between'][1]})")
    if res.reason:
        add(f"\ncaveat: {res.reason}")

    warn = sorted({w for p in res.curve for w in p.warnings})
    if warn:
        add("\nsampling:")
        for w in warn:
            add(f"  ! {w}")

    r = res.rank()
    if r:
        m = res.market
        add(f"\nwhole bill at {m.in_per_request:,.0f} in / {m.out_per_request:,.0f} out"
            f"   (OpenRouter {m.as_of})")
        add(f"  {'#':>3}  {'provider':<14}{'eff-in $/M':>12}{'out $/M':>9}{'$/1k req':>10}")
        for row in r["board"]:
            mark = "  <--" if row["us"] else ""
            add(f"  {row['rank_bill']:>3}  {row['provider']:<14}{row['eff_in']:>12.4f}"
                f"{row['out']:>9.3f}{row['bill_1k']:>10.2f}{mark}")

    b = res.best
    add("")
    add(f"capacity: one {sim.n_gpu}x{sim.gpu} node serves {b.capacity_per_day:,.0f} "
        f"requests/day = {b.share_per_node:.2%} of the market at {sim.util:.0%} utilisation.")
    for tgt in (0.01, 0.05, 0.10):
        add(f"  {tgt:>5.0%} share needs {tgt / b.share_per_node:>5.1f} nodes "
            f"(${tgt / b.share_per_node * sim.n_gpu * 24 * sim.rate_per_gpu_hour:,.0f}/day)")
    add("")
    add("Price below saturation is a hyperbola in share that the serving stack")
    add("does not enter: the stack sets only where the curve stops falling.")
    return "\n".join(L) + "\n"
