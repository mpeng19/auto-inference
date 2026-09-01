"""Pull OpenRouter's real traffic and latency data for a model.

    uv run python scripts/market_pull.py qwen/qwen3.8-27b

The public API (`/api/v1/models/<slug>/endpoints`) gives listed prices and
uptime only. Everything that matters -- per-provider latency percentiles and
the daily token/cache/request series -- is streamed into the page as a Next.js
React Server Component payload, which is why every JSON endpoint guess 404s.
Requesting the page with an `RSC: 1` header returns that payload directly.

Two things this exists to prevent:

  * Sizing the SLO against the page's *median* latency. The market's p50 TTFT
    is 0.4-3.8s but its **p99 is 3.8-62s**. We were benchmarking against a
    2s p99, stricter than any provider on the board.
  * Sizing the workload against TraceLab. Real traffic is 20.6k in / 2.1k out
    (9.9:1); our replay was 132k / 454 (291:1), which put 12% of the modelled
    cost on output tokens when the real figure is 70-81%.
"""
from __future__ import annotations

import json
import sys
import urllib.request

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")


def fetch(slug: str) -> str:
    req = urllib.request.Request(
        f"https://openrouter.ai/{slug}",
        headers={"RSC": "1", "Accept": "*/*", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def _array(payload: str, key: str) -> list[dict]:
    marker = f'"{key}":'
    i = payload.index(marker)
    arr, _ = json.JSONDecoder().raw_decode(payload[i + len(marker):])
    return arr


def latency(payload: str) -> list[dict]:
    """Per-provider throughput/latency percentiles over a 30-minute window."""
    import re
    names = [(m.start(), m.group(1)) for m in
             re.finditer(r'"provider_display_name":"([^"]+)"', payload)]
    out = []
    for m in re.finditer(r'"stats":\{"endpoint_id":"[^"]+",[^}]+\}', payload):
        d = json.loads("{" + m.group(0).split('"stats":{', 1)[1])
        prior = [n for p, n in names if p < m.start()]
        out.append({"provider": prior[-1] if prior else "?", **d})
    return out


def daily(payload: str) -> list[dict]:
    """Market-wide daily totals. `model_chart` is all traffic for the model."""
    rows = {}
    for r in _array(payload, "model_chart"):
        rows[r["date"][:10]] = r
    return [rows[d] for d in sorted(rows)]


def summarise(days: list[dict]) -> dict:
    p = sum(r["total_prompt_tokens"] for r in days)
    c = sum(r["total_native_tokens_cached"] for r in days)
    o = sum(r["total_completion_tokens"] + r["total_native_tokens_reasoning"]
            for r in days)
    n = sum(r["count"] for r in days)
    return {"days": len(days), "prompt_tokens": p, "cached_tokens": c,
            "output_tokens": o, "requests": n, "cache_hit_rate": c / p,
            "in_per_request": p / n, "out_per_request": o / n,
            "in_out_ratio": p / o}


def main(slug: str) -> None:
    payload = fetch(slug)
    lat, days = latency(payload), daily(payload)
    s = summarise(days)

    print(f"{slug}   {s['days']} days of traffic\n")
    print(f"{'provider':<14}{'p50':>7}{'p90':>8}{'p99 TTFT':>10}{'tps':>6}{'reqs':>8}")
    print("-" * 53)
    for d in sorted(lat, key=lambda d: d["p99_latency"]):
        print(f"{d['provider'][:13]:<14}{d['p50_latency']:>7.0f}"
              f"{d['p90_latency']:>8.0f}{d['p99_latency']:>10.0f}"
              f"{d['p50_throughput']:>6.0f}{d['request_count']:>8}")
    p99 = sorted(d["p99_latency"] for d in lat)
    print(f"\nmarket p99 TTFT: best {p99[0]:.0f}ms  median {p99[len(p99)//2]:.0f}ms"
          f"  worst {p99[-1]:.0f}ms")

    print(f"\n{'date':<12}{'prompt':>9}{'hit':>7}{'reqs':>10}{'in/req':>8}{'out/req':>9}")
    print("-" * 55)
    for r in days:
        o = r["total_completion_tokens"] + r["total_native_tokens_reasoning"]
        print(f"{r['date'][:10]:<12}{r['total_prompt_tokens']/1e9:>8.1f}B"
              f"{r['total_native_tokens_cached']/r['total_prompt_tokens']:>7.3f}"
              f"{r['count']:>10}{r['total_prompt_tokens']/r['count']:>8.0f}"
              f"{o/r['count']:>9.0f}")
    print(f"\ncache hit {s['cache_hit_rate']:.3f} | in/req {s['in_per_request']:,.0f}"
          f" | out/req {s['out_per_request']:,.0f} | in:out {s['in_out_ratio']:.1f}")

    out = f"market-{slug.replace('/', '-')}.json"
    with open(out, "w") as f:
        json.dump({"slug": slug, "summary": s, "latency": lat, "daily": days},
                  f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.8-27b")
