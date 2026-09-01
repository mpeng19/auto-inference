"""Research-loop health: is the search working, or just busy?

    uv run python scripts/ledger_report.py
    uv run python scripts/ledger_report.py --new 10     # recent digest only
"""
from __future__ import annotations

import argparse
import json

from autoinf.ledger import Ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="runs/ledger.jsonl")
    ap.add_argument("--new", type=int, default=5, help="how many recent experiments to show")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    led = Ledger(a.path)
    r = led.report()
    if a.json:
        print(json.dumps(r, indent=2, default=str))
        return 0

    if not r.get("n"):
        print("No experiments recorded yet.")
        print(f"(ledger: {a.path})")
        return 0

    print(f"experiments      {r['n']}  ({r['n_scored']} scored)")
    print(f"baseline         {r['baseline_goodput']}")
    print(f"best             {r['best_goodput']}   ({r['improvement_pct']:+.2f}%)")
    print(f"since best       {r['experiments_since_best']} experiments")
    print(f"cost             ${r['total_cost_usd']}"
          + (f"   (${r['cost_per_pct_usd']}/1% gained)" if r.get("cost_per_pct_usd") else ""))
    print()
    print(f"novelty (recent) {r['mean_novelty_recent']}   "
          "0 = repeating earlier work, 1 = unexplored")
    print(f"knob diversity   {r['knob_diversity']}   "
          "0 = one knob only, 1 = attention spread evenly")
    print(f"file diversity   {r['file_diversity']}")

    if r["knobs_touched"]:
        print("\nknobs touched:")
        for k, v in list(r["knobs_touched"].items())[:10]:
            print(f"  {k:<32}{v}")
    if r["files_touched"]:
        print("\nfiles overlaid:")
        for k, v in list(r["files_touched"].items())[:10]:
            print(f"  {k:<48}{v}")

    print(f"\nverdict: {r['verdict']}")
    if r["flags"]:
        for f in r["flags"]:
            print(f"  ! {f}")

    print("\n--- what's new ---")
    print(led.whats_new(k=a.new))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
