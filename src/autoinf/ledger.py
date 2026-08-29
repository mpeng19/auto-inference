"""Experiment ledger: is the research loop actually making progress?

An autonomous search can fail in ways that look like work. It can re-propose
the same change with cosmetic differences; it can grind one knob while ignoring
the rest of the space; it can plateau for fifty runs; it can "win" by breaking
correctness. All of these produce a healthy-looking stream of completed
experiments, so the number of experiments run tells you nothing.

This records every experiment and scores the *search*, not the result:

  * **Novelty** -- distance from the nearest previous experiment. A run of low
    novelty means the loop is circling.
  * **Diversity** -- entropy over which knobs and files have been touched. High
    novelty with low diversity means it is exploring one dimension thoroughly
    and ignoring the others.
  * **Progress** -- best-so-far, and how long since it last moved.
  * **Integrity** -- goodput gains that coincide with canary divergence or
    dropped requests are the signature of reward hacking, not improvement.
  * **Cost** -- dollars per percent of improvement, so a stalled search is
    visible as a rising price rather than a flat graph.

The ledger is append-only JSONL: an experiment that happened cannot be quietly
revised once its result is known.
"""
from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import asdict, dataclass, field

# Numeric knobs, with the range used to normalise distance. Two configs
# differing by 8 on max_running_requests (range 1024) are far more similar than
# two differing by 8 on tp_size (range 8).
_NUMERIC_RANGES = {
    "tp_size": 8, "dp_size": 8, "ep_size": 8,
    "mem_fraction_static": 0.5, "max_total_tokens": 500_000,
    "max_running_requests": 1024, "chunked_prefill_size": 32768,
    "schedule_conservativeness": 2.0, "context_length": 262144,
}
_CATEGORICAL = ("schedule_policy", "enable_prefix_caching", "model", "gpu", "n_gpu")


@dataclass
class Experiment:
    """One evaluated idea."""
    id: str
    ts: float
    hypothesis: str                     # why this was worth trying, in words
    config: dict                        # ServingConfig as a dict
    overlay_digest: str = ""
    overlay_files: tuple[str, ...] = ()
    parent_id: str | None = None        # what it was derived from
    # results
    goodput_rps: float | None = None
    p99_ttft_ms: float | None = None
    good_frac: float | None = None
    n_failed: int = 0
    canary_exact_rate: float | None = None
    cost_usd: float = 0.0
    wall_s: float = 0.0
    status: str = "ok"
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def _config_distance(a: dict, b: dict) -> float:
    """0 = identical, 1 = maximally different over the knobs we search."""
    total, seen = 0.0, 0
    for k, rng in _NUMERIC_RANGES.items():
        x, y = a.get(k), b.get(k)
        if x is None and y is None:
            continue
        seen += 1
        if x is None or y is None:
            total += 1.0
        else:
            total += min(1.0, abs(float(x) - float(y)) / rng)
    for k in _CATEGORICAL:
        if k not in a and k not in b:
            continue
        seen += 1
        total += 0.0 if a.get(k) == b.get(k) else 1.0
    return total / seen if seen else 0.0


def _overlay_distance(a: Experiment, b: Experiment) -> float:
    """Code changes: identical digest is 0, disjoint file sets is 1."""
    if a.overlay_digest and a.overlay_digest == b.overlay_digest:
        return 0.0
    fa, fb = set(a.overlay_files), set(b.overlay_files)
    if not fa and not fb:
        return 0.0
    if not fa or not fb:
        return 1.0
    jac = len(fa & fb) / len(fa | fb)
    # Same files but different digests: a real edit, but in familiar territory.
    return 1.0 - jac * 0.5


def novelty(e: Experiment, prior: list[Experiment]) -> float:
    """Distance to the closest thing already tried. 0 = a repeat."""
    if not prior:
        return 1.0
    return min(
        0.5 * _config_distance(e.config, p.config) + 0.5 * _overlay_distance(e, p)
        for p in prior
    )


def _entropy(counts: dict) -> float:
    """Normalised Shannon entropy: 1 = attention spread evenly."""
    tot = sum(counts.values())
    if tot <= 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts.values() if c > 0)
    return h / math.log(len(counts))


class Ledger:
    def __init__(self, path: str | pathlib.Path = "runs/ledger.jsonl"):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Experiment]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                d["overlay_files"] = tuple(d.get("overlay_files") or ())
                out.append(Experiment(**d))
        return out

    def append(self, e: Experiment) -> Experiment:
        with self.path.open("a") as f:
            f.write(e.to_json() + "\n")
        return e

    # ── analysis ─────────────────────────────────────────────────
    def report(self, plateau_window: int = 8, novelty_floor: float = 0.05) -> dict:
        xs = self.load()
        if not xs:
            return {"n": 0, "verdict": "no experiments yet"}

        xs.sort(key=lambda e: e.ts)
        nov = [novelty(e, xs[:i]) for i, e in enumerate(xs)]

        # What has been touched, and how evenly.
        knob_counts: dict[str, int] = {}
        base = xs[0].config
        for e in xs[1:]:
            for k in list(_NUMERIC_RANGES) + list(_CATEGORICAL):
                if e.config.get(k) != base.get(k):
                    knob_counts[k] = knob_counts.get(k, 0) + 1
        file_counts: dict[str, int] = {}
        for e in xs:
            for f in e.overlay_files:
                file_counts[f] = file_counts.get(f, 0) + 1

        scored = [e for e in xs if e.goodput_rps is not None and e.status == "ok"]
        best_curve, best = [], float("-inf")
        for e in scored:
            best = max(best, e.goodput_rps)
            best_curve.append(best)
        since_best = 0
        if best_curve:
            top = max(best_curve)
            # best_curve is a running maximum, so it stays flat at `top` once
            # reached. The plateau length is measured from where the best was
            # FIRST achieved; taking the last matching index would always give 0.
            first_best = min(i for i, v in enumerate(best_curve) if v == top)
            since_best = len(best_curve) - 1 - first_best

        baseline = scored[0].goodput_rps if scored else None
        improvement = ((best - baseline) / baseline * 100
                       if baseline and best > float("-inf") else 0.0)
        cost = sum(e.cost_usd for e in xs)

        # Reward hacking: gains that arrive alongside broken outputs.
        suspect = [
            e.id for e in scored
            if baseline and e.goodput_rps > baseline * 1.02
            and ((e.canary_exact_rate is not None and e.canary_exact_rate < 0.9)
                 or e.n_failed > 0)
        ]

        recent_nov = nov[-plateau_window:] if len(nov) >= 2 else nov
        mean_recent_nov = sum(recent_nov) / len(recent_nov) if recent_nov else 1.0
        knob_div = _entropy(knob_counts)

        flags = []
        if mean_recent_nov < novelty_floor:
            flags.append(f"CIRCLING: last {len(recent_nov)} experiments average "
                         f"novelty {mean_recent_nov:.3f} — near-repeats of earlier work")
        if len(xs) >= 5 and knob_counts and knob_div < 0.35:
            top_knob = max(knob_counts, key=knob_counts.get)
            flags.append(f"NARROW: knob diversity {knob_div:.2f}; attention "
                         f"concentrated on '{top_knob}'")
        if since_best >= plateau_window:
            flags.append(f"PLATEAU: no new best in {since_best} experiments")
        if suspect:
            flags.append(f"INTEGRITY: {len(suspect)} experiment(s) gained goodput "
                         f"while failing canaries or dropping requests — "
                         f"{suspect[:3]}")
        if improvement > 0 and cost > 0 and cost / max(improvement, 1e-9) > 5.0:
            flags.append(f"EXPENSIVE: ${cost / improvement:.2f} per 1% improvement")

        return {
            "n": len(xs),
            "n_scored": len(scored),
            "verdict": "HEALTHY" if not flags else "; ".join(flags),
            "flags": flags,
            "baseline_goodput": baseline,
            "best_goodput": best if best > float("-inf") else None,
            "improvement_pct": round(improvement, 2),
            "experiments_since_best": since_best,
            "mean_novelty_recent": round(mean_recent_nov, 3),
            "knob_diversity": round(knob_div, 3),
            "file_diversity": round(_entropy(file_counts), 3),
            "knobs_touched": dict(sorted(knob_counts.items(), key=lambda kv: -kv[1])),
            "files_touched": dict(sorted(file_counts.items(), key=lambda kv: -kv[1])),
            "total_cost_usd": round(cost, 2),
            "cost_per_pct_usd": round(cost / improvement, 2) if improvement > 0 else None,
            "suspect_ids": suspect,
        }

    def whats_new(self, k: int = 5) -> str:
        """Human-readable digest of the most recent experiments."""
        xs = sorted(self.load(), key=lambda e: e.ts)
        if not xs:
            return "No experiments recorded yet."
        prior_by_index = {e.id: xs[:i] for i, e in enumerate(xs)}
        lines = []
        for e in xs[-k:]:
            n = novelty(e, prior_by_index[e.id])
            delta = ""
            scored = [x for x in xs if x.goodput_rps is not None]
            if scored and e.goodput_rps is not None:
                b = scored[0].goodput_rps
                if b:
                    delta = f"  ({(e.goodput_rps - b) / b * 100:+.1f}% vs baseline)"
            lines.append(
                f"[{e.id}] novelty {n:.2f}  goodput "
                f"{e.goodput_rps if e.goodput_rps is not None else '--'}{delta}\n"
                f"    hypothesis: {e.hypothesis[:150]}\n"
                f"    changed: {', '.join(e.overlay_files) or 'config only'}"
                + (f"\n    NOTE: {e.notes}" if e.notes else "")
            )
        return "\n".join(lines)
