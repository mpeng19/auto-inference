"""What could we sell this inference for?

The objective is an OpenRouter listing price, not a latency number. Getting
there needs three things the harness can measure and one it cannot:

  1. **The SLO frontier.** The largest offered load that still meets the
     marketplace's latency targets. Measured (`sweep_concurrency`).
  2. **Phase-split GPU time.** How many GPU-seconds prefill and decode each
     consume, read straight off SGLang's CUDA-event device timer. Measured.
     There is no regression here any more: splitting input cost into cached
     and uncached is needed only to re-blend at a competitor's hit rate, and
     caching well *is* serving well, so we price at our own (HANDOFF SS6b).
  3. **A cost basis.** Dollars per GPU-hour. *Not* measured — an input.
  4. **Utilisation.** What fraction of paid-for capacity carries traffic. Not
     measurable here at all; it depends on how much traffic the marketplace
     sends. It is the single largest lever on the answer, so it is an explicit
     parameter that must be stated with every result.

Modal's $3.95/hr is serverless retail and is deliberately excluded from the
default bases: nobody serving at scale pays it. The optimisation target is
therefore the **hardware-independent** quantity -- SLO-constrained tokens per
GPU-second -- and price is a reporting layer over it. A stack improvement shows
up in the former; a procurement decision only moves the latter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Reference costs, $/GPU-hour. Nebius list prices fetched 2026-08-30; the
# committed figure applies their stated "up to 35% off on-demand".
COST_BASES: dict[str, tuple[float, str]] = {
    "nebius-h100-committed": (2.50, "on-demand less 35% commitment discount"),
    "nebius-h100-preemptible": (2.15, "preemptible; cheapest, can be reclaimed"),
    "nebius-h100-ondemand": (3.85, "list on-demand"),
    "nebius-h200-committed": (2.93, "on-demand less 35%"),
    "nebius-h200-preemptible": (2.45, "preemptible"),
    "modal-h100": (3.95, "serverless retail -- for our own spend, NOT a serving cost basis"),
}
# Agreed with the project stakeholder 2026-09-01: price at $3.00/GPU-hr and
# 50% utilisation. Both are decisions, not estimates -- $3.00 sits between
# Nebius committed ($2.50) and list on-demand ($3.85), covering a realistic
# on-demand/preemptible mix; 50% is close to the 48% mean/peak this model's
# own traffic implies for a single-model deployment sized for peak.
COST_BASES["agreed-2026-09"] = (3.00, "agreed basis: on-demand/preemptible mix")
DEFAULT_BASIS = "agreed-2026-09"
DEFAULT_UTILISATION = 0.50
DEFAULT_MARGIN = 0.0                 # report break-even; margin is stated separately
















MIN_R2 = 0.90










def effective_in(price_in: float, price_cached: float, hit_rate: float) -> float:
    """Blended input price at a given cache hit rate -- the marketplace metric."""
    return hit_rate * price_cached + (1.0 - hit_rate) * price_in




# ── utilisation from burstiness, instead of an assumed constant ──────────





# ── pricing without decomposition: hit rate as an outcome, not a control ──

@dataclass(frozen=True)
class DirectPrice:
    """Prices read straight off measured totals, with no regression.

    `effective_in` here is what we would actually charge, at the cache hit rate
    our system actually achieved -- not re-blended to match a competitor.
    """
    effective_in_per_m: float
    out_per_m: float
    hit_rate: float
    input_tokens: float
    output_tokens: float
    gpu_seconds_input: float
    gpu_seconds_output: float
    basis: str
    usd_per_gpu_hour: float
    utilization: float
    margin: float

    @property
    def bill_per_1k_requests(self) -> float:
        raise NotImplementedError("needs a request count; use bill_per_request")

    def bill_per_request(self, in_tok: float, out_tok: float) -> float:
        return (self.effective_in_per_m * in_tok
                + self.out_per_m * out_tok) / 1e6


def price_direct(gpu_seconds_input: float, gpu_seconds_output: float,
                 input_tokens: float, output_tokens: float,
                 cached_tokens: float,
                 basis: str = DEFAULT_BASIS,
                 utilization: float = DEFAULT_UTILISATION,
                 margin: float = DEFAULT_MARGIN) -> DirectPrice:
    """Effective input price at the hit rate the system actually achieved.

    **Cache hit rate is an outcome we optimise, not a control we hold fixed.**
    Providers on the same marketplace traffic realise anywhere from 0.0% to
    87.4%, so it is overwhelmingly a property of the serving system -- caching
    well *is* serving well. Re-blending our price to a competitor's hit rate
    would normalise away exactly the thing we are trying to improve.

    That also makes the decomposition unnecessary. Splitting input cost into
    cached and uncached is required only to re-blend at somebody else's hit
    rate; at our own,

        effective input price = input GPU-seconds / input tokens x rate / util

    is directly measurable. No mixes, no NNLS, no identifiability gate, and no
    per-sequence-versus-per-context confound.

    The price of that simplicity: the hit rate becomes load-bearing, so it has
    to be earned on representative traffic rather than normalised away.

    `gpu_seconds_input` / `gpu_seconds_output` come from the device timer's
    phase split (`SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`); wall-clock GPU
    seconds are not a valid substitute below saturation, because idle time
    would be charged to whatever tokens happened to flow.
    """
    if basis not in COST_BASES:
        raise KeyError(f"unknown cost basis {basis!r}; have {sorted(COST_BASES)}")
    if not 0 < utilization <= 1:
        raise ValueError(f"utilisation must be in (0, 1], got {utilization}")
    if input_tokens <= 0 or output_tokens <= 0:
        raise ValueError("need positive token counts")
    usd_hr, _ = COST_BASES[basis]
    per_s = usd_hr / 3600.0
    scale = per_s / utilization * (1.0 + margin) * 1e6
    return DirectPrice(
        effective_in_per_m=gpu_seconds_input / input_tokens * scale,
        out_per_m=gpu_seconds_output / output_tokens * scale,
        hit_rate=cached_tokens / input_tokens,
        input_tokens=input_tokens, output_tokens=output_tokens,
        gpu_seconds_input=gpu_seconds_input,
        gpu_seconds_output=gpu_seconds_output,
        basis=basis, usd_per_gpu_hour=usd_hr,
        utilization=utilization, margin=margin)


# ── the gate: refuse to price a run that cannot be priced ────────────────

def gpu_seconds_per_request(gpu_seconds_input: float, gpu_seconds_output: float,
                            input_tokens: float, output_tokens: float,
                            in_per_request: float, out_per_request: float) -> float:
    """Forward GPU-seconds for one *market-sized* request.

    The bridge between a measured level -- whose requests are whatever length
    the trace produced -- and the market's average request. Everything in
    `price.market.Economics` derives from this one number.
    """
    if input_tokens <= 0 or output_tokens <= 0:
        raise ValueError("need positive token counts")
    return (gpu_seconds_input / input_tokens * in_per_request
            + gpu_seconds_output / output_tokens * out_per_request)


def usable(level: dict, n_gpu: int = 1) -> tuple[bool, str]:
    """Can this level be priced at all? Returns (ok, reason-if-not).

    Five earlier attempts at per-token cost all produced *plausible-looking*
    numbers from unusable data (HANDOFF SS4), which is worse than producing
    none. An automated caller cannot sanity-check a price, so refusing has to
    be the default rather than a warning.

    The direct method's failure modes are different from the regression's, so
    the checks are too:

      * the device timer was off, so the phase split is missing entirely;
      * forward time exceeds wall time x n_gpu, which is physically impossible
        and means the counter is aggregated differently than assumed (this is
        exactly the bug that doubled every output price once);
      * the GPU was mostly idle, so forward time is a tiny slice of a window
        and the level is dominated by whatever else was happening;
      * no tokens flowed.
    """
    c = level.get("server_counters") or {}
    ext = c.get("sglang:forward_execution_seconds_total[extend]")
    dec = c.get("sglang:forward_execution_seconds_total[decode]")
    if ext is None or dec is None:
        return False, ("no phase-split forward time; SGLANG_ENABLE_METRICS_DEVICE_TIMER "
                       "was not set, so the counter is declared but never incremented")
    if not level.get("prompt_tokens") or not level.get("output_tokens"):
        return False, "no tokens recorded for this level"
    wall = level.get("wall_s") or 0.0
    if wall > 0:
        busy = (ext + dec) / (wall * max(1, n_gpu))
        if busy > 1.02:
            return False, (f"forward time is {busy:.2f}x wall x n_gpu, which is "
                           "impossible; the counter is not aggregated as assumed")
        if busy < 0.05:
            return False, (f"GPU busy only {busy:.1%} of the window; forward time "
                           "is too small a slice of this level to price from")
    return True, ""
