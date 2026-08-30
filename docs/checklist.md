# Build checklist

Three tracks: the **objective** (what we hillclimb), the **harness** (what does
the climbing), and **cleanup** (where things live). Track 1 gates the others —
an autoresearch loop optimising the wrong number is worse than no loop.

---

# 1. Objective: effective input price on OpenRouter

## The goal, restated

A list of reproducible diffs to SGLang's `srt/` that put us in the **top 5–10
effective input price** for `qwen/qwen3.8-27b`, serving a coding workload with a
high cache hit rate, while meeting marketplace SLOs.

## What the live market actually looks like

11 providers, fetched 2026-08-30:

| | raw input $/M | cache read $/M | eff @ 90% hit |
|---|---|---|---|
| best | 0.250 | 0.025 | 0.0475 |
| worst | 0.480 | 0.250 | 0.2730 |
| **spread** | **1.9x** | **10.0x** | **5.7x** |

**Cache read price is the competition, not input price.** At 90% hit rate the
ranking reorders: Cloudflare climbs 9 -> 5, Phala falls 5 -> 8. Raw input price
is nearly undifferentiated.

This means the optimisation target is narrower and sharper than "make serving
faster": **make a cache hit cheap**. Which collides with something already
measured on this stack —

> A full prefix-cache hit is **1.57x slower** than a cold prefill
> (115ms vs 73ms on a 1000-token prompt, CV ~1%). A partial hit is 1.76x
> *faster*. See handoff.md, 2026-08-29.

If serving a cached token costs nearly what an uncached one costs, we cannot
price cache reads at a tenth of input. That anomaly is now the highest-value
target we have, and it was found by accident rather than by looking.

## Methodology for estimating our price

Five steps. Each is measurable; none requires guessing at competitors' costs.

- [ ] **Step 1 — find the SLO frontier.** Sweep concurrency N. For each N
      record goodput and p99 TTFT / p99 TPOT. `N*` is the largest N meeting
      **TTFT < 1s** (stretch: 0.5s) and **TPOT < 50ms**. This is the "rightmost
      point on the curve" — we already have the machinery (`staircase`,
      `detect_collapse`). *Caveat established the hard way: near saturation the
      server is metastable, so `N*` must be the largest N that holds SLOs
      **reliably across repeats**, not in one lucky run.*

- [ ] **Step 2 — attribute GPU-seconds to token classes.** One workload gives
      one equation and three unknowns, so regress across workloads with
      deliberately different mixes:

          gpu_seconds = a * uncached_in + b * cached_in + c * out_tokens

      `prefill_heavy`, `decode_heavy`, `prefix_heavy` and `short_chat` already
      span this space; solve by non-negative least squares. `b/a` is the
      quantity the whole business case rests on — how much cheaper a cached
      token really is — and today it is close to 1, which is the problem.

- [ ] **Step 3 — convert to cost.** `cost_x = coefficient * ($/gpu-hour / 3600)`.

- [ ] **Step 4 — apply utilisation and margin.** `price = cost / utilisation *
      (1 + margin)`. Utilisation dominates: at 40% vs 80% the price doubles for
      an identical stack. Treat it as an explicit input, never a hidden default.

- [ ] **Step 5 — effective input price.** `eff_in = h * price_cached +
      (1 - h) * price_in`, with `h` measured from the real coding workload
      (see the OpenHands capture track), not assumed.

- [ ] **Step 6 — rank against the live table.** Refetch OpenRouter endpoints and
      report our position. Ranking is the score; the dollar figure alone is not.

## Two risks that would invalidate the number

- [ ] **Modal retail pricing is not a provider's cost basis.** $3.95/hr for an
      H100 is serverless retail; a provider on reserved or owned hardware pays
      far less. So **optimise the hardware-independent quantity — SLO-constrained
      tokens per GPU-second — and treat $/hr as a reporting parameter.** Report
      price under several cost assumptions rather than one.
- [ ] **Cache hit rate must come from the real workload.** `h` moves the answer
      more than most stack changes do. Measure it from captured agent traffic.

## Target model

- [ ] Decide whether to serve `qwen/qwen3.8-27b` itself, or develop on a
      cheaper proxy and port. All current calibration is for Qwen3-30B-A3B on
      H100 and Qwen3-4B on L40S; neither transfers.
- [ ] Confirm the licence permits commercial serving.
- [ ] Re-derive the SLO frontier for the real model. Every rate in the suite is
      hardware- and model-specific.

---

# 2. Harness architecture

## The services you listed

- [ ] **MemoryAPI / MemoryDbService** — temporal + relational experiment store.
- [ ] **ContextManagerService** — full JSONL agent traces, implementation detail.
- [ ] **AgentOrchestrationService** — ~10 parallel agent loops.
- [ ] **AgentService** — one loop: seeding, divergence control, retries.

## The constraint that should shape all of it

**GPUs are the scarce resource, not agents.** An experiment costs 90–505s of
model load plus trace time — call it 10–30 minutes — and Modal's plan caps GPU
concurrency at 10 (an 8xH100 run consumes 8 of those). So:

- 10 agents does **not** mean 10 concurrent experiments. It probably means 1–3.
- An agent's iteration latency is dominated by queue wait, not by thinking.
- Read-every-5-min / write-every-30-min is comfortably fast enough. Memory is
  not the bottleneck and should not be optimised as if it were.

## Services I would add

- [ ] **ExperimentQueue (the important one).** Central GPU scheduler: priority,
      admission, and **deduplication before spending**. With 10 agents,
      overlapping proposals are certain. Key every experiment on
      `(serving_config_digest, overlay_digest, workload_digest)` and return the
      cached result instead of re-running — the ledger already computes all
      three. Agents must be written to tolerate waiting.

- [ ] **VerificationService, as a separate authority.** Correctness gating must
      not live inside AgentService, or an agent can learn to skip it. A search
      maximising goodput can "win" by dropping slow requests — we have already
      simulated exactly that and the ledger caught it (+57% goodput, canaries at
      0.35, 310 requests dropped). Verification should be the thing that decides
      whether a result is admissible at all.

- [ ] **BudgetGovernor.** Hard per-agent and per-run caps. Ten agents times GPU
      experiments is the difference between $50 and $5000. Should be able to
      pause the whole fleet, not just decline one job.

- [ ] **ScoreService.** One authoritative implementation of the objective from
      Track 1. If each agent computes its own score, they optimise subtly
      different things and results stop being comparable.

## Questions worth settling before writing code

- [ ] What does an agent actually emit — a config, a patch to `srt/`, or both?
      The overlay mechanism already supports arbitrary Python edits.
- [ ] Who owns the baseline? A moving baseline makes the ledger's progress curve
      meaningless.
- [ ] How does an agent learn from a *failed* experiment? Most will fail; the
      failures carry most of the information.
- [ ] What stops ten agents converging on the same idea? Diversity has to be
      enforced somewhere — the ledger measures it but does not act on it.

---

# 3. Cleanup and layout

- [ ] Move to `archive/`: `test.py` (Modal hello-world), and anything the
      research loop will not call.
- [ ] Keep but mark as unrun: `virtual_users.py` (realism track), `suite_8x`.
- [ ] Decide the retention policy **before** generating artifacts. A run writes
      per-request records; at 30k requests that is ~13MB per suite run, and ten
      agents iterating will produce gigabytes.

Proposed layout:

    runs/
      <run_id>/                      one orchestration run
        ledger.jsonl                 append-only experiment record
        score.json                   objective definition in force
        agents/
          <agent_id>/
            trace.jsonl              ContextManagerService
            experiments/<exp_id>/
              config.json
              overlay.diff
              result.json            metrics, no per-request rows
              per_request.parquet    bulk rows, separately prunable
    overlays/                        current working overlay set
    artifacts/<sha256>/              content-addressed diffs, deduped

Content-addressing the diffs matters: ten agents will produce near-identical
patches, and the same digest already keys the result cache.

---

# Order of work

1. **Step 1–2 of the objective** (SLO frontier, GPU-second attribution). Nothing
   downstream means anything without a trustworthy score.
2. **The cached-token cost gap.** `b/a` near 1 is both the biggest measured
   anomaly and the thing the market prices at 10x. Best first experiment.
3. **ExperimentQueue + VerificationService.** The two pieces the loop cannot
   safely run without.
4. **Everything else in the orchestration design.**
