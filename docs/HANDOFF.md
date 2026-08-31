# Handoff — resume here

Written 2026-08-31. Companion to `handoff.md` (the running log) and
`docs/checklist.md` (the plan). This file is what a fresh session needs to
pick up without re-deriving anything.

---

## 0. Spend — read first

    metered $95.65   billed $64.09   free credits exhausted

The free $30 ran out mid-session. Everything since is real money, and 8xH100
runs at **$31.60/hr** dominate: three attempts (two of which failed in their
first minutes) plus one successful ~20-minute run account for most of the
$64. **Check `make spend` before and after any 8-GPU work.** The daily digest
emails from `onboarding@resend.dev` land in spam.

---

## 1. The objective

Produce reproducible diffs to SGLang's `srt/` that reach a **top 5-10
effective input price** for `qwen/qwen3.8-27b` on OpenRouter, serving
coding-agent traffic, while meeting marketplace SLOs.

**Effective input price** is the metric because for high-cache-hit traffic it
is what competition runs on:

| | listed input | cache read | spread |
|---|---|---|---|
| across 11 providers | $0.35–$0.48 | $0.025–$0.25 | **1.9x vs 10x** |

Cache-read price varies 10x and reorders the leaderboard; listed input barely
varies. So the objective reduces largely to *serve cached tokens cheaply and
hit the cache often*.

    eff_in = hit_rate * cache_read_price + (1 - hit_rate) * listed_input_price

**This formula is verified**, not assumed: OpenRouter publishes both inputs and
realised effective prices, and it reproduces theirs to <0.1% on 9 of 11
providers (`tests/test_market.py`).

### The chain, and what is trustworthy in it

    1. measure GPU-seconds per token class   <- our harness, HARD (see §4)
    2. x $/GPU-hr, / utilisation, x margin   <- assumptions, unverifiable
    3. blend by cache hit rate               <- VERIFIED against OpenRouter
    4. rank against competitors              <- the score

Step 2 can never be validated internally: utilisation depends on traffic a
marketplace sends us. **So hillclimb the hardware-independent quantity —
SLO-constrained GPU-seconds per token — and treat price as a reporting layer.**

---

## 2. Ground truth we have

**OpenRouter model page** (`MARKET_QWEN38_27B`, `MARKET_REALISED` in
`modal_app.py`). Gives listed prices, realised effective prices, and each
provider's actual cache hit rate.

- Best realised effective input: **Novita $0.1272** at 87.4% hit (2026-08-31).
  **This moves.** Two days earlier it was Chutes at $0.1310/69.5% hit; Chutes'
  hit fell to 65.4% and Novita's rose to 87.4%, and the leader changed. Listed
  prices did not move at all — verified against
  `/api/v1/models/qwen/qwen3.8-27b/endpoints`. Both snapshots are kept in
  `MARKET_SNAPSHOTS`; any rank claim needs its date.
- Weighted average actually paid: $0.2116 in / $2.868 out.
- **Hit rate is a serving-system property**: same model, same traffic, Novita
  realises 81.8% while Venice and Cloudflare realise **0.0%**. Failing to
  implement prompt caching costs them 3x on effective price.
- Real provider latency: TTFT 0.55–2.64s, 36–138 tok/s. A 1s TTFT SLO is
  stricter than most of them meet; 2s is realistic.

**TraceLab** (`UW-SyFI/TraceLab`, CC-BY-4.0, 665k real Claude Code / Codex
invocations). Attribution required; `tracelab.CITATION` carries it.

|  | our synthetic | TraceLab |
|---|---|---|
| median input tokens | 537 | **132,092** |
| input:output ratio | 2.3 | **291:1** |
| cache hit rate | 0.96 | 0.956 |

Our workloads were wrong by ~250x on input length. Output length was right,
which is exactly why the ratio was so far off. Replay via
`--trace-scale`; `scale_sessions()` shrinks contexts while preserving hit rate
and in:out ratio, because full-size contexts often do not fit.

---

## 3. What is measured and reproducible

**Qwen3-4B on L40S** (dev model, cheap iteration):

- N* = 128 concurrent users, goodput ~7.2-7.7 rps. **Reproduced three times.**
- Noise floor: goodput CV **0.0003** over 5 separate launches. Effects below
  1% are detectable.
- Canary floor: 6/6 exact — outputs bitwise reproducible.
- Metastable collapse at ~88% utilisation: two identical runs gave goodput
  30.29 and 0.54. `metrics.detect_collapse` measures it rather than averaging
  it away.
- cache_discount **0.197** (fit 0.959, condition 3.4) — a cached token costs
  ~a fifth of an uncached one. Roofline agrees independently: measured rates
  are 40-44% of ceiling.

**Qwen3.8-27B on 8xH100 TP=8/EP=8, full-scale TraceLab** (run
`1788189825`, the first trustworthy one):

- N* = 4 (TTFT-bound at 2s), goodput 0.79 rps, TPOT 8.7ms, cache hit 0.88.
- TP=8 fixed decode entirely: TPOT 54ms on 1 GPU -> 8.7ms on 8.
- **TTFT is the binding constraint**, and it is what caps N* at 4. At 88% hit
  on a 132k context you still prefill ~16k new tokens per turn. **Raising N*
  by improving TTFT is the clearest optimisation target.**
- Cost attribution, all three classes identified (spans 472x / 348x / 35x,
  fit 0.9901, condition 1.3):

      uncached input  1.403e-04 GPU-s/token  -> $0.0974/M raw
      cached input    4.487e-05              -> $0.0312/M raw
      output          2.331e-03              -> $1.6188/M raw

  **cache_discount = 0.320** — a cached token costs ~a third of an uncached
  one. Cheaper, as the thesis requires, but less so than Qwen3-4B's 0.197 and
  well above the ~0.10 the market prices cache reads at. The linear-attention layers keep fixed-size
  state rather than a growing cache, so only 16 of 64 layers benefit from a
  prefix hit at all — a plausible mechanism, still unverified.

- **The cache discount is the whole objective.** To match the market at a
  realistic 40% utilisation on real coding traffic we need ~**0.12**; we
  measure **0.320**. Utilisation scales all token classes equally and cannot
  fix a ratio. Caveat: our 0.88 hit rate came from replaying a trace
  back-to-back, so it assumes traffic we do not yet have — real providers'
  hit rates track their traffic volume (§3b).

### 3c. OUR WORKLOAD AND SLO DO NOT MATCH THE MARKET — read before trusting §3

`scripts/market_pull.py` pulls OpenRouter's real numbers (they are streamed as
a Next.js RSC payload; request the page with an `RSC: 1` header — no JSON API
exposes them). Snapshot in `data/market-qwen-qwen3.8-27b.json`. Two findings
invalidate the measurements above.

**1. Our SLO is far stricter than any provider meets.**

    market p99 TTFT   best 3844ms (Parasail)   median 13225ms   worst 62475ms
    our SLO           2000ms

Novita — *today's cheapest effective price* — runs p99 TTFT of 52-56s and p50
of 3.8s. We capped N* at 4 using a 2s p99 that nobody on the board delivers.
Our 16-user level measured p99 3294ms, which would beat 10 of 11 providers,
at ~1.9x the throughput. **N* = 4 is an artefact of the SLO, not the hardware.**

**2. Our replayed workload is wrong on every axis.**

    |                    | our replay | real market | factor |
    |--------------------|-----------:|------------:|-------:|
    | cache hit rate     |      0.956 |       0.394 |   2.4x |
    | input tokens/req   |    132,092 |      20,583 |   6.4x |
    | output tokens/req  |        454 |       2,076 |   0.2x |
    | input:output       |      291:1 |       9.9:1 |  29.3x |

TraceLab is Claude Code traffic; OpenRouter's traffic for this model is a mix
(pi, Hermes Agent, Claude Code, DeepSeek Harness, LangChain) with far more
output. We over-corrected: the old synthetic workload was 250x too *short*,
TraceLab is 6.4x too *long*.

**Consequence — the optimisation target inverts again.** Share of modelled cost:

    traffic model                    uncached in   cached in   output
    our replay (132k/454, hit .88)          26%         61%      12%
    real market (20.6k/2.1k, hit .87)        6%         13%      81%

**Output tokens are 70-81% of the bill on real traffic, not 12%.** The cache
discount — called "the whole game" one section above — is 5-13%. And the output
coefficient is our *least identified* column (35x span vs 472x/348x).

**3. Decode runs ~10x off the memory-bandwidth roofline.** An earlier version
of this section claimed the output coefficient was "measured at the wrong
context" because 2.331e-03 matches the 132k-context roofline. **That was
wrong** — the decode mix that identifies the output column used a *64-token*
prompt, so the match was a coincidence. Checked properly:

    context   batch   roofline GPU-s/tok   vs measured 2.331e-03
      1,000      32             2.35e-04                    9.9x
      1,000       3             2.32e-03                    1.0x
     20,583      32             6.18e-04                    3.8x

At the decode mix's own context and its nominal batch of 32, roofline says
2.4e-04 and we measured ten times that — a figure consistent with an
**effective decode batch of ~3, not 32**. Either the scheduler is not batching
decode, or the mix never reached the concurrency it asked for.

That is the single highest-value thing to find out, because output tokens are
70-81% of the bill on real traffic. `server_metrics.BatchSampler` now samples
`num_running_reqs` every 2s during every measured window (one local HTTP GET,
unlike the nvidia-smi polling that starved the load generator), and every level
and mix records `batch`. The phase-B mixes now hold context fixed at the
marketplace's ~20k and vary only output length, so decode is measured in the
regime the marketplace actually runs.

**Next run must be: market-realistic workload (20.6k in / 2.1k out / 9.9:1) at
a market-realistic SLO (p99 TTFT 4s), then re-attribute.** Every cost and rank
in §3 was produced under a workload the market does not send and an SLO no
competitor meets.

### 3b. Hit rate may be driven by traffic, not only by the serving stack

Across the two snapshots 7 of 9 providers moved as price-driven routing would
predict (cheaper -> more share; correlation -0.80). But listed prices never
moved, so effective price moved *only* via hit rate — and the likelier causal
direction is the reverse: more traffic keeps prefixes warm, which raises hit
rate, which lowers effective price. The two are observationally identical here.

If that is right, a new entrant starts cold: thin traffic -> cold cache -> high
effective price -> less routing. **Measure hit rate as a function of arrival
rate** rather than at one replay density; that turns 0.88 into a curve, which
is what an entry-price estimate actually needs.

- Sanity: market *listed* input ($0.35-$0.48/M) is 3.6-4.9x our raw cost,
  which is the plausible size of a utilisation-plus-margin gap.

- **The binding constraint is the cache discount, not utilisation.**
  Scoring the buyer's whole bill (132k in / 454 out) against all 11 providers:

      hit rate      util 50%      util 60%      util 70%      util 80%
      0.695 (Chutes)  2/12          1/12          1/12          1/12
      0.956 (real)    6/12          6/12          2/12          2/12

  At 50% utilisation we are **never** first, and on real coding traffic
  (TraceLab, 95.6% reuse) we cost **1.70x** the leader. Earlier drafts of this
  doc claimed 60% utilisation beat every provider; that holds only at moderate
  hit rates and is **retracted** for the high-hit regime that matters.

  Cause: our cached/uncached cost ratio is **0.320**; Chutes prices cache reads
  at **0.100** of listed. At 95.6% hit nearly every token is cached, so that
  ratio dominates. Utilisation scales all three token classes equally and
  cannot fix a ratio. **Therefore cached-token cost outranks TTFT as the
  optimisation target** — TTFT work raises N*, which raises capacity, which
  *lowers* utilisation when we are demand-limited (§3a).

- Caveat both ways: $0.035 is Chutes' *price*, not their cost, and OpenRouter's
  leaderboard sorts on effective input price, so pricing cache reads near zero
  is the cheapest way to top the list. We must beat the price regardless.

### 3a. Market size — we would be demand-limited, not capacity-limited

    our capacity     9.0B input tokens/day per 8xH100 node
    entire market   17.6B input tokens/day (OpenRouter activity, 1 day)
    -> the whole market for this model is 2.0 nodes

Capturing today's *largest* provider's share (Phala, 20.6%) yields only **40%
utilisation** on one node. Routing is not greedy on price — correlation between
effective price and token share is **-0.71**, the cheapest provider holds 9.5%
while one at 1.74x the price holds 20.6%, and Venice/Cloudflare hold share at
$0.45 with 0% cache hit. So share cannot be bought with price.

Consequence: incumbents multiplex dozens of models over one fleet and stay busy;
a single-model dedicated node cannot. If demand-limited, cost/token =
node_cost / demand, so **the smallest SLO-meeting deployment beats the fastest
one** — 4xH100 at half the cost would halve $/token at fixed demand. Untested.

- Volume: 100B input tokens/week is 1.25 req/s, about 1.6 nodes of 8xH100,
  roughly $23k/month of compute at $2.50/GPU-hr.

---

## 4. Measuring per-token cost: five failed attempts, and why

This is the hardest part of the harness and the most likely thing to get wrong
again. All five failures produced *plausible-looking* output.

1. **Wall time as denominator.** Every level ran a fixed 90s, so GPU-seconds
   was near-constant while token counts varied 10x. r2 = -2.9 — worse than
   predicting the mean — yet it printed `$0.11/M input`, entirely reasonable
   and meaningless.
2. **`sglang:forward_execution_seconds_total`.** Declared as a Counter in the
   SGLang source, never emitted, even with
   `--enable-metrics-for-all-schedulers`. Returned zeros.
3. **`nvidia-smi utilization.gpu`.** Reports *kernel residency*, not work: read
   0.99 at N=4 and 0.95 at N=32 while doing 5x the work. Worse, polling it
   4x/second starved the load generator and moved N* from 128 to 32 — the
   instrument changed what it measured, worst at high load.
4. **Fixed-duration saturated mixes.** Still constant GPU-seconds by
   construction. r2 = -36.7.
5. **Rate form (current).** Divide the cost equation by GPU-seconds and
   duration cancels: `1 = a*(U/s) + b*(C/s) + c*(O/s)`. This works —
   `attribute_saturated()`.

**The lesson**: (1)–(4) looked like four different problems and were one — the
experimental design, not the denominator. Fixed-duration experiments cannot
identify per-token costs however you measure the denominator.

**`usable()` gates every result on fit quality and refuses to print a price
from a bad fit.** Keep that. It caught all four failures; without it each would
have produced a confident wrong number.

---

## 5. Open problems — start here

### 5a. Phase-B mixes did not span the space — FIXED, needs re-running

The 8xH100 attribution returned `cache_discount = 1.52` (cached tokens cost
*more* than uncached) with fit 0.95 and condition 5.4 — passing every check.
**It is an artefact.** The mix compositions:

    mix              unc/s   cach/s   out/s    hit
    prefill_heavy    945.4   1049.8   107.7   0.53   <- 53% cached!
    decode_heavy      68.1    132.0   166.2   0.66
    cache_heavy      155.4    965.6   132.2   0.86
    balanced         195.1    526.9   136.4   0.73

Two defects:

- **No genuinely uncached observation.** `run_concurrent_users` loops users
  through a 600-session pool; within a 120s window sessions repeat, so prompts
  are cached by *repetition* rather than by conversation structure. Even
  `prefill_heavy` is 53% cached.
- **Output rate barely varies** (108–166, 1.5x). That column is nearly
  constant, so the output coefficient is unidentified — hence the absurd
  5.9e-3 GPU-s per output token.

**Fixed 2026-08-31.** `pricing.identifiability()` requires each token class to
vary >=3x across mixes, and `usable()` refuses the result otherwise; it rejects
the 8xH100 run and accepts the 4B one (`tests/test_identifiability.py`). The
mixes are rebuilt to span: `uncached` generates a fresh prompt per request so
nothing caches by repetition, and output lengths differ 250x across mixes.
Reported in the run log too, so a degenerate design shows up while the GPUs are
still warm.

**Still to do: re-run the 8xH100 attribution with the new mixes (~$10).** The
`cache_discount = 1.52` result is retracted; there is currently no trustworthy
per-token cost for the target model.

### 5b. Known-wrong or unverified

- `prices()` multiplied the hourly rate by `n_gpu` while `gpu_seconds` already
  aggregated the node — every 8xH100 price was **8x too high**. Fixed, with a
  regression test, but any earlier recorded price is wrong.
- 1xH100 fails the SLO at N=2 (TPOT 54ms). Measured and real. My *explanation*
  ("dense weights force a 7ms floor") was wrong — see §6 — so **why** remains
  open.
- `virtual_users.py` (LLM-driven realism track) has never been run.
- The OpenHands gateway is deployed and captures traces, but no real agent
  session has been recorded through it.

---

## 6. Corrections — things stated confidently and wrongly

Recorded because they were all presented as findings before being checked.

- **Qwen3.8-27B is dense, and this entry previously said the opposite.** The
  first version claimed dense-with-KV-on-all-64-layers; the correction claimed
  hybrid MoE; the config says **`has_moe: false`, `intermediate_size: 17408`**
  — dense FFN, hybrid *attention*. What survives from the correction is the KV
  half: `layer_types` gives full attention every 4th layer, so **only 16 of 64
  layers hold growing KV**, KV is **64 KiB/token not 256**, and one 132k
  conversation is 8.66 GB. What was wrong: 128 experts x 512 implies a **64B**
  model, more than twice the 27B on the tin, and it underestimated FFN FLOPs
  ~4.3x. `--ep-size` is meaningless on this model and we passed `--ep 8`.
  Empirical measurements are unaffected; `docs/capacity.md` roofline numbers
  for this model are not. `tests/test_model_specs.py` now checks every spec
  against the parameter count in its own name.
- **`ServingConfig.n_gpu` allocates nothing** — it only sets SGLang's
  `--tp-size`. Modal resources must be overridden at call time
  (`.with_options`). Requesting TP=8 without it died with "invalid device
  ordinal".
- **MoE + FP8 has a block constraint**: `moe_intermediate / (tp/ep)` must be a
  multiple of 128. `ServingConfig.validate()` now refuses invalid combinations
  before spawning, and reports which `ep_size` values are legal.

---

## 7. Commands

    make test                 # 118 local tests, no GPU
    make suite                # eval suite on the dev model
    make noise                # noise floor
    make ledger               # research-loop health
    make spend                # current spend

    # frontier + cost attribution + price, spawned so it outlives the terminal
    uv run python scripts/launch.py frontier \
      --model Qwen/Qwen3.8-27B-FP8 --gpu H100 --n-gpu 8 --ep 8 \
      --levels 4,8,16,32,64 --seconds 120 --trace-scale 1.0 \
      --ttft-ms 2000 --tpot-ms 50
    uv run python scripts/launch.py collect <call_id>

    uv run modal run scripts/results.py::ls
    uv run modal run src/autoinf/modal_app.py::traces    # agent captures

**Always `.spawn()` for long runs** (`launch.py` does). `modal run --detach`
keeps the app alive but a `local_entrypoint`'s in-flight call still dies with
the client — that cancelled a sweep three levels in.

---

## 8. Assets

- Modal Volumes: `auto-inference-hf-cache` (Qwen3-4B, Qwen3-30B-A3B,
  Qwen3.8-27B-FP8 30.9GB, Qwen3-235B-A22B 236GB — **~$24/month**, delete what
  is idle), `auto-inference-results`.
- Secrets: `huggingface`, `auto-inference-resend`, `auto-inference-anthropic`,
  `auto-inference-gateway`, `auto-inference-modal-token`.
- Deployed: `auto-inference` (frontier, bench, agent_endpoint at
  `https://mpeng19--auto-inference-agent-endpoint.modal.run`),
  `auto-inference-spend-monitor` (daily 16:00 UTC).

---

## 9. Next steps

1. **Fix the mix design (§5a).** Nothing about the objective is trustworthy
   until per-token costs are identified. Needs unique prompts for the uncached
   observation, genuinely varied output lengths, and a per-column
   identifiability check to replace the too-lenient condition number.
2. **Re-run the 8xH100 attribution** once mixes span the space. ~$10.
3. **Capture real OpenHands traffic** through the deployed gateway and compare
   its `prefix_reuse_frac` against TraceLab's 95.6%.
4. **Then** the first optimisation experiment. TTFT is the binding constraint
   on the target model, and 88% of a 132k context is cached, so prefix-cache
   and prefill-scheduling work is where the headroom is.
5. The research-loop architecture (`docs/checklist.md` §2) — still awaiting
   the system design.
