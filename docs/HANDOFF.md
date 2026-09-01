# Handoff — resume here

Written 2026-08-31. **The** handoff document — a second one at the repo
root was retired into the appendix below on 2026-08-31. Companions:
`docs/methodology.md` (how the simulator works and what it assumes) and
`docs/checklist.md` (the plan). This file is what a fresh session needs
to pick up without re-deriving anything.

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

**RULE: all serving-physics analysis is on `Qwen/Qwen3.8-27B-FP8`.** The GPU
may change when there is a reason (cost, or to vary a parameter); the model may
not. Cost per token varies with batch only through the weight read, and a 4B
has 4 GB of weights against the target's 23 GB — so batching, cache behaviour
and the cost model itself live in a different regime there. The dev model is
for *harness* validation only: noise floors, determinism, load-generator
behaviour. `simulator.TARGET_MODEL` and `test_calibration_is_target_model_only`
enforce this.

**Qwen3-4B on L40S** (dev model — harness validation ONLY, not serving physics):

- N* = 128 concurrent users, goodput ~7.2-7.7 rps. **Reproduced three times.**
- Noise floor: goodput CV **0.0003** over 5 separate launches. Effects below
  1% are detectable.
- Canary floor: 6/6 exact — outputs bitwise reproducible.
- Metastable collapse at ~88% utilisation: two identical runs gave goodput
  30.29 and 0.54. `metrics.detect_collapse` measures it rather than averaging
  it away.
- cache_discount 0.197 was also measured here. **Do not compare it to the
  target's.** Cache discount varies 0.101 -> 0.324 across TP *on the target
  model alone* (§3e), so a cross-model comparison carries no information.

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
  one at TP=8, and **0.101 at TP=1** (§3e), against the ~0.10 the market prices
  cache reads at. An earlier version compared this to the 4B dev model's 0.197;
  that comparison was meaningless and is withdrawn. The linear-attention layers keep fixed-size
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

### 3e. GPU-count sweep — 8xH100 was the wrong deployment

Same market workload and SLO at TP=2/4/8 (runs `1788206950`, `1788206719`,
`1788203369`). **Fewer GPUs are cheaper per token, not more.**

    GPUs  cache disc  goodput  rps/GPU  $/1k req @60% util  break-even util
       2       0.127     0.57    0.285                4.55             27%
       4       0.222     0.92    0.230                5.22             31%
       8       0.324     1.58    0.198                8.72             52%

**2xH100 is 1.9x cheaper per request than 8xH100** and 44% more efficient per
GPU. Break-even utilisation falls from 52% to **27%**.

And that reverses §3a's conclusion. Capacity vs the market's 17.9B tokens/day:

    2 GPU  $5.00/hr   1.0B tok/day   break-even needs ~1.5% market share
    8 GPU  $20.00/hr  2.8B tok/day   break-even needs ~8.2% market share

§3a said we would be demand-limited because the market is only ~2 nodes of
8xH100. At the right node size the market is ample: **1.5% share fills a 2-GPU
node to break-even**, against a largest provider holding 15.9%.

**The cache-discount problem largely dissolves too.** 0.127 at TP=2 against the
market's ~0.10 — versus 0.324 at TP=8. An earlier section called the cache
discount "the whole game" and "the strongest result we have, replicating to 1%".
**That replication was across two 8xH100 runs only.** Across TP sizes it varies
2.5x, so it is a property of the deployment, not of the serving stack alone.

### 3f. The simulator, and its measured fidelity

`src/autoinf/simulator.py` predicts decode cost from two parameters: the
fraction of HBM bandwidth decode realises and the fraction of peak FLOPs
prefill realises (calibrated 0.277 and 0.312 on the 8xH100 run). Two constants
cannot memorise four configurations, so `validate_loo` -- calibrate on the
others, predict the held-out one -- is a real test.

**It fails, informatively. Mean absolute error 27%, worst 49%:**

    held out   calibrated on   predicted    measured   error
          2g          [4, 8]   1.521e-03   1.017e-03    +50%
          4g          [2, 8]   1.201e-03   1.042e-03    +15%
          8g          [2, 4]   9.248e-04   1.775e-03    -48%

Cause: the model assumes tensor parallelism divides bytes and bandwidth
equally, so cost per token is TP-invariant. Measured decode efficiency instead
**falls by half** as TP grows -- 0.596 at TP=2, 0.510 at TP=4, 0.288 at TP=8.

The stronger empirical regularity is that **decode step time is ~21ms
regardless of batch or GPU count** (20.6 / 20.3 / 21.5ms at batches 40.6 / 78.1
/ 96.7). We are not bandwidth-bound: at TP=8 roofline says 6.2ms and we take
21.5ms. Fitting step = roofline + overhead gives overhead 8.2 / 9.9 / 15.3ms at
TP 2/4/8 -- growing with TP, consistent with per-layer synchronisation across
64 layers rather than bandwidth (all-reduce volume at TP=8 is ~111MB/step,
which NVLink moves in 0.12ms).

**Resolved by the TP=1 run (`1788207351`).** Decode step time is ~constant
across the whole sweep:

    GPUs  batch  batch/GPU  step ms   GPU-s/out token   cache discount
       1   15.6       15.6     23.0         1.473e-03            0.101
       2   40.6       20.3     20.6         1.017e-03            0.127
       4   78.1       19.5     20.3         1.042e-03            0.222
       8   96.7       12.1     21.5         1.775e-03            0.324

    mean 21.4ms, sd 1.0ms -- stable across 8x the GPUs and 6x the batch

So cost per output token is **`step * n_gpu / batch`**. One parameter:

    LOO, null (predict the average)   31% mean, 41% worst   0 parameters
    LOO, roofline model               27% mean, 49% worst   2 parameters
    LOO, constant-step model           5% mean,  9% worst   1 parameter

Roofline's two parameters buy essentially nothing over guessing the average,
and it is worse in the tail. 27% also exceeds our 20-50% margin over the
market leader, so it cannot answer the question we need answered.

**The simulator's job therefore reduces to predicting the batch a
configuration sustains** -- given batch, cost follows. `StepModel`,
`calibrate_step`, `validate_loo_step`, and the sweep data are in
`simulator.py`; `tests/test_simulator.py` pins the comparison.

Two honest limits. The 21.4ms is an **empirical invariant, not a derivation** —
roofline says TP=8 should reach 6.2ms, so we sit 3.5x off, and *that gap is the
optimisation headroom*. And it is calibrated for this model, GPU and workload;
re-measure it if any of those change. The roofline path stays in the module as
a pinned negative result, because the distance between the two is the target.

**Cache discount is monotonic in TP** — 0.101 / 0.127 / 0.222 / 0.324 at TP
1/2/4/8. At TP=1 it exactly matches the market's ~0.10.

**Deployment economics**, break-even against the best provider, and the share
of the market's 17.9B tokens/day needed to reach it:

    GPUs  $/hr   capacity   break-even util   market share needed
       1  2.50   0.32B/day             33%                 0.60%
       2  5.00   1.01B/day             27%                 1.54%
       4 10.00   1.64B/day             31%                 2.84%
       8 20.00   2.81B/day             52%                 8.16%

**2xH100 has the best break-even utilisation; 1xH100 needs the least absolute
demand.** Either is viable where 8xH100 is not. Caveat: the TP=1 run never
found its N\* — all four levels passed — so 1 GPU is under-characterised and
may be better than measured. Re-run it with higher levels.

### 3d. The market-realistic run — run `1788203369`, and it supersedes §3

8xH100 TP8/EP8, 20,583 in / 2,076 out (9.9:1), p99 TTFT 4s, batch sampled.
`--ep 8` is still required: SGLang runs its quantized-MoE block check on this
dense model using a fallback `moe_intermediate_size=512` (see §6).

    users  goodput  thruput  SLO%  TTFT p99  TPOT p99   hit   batch
        8     0.29     0.29 100.0       684       8.1  0.87     3.9
       16     0.55     0.55 100.0       649      10.2  0.86     8.8
       32     0.88     0.89  99.2       778      21.1  0.84    16.1
       64     1.58     1.60  99.1      1066      37.8  0.81    35.6   <- N*
      128     2.15     2.20  97.8      1061      82.9  0.78    69.0   MISS

**N\* = 64, not 4** — 16x the old figure, and goodput 1.58 rps against 0.79.
The old N\*=4 was an artefact of the 132k replay and the 2s SLO, not the
hardware.

**TPOT binds, not TTFT.** At 128 users TTFT p99 is 1061ms — a quarter of the
SLO — while TPOT p99 is 82.9ms against a 50ms limit. §3's "TTFT is the binding
constraint, raising N\* by improving TTFT is the clearest optimisation target"
was true only of the 132k replay. **At market context the binding constraint is
decode speed.**

**Attribution** (fit 0.932, condition 2.1, spans 23x / 688x / 129x, all
identified):

    |                | run 1 (132k/454) | run 2 (20.6k/2076) |
    | uncached_in    |        1.403e-04 |          1.549e-04 |
    | cached_in      |        4.487e-05 |          5.026e-05 |
    | out            |        2.331e-03 |          1.775e-03 |
    | cache discount |            0.320 |          **0.324** |

**The cache discount replicates to 1% across two runs with different workload,
context, SLO and mix design.** It is a real property of the serving stack, not
an artefact — the strongest result we have.

**Decode does batch — the batch-of-3 hypothesis is dead.** The decode mix ran
at mean batch 96.7 (p50 111, max 130) with ~95 more queued. Measured against
roofline at that batch and context:

    roofline  4.739e-04 GPU-s/token
    measured  1.775e-03      -> 3.7x off = **27% of memory bandwidth**

So the gap is decode *efficiency*, not scheduling. 27% of HBM roofline is low
but not absurd for a production stack; closing it toward 50-60% is the single
highest-value optimisation, because output tokens are 70-81% of the bill on
real traffic.

**Economics on the whole bill** (20,583 in / 2,076 out, $2.50/GPU-hr, 25%
margin), break-even utilisation against the best provider:

    hit 0.394 (market-wide average)   52%
    hit 0.874 (best provider today)   61%

Better than the 60-73% of the previous section but still the number everything
rests on, and still unmeasurable from inside the harness.

**Instrument caveat.** The `cached` mix sampled batch 0.0 at 100% idle. That is
aliasing, not idleness: its requests are a fully-cached prompt plus 8 output
tokens, so they complete well inside the 2s sampling interval. `BatchSampler`
undersamples short requests and its numbers should only be trusted for mixes
with long generations.

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

## 6a. Direction: this becomes an API the auto-research harness calls

Stated 2026-09-01. Not to be built yet — recorded so the shape is not
foreclosed by decisions made now.

The eventual consumer is the auto-research loop: an agent proposes a diff to
SGLang's `srt/`, and wants back a price without knowing anything about mixes,
NNLS or Modal. So the surface is roughly

    evaluate(diff_or_config) -> {effective_in, out, n_star, hit_rate, ...}

Pieces that already exist and should be kept API-shaped:

- `ServingConfig.digest()` — a stable content hash, so identical requests can
  be served from cache instead of re-running 20 GPU-minutes.
- `overlays/` with staleness detection — how a diff is applied and provenance
  recorded. A run that cannot attribute itself to a specific version of the
  serving code is worthless to a search loop.
- `.spawn()` + results Volume — runs already outlive the caller, which any
  async API needs.
- `usable()` — the gate that refuses to return a price from a bad fit. An
  automated caller cannot sanity-check a number, so refusing must stay the
  default rather than a warning.

Two things to settle before productionising: a run takes **~20-25 min**, so the
API is necessarily asynchronous (submit, poll, collect); and a search loop will
want variance, not just a point estimate — the noise floor is measured on the
dev model (goodput CV 0.0003) but not on the target.

**Deliberately deferred:** `ignore_eos: true` forces exact output lengths,
making runs deterministic and canaries bitwise-reproducible. Real generations
stop at EOS at varying lengths, so our output-length distribution is a point
mass where reality has a spread — which matters for the p99 tail, and the tail
is what sets N*. Reproducibility is worth more than realism while hill-climbing;
expose it as an argument when the research environment settles.

## 6b. THE CURRENT METHOD — supersedes §3 and §3c-3f

Settled 2026-09-01. Everything before this describes how we got here; this is
what to actually run.

### The standard environment

    model        Qwen/Qwen3.8-27B-FP8      (never a substitute; see §3c)
    hardware     1 x H100                   assume $3.00/GPU-hr
    utilisation  0.50                       margin 0.0 -> report BREAK-EVEN
    traffic      TraceLab rescaled to 20,583 in / 2,076 out per request

**Why 1xH100 and not something else.**

*Not A100 80GB* (cheaper per hour): Ampere is SM80 and has **no FP8 tensor
cores**. This checkpoint is block-quantised FP8, so it would dequantise to
bf16 — a different machine from the one we are pricing. Same trap as the A10G
check in §6.

*Not L40S* (cheaper still): decode is bandwidth-bound, so L40S is half the
hourly price and **~1.9x the cost per token**, and its 48 GB holds only ~12
conversations of KV. Renting slow GPUs is a false economy here.

*Not 2xH100*, even though it may price lower. Two effects fight:

    at the same batch   1 GPU is cheaper -- TP costs n^0.29 = 1.22x
    at its own ceiling  2 GPUs reach ~42 convs vs ~17, so ~1.34x cheaper
                        IF it reaches that ceiling, which the SLO has
                        never allowed (measured batch 12-36, not 42)

The decisive argument is not price. **`tp_decay = 0.29` is a fitted term that
exists only to describe TP loss, and at n_gpu = 1 it is exactly 1.0 and drops
out.** A 1-GPU baseline therefore has strictly fewer unmodelled effects, and a
diff measured against it cannot be confounded by TP scaling. It is also half
the cost per run, which matters because **every evaluation needs its own
sweep**.

Optimise on 1xH100; re-measure on 2 before committing to a deployment.

**Batch is deliberately left unbounded** (`max_running_requests = 256`, never
binding). In production the scheduler admits whatever fits, so batch is an
outcome that tracks load. Capping it would measure a system we would not
deploy. The cap remains useful as a *diagnostic* — fix offered load, sweep the
cap, and cost-vs-batch falls out with everything else held constant — but note
that capping trades one SLO for the other: a smaller batch gives faster steps
(better TPOT) and a longer queue (worse TTFT), so the feasible region is a
band in (offered load, batch cap), not a point.

**On 1xH100 expect memory to bind sooner than it did on two:**

    68 GB reserved at mem-fraction 0.85
    23.1 GB weights (FP8, unsharded -- only one GPU)
    45 GB KV pool / 1.50 GB per 20.6k conversation
    -> ~30 conversations, ~17 after the scheduler's decode reserve

and price to land between ~$1.55/M output (batch 30) and ~$3.20/M (batch 10),
depending entirely on the batch sustained.

### The SLOs, taken from what the market actually publishes

    p90 TTFT <= 2818 ms    median provider's p90 latency
    p50 TPOT <=   20 ms    median provider's p50 throughput (49 tok/s)

Both are measured values from `scripts/market_pull.py`, not guesses.

Three things had to be discarded to get here:

1. **Generic guidance (p90 TTFT 300-500 ms) is unreachable for this workload.**
   A cold prefill of 20,583 tokens is ~770 ms of compute on 2 GPUs. Our p90
   TTFT sits at ~450 ms and **does not move with load** (471 ms at 8 users,
   411 ms at 32) because sessions are multi-turn and TTFT is bimodal: first
   turns prefill everything, later turns prefill ~20%. The 90th percentile
   lands on the boundary between the two modes. That guidance assumes ordinary
   chat prompts, not 20k-token agentic contexts.
2. **A p99 TPOT cannot be read off market data.** Throughput percentiles run
   the wrong way — `TPOT = 1/throughput` is decreasing, so `p99_throughput` is
   the FASTEST 1%, mapping to p1 TPOT. The slow tail would be
   `p1_throughput`, which nobody publishes.
3. **Fitting the tail does not rescue it.** A lognormal fitted to the TPOT
   quantiles we do have (p1-p50) gives p90/p50 ratios of only 1.16-1.61x,
   while our own measurements show 1.2-3.4x and their published TTFT shows
   2-9x. Latency tails are heavy for reasons the body cannot predict —
   queueing, preemption, eviction. Only 4 of 11 providers even reached
   r2 > 0.95 on the fast half.

Any TPOT *tail* bound is therefore **our choice, not an industry standard**.
p99 TPOT ~60 ms is the same service as a 20 ms p50 given a 3x tail; label it
as a choice wherever it is quoted.

### The measurement, in four steps

1. **Sweep concurrency** on real traffic until the SLOs stop holding. Take the
   last passing level as N*. **Every evaluation needs its own sweep** — a diff
   moves N*, and pricing it at the baseline's N* would systematically
   understate every latency win (a diff that moved N* 16 -> 32 with no change
   to step time still cuts output cost 22%).
2. **Read phase-split GPU time** at N*: `forward_execution_seconds_total`
   labelled `extend` (prefill) and `decode`. Requires
   `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`, already set in `_server_env`.
   The counter is **already summed across TP ranks** — do not multiply by
   n_gpu (that bug doubled every output price once).
3. **Divide**: `eff_in = extend_gpu_s / ALL input tokens`,
   `out = decode_gpu_s / output tokens`. No regression, no mixes, no
   identifiability gate. `pricing.price_direct()`.
4. **x rate / utilisation** -> break-even price; blend is already implicit
   because cached tokens cost no prefill and sit in the denominator.

**Cache hit rate is an OUTCOME, not a control (§5e).** Do not re-blend to a
competitor's hit rate — caching well *is* serving well, and normalising it
away removes what we are trying to optimise.

### What phase B was for, and why it is gone

The four-mix NNLS regression existed to split input cost into cached and
uncached, which is needed **only** to re-blend at someone else's hit rate. It
also had to run at `sat_users`, past N*, so that wall-clock would equal work
time — which measured decode at a batch the SLO does not permit and
**understated output cost ~4x**. The device timer removes the need for both.

Kept as a cross-check, not a dependency: on run `1788247497` the two methods
agreed to 3% on uncached input and 0% on output, and `busy_frac` came back
0.98-1.00, confirming wall-clock was a valid denominator at saturation.

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

---

# Appendix: earlier decisions and session log

Carried over from the root `handoff.md`, retired 2026-08-31 so there
is a single handoff document. Everything below predates the market-
realistic workload (§3c) and the GPU sweep (§3e), so its *numbers* are
superseded — it is kept for the decisions and their reasoning.

## Decisions

**One server launch per suite, not per workload.** Model load is ~250s warm
(349s cold, including the 31GB download) and would otherwise dominate a
2-minute trace. `bench()` takes a *list* of workloads and replays each against
the same server. Anything that varies per launch — a `ServingConfig` field, an
overlay — still needs its own call; traffic shape does not.

That ~250s is the real constraint on sweep size: every config change costs
about $0.27 of pure loading before any measurement happens, so a 50-config
sweep spends ~3.5 GPU-hours just loading. Worth knowing that several
interesting knobs (`schedule_policy`, `max_running_requests`,
`chunked_prefill_size`) are launch-time only in stock SGLang but could be made
runtime-tunable *via an overlay* — which would collapse a whole sweep into one
model load. That is probably the highest-leverage use of the overlay mechanism
and should come before any large sweep.

**Correctness is judged against a measured floor, not against exact match.**
Greedy decoding is not bitwise deterministic across batch compositions —
reduction order shifts and near-tied tokens flip. An exact-match test would
fail constantly and train us to ignore it. So `noise` measures how much two
runs of the *same* config diverge, and every other config is judged against
that floor. Equal divergence is ordinary non-determinism; materially more is
suspect.

**Fail fast on a dead server.** `wait_until_ready` now watches the server
process and aborts the moment it exits, and echoes the load log every 30s. The
previous version would poll a corpse for the full 40-minute timeout — roughly
$2.60 of GPU time to learn the server never started, with no visibility while
it happened. That matters specifically for unattended runs.

**Serving research means editing code, not just turning flags.** SGLang's CLI
knobs are a thin slice of the design space; a better scheduler, KV eviction
policy, batch former or dataloader is code, not configuration. `overlay.py`
makes any file under `sglang/` replaceable:

    overlays/sglang/srt/managers/schedule_policy.py
        replaces
    site-packages/sglang/srt/managers/schedule_policy.py

This is cheap because SGLang's serving layer is pure Python — verified by
probe: `managers/{scheduler,schedule_policy,schedule_batch,tp_worker}.py` and
`mem_cache/{radix_cache,memory_pool}.py` all exist as plain modules, while the
compiled code sits in separate packages (`sgl_kernel`, `sgl_deep_gemm`,
`sgl_deep_ep`). So overlays need no recompilation, and Modal mounts the overlay
directory at runtime (`copy=False`), so an edit costs a container start rather
than an image rebuild.

Every overlay records the SHA of the upstream file it came from. If SGLang is
upgraded and that file moves, `apply()` refuses to run — a stale overlay would
silently revert upstream changes while still looking like a valid experiment.
The run record carries the overlay digest, so a result is attributable to a
specific version of the serving *code*, not just a config.

Tiers, cheapest first: **0** config flags; **1** overlaid Python modules;
**2** wholly new components injected via overlay; **3** CUDA kernels, which do
need compilation and an image rebuild, and are deliberately out of scope now.

**Dev on Qwen3-30B-A3B-Instruct-2507-FP8, single H100 — not the 235B target.**
~31GB of weights leaves ~49GB for KV on one H100, at $3.95/hr instead of
$31.60/hr. It is still a 128-expert MoE, so expert routing and imbalance stay
in the picture. The first ~50 runs will be harness bugs; paying 8× for that is
pointless. The model is a parameter, the harness is the product.

**Client and server share one container, talking over loopback.**
A laptop-side client would put 20-80ms of WAN latency ahead of every TTFT
measurement — larger than most effects we want to detect. The cost is CPU
contention, mitigated by `cpu=8.0` and monitored by `client_dispatch_lag_ms`,
which is reported on every run. If that lag grows during a run, the client was
the bottleneck and the run is void.

**Open-loop load generation.** Arrival times are drawn from a Poisson process
and fixed before the run. Closed-loop generators silently convert overload into
slowdown, which flatters a bad scheduler and hides queueing.

**Goodput is the headline metric, not throughput.** A server can post excellent
tokens/sec while missing every latency target. A request counts only if it meets
*both* the TTFT and TPOT targets; failures never count.

**FP8 over BF16.** BF16 at ~61GB on one 80GB H100 leaves almost nothing for KV
cache, which would make the batching knobs — the ones we actually want to
study — uninteresting.

**HF token stored anyway.** Both Qwen3 repos are Apache-2.0 and ungated, so it
is not required, but it is set up and verified in case of anonymous rate limits
or a later gated model.

**The eval suite varies arrival *shape*, not just rate.** `sustained` and
`bursty` carry the same mean rate by construction — the idle gap is derived as
`on * (burst_factor - 1)` so the mean is preserved exactly — which makes any
difference between them attributable to burstiness alone. Queueing, admission
control and batch formation all respond to variance rather than to the mean, so
a suite that only varies rate would miss most of the interesting failures.
Time-varying processes use Lewis-Shedler thinning, which is exact for any
bounded rate function.

**Spend alerting is ours to build.** Modal's workspace budget is a hard stop
configured in the dashboard, with no email and no CLI. `modal.Workspace.billing`
gives month-to-date and per-object costs, so the monitor polls that and emails
through Resend.


## Session log

### 2026-08-29 — metastable congestion collapse (the big one)

Two **identical** 30-minute suite runs, same seed, same config, same hardware:

    workload        run A    run B
    sustained       30.47    19.62
    constant        30.29     0.54
    bursty          27.74     4.73
    human           21.18     1.57
    prefill_heavy    1.82     1.82
    decode_heavy     6.17     6.14
    prefix_heavy    19.28    19.27
    short_chat      77.52    77.37

The low-rate workloads agree to three digits. The ~30 rps ones disagree by up
to 98%.

**Server-side metrics identified it.** Same throughput (30.47 vs 30.39), same
decode speed (ITL p99 60 vs 56ms), same GPU and driver, and **queue time
negligible in both** (1ms vs 5ms) — yet server-reported TTFT p99 was 99ms in A
and 3616ms in B. Not a slow host, and not queueing as SGLang accounts for it.

The time series settled it. TTFT p50 per bucket:

    t(s)     0    40    81   111   121   131   142
    A       35    39    39    41    41    39    42
    B       96    69   366   616  1219  1521  2704

**A is flat for 152s. B enters a runaway at t~110 and never recovers.**

This is metastable congestion collapse. At ~30 rps against ~34 rps capacity we
sit at ~88% utilisation, where there is no slack to drain a transient. One
unlucky excursion builds a backlog that never clears; TTFT escalates without
bound while throughput stays pinned, because the server is still completing
requests, just increasingly late ones.

**It also explains `spike`**: 10s of overload costing 71% of a 90s trace with no
recovery is the same mechanism.

Two consequences.

*A design error of mine.* Calibrating the suite to sit just below the knee
maximised sensitivity and put it inside the metastable region, where a single
run measures which basin it fell into. Reproducibility and sensitivity were in
conflict and I optimised only for sensitivity. General-purpose workloads now
sit at ~60% of measured max throughput (20 rps), outside that region.

*A research target.* A server that collapses unboundedly at 88% load has no
admission control. `metrics.detect_collapse` makes this a first-class metric —
whether and when TTFT runs away — so the instability is measured rather than
averaged into a number that never occurs. A new `stress` workload sits
deliberately inside the region; there the metric is collapse *rate* over
repeats, never single-run goodput.

The detector was itself wrong on first write: it required most bucket
transitions to rise, which a collapse with a flat prefix fails. It now tests
that the final third is far above the first *and* the trace ends near its peak,
which separates a runaway from a spike that recovered. Validated against the
real run B: onset 109.8s, 28.2x escalation.

### 2026-08-29 — noise floor: 0.03% on goodput

Five separate server launches, same config, same trace — fresh container and
allocator each time, possibly different physical hosts:

    goodput_rps   median 26.073  min 26.068  max 26.085  CV 0.0003
    p99_ttft_ms   median 63.475  min 60.234  max 68.042  CV 0.0399

**Goodput CV is 0.03%.** The whole spread is 0.017 rps out of 26. Effects below
1% are detectable, which is far better than the 10% I assumed when planning.
p99 TTFT is looser at 4% CV, so tail-latency claims need roughly a 10% effect.

This retires the biggest existential risk. It also retroactively validates two
results that looked too small to trust: `bursty` vs `sustained` at -2.6% is
**~80x the noise floor**, and `sustained` vs `constant` at -1.7% is ~50x. Both
are real effects, not measurement drift.

**Canary floor: 6/6 exact match.** Outputs are bitwise identical across runs of
the same config, so the batching non-determinism I expected did not materialise
here. The correctness gate is therefore sharper than designed for: *any*
divergence is signal, not noise. One caveat — canaries run on an idle server
before the workload, so batch composition is trivially identical between runs.
Divergence may still appear for requests measured under load, and the gate
should be re-validated once an overlay actually changes scheduling.

Also observed: model load times of 90, 93, 209, 225, 247, 253, 349 and 505
seconds for the same weights. A 5.6x spread in Volume read throughput. It does
not affect measurements (the server is warm before anything is recorded) but it
makes sweep duration hard to predict.

### 2026-08-29 — recalibrated suite: it finally discriminates

    workload         goodput   thruput   p99 TTFT  p99 TPOT  failed  lag p99
    sustained          26.13     26.13         68      24.0       0        2
    constant           26.57     26.57         74      25.9       0        2
    bursty             25.45     27.11       1233      46.1       0       90
    ramp               13.25     31.43      11847      29.1       0        2
    spike               8.79     30.17      19804      71.0       0        3
    prefill_heavy       1.80      1.80         89       7.8       0        3
    decode_heavy        5.42      5.42         41      14.0       0        2
    prefix_heavy       19.11     19.11        185      22.9       0        3
    short_chat         76.17     76.17         45      25.2       0        2

Goodput now diverges from throughput in three workloads. There is signal.

**`spike` is the standout target.** Ten seconds of 4x overload inside a
90-second trace costs **71% of all requests** their SLO (8.79 vs 30.17), and
p99 TPOT breaches its 40ms target at 71ms. The damage vastly outlasts the
spike — the server does not recover. That is the single most promising
optimisation target we have: a clear mechanism (no admission control, unbounded
prefill queue), a clean measurement, and a large effect.

**`bursty` vs `sustained` at identical 32 rps mean:** goodput moves only -2.6%
(25.45 vs 26.13) while p99 TTFT goes 68ms -> 1233ms, an 18x tail blowup.
Burstiness costs almost nothing in aggregate and everything in the tail. Had we
tracked goodput alone we would have called burstiness harmless; the
constant/sustained/bursty triple at matched mean rate is what makes this
visible.

**`sustained` vs `constant`:** 26.13 vs 26.57 — arrival randomness alone costs
~1.7%.

**Caveat, not glossed:** `bursty` shows client dispatch lag p99 of 90ms (vs
~2ms elsewhere) — at 128 rps peaks the load generator strains. The 1233ms TTFT
is an order of magnitude above that so the signal survives, but bursty is the
least trustworthy row in the table. Bumped the bench container to 16 CPUs; CPU
is $0.047/core/hr against $3.95 for the H100, so removing a measurement
confound is nearly free.

### 2026-08-29 — saturation knee found

`saturate` ramped 5 -> 160 rps over 300s, 24807 requests, bucketed by arrival
window:

    window          offered     ok  met SLO  p99 TTFT  p99 TPOT
        0-25           11.2    279     100%       138      15.1
       25-50           26.1    652     100%        67      21.5
       50-75           35.6    890     100%       382      30.0
       75-100          49.1   1228      24%      3833      29.8
      100-125          63.5   1588       0%     20775      49.4
      275-300         152.9   3823       0%    421007      27.1

**The knee is between 35.6 and 49.1 rps.** Sustained max throughput 34.3 rps.
The collapse is violent: 100% -> 24% -> 0% across two windows.

The failure mode matters as much as the number. **p99 TPOT stays flat at
27-53ms throughout while p99 TTFT reaches 418 seconds.** Decode keeps up fine;
prefill queueing is what collapses. That is an admission-control failure, and
it is precisely the regime where scheduler and admission policy changes should
be observable — so it is the right operating point for the whole project.

Caveat: client dispatch lag p99 rose to 55ms in this run (vs ~2ms elsewhere) at
24807 requests. Negligible against a 418s TTFT so the finding stands, but the
client is no longer free at these volumes and should be watched.

Suite rates recalibrated from this: `sustained` 4 -> 32 rps, `short_chat`
24 -> 80, `prefix_heavy` 6 -> 20, `decode_heavy` 2 -> 7, `prefill_heavy`
1.5 -> 1.8 (already near its own prefill-bound ceiling of ~2.2 req/s).
Per-workload rates are set near each workload's own bottleneck, not to one
global number.

### 2026-08-29 — full suite green, and calibrated far too low

All 9 workloads ran against one server launch in 13 minutes (209s load, then
every workload). Zero failures anywhere, client dispatch lag p99 ~2ms
throughout, so the load generator was never the constraint.

    workload         goodput   thruput   p99 TTFT  p99 TPOT  failed
    sustained           3.87      3.87         64       9.1       0
    constant            3.72      3.72         37       8.4       0
    bursty              3.63      3.63         49      11.8       0
    ramp               16.40     16.40         84      20.0       0
    spike               7.71      7.71         63      23.4       0
    prefill_heavy       1.34      1.34         87       7.7       0
    decode_heavy        1.43      1.43         30       9.4       0
    prefix_heavy        5.64      5.64         87       9.6       0
    short_chat         23.15     23.15         28      11.1       0

**Goodput equals throughput in every row.** 100% of requests met both SLOs
everywhere, including the ramp to 32 rps and short_chat at 23 rps, with p99
TTFT peaking at 87ms against a 500ms target. The suite is measuring an idle
server.

A 30B MoE with 3.3B active params on an H100 is far more capable than the rates
I picked assumed. **No scheduler or cache change can show up in a system that
is not stressed**, so the suite rates have to be recalibrated against the real
saturation point before any optimisation work means anything. `saturate` ramps
5 -> 160 rps and buckets per-request results by arrival time to locate the knee;
success there looks like *failure*, a region where goodput falls below
throughput.

Note also that `bursty` (3.63) came out slightly below `sustained` (3.87) at
identical mean rate — the expected direction for burstiness, but far too small
a gap to claim as a result at this load level.

### 2026-08-29 — prefix cache anomaly resolved

`probe_prefix` separated the confounds. Three findings.

**1. TTFT scales with prefill work.** Distinct prompts, nothing cached:

    tokens    10    100    500   1000   2000   4000
    TTFT ms 18.0   21.1   44.0   73.2  181.6  479.1

So TTFT is ~18ms of fixed overhead plus real compute. My earlier guess that it
was overhead-dominated was wrong.

**2. A FULL cache hit is ~1.57x SLOWER than a cold prefill.** Reproduced three
ways on a 1000-token prompt, all at CV ~1-2%:

    cold prefill (first request)        73-74 ms
    same prompt repeated, no flush     114.7 ms
    first request after /flush_cache    73.6 ms
    second request after flush         116.9 ms

The flush is irrelevant — the effect reproduces with no flush anywhere. What
matters is whether the request has prefill work to do. The likely mechanism:
with a 100% prefix match there is nothing to prefill, so the request takes a
different scheduling path and waits for a batch tick, costing ~40ms.

**3. A PARTIAL prefix match works exactly as advertised**, 1.76x faster:

    700 tokens sharing a 500-token head    29.8 ms
    700 tokens fully unique                52.5 ms

**Why the earlier measurement looked backwards.** `probe_serve`'s idle test
sent `"Say hi."` twenty times — the *same* prompt — so every request after the
first was a full cache hit, landing on the slow path at 100.3ms. The "cold"
prefix measurement used a distinct 1213-token prompt and got 36.2ms. Not a
contradiction once full-hits are known to be slow; the two tests were measuring
different regimes and I had read them as one.

**Consequences.** `prefix_heavy` generates a 1024-token shared prefix plus a
~315-token unique body — a partial match, the regime where the cache helps. It
is trustworthy as written. And the full-hit penalty is itself a concrete
optimisation target: ~40ms of avoidable fixed cost on every fully-cached
request, which is a strong candidate for the first real overlay experiment.

### 2026-08-29 — first real bench run

`smoke` green end to end: 60 requests, 0 failures, goodput 1.57 rps, p99 TTFT
38ms, p99 TPOT 7.1ms, all 6 canaries returned. **Client dispatch lag p99 was
2.6ms**, so the load generator was nowhere near saturated and the server
numbers are trustworthy. At 1.6 rps nothing is stressed — the SLOs are 13x
away from binding — which is what a smoke test should look like.

Fixed along the way:

- **The HF token was never attached.** We created the secret and left the
  `secrets=[...]` line commented out, so SGLang was making unauthenticated Hub
  requests. Now wired into every GPU function.
- **fd limit raised to 65536.** The `ramp` workload issues ~3000 requests and
  the client holds a socket per in-flight request; a default 1024 cap would
  have surfaced as connection errors *attributed to the server*, which is a
  wrong conclusion drawn from a client-side limit.
- **Fast-fail on a dead server** plus a load-progress echo every 30s. The old
  path polled `/health` for the full 40-minute timeout with no visibility —
  ~$2.60 of GPU time to discover the server never started. Found this while
  blind-waiting on the smoke run and unable to tell loading from crashed.
- `hf_cache.commit()` after load, so a fresh download persists.

**Model load time is highly variable: 247s, 349s, 505s across runs.** The
weights are cached (the log confirms `skipping download`), so this is Volume
read throughput, not network. It is the dominant per-experiment cost and it is
not stable, which matters for planning sweeps.

### 2026-08-29 — probe_serve + overlay architecture

- `probe_serve` PASS. `ignore_eos` honoured (64 asked, 64 returned; 12 without).
  Usage present in stream. Deltas != tokens (62 vs 64) — vindicates using the
  server's `usage` rather than counting stream chunks.
- **Noise floor measured: idle TTFT CV 0.022, total CV 0.009 over n=20.**
  Modal hardware is consistent enough to detect a 10% effect. Biggest
  existential risk to the project retired.
- **Prefix cache is reproducibly backwards** (cold 36.2ms vs warm 105.9ms,
  CV ~1%, n=6 with `/flush_cache` between trials). First run had suggested this
  at n=1 and was rightly dismissed as noise; repeated measurement made it real.
  Logged as open question 9.
- Reworked the probe to repeat every latency measurement and compare medians —
  the n=1 version produced a nonsense answer that looked like a finding.
- **Built the overlay system** in response to the point that the harness should
  not be limited to SGLang's exposed flags. Vendored
  `srt/managers/schedule_policy.py` (1500 lines) as the first editable target.
- `modal_app.bench` now applies overlays before launch and records their digest
  in every run record.

### 2026-08-29 — probe_env results

Ran on a real H100 for **$0.03** (credit-covered). Verdict **PASS**, and it
paid for itself immediately:

- Hardware: H100 80GB HBM3, 81559 MiB, driver 580.95.05, compute 9.0, PCIe gen4.
  **78.7 of 79.2 GiB free** — so the ~31GB FP8 dev model leaves ~47GB for KV.
- Image builds in 228s: sglang 0.5.18, torch 2.13.0+cu130, flashinfer 0.6.17.
- **All 12 SGLang flags exist.** No silent-ignore risk in the current config.
- **`--schedule-policy` accepts seven values, not the four we assumed:**
  `{lpm, random, fcfs, dfs-weight, lof, priority, routing-key}`. `lof`,
  `priority` and `routing-key` are search-space dimensions we would have
  missed entirely. Recorded in `config.py`.
- Two deprecated env vars caught and fixed: `HF_HUB_ENABLE_HF_TRANSFER` ->
  `HF_XET_HIGH_PERFORMANCE`, and `SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK`
  (removed; SGLang inverted it to `SGLANG_ENABLE_...`).
- Bug fixed: `torch.__version__` is a `str` *subclass*, so returning it from a
  Modal function fails to deserialize locally with
  `DeserializationError: 'torch' module is not available`. The same bug was
  present in `modal_app.py::_provenance`, where it would have broken every
  benchmark return value. Coerced to plain `str`.
- Workspace is on the **legacy 2023.12 image builder**; upgrading is a
  deliberate choice (forces rebuilds), logged as an open question.

### 2026-08-29 — evals, secrets, probe

- Stored HF + Resend keys as Modal Secrets (`huggingface`,
  `auto-inference-resend`, `auto-inference-modal-token`). Values live only in
  Modal, never in this repo. HF token verified (`mpeng19`, write role).
- Spend monitor verified **end to end** — a real digest email was delivered.
- Built the eval suite: 9 arrival/length patterns plus a merged `mixed` stream,
  with 12 more tests. Verified `bursty` preserves the mean rate of `sustained`
  (3.7 vs 3.9 rps) at CV 4.08 vs 0.91, and that `ramp` matches the analytic
  integral of its rate function per 50s window.
- Wrote `probe.py` to answer the open questions on a real H100 in two stages,
  cheap first. `probe_env` checks all 12 SGLang flags we emit against the
  server's own `--help`.
- Restored `test.py`, which vanished mid-session; cause never established. It
  was untracked, so git could not protect it. **The repo still has no commits.**

### 2026-08-29 — bootstrap

- Sorted a broken Python path: 7 interpreters, conda `base` auto-activating,
  and `modal`'s CLI installed somewhere off-PATH. Fixed via a project venv and
  by removing two python.org framework entries from `.zprofile`; nothing was
  deleted. `uv` relocated to `~/.local/bin` (it had been living only inside
  miniconda, so removing miniconda would have silently broken the project).
- Verified `modal run test.py` executes remotely.
- Pulled live pricing from the account: H100 $3.95/hr, Volumes $0.09/GiB/mo.
  Starter plan: $30/mo credits, GPU concurrency 10.
- Confirmed both Qwen3 FP8 repos exist, Apache-2.0, ungated.
- Built the harness: config, workload, metrics, bench, Modal app.
- 17 local tests, including proof the generator is genuinely open-loop
  (20 requests at 100/s against a 200ms server finish in <1s, not 4s).
- Built and live-verified the spend monitor. Not yet deployed.
