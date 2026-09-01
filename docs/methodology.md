# Methodology: how the serving-cost simulator was built and checked

Written 2026-08-31. Companion to `docs/HANDOFF.md` (resume state) and
`docs/checklist.md` (plan). This document states what the simulator computes,
how it was built, everything we have measured, and every assumption it rests
on — separated by whether the assumption is verifiable.

---

## 1. What we are trying to do

Produce reproducible diffs to SGLang's `srt/` that reach a top 5–10 **effective
input price** for `qwen/qwen3.8-27b` on OpenRouter, serving coding-agent
traffic, while meeting marketplace SLOs.

Effective input price is the metric because for cache-heavy traffic it is what
competition runs on:

    eff_in = hit_rate * cache_read_price + (1 - hit_rate) * listed_input_price

Across 11 providers, listed input varies 1.9x ($0.35–$0.48) while cache-read
price varies 10x ($0.035–$0.25). Cache-read price reorders the leaderboard.

**This formula is verified, not assumed.** OpenRouter publishes both the inputs
and the realised effective prices, and it reproduces theirs to <0.1% on 9 of 11
providers (`tests/test_market.py`).

### The chain, and what is trustworthy in it

    1. measure GPU-seconds per token class    MEASURED   (§4)
    2. predict them for unseen configurations SIMULATED  (§3, 5% error)
    3. x $/GPU-hr, / utilisation, x margin    ASSUMED    (§6.3)
    4. blend by cache hit rate                VERIFIED   (<0.1%)
    5. rank against competitors               SCORED

Step 3 is the weak link and cannot be closed from inside the harness:
utilisation is a property of traffic a marketplace chooses to send us.

---

## 2. What the simulator computes

Given `(model, GPU type, GPU count, workload)` it returns GPU-seconds per token
for three classes — uncached input, cached input, output — which multiply
through §1's chain to a price and a rank.

Its central result:

    cost per output token = step_s * n_gpu / batch

Each decode step advances every sequence in the batch by one token, and costs
the same wall time regardless of how many GPUs or how large the batch. So the
optimisation problem reduces to **maximising batch per GPU**.

---

## 3. How it was constructed, and why that way

### 3.1 The falsifiability constraint

A simulator that cannot be caught being wrong is worth nothing. Any model with
enough free parameters fits any dataset, so the design rule was: **fewer free
parameters than configurations to predict.**

We have four measured configurations (TP = 1, 2, 4, 8). The roofline model has
two parameters. Leave-one-out — calibrate on three, predict the fourth — is
therefore a real out-of-sample test, and its error is the fidelity number.

### 3.2 Attempt 1: memory-bandwidth roofline (rejected, 27% error)

Decode is memory-bound: each step re-reads all weights, plus the KV cache of
every sequence in the batch.

    bytes_per_step = active_params * 1 byte  +  batch * context * 64 KiB
    step_s         = bytes_per_step / (hbm_bandwidth * n_gpu * f_decode)

Parameters: `f_decode` (fraction of HBM bandwidth realised) and
`f_prefill` (fraction of peak FLOP/s realised). Calibrated on the 8xH100 run:
0.277 and 0.312.

Leave-one-out result:

    held out   calibrated on   predicted    measured   error
          2g          [4, 8]   1.521e-03   1.017e-03    +50%
          4g          [2, 8]   1.201e-03   1.042e-03    +15%
          8g          [2, 4]   9.248e-04   1.775e-03    -48%
                                        mean 27%, worst 49%

(An earlier version of this table quoted 38%/50%; that was computed on three
configurations, before the TP=1 run landed. With all four it is 27%/49%.)

**Against a null.** A zero-parameter model that ignores everything and predicts
the mean cost scores **31% mean / 41% worst**. Roofline's two parameters buy
essentially nothing over guessing — and it is *worse* in the tail. That, not
the absolute number, is the verdict.

**Why 27% is disqualifying here.** Our margin over the market leader runs
20-50% (§4.2), so an error of this size cannot tell us whether we beat the
market or lose to it. The error is larger than the effect being measured.

**Why it fails:** it treats tensor parallelism as free — TP divides both the
bytes and the bandwidth by `n`, so cost per token should be TP-invariant.
Measured decode efficiency instead falls by half:

    TP=2  f_decode 0.596
    TP=4  f_decode 0.510
    TP=8  f_decode 0.288

Retained in `simulator.py` as a **pinned negative result**, because the gap
between roofline and measurement is the optimisation headroom (§8.1). It must
not be used to predict cost; `test_roofline_still_fails_the_held_out_test`
asserts it keeps failing.

### 3.3 Attempt 2: constant step time (accepted, 5% error)

The sweep shows step time barely moving across the whole range:

    GPUs  batch  batch/GPU  step ms   GPU-s/out token
       1   15.6       15.6     23.0         1.473e-03
       2   40.6       20.3     20.6         1.017e-03
       4   78.1       19.5     20.3         1.042e-03
       8   96.7       12.1     21.5         1.775e-03

    mean 21.4ms, sd 1.0ms, spread 1.13x — across 8x GPUs and 6x batch

One parameter. Leave-one-out:

    held out 1g: predicted 1.334e-03  measured 1.473e-03   -9%
    held out 2g: predicted 1.064e-03  measured 1.017e-03   +5%
    held out 4g: predicted 1.111e-03  measured 1.042e-03   +7%
    held out 8g: predicted 1.764e-03  measured 1.775e-03   -1%
                                          mean 5%, worst 9%

### 3.3.1 How strong is this result, honestly

Weaker than "5% error" makes it sound. Three qualifications, all of which
should be read before quoting the number.

**(a) The formula is an identity, not a physical law.** In wall time `T` at
batch `B`, a configuration emits `B*T/step` output tokens and burns `n*T`
GPU-seconds, so

    cost = n*T / (B*T/step) = step * n / B

`T` cancels. Given a stable batch and GPU time going to decode, `cost = step *
n / batch` is **algebra**. Its entire empirical content is the separate claim
that `step` is constant across configurations. The formula cannot be wrong; the
constancy can.

**(b) The "5% LOO error" is the coefficient of variation of four numbers.**
Step times are 23.0 / 20.6 / 20.3 / 21.5 ms — mean 21.36, sd 1.18, **CV 5.5%**.
For a constant-mean model, leave-one-out error *is* the CV; the two agreeing is
arithmetic, not corroboration. It says the four step times are close together.
It does not establish that the model extrapolates.

Against a null of "cost is constant", the model does earn its keep — R² +0.963
versus +0.000 — but that null is very weak, since cost varies 1.7x across the
sweep and any downward function of batch would beat it.

**(c) `n_gpu` and `batch` were not independently varied.**
`corr(log n_gpu, log batch) = +0.965`. `sat_users` was derived from `N*`, which
grows with GPU count, so batch rose with GPUs by construction. **These four
points cannot distinguish a `1/batch` law from an `n_gpu` law** — only their
ratio was observed. The functional form is assumed, not measured.

**What would settle (c):** hold `n_gpu` fixed, hold mix composition fixed,
and sweep concurrency so batch varies alone. The frontier levels cannot serve
for this — at low concurrency the GPU idles, so wall-clock GPU-seconds charge
idle time to the few tokens that flowed, failure mode (1) of §3.4. Only
saturated phase-B windows are valid.

### 3.3.2 That experiment was run, and it did not settle it

Three runs at n_gpu=2, identical mix composition, `sat_users` 24 / 64 / 192,
plus the existing 128 (runs `1788210264`, `1788210446`, `1788210918`):

    sat_users   batch   GPU-s/out    fit   implied step
           24    21.8   1.478e-03  0.814        16.1ms   REJECTED by usable()
           64    37.6   1.098e-03  0.883        20.6ms   REJECTED by usable()
          128    40.6   1.017e-03  0.921        20.6ms
          192    42.7   9.572e-04  0.944        20.4ms

**It failed for a reason worth more than the experiment.** Offered load went up
8x and the running batch went 21.8 -> 42.7, saturating near 42 while the
*queue* grew 1.5 -> 14.9 -> 89.1. The two points that passed the fit gate span
**1.05x in batch** — no leverage at all, so the `1/batch` form remains
untested.

**Batch is not a free variable: the scheduler pins it.** ~42 on 2 GPUs, ~97 on
8, regardless of load, and in both cases far below what either the SLO or KV
capacity permits (2 GPUs: memory allows ~76 at this context; 8 GPUs: the TPOT
budget allows 251). So `batch` and `n_gpu` are **structurally** coupled in
SGLang's default configuration, not merely correlated by our choice of
`sat_users` — and no amount of load variation will separate them.

Consequences:

1. The confound of §3.3.1(c) **cannot** be broken by varying offered load. It
   needs the batch cap itself moved — `--schedule-conservativeness` (1.0 here)
   and the chunked-prefill interaction are the candidates, since neither
   `--max-running-requests` (256) nor memory is binding.
2. That is the same investigation as §8.2, and it is now the critical path for
   *both* the simulator's validity and the largest known cost saving.
3. Weak positive signal only: implied step time at fixed n_gpu=2 is 20.6 /
   20.6 / 20.4 ms across three of the four points — tight, and close to the
   21.4ms from the cross-TP sweep. But over a 1.05x batch range that is
   consistency, not confirmation.

**Two of four runs were rejected by `usable()`** (r2 0.814 and 0.883 against a
0.9 gate). The guard behaving as designed: a bad fit still yields
plausible-looking prices, and reporting nothing is better.

**Also unverified:** 21.4ms has no first-principles justification, and roofline
says TP=8 should manage 6.2ms. It is calibrated for this model, this GPU, this
workload shape. Re-measure if any changes, and treat a change as a finding.

### 3.5 Experiment-design rules, each earned by a wasted run

Three failures in this project share one shape: **an experiment launched
without first checking it could produce the signal it was looking for.**

**Rule 1 — the model is fixed; the GPU is not.** All serving-physics
measurement is on `Qwen/Qwen3.8-27B-FP8`. Cost per token varies with batch only
through the weight read, and a 4B carries 4 GB of weights against this model's
23 GB, so batching and cache behaviour sit in a different regime entirely. A
batch sweep was launched on the 4B dev model to save money; its predicted cost
variation was **1.21x over a 2.2x batch range — inside the 8% fit noise**, so
it was unresolvable by construction and was cancelled before completing. The
dev model is for harness validation only (noise floor, determinism, load
generator). Enforced by `simulator.TARGET_MODEL` / `assert_target_model`.

**Rule 2 — check the effect exceeds the noise before spending.**
`simulator.effect_size()` computes predicted variation for a planned sweep and
refuses anything under 4x the fit noise. A 2x threshold was tried first and let
the bad dev-model experiment through: 21% effect against +-8% per endpoint is a
signal-to-noise of 1.8, enough to notice a difference but not to *fit an
exponent*, which is what we are doing.

**Rule 3 — check the independent variable can actually move.** A sweep raising
offered load 8x to vary batch produced 21.8 -> 42.7 (and only 1.05x among
points that passed the fit gate), because batch is a memory ceiling, not a
function of load (§8.2). Reading `schedule_policy.py` first would have shown
this for free. Prefer reading the source to buying a data point.

**Corollary — the cheapest experiment is often not the informative one.** Rules
1 and 3 both came from optimising for GPU cost before checking whether the
result would answer the question. A run that cannot resolve its effect costs
its full price and returns nothing.

### 3.3.3 Attempt 3: affine step time — ACCEPTED, 2-3% error

The fixed-GPU batch sweep (1xH100, mem-fraction 0.45/0.65/0.90, batches
3.06/12.70/23.30 — a **7.6x range with the n_gpu confound finally broken**)
settled it. Neither earlier model was right; the truth is between them:

    step(B) = 14.59ms + 0.1505ms * B        residuals < 0.3ms

Term by term against roofline at 1xH100, 6k context:

    fixed term  14.59ms   vs weights/BW  6.89ms   ->  47% of bandwidth
    slope      0.1505ms   vs per-seq/BW  0.164ms  -> ~100% of bandwidth

**The per-sequence KV read is at the bandwidth bound. Only the fixed per-step
cost is off, by 2.1x.** That is why the earlier tests disagreed: at small batch
the fixed term dominates and the system looks constant-step; at large batch the
KV term dominates and it looks roofline. The three-point discriminator returned
a dead tie (log-margin 0.007) for exactly this reason.

The model, with `f_kv` pinned at the roofline it was measured at, leaving one
free parameter for the fixed term plus one for TP scaling:

    step = [W/(BW*f_w) + per_seq(ctx)*batch/(BW*f_kv)] / (n_gpu * n_gpu**-tp_decay)
    f_w = 0.475    f_kv = 1.00    tp_decay = 0.29

Leave-one-out over all 7 measured configurations:

    fixed GPU count (batch + context physics)   mean  2%   worst  4%
    across TP 1/2/4                             mean  3%   worst  7%
    across TP 1/2/4/8 (incl. anomaly)           mean 12%   worst 33%

    for comparison, on the same 7 points:
      constant-step  mean 15%      roofline  mean 16%      null  mean 31%

**At fixed GPU count the model is essentially exact across a 7.6x batch range
and a 3.8x context range, with one free parameter.** The batch-and-context
physics is solved.

**TP=8 is flagged, not fitted.** It requires g=0.36 where the TP 1/2/4 trend
gives 0.55 — a 35% shortfall *on top of* its batch under-fill (§8.2). So TP=8
has two distinct problems, and smoothing them into a scaling law would hide
both. `test_tp8_is_flagged_not_fitted` fails if the gap ever closes, which is
the intended win.

**The 2.1x gap on the fixed term is now the whole optimisation target.** It is
the weight read plus per-step overhead, it dominates at the batch sizes we
actually run, and unlike the KV term there is headroom in it.

### 3.3.4 Prospective test — FAILED at -15%, and that is the real number

Leave-one-out is out-of-sample across configurations, but every held-out point
still sits inside the convex hull of the rest, and all seven come from one
apparatus and one SGLang build. So a prediction was **sealed in git before the
confirming run** (`docs/prediction-2026-09-01.json`, commit `a70f3ae`) for a
combination absent from training: **n_gpu=2 at 6k context** (the n=2 training
point is at 22.7k; every 6k point is n=1).

    batch observed   59.9
    PREDICTED        4.996e-04 GPU-s per output token
    MEASURED         5.878e-04
    error            -15.0%        criterion: within 15%
    VERDICT          FAIL (marginal)

The model **under-predicts cost** one step outside its envelope. So the honest
fidelity statement is:

    within the training envelope   2-3%   (leave-one-out, §3.3.3)
    one step outside it            15%    (prospective, this section)

Quote the second number. The first is optimistic for exactly the reason this
test was run: interpolation is easier than extrapolation, and LOO cannot
detect an error shared by every training point.

Direction of the miss is informative: predicting *too cheap* at a new
(n_gpu, context) pair suggests `tp_decay` is too small — TP loses more than
`n**-0.29` at short context, where the fixed per-step term dominates and
per-step synchronisation is a larger share of the total. Untested.

### 3.4 How the inputs to the fit are measured

The per-token costs come from **rate-form NNLS regression** over saturated
workload mixes. Four earlier approaches all failed, all producing
plausible-looking numbers (`docs/HANDOFF.md` §4). The root cause was one thing:
fixed-duration experiments hold GPU-seconds nearly constant while token counts
vary, so nothing is identifiable. Dividing through by GPU-seconds cancels
duration:

    1 = a*(U/gpu_s) + b*(C/gpu_s) + c*(O/gpu_s)

Guards that must stay:
- `usable()` refuses to print a price from a bad fit.
- `identifiability()` requires each token class to vary >=3x across mixes.
  Overall condition number is not enough: an 8xH100 run reported cached tokens
  costing *more* than uncached, at fit 0.95 and condition 5.4, because the
  output column spanned only 1.5x.
- `BatchSampler` records the running batch during each window, since the whole
  cost model turns on batch.

---

## 4. Everything measured

### 4.1 The GPU sweep (2026-08-31, market workload, p99 TTFT 4s / TPOT 50ms)

| TP | run | N* | goodput | uncached_in | cached_in | out | cache disc | fit |
|---|---|---|---|---|---|---|---|---|
| 1 | 1788207351 | >=16 | 0.18 | 6.163e-05 | 6.204e-06 | 1.473e-03 | 0.101 | 0.94 |
| 2 | 1788206950 | 32 | 0.57 | 7.644e-05 | 9.737e-06 | 1.017e-03 | 0.127 | 0.92 |
| 4 | 1788206719 | 32 | 0.92 | 1.013e-04 | 2.244e-05 | 1.042e-03 | 0.222 | 0.92 |
| 8 | 1788203369 | 64 | 1.58 | 1.549e-04 | 5.026e-05 | 1.775e-03 | 0.324 | 0.93 |

All four passed identifiability and conditioning.

**Cache discount is monotonic in TP** — 0.101 / 0.127 / 0.222 / 0.324. At TP=1
it exactly matches what the market prices cache reads at (~0.10). An earlier
claim that the cache discount "replicates to 1%" was true only across two
8xH100 runs; across TP it varies 2.5x. It is a property of the *deployment*,
not of the serving stack alone.

**TPOT binds, not TTFT.** At TP=8/128 users, TTFT p99 is 1061ms against a
4000ms SLO while TPOT p99 is 82.9ms against 50ms.

### 4.2 Deployment economics

$2.50/GPU-hr, 25% margin, market traffic, against the best provider:

| GPUs | $/hr | capacity | break-even util | market share needed |
|---|---|---|---|---|
| 1 | 2.50 | 0.32B tok/day | 33% | 0.60% |
| 2 | 5.00 | 1.01B tok/day | **27%** | 1.54% |
| 4 | 10.00 | 1.64B tok/day | 31% | 2.84% |
| 8 | 20.00 | 2.81B tok/day | 52% | 8.16% |

**8xH100 was the wrong deployment.** 2xH100 is 1.9x cheaper per request and
44% more efficient per GPU.

### 4.3 Reproducibility (dev model, Qwen3-4B on L40S)

- N* = 128, goodput 7.2–7.7 rps, reproduced three times.
- Noise floor: goodput CV 0.0003 over 5 launches — 1% effects are detectable.
- Canary: 6/6 exact, outputs bitwise reproducible.
- Metastable collapse at ~88% utilisation: two identical runs gave goodput
  30.29 and 0.54. `detect_collapse` measures it rather than averaging it away.

---

## 5. Market ground truth

Pulled by `scripts/market_pull.py`. The public API gives listed prices and
uptime only; per-provider share, cache hit rate, latency percentiles and the
daily series are streamed as a Next.js RSC payload, retrieved by requesting the
page with an `RSC: 1` header.

### 5.1 Real traffic (17 days, `model_chart`)

|  | our first replay | TraceLab | **real market** |
|---|---|---|---|
| cache hit rate | 0.96 | 0.956 | **0.394** |
| input tokens/req | 537 | 132,092 | **20,583** |
| output tokens/req | ~230 | 454 | **2,076** |
| input:output | 2.3 | 291:1 | **9.9:1** |

Daily volume swings 3.0B–42.8B prompt tokens; recent ~17.9B.

**Consequence:** under TraceLab's ratio, output tokens are 12% of modelled
serving cost; under the real one they are **70–81%**. The optimisation target
follows the ratio, so getting this wrong pointed all previous work at the wrong
term.

### 5.2 Provider latency — our SLO was stricter than the whole market

    market p99 TTFT   best 3844ms (Parasail)   median 13225ms   worst 62475ms

Novita — currently the cheapest effective price — runs p99 TTFT of 52–56s.
Our original 2s p99 SLO is met by nobody on the board; it capped N* at 4.

### 5.3 Prices (2026-08-31)

- Best realised effective input: **Novita $0.1272** at 87.4% hit.
- Two days earlier it was Chutes $0.1310 at 69.5%. **Listed prices did not
  change** — only hit rates did. The target moves without anyone repricing.
- Weighted average actually paid: $0.2116 in / $2.868 out.

---

## 5a. Why the "#1 rank" is not a finding — read before quoting any price

We run **stock SGLang 0.5.18 with no diffs applied**, yet the model says we
would rank first on effective input price by ~3x. That should be treated as a
warning, not a result.

**The comparison is our COST against their PRICE.** Those are different
quantities. Against every provider's realised effective input price, at their
own hit rate:

    provider     their eff-in   their hit   our raw cost   multiple
    Novita            $0.1272       87.4%        $0.0126      10.1x
    Chutes            $0.1439       65.4%        $0.0228       6.3x
    Parasail          $0.1773       57.5%        $0.0264       6.7x
    CoreWeave         $0.1988       80.5%        $0.0158      12.6x
    Reka              $0.2257       41.4%        $0.0339       6.7x

Everyone charges 6-13x raw GPU cost. **A provider at 30% utilisation with a 2x
margin lands at 6.7x** — entirely ordinary. The gap is utilisation, margin and
overhead, not serving efficiency, and we have built no serving efficiency: no
`srt/` diff has been applied.

**What our cost basis omits that a real provider pays for:**

- redundancy for an uptime SLA (Chutes 99.74%, Parasail 99.97%)
- provisioning for **peak**, not mean — daily volume swings 3.0B to 42.8B (14x)
- failed and retried requests, cold starts, model swaps
- gateway, load balancer, control plane, egress, weight storage
- engineering and on-call labour, multi-region presence

The peak-vs-mean point is the largest: a 14x swing means capacity sized for
peak runs far below 60% mean utilisation, and **utilisation is already the
assumption every price rests on** (§6.3).

**Our serving may be worse than theirs, not better.** Decode runs at **47% of
the memory-bandwidth roofline**. We have no measurement of a competitor's, but
47% is not a figure that suggests we are ahead of specialists.

### What the simulator is actually for

Not absolute price prediction — that needs utilisation and overheads we cannot
observe. Its value is **differential**: it isolates `f_weights`, the fraction
of bandwidth the fixed per-step read achieves, which is a property of the
serving stack alone. Utilisation, margin, GPU price and overheads all cancel
when comparing two configurations of the same deployment.

So the question the harness can answer is *"did this diff move `f_weights` from
0.47 toward 0.70?"* — measurable to 2-3% (§3.3.3) and independent of every
unverifiable business assumption. The question it cannot answer is *"would we
beat Chutes?"*

## 5b. Utilisation, bounded — and the worry was backwards

Utilisation cannot be *fitted* from price. A power law of share against price
over 20 provider-snapshots gives **R^2 = -0.36** — worse than predicting the
mean, so price does not determine share well enough to build on.

But it can be **bounded**, which needs no model. Every provider holding >1%
share, across both snapshots, sits between **1.7% and 20.6%** (median 12.2%).
Against 17.9B input tokens/day and our measured capacities:

    share    tokens/day    1 GPU    2 GPU    4 GPU    8 GPU
     5.0%         0.90B     100%      89%      55%      32%
     7.5%         1.34B     100%     100%      82%      48%
    10.0%         1.79B     100%     100%     100%      64%
    20.6%         3.69B     100%     100%     100%     100%

**Even the smallest share any real provider holds saturates a 1-2 GPU node.**
So §3a's demand-limitation conclusion was an artefact of sizing at 8xH100. At
the node size the cost model prefers we are **capacity-limited**, and the
question becomes how large a node we can fill, not whether we can fill one.

What this still cannot say is *where* in the 1.7-20.6% range a new entrant
lands — that is the part price does not predict.

## 5c. Margin should not be inside the cost model

Margin was carried as 25% throughout. It is dead weight for the engineering
question and actively misleading in comparisons:

- **It cancels** in every internal comparison (TP=2 vs TP=8, before-diff vs
  after-diff), so it can only distort, never inform.
- **It is a free parameter that moves conclusions.** 1.25x vs 2.0x flipped
  "we rank #1" into "we lose to Chutes" with no change to any measurement.
- **It is not ours to choose** when comparing against a competitor whose own
  margin is unobservable.

**Report break-even instead** (margin = 1.0): the lowest price that does not
lose money. The comparison then becomes "is our break-even below their price?",
which needs no guess about anyone's profit target:

    our break-even eff-in, 2xH100 @ 100% util, hit 65.4%:   $0.0228/M
    Chutes' published effective input price:                $0.1439/M

Margin belongs at the end as an explicitly stated business overlay, never
buried inside a cost figure.

## 6. Assumptions

### 6.1 Verified against external data

| Assumption | Evidence |
|---|---|
| `eff_in = h*cache_read + (1-h)*listed` | reproduces OpenRouter's published figures to <0.1% on 9/11 providers |
| Model architecture | `config.json` fetched raw: dense, 64 layers (16 full-attn + 48 linear), hidden 5120, intermediate 17408, head_dim 256, vocab 248320 |
| Workload shape | 17 days of OpenRouter totals |
| Achievable cache hit rate | providers realise 0.0%–87.4% on identical traffic |

### 6.2 Measured internally, held-out tested

| Assumption | Status |
|---|---|
| `cost/out token = step * n_gpu / batch` | 5% mean LOO error over TP 1/2/4/8 |
| step time = 21.4ms | sd 1.0ms across the sweep |
| per-token costs from rate-form NNLS | fit 0.92–0.94, all columns identified |

### 6.3 Assumed, and NOT verifiable from inside the harness

These are where the price numbers are soft. Each is a choice, not a finding.

| Assumption | Value | Why it is uncertain |
|---|---|---|
| **Utilisation** | 60% quoted, 27–52% break-even | Depends entirely on traffic a marketplace routes to us. Nothing measurable here. This is the single largest uncertainty. |
| **GPU price** | $2.50/GPU-hr, Nebius committed | Chutes runs decentralised compute, possibly far cheaper. Providers who own hardware have a depreciation basis, not a rental one. |
| **Margin** | 25% | Our choice. Competitors may run thinner or at a loss for share. |
| **KV dtype** | bf16, 2 bytes/element | SGLang default; FP8 KV would halve KV bytes and change the batch/cost curve. |
| **Weight dtype** | FP8, 1 byte/param | Matches the served checkpoint. |
| **Competitor cost** | inferred from their price | Their $0.035 cache read is a *price*. OpenRouter's leaderboard sorts on effective input price, so pricing cache reads near zero is the cheapest way to top it. |
| **Cache hit rate to price at** | 0.394 or 0.874 | Market-wide average vs what a good provider achieves. A pricing choice, not a measurement. |

### 6.4 Structural assumptions worth naming

- **Modelling this model on one dedicated node.** Incumbents multiplex dozens
  of models over a shared fleet, so their utilisation is high even when any one
  model's demand is thin. We would not. This is a disadvantage no SGLang diff
  fixes; a right-sized node (§4.2) is the mitigation.
- **Hit rate may be partly bought with traffic.** Effective prices moved across
  two snapshots while listed prices did not, i.e. hit rates moved. More traffic
  keeps prefixes warm, raising hit rate, lowering effective price — which is
  observationally identical to "cheaper wins traffic" and, if true, means a new
  entrant starts cold. Our measured hit rates assume traffic we do not have.
- **The market does not price on cost.** Predicted a 22% output-price gap
  between fastest and slowest providers (smaller batch ⇒ costlier tokens);
  observed 1.00x. Prices cluster on $2.55 / $3.00 / $3.20. Correlation between
  price and traffic share is -0.71 to -0.80, yet the cheapest provider holds
  7.5% share while one at 1.7x the price holds 15.9%. **Reaching the cost
  frontier is achievable; buying share with price is not demonstrated.**

---

## 7. Things we got wrong, and how they were caught

Recorded because each was stated confidently before being checked.

| Claim | Reality | Caught by |
|---|---|---|
| Qwen3.8-27B is dense with KV on all 64 layers | 16 of 64 hold KV; 64 KiB/token not 256 | reading `layer_types` |
| ...correction: it is a hybrid MoE | It is dense. No moe/expert field exists | raw `config.json` via curl |
| `cache_discount = 1.52` | Artefact of a degenerate mix design | `identifiability()` |
| `prices()` per-GPU scaling | 8x too high; `gpu_seconds` already aggregated | regression test |
| TTFT is the binding constraint | TPOT binds at market context | market-realistic run |
| N* = 4 | N* = 64; the 4 was an artefact of a 132k replay and a 2s SLO | §4.1 |
| Cache discount replicates to 1% | Only across 8xH100 runs; varies 2.5x with TP | the sweep |
| Output coefficient measured at wrong context | Coincidence; the decode mix used a 64-token prompt | checking the mix definition |
| Decode is not batching (effective batch ~3) | It batches at 96.7 | `BatchSampler` |
| Roofline predicts cost | 27% out-of-sample error, barely beating a 31% null | `validate_loo` |

**Two methodological lessons.** WebFetch's summariser is not a source for
config data — it returned invented fields (`has_moe: false`) that read as
authoritative and put a wrong architecture into the repo twice; fetch raw. And
correcting a spec silently disabled a validation guard gated on `n_experts > 1`,
which cost a run — guards should key on the constraint they enforce, not on a
proxy for it.

---

## 8. Open questions

### 8.1 The 3.5x decode gap — the largest known headroom

Step time is 21.4ms; roofline at TP=8 says 6.2ms. Since output tokens are
70–81% of the bill on real traffic, closing this is worth more than everything
else on this list. Fitting `step = roofline + overhead` gives overhead of
8.2 / 9.9 / 15.3ms at TP 2/4/8 — growing with TP, but **not** bandwidth-bound
communication: all-reduce volume at TP=8 is ~111MB/step, which NVLink moves in
0.12ms. Consistent with per-layer synchronisation across 64 layers. Unproven.

### 8.2 What determines the sustainable batch — ANSWERED, and it is memory

Read out of `schedule_policy.py` rather than guessed. SGLang admits requests
until the KV pool is exhausted, charging each running request its prompt plus a
reservation for future decode tokens:

    _get_running_request_total_token_offset(req) =
        min(max_new_tokens - len(output_ids), CLIP_MAX_NEW_TOKENS=4096)
        * new_token_ratio

`new_token_ratio` scales with `--schedule-conservativeness`. So **batch is the
KV-pool ceiling**, and since the pool scales with GPU count, so does batch —
which is the *mechanism* behind the §3.3.1(c) confound, not an artefact of how
we chose `sat_users`. No load sweep can separate them.

`predict_batch()` implements this. Against the sweep:

    GPUs  predicted  measured  error
       1         15      15.6    -4%
       2         38      40.6    -6%
       4         85      78.1    +9%
       8        179      96.7   +85%

**TP 1/2/4 sit at their memory ceiling within 9%. TP=8 reaches only ~54% of
it.** The model is right and TP=8 is the anomaly.

**This is the largest identified saving in the stack.** Cost is
`step * n_gpu / batch`, so TP=8 running at 96.7 instead of ~179 costs nearly
double what the hardware allows. It also revises §4.2: 8xH100 is not inherently
worse than 2xH100 — it is *under-filled*. At its ceiling, TP=8 would reach
`0.0214 * 8 / 179 = 9.6e-04` GPU-s per output token, comparable to 2 GPUs but
with four times the capacity.

What to try, in order: `--schedule-conservativeness` below 1.0 (raises
admission directly), `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` below 4096
(shrinks the per-request reservation), and `--mem-fraction-static` above 0.85.
`tests/test_simulator.py::test_tp8_underfills_its_own_memory_ceiling` pins the
gap; if it starts failing because the gap closed, that is the win.

### 8.2b The 1.77x reservation factor

`predict_batch` needs `RESERVATION_FACTOR = 1.77` to match: measured
bytes-per-sequence is 2.9 GB against 1.64 GB modelled (KV at 22.7k context plus
155 MB linear state). The gap is the decode reservation (4096 tokens ~ 268 MB)
plus page rounding and fragmentation — which accounts for roughly half of it.
The remainder is unexplained and the factor is fitted, so it should be
re-derived rather than trusted across a change of model or context.

### 8.3 Smaller

- TP=1 never found its N*: all four levels passed. Under-characterised, and
  possibly better than measured. ~$2.50/hr to redo.
- `BatchSampler` aliases on short requests: the `cached` mix sampled batch 0.0
  at 100% idle because its requests finish inside the 2s interval. Trust it
  only for mixes with long generations.
- Simulator validated only against our own stack, one model, one GPU type.
- No real agent traffic captured through the deployed OpenHands gateway.

---

## 9. Using it

    from autoinf.simulator import (SWEEP_2026_08_31, calibrate_step,
                                   validate_loo_step)

    model = calibrate_step(SWEEP_2026_08_31)      # -> StepModel(step_s=0.0214)
    model.gpu_s_out(n_gpu=2, batch=40)            # cost per output token
    validate_loo_step(SWEEP_2026_08_31)           # the fidelity number

Re-measure `step_s` whenever the model, GPU, or workload shape changes — and
treat a change in it as a finding, since §8.1 is the target.

    uv run python scripts/market_pull.py qwen/qwen3.8-27b   # refresh §5
    uv run python scripts/launch.py frontier --market ...   # new measurement
