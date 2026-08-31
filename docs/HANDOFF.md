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

- Best realised effective input: **Chutes $0.1310** at 69.5% hit. That is the
  number to beat.
- Weighted average actually paid: $0.2149 in / $2.866 out.
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

**Qwen3.8-27B on 8xH100 TP=8/EP=8, full-scale TraceLab:**

- N* = 4 (TTFT-bound at 2s), goodput 0.68 rps, TPOT 7.3ms.
- TP=8 fixed decode entirely: TPOT 54ms on 1 GPU -> 7.3ms on 8.
- **TTFT is the binding constraint.** At 88% hit on a 132k context you still
  prefill ~16k new tokens per turn.
- **The cost attribution from this run is NOT trustworthy** — see §5.

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

- **Qwen3.8-27B is not dense.** It is a hybrid MoE. `layer_types` interleaves
  three linear-attention layers per full-attention one, so **only 16 of 64
  layers hold growing KV**. KV is **64 KiB/token, not 256** — a 4x
  overestimate. One 132k conversation is 8.66 GB, not 34.6. An L40S holds ~2,
  not "cannot hold one". It is the *most* memory-efficient of our three
  models, not the least.
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
