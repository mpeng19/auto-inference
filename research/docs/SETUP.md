# Setup

Everything needed to run the harness. Items marked **[you]** need a human —
a browser, a signup, or a payment method — and cannot be scripted.

## Status of the four things you asked about

| # | Item | Status |
|---|------|--------|
| 1 | Hugging Face token | **Done.** Stored as Modal Secret `huggingface`. Verified valid (user `mpeng19`, write role). Not strictly required — both models are ungated Apache-2.0 — but set up anyway. |
| 2 | Spend monitoring + email | **Done and verified.** Test email delivered via Resend. Secret `auto-inference-resend`. |
| 3 | Other APIs | Only **Resend**. Done. |
| 4 | Harness | **Built** (`src/autoinf/`). 29 local tests pass. Not yet run on a GPU. |
| 5 | Eval suite | **Built.** 9 workload patterns + a mixed stream. |

---

## 1. Local environment — done

```bash
uv sync --group dev      # 3.12.5 venv, modal 1.5.5, pytest, aiohttp
uv run pytest -q         # 17 passed
```

Modal is authenticated as profile `mpeng19`.

## 2. Modal account — **[you]**

- [ ] **Set a workspace budget.** Dashboard → Settings → Usage & Billing.
      This is a **hard stop**, not an alert: Modal kills running workloads
      when it is hit. It is the only thing standing between a forgotten
      8×H100 container and a four-figure bill. Suggested first value: **$100**.
- [ ] **Add a payment method** if you want to exceed the $30/month Starter credits.
      $30 is roughly *one hour* of 8×H100 ($31.60/hr), so this is needed before
      any serious Phase-2 work.
- [ ] Note the plan limit: **Starter caps GPU concurrency at 10.** An 8×H100 run
      consumes 8 of those, so you can run exactly one at a time and cannot
      parallelise a sweep. Team ($250/mo) raises it to 50. Not needed for
      Phase 0/1 on a single H100.

## 3. Email alerts — **done**

Modal sends no spend email of its own, so the monitor uses [Resend](https://resend.com)
(free tier: 100/day, 3,000/month; `onboarding@resend.dev` sends without domain
verification).

Three Secrets exist in the workspace — `huggingface`, `auto-inference-resend`,
`auto-inference-modal-token`. Values live only in Modal, never in this repo.
A test digest was delivered successfully.

- [ ] **Still to do:** deploy the daily digest (16:00 UTC):

```bash
uv run modal deploy scripts/spend_monitor.py
```

Test it without sending anything first:

```bash
uv run python scripts/spend_monitor.py           # dry run, prints the email
uv run python scripts/spend_monitor.py --send    # actually sends
```

## 4. Hugging Face — done

Secret `huggingface` holds `HF_TOKEN`; verified valid. Not needed for the
current ungated models, but available. To use it, uncomment the `secrets=[...]`
line in `src/autoinf/modal_app.py`.

## 5. Running things

Everything is a `make` target. All run on one H100 unless stated.

```bash
make test        # 57 local tests, no GPU
make suite       # eval suite, ~10 min of traces (+4 min model load)
make suite30     # 30 min of traces -- the consistency run
make staircase   # step to 100% of roofline, stop when SLO collapses
make noise       # same config 5x: the noise floor
make realism     # LLM-driven virtual users (multi-turn, abandonment)
make results     # list stored runs
make ledger      # research-loop health
make spend       # current Modal spend
```

Every `suite` run appends to `runs/ledger.jsonl` and prints a health verdict.

### Consistency

`make noise` measures run-to-run variance: five separate server launches of an
identical config. Last measured **goodput CV 0.0003, p99 TTFT CV 0.040** — so
effects below 1% are detectable. Re-run this after any change to hardware,
model, engine version or the client, because every comparison downstream is
only as trustworthy as that number.

Each workload also carries a `client_health` verdict. A run marked `SUSPECT`
means the load generator could not keep up and its server metrics are not
evidence — discard it rather than interpreting it.

### Reading results

```bash
uv run modal run scripts/results.py::ls
uv run modal run scripts/results.py::show --name <file>
uv run modal run scripts/results.py::compare --a <file> --b <file>
```

`compare` refuses to treat two runs as comparable unless their **trace digests
match**. Measuring two configs against different traffic is the easiest way to
manufacture an improvement that is not there.

## 6. Modifying the serving code

Not limited to SGLang's CLI flags. Any module under `sglang/` can be replaced:

```bash
# Pull a stock module into overlays/ (records its upstream SHA)
uv run modal run scripts/vendor.py::main --path srt/managers/scheduler.py
uv run modal run scripts/vendor.py::ls --pattern mem_cache   # browse

# Edit overlays/sglang/srt/managers/scheduler.py, then just run a bench.
# No image rebuild -- overlays are mounted at container start.
```

Already vendored: `srt/managers/schedule_policy.py` (1500 lines),
`srt/mem_cache/radix_cache.py` (862 lines).

If SGLang is upgraded and an overlaid file changes upstream, `overlay.apply()`
**refuses to run**. A stale overlay would silently revert upstream fixes while
still looking like a valid experiment.

### Before trusting any overlay result

Run `noise` first. It reports a **canary floor**: how much the model's output
differs between two runs of the *same* config. Greedy decoding is not bitwise
deterministic across batch compositions, so some divergence is normal. A config
that diverges materially more than that floor is suspect, and its goodput gain
should not be believed.

## 7. Scaling to 8xH100 (Phase 2, not yet run)

```bash
# One-time: pull the 235B weights. CPU-only -- this needs no GPU, and running
# it on 8xH100 would cost $31.60/hr for a network transfer.
uv run modal run src/autoinf/modal_app.py::prefetch_big

# Full suite, 8xH100, TP=8 EP=8 -- ~$31.60/hr
uv run modal run src/autoinf/modal_app.py::suite_8x
```

`Qwen3-235B-A22B-Instruct-2507-FP8` is **236.4 GB across 24 shards**, about
**$20/month** to keep on a Modal Volume. Delete it when not in active use.

Consumes 8 of the Starter plan's 10 GPU concurrency, so nothing runs alongside
it. Pin `--region` once a baseline exists so later comparisons are like-for-like.

**Re-derive the suite rates before trusting any 8-GPU number.** The current
ones are calibrated to the 1xH100 knee and mean nothing at that scale; run
`staircase` first.

## 8. Research-loop monitoring

```bash
make ledger
```

`runs/ledger.jsonl` is append-only: one line per experiment, with the config,
overlay digest, result, and attributed cost. The report scores the **search**
rather than the result, and flags:

| flag | meaning |
|---|---|
| `CIRCLING` | recent experiments are near-repeats of earlier ones |
| `NARROW` | attention concentrated on one knob; the rest of the space untouched |
| `PLATEAU` | no new best for N experiments |
| `INTEGRITY` | goodput gained while canaries diverged or requests were dropped |
| `EXPENSIVE` | dollars per 1% of improvement has gone up |

`INTEGRITY` is the one that matters most. A search maximising goodput can "win"
by dropping slow requests, and that looks identical to a real improvement from
the outside.

## 9. Capturing real agent traffic

A deployed OpenAI-compatible endpoint with a recording proxy in front of
SGLang. Point a real agent at it, let it work, and it records the traffic
*shape* without ever being in the measurement loop.

    https://mpeng19--auto-inference-agent-endpoint.modal.run/v1

OpenHands: Settings -> LLM

| field        | value |
|---|---|
| Custom Model | `openai/Qwen/Qwen3-4B-Instruct-2507-FP8` |
| Base URL     | the URL above |
| API Key      | the `auto-inference-gateway` secret |

Run OpenHands in **Docker mode with a scratch directory mounted**, not this
repo. An agent doing a coding task has no reason to touch the harness measuring
it, and "no reason to" is not isolation.

```bash
uv run modal run src/autoinf/modal_app.py::traces                 # list
uv run modal run src/autoinf/modal_app.py::traces --name <file>   # summarise
```

The number to look at is `prefix_reuse_frac`: what fraction of each prompt is a
verbatim repeat of the previous turn. Chat sits around 0.5; a coding agent
resending file contents and tool output should be far higher, and that is what
makes prefix caching decisive rather than incidental for this traffic.

Why this matters: our synthetic workloads are chat-shaped -- ~500 tokens in,
decode-bound. A coding-agent turn is 8k-90k tokens in and ~99% prefill-bound,
183x to 768x the work per request, on the opposite side of the roofline. An
optimisation tuned on the current suite could be irrelevant, or actively wrong,
for agentic traffic.

## 10. GPU probes -- already run, re-run after any SGLang upgrade

```bash
uv run modal run src/autoinf/probe.py::probe_env     # image, hardware, flags
uv run modal run src/autoinf/probe.py::probe_serve   # ignore_eos, usage, cache
uv run modal run src/autoinf/probe.py::probe_prefix  # prefix-cache diagnostic
```

---

## Cost reference (live from your account, 2026-08-29)

| GPU | $/hr | ×8 |
|-----|------|-----|
| H100 | $3.95 | $31.60 |
| H200 | $4.54 | $36.32 |
| B200 | $6.25 | $50.00 |
| A100 80GB | $2.50 | $20.00 |
| L40S | $1.95 | $15.60 |

Volumes $0.09/GiB/month — the 30B FP8 dev model (~31GB) is ~$2.80/mo to park,
the 235B target (~235GB) ~$21/mo.

Refresh with `uv run modal billing rates`.
