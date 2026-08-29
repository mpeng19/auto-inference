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

## 5. The 1-GPU test case

This is the day-to-day loop. One H100 ($3.95/hr), the 30B FP8 model, weights
already cached in a Volume.

```bash
# Smallest end-to-end run: 60 requests, ~10 min, ~$0.70
uv run modal run src/autoinf/modal_app.py::smoke

# Full eval suite -- all 9 workloads against ONE server launch
uv run modal run src/autoinf/modal_app.py::suite

# Shorter/longer: scale multiplies request counts
uv run modal run src/autoinf/modal_app.py::suite --scale 0.3

# Noise floor: same config, N separate server launches
uv run modal run src/autoinf/modal_app.py::noise --repeats 5
```

The suite runs every workload against a **single** server launch. Model load is
~350s cold and dominates a short trace, so launching per workload would spend
most of the budget loading rather than measuring. Anything that varies per
launch (a `ServingConfig` field, an overlay) needs a separate call; traffic
shape does not.

### Reading results

Every run is written to the `auto-inference-results` Volume as self-describing
JSON -- config, trace digest, overlay digest, provenance, per-request records.

```bash
uv run modal run scripts/results.py::ls
uv run modal run scripts/results.py::show --name <file>
uv run modal run scripts/results.py::compare --a <file> --b <file>
uv run modal run scripts/results.py::pull      # copy them all locally
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

## 7. Scaling to 8xH100 (Phase 2)

```bash
# One-time: pull the 235B weights (~235GB, ~$21/mo to keep parked)
uv run modal run src/autoinf/modal_app.py::prefetch_big

# Full suite, 8xH100, TP=8 EP=8 -- ~$31.60/hr
uv run modal run src/autoinf/modal_app.py::suite_8x
```

Consumes 8 of the Starter plan's 10 GPU concurrency, so nothing else can run
alongside it. Pin `--region` once a baseline exists so later comparisons are
like-for-like.

## 8. GPU probes -- already run, re-run after any SGLang upgrade

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
