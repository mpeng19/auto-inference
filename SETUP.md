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

## 5. GPU probes — run these before anything else

The probe answers the questions that would otherwise invalidate every later
experiment. Cheap stage first:

```bash
# ~2 min once the image is built, no weights downloaded. Checks the image
# builds, what hardware we get, and whether all 12 SGLang flags we emit are real.
uv run modal run src/autoinf/probe.py::probe_env

# ~15 min including a 31GB download. Checks ignore_eos, streamed usage
# accounting, and whether the prefix cache actually helps.
uv run modal run src/autoinf/probe.py::probe_serve
```

Then the benchmark itself:

```bash
uv run modal run src/autoinf/modal_app.py::prefetch --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
uv run modal run src/autoinf/modal_app.py     # 60-request smoke run
```

## 6. Eval suite

Nine patterns plus a mixed stream, in `src/autoinf/workload.py::suite()`.
`sustained` and `bursty` deliberately carry the same mean rate so any
difference between them is attributable to burstiness alone.

| workload | shape | stresses |
|---|---|---|
| `sustained` | Poisson, CV~0.9 | baseline |
| `constant` | clockwork, CV=0 | control: cost of arrival variance alone |
| `bursty` | 4x bursts, CV~4.1 | queueing, batch formation |
| `ramp` | 2 -> 32 rps | locates the saturation knee |
| `spike` | 10x step for 10s | admission control and **recovery** |
| `prefill_heavy` | ~3800 in / ~30 out | chunked prefill, prefill/decode interference |
| `decode_heavy` | ~55 in / ~890 out | KV growth, batch residency, ITL |
| `prefix_heavy` | 89% shared prefixes | radix cache, cache-aware routing |
| `short_chat` | ~36/53 tok at 24 rps | per-request scheduling overhead |
| `mixed` | merge of four | class interference — where schedulers usually fail |

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
