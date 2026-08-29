# Handoff

Running log. Append at the top of **Session log**; keep **Status**, **Next**,
and **Open questions** rewritten to reflect current reality rather than history.

---

## Status — 2026-08-29

**Phase 0 (substrate) complete. Secrets, spend alerting and the eval suite are
done and verified. First H100 probe in flight.**

The harness runs end-to-end against a fake server and has a 10-pattern eval
suite. It has not yet met SGLang.

## Where things stand

| Piece | State |
|---|---|
| Local env (`.venv`, 3.12.5, modal 1.5.5) | done, verified |
| Modal auth (`mpeng19`) | done, `modal run` verified remotely |
| Config / search space (`config.py`) | done |
| Trace generator (`workload.py`) | done, tested |
| Metrics + goodput (`metrics.py`) | done, tested |
| Load generator (`bench.py`) | done, tested against a fake SSE server |
| Eval suite (9 patterns + mixed) | done, tested |
| Modal app (`modal_app.py`) | written, **never executed** |
| H100 probe (`probe.py`) | `probe_env` **PASS**; `probe_serve` not yet run |
| Spend monitor (`spend_monitor.py`) | done, **email delivery verified**; not deployed |
| Secrets (HF, Resend, Modal token) | done, both keys verified live |
| Weights in a Volume | not started |
| Baseline / noise floor | not started |

`uv run pytest -q` → **29 passed**.

## Next

1. **Set the Modal workspace budget** ($100) — still not done, still the only
   hard stop that exists. `SETUP.md` §2.
2. Deploy the spend monitor: `uv run modal deploy scripts/spend_monitor.py`.
3. Read `probe_env` output; fix any SGLang flags reported MISSING.
4. Run `probe_serve` — settles ignore_eos, streamed usage, prefix-cache benefit.
5. `prefetch` weights, then one smoke run.
6. **Noise floor**: identical baseline 5x, report σ of p99 TTFT and goodput.
   Go/no-go gate — if σ exceeds the effect size we care about, no amount of
   searching produces a trustworthy result. Also the direct test of whether
   Modal hands us consistent hardware.
7. Only then start turning knobs.

## Decisions

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

## Open questions

1. ~~Does the image build?~~ **Answered: yes.** 228s, sglang 0.5.18,
   torch 2.13.0+cu130, flash-attn-4, flashinfer 0.6.17.
2. ~~Are the SGLang flag names right?~~ **Answered: all 12 present.**
3. **Is `ignore_eos` honoured** by SGLang's OpenAI-compatible endpoint? Without
   it, output lengths depend on when the model emits EOS and the decode
   workload stops being controlled. `probe_serve` tests this directly by asking
   for exactly 64 tokens on a prompt the model would answer in a handful.
4. **Does Modal give consistent hardware run to run?** The noise floor answers
   this directly. If it is bad, the serverless abstraction may make Modal the
   wrong substrate for measurement, and a dedicated node elsewhere becomes the
   better call.
5. **Starter's GPU concurrency limit of 10** means one 8×H100 run at a time and
   no parallel sweeps. Fine for Phase 0/1; a Phase-2 blocker.
6. **Is `stream_options.include_usage` supported?** The client falls back to
   counting deltas, which is an approximation — a delta is not always one token.
7. **Should we upgrade the Modal image builder?** The workspace is on the legacy
   `2023.12` builder and Modal suggests upgrading. Doing so changes the base
   image for everything and forces a rebuild, so it is a deliberate choice, not
   a free one. Decide before the baseline is measured — not after.
8. **Only 4 CPUs by default.** `probe_env` asked for `cpu=4.0` and got exactly
   4, so the request is honoured. The bench function asks for 8 because it runs
   client and server together; confirm that lands, and watch
   `client_dispatch_lag_ms` on the first real run.

## Session log

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
