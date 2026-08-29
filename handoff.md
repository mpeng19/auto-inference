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
| Modal app (`modal_app.py`) | **first end-to-end run green** |
| H100 probe (`probe.py`) | both stages **PASS** |
| Source overlays (`overlay.py`) | built; `schedule_policy.py` vendored |
| Correctness gating | canaries built, **floor measured: 6/6 exact** |
| Results tooling | `scripts/results.py` — ls / show / compare / pull |
| 1-GPU workflow | `smoke`, `suite`, `noise` entrypoints |
| 8xH100 path | `suite_8x`, `prefetch_big` — written, never run |
| Spend monitor (`spend_monitor.py`) | done, **email delivery verified**; not deployed |
| Secrets (HF, Resend, Modal token) | done, both keys verified live |
| Weights in a Volume | not started |
| Baseline / noise floor | **done — goodput CV 0.0003** |

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

## Open questions

1. ~~Does the image build?~~ **Answered: yes.** 228s, sglang 0.5.18,
   torch 2.13.0+cu130, flash-attn-4, flashinfer 0.6.17.
2. ~~Are the SGLang flag names right?~~ **Answered: all 12 present.**
3. ~~Is `ignore_eos` honoured?~~ **Answered: yes.** Asked 64, got 64.
4. ~~Does Modal give consistent hardware?~~ **Answered: yes, remarkably.**
   Under real load across 5 separate server launches, goodput CV is **0.0003**
   and p99 TTFT CV is 0.0399. Effects below 1% are detectable.
5. **Starter's GPU concurrency limit of 10** means one 8×H100 run at a time and
   no parallel sweeps. Fine for Phase 0/1; a Phase-2 blocker.
6. ~~Is `stream_options.include_usage` supported?~~ **Answered: yes**, and
   deltas != tokens (62 vs 64), so counting deltas would have biased TPOT ~3%
   on every request. `bench.py` uses `usage`, which was the right call.
9. ~~Why is a warm prefix cache SLOWER?~~ **Answered — and the cache is fine.**
   See the session log below. Short version: a *full* cache hit is slower than
   a cold prefill, but a *partial* prefix match is much faster, and
   `prefix_heavy` generates partial matches. It is trustworthy.
10. ~~Correctness gating does not exist.~~ **Built, and the floor is 6/6
    exact.** See below — outputs are bitwise reproducible, so the gate is
    sharper than expected. Original concern retained for context: Overlays make the serving code
    editable, which means it is now possible to "win" by breaking correctness —
    truncating outputs, dropping requests, altering sampling. Output-equivalence
    checks at temp=0 against a stock baseline must land before any overlay is
    trusted, and certainly before an agent is allowed to write one.
7. **Should we upgrade the Modal image builder?** The workspace is on the legacy
   `2023.12` builder and Modal suggests upgrading. Doing so changes the base
   image for everything and forces a rebuild, so it is a deliberate choice, not
   a free one. Decide before the baseline is measured — not after.
8. **Only 4 CPUs by default.** `probe_env` asked for `cpu=4.0` and got exactly
   4, so the request is honoured. The bench function asks for 8 because it runs
   client and server together; confirm that lands, and watch
   `client_dispatch_lag_ms` on the first real run.

## Session log

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
