# auto-inference

Price an LLM serving stack the way a marketplace would.

Give it an **inference stack** — stock SGLang, or stock plus a set of diffs to
`srt/` — and it answers one question: *what effective price can this stack
serve marketplace traffic at, and how much of that market can it hold?*

```python
from simulator import Simulator, InferenceStack

sim = Simulator(root_dir="runs/my-diff",
                stack=InferenceStack.from_dir("my-diff/"))   # or .stock()
result = await sim.eval()
print(result.summary())
```

```
N* = 12 users   batch 5.0   hit 0.748   7.34 GPU-s per market request
  effective input  $0.0294/M
  output           $5.6021/M
  whole bill       $12.23 per 1k requests   rank 9/12
  one node serves  0.42% of the market  (5,885 req/day)
  on effective input alone: rank 1/12  -- the metric OpenRouter sorts on,
  not what a buyer pays
```

Every artifact lands in `root_dir`: the raw sweep, the priced curve, a report,
and one figure per question — a `slo-<bound>.png` for each SLO bound (aggregate
throughput on y, the constraint on x, one point per concurrency, the forbidden
region shaded), plus `price-vs-share.png` and `price-vs-demand.png`.

## Setup, from a fresh clone

```bash
uv sync                    # dependencies
uv run modal token new     # your own Modal account -- nothing here is shared
make deploy                # push the runner
make test                  # 112 tests, no GPU, no network
```

That is the whole setup. No Hugging Face token is needed: the default
checkpoint is Apache-2.0 and ungated. No account-specific names are baked in —
the Modal volumes are created on first use in whichever workspace you are
logged into. Override any of it if you want to:

```bash
export SIMULATOR_HF_SECRET=huggingface     # only for a gated model
export SIMULATOR_APP_NAME=my-app
export SIMULATOR_MARKET_DATA=/path/to/snapshot.json
```

To price a **different model**, set `model=` and pull that model's market data
with `make market` — the snapshot supplies the demand denominator and the
provider board, and both are model-specific.

Start with **`docs/example.ipynb`**, which runs the whole thing end to end
against a stored sweep in a few seconds. `docs/examples/baseline-1xh100/` is
what a real run leaves behind.

## The method

1. **Sweep offered load** on real coding-agent traffic rescaled to the
   marketplace's own token mix (20,583 in / 2,076 out), until the SLOs stop
   holding. The last level that held is `N*`. Every evaluation needs its own
   sweep: a diff moves `N*`, and pricing it at the baseline's would understate
   every latency win.
2. **Read phase-split GPU time** at `N*` from SGLang's CUDA-event device timer
   (`forward_execution_seconds_total`, labelled `extend` and `decode`).
3. **Divide.** `eff_in = extend / ALL input tokens`, `out = decode / output
   tokens`. No regression. Splitting input into cached and uncached is needed
   only to re-blend at someone else's hit rate, and caching well *is* serving
   well — so we price at our own. **Cache hit rate is an outcome, never a
   control.**
4. **Score both ways.** Effective input price is what OpenRouter sorts on; the
   whole bill is what a buyer pays. On the current baseline those give rank 1
   and rank 9, so the report always carries both.

## The auto-research harness

`src/harness/` is a fleet of agents that propose diffs to SGLang's `srt/` and
price them with the simulator. Four services, each defined by a Protocol in
`harness.contracts` and each replaceable without the others noticing:

| service | question it answers |
|---|---|
| `MemoryService` | has anyone tried this, and what happened? |
| `ContextService` | how exactly did they do it? |
| `EvalService` | the queue in front of the GPUs |
| `SessionStore` | the seam between a running fleet and anything watching it |
| `AgentService` | one idea, iterated, with divergence and retry policy |
| `OrchestrationService` | N of those at once, kept diverse and in budget |

**Agents are Claude Code processes.** Each runs as `claude -p` in its own
workspace directory, edits the files there, and the harness reads the diff
back. Rebuilding a coding agent badly is weeks of work for something worse;
what this adds is the part Claude Code does not have — a fleet, a shared memory
of every experiment, and a priced evaluation of the diff. Token usage comes
back in the JSON envelope, which is what makes per-agent cost real on the
dashboard rather than estimated.

Prefer `--model sonnet` or `opus`. A reasoning-heavy frontier model spends its
budget thinking about a task whose difficulty lives in the codebase, not in the
prompt.

```bash
harness start --agents 10 --evals 3 --model sonnet --budget 500   # detached
harness tui                        # watch and steer it
harness status                     # one-shot, scriptable, --json
harness scale 6                    # add or remove agents in flight
harness agent pause a03            # pause / resume / kill one agent
harness stop                       # graceful: finish paid work, then wind up
harness kill                       # flat: everything, now
```

`start` is asynchronous: a fleet runs for hours and must outlive the terminal.
Everything after it talks to a SQLite session store, so the CLI and the TUI are
two clients of the same interface and neither needs the other to exist. Add
`--dry-run` to fake the GPUs (saves dollars) and `--fake-agents` to fake Claude
Code too (saves subscription usage); together they exercise the whole fleet for
free.

The TUI is deliberately small — one table, one detail pane, six keys
(`p`/`r`/`k` per agent, `+`/`-` to scale, `s` to stop). It answers four
questions and stops: who is running and what is each doing right now, what has
it cost in dollars and tokens per agent, are the GPUs busy or is the fleet
stalled behind them, and which agent do I want to pause.

```
demo   running   4/4 agents   $28.00 of $60   updated 0s ago
evals: 2 running, 4 queued, 35 done, 0 deduped, 100% GPU utilisation

agent  status      idea                  att      Δ%       $  activity
a00    evaluating  prefill chunking        3   -10.3    2.60  attempt 3: full running; studying meanwhile
a03    evaluating  queue ordering          1   -11.7    5.60  attempt 1: screen running; studying meanwhile
```

**Tools agents can call.** An agent's feedback loop is otherwise one bit every
25–60 minutes. These put signal in front of that:

```bash
harness tool recall "raise chunked prefill"    # what the fleet already tried
harness tool roofline --batch 12               # predicted step time and $/M
harness tool preflight --workspace agents/a01  # parse + undefined-name check
```

`recall` matters most — memory is injected once per attempt, but an agent with
a surprising result should be able to *ask*. `preflight` is the cheap half of
an evaluation: a NameError costs six GPU-minutes to find on a GPU and nothing
to find here.

**Traces.** Every agent run writes append-only JSONL, one turn per line, each
line self-contained so `cat traces/*.jsonl | loader` loses nothing.
`docs/trace-schema.md` is the spec a downstream profile database can build
against.

```bash
harness traces list                       # what exists, by session and agent
harness traces show <id> --kind eval_submit --full
harness traces export --out DIR           # + a manifest with line counts
```

An agent's whole interface to the code is a `Workspace`:

```python
ws = Workspace("agents/a01")
ws.replace("srt/managers/schedule_policy.py", old, new)   # refuses if ambiguous
ok, why = ws.check()                                      # parses? actually changed?
result = await Simulator(root_dir=ws.run_dir(), stack=ws.stack()).eval()
```

Stock SGLang is fetched from the pinned wheel by URL and cached across the
fleet — 0.5.18 publishes only manylinux wheels and no sdist, so `pip download`
on a Mac cannot get it, and an agent needs the source, not an install.

**On isolation:** the modified SGLang only ever executes inside a fresh Modal
container, so the isolation that matters is already paid for. Agents write text
and wait on an API. The resource they genuinely contend over is **GPU
concurrency and money** — every attempt rents an H100 for 25–60 minutes — so
per-agent directories are enough.

**On keeping agents busy.** That scarcity is a *queue*, not a lock. An agent
submits and keeps working; it blocks only on `collect`, and only after it has
run out of useful things to do:

```python
ticket = evals.submit(EvalRequest(stack=ws.stack(), tier="screen"))
proposer.study(...)          # read other agents' traces, refine the hypothesis
rec = evals.collect(ticket.id)
```

Three mechanisms keep utilisation high, in descending order of what they buy:

1. **Tiers.** A screening run is a fraction of a full sweep and most candidates
   die in it. Capacity is reserved for screens so a backlog of confirmations
   cannot starve the tier that does the filtering.
2. **Dedup by content.** A stack digest is a content hash, so identical
   proposals share one GPU run. Writing the tests surfaced how common this is:
   ten agents seeded from one baseline produced **twenty proposals and one
   run**. Only successes are cached — a failure is a property of the moment,
   and memoising one turns an infra retry into an infinite loop.
3. **Fair ordering.** FIFO within a priority band, so a fast agent cannot barge
   repeatedly and starve a slow one.

`AgentOutcome.idle_s` and `QueueStats.utilisation` report whether it is
working, and tests assert that every `study` call happens with an evaluation
actually in flight.

## Quality, not just speed

Every sweep scores **GSM8K** on an idle server before load, pinned by dataset
revision. The reason is structural: an agent maximising goodput has an obvious
cheat — serve worse answers faster — and nothing in the price model can see it.

```
N* = 12 users   batch 5.0   hit 0.748
  whole bill       $12.23 per 1k requests   rank 9/12
  quality gsm8k:   72.0%  (+0.0 pts)
```

A regression beyond 10 percentage points is rejected as a **failed hypothesis**,
not an infra failure — re-running would reproduce it. The tolerance is wide
because FP8 greedy decoding is not bitwise deterministic: two stock sweeps on
the same 50 items scored 62% and 70%. The gate catches a stack that serves a
different model, not a subtle drift.
`canary.py` still runs, but it digests six short outputs and would miss a
subtle numerical degradation; this is the gate that catches it.

Set `quality_suites=("gsm8k", "mmlu")` for both. MMLU is one token per item so
it is cheap, but a single token is a weak per-item signal; GSM8K's multi-step
reasoning compounds a small numerical error into a wrong answer, which is the
sensitivity worth paying for.

**Cross-check.** `measure/crosscheck.py` runs `sglang.benchmark.serving`
against the same server and compares TTFT/TPOT percentiles with ours. Two
independently written clients agreeing is real evidence; disagreeing means one
has a bug. It is deliberately not a dependency — SGLang moved
`sglang.bench_serving` to `sglang.benchmark.serving` in the version we pin, and
an actively reorganised surface is a bad thing to put in the path of a price.

## GPU profiles

`src/tracedb/` is a queryable database for GPU profiling traces, built for
agents. A kineto trace is hundreds of thousands of events; no agent reads one.
But trace debugging is pattern matching, and those are queries.

```bash
simulate run --root runs/x --profile-level 8      # capture during the sweep
simulate profile --dir <dir from the record>      # download and ingest
tracedb --db profiles/trace.sqlite summary
tracedb --db profiles/trace.sqlite gaps attn_out mlp_in --min-gap 100
tracedb --db profiles/trace.sqlite idle --min-gap 50
```

This is what turns *"decode runs at 28% of memory bandwidth"*
(`docs/methodology.md` §8.3) into *"and here is the kernel where it goes"*.
Profiling perturbs what it measures, so it runs at one level only and the price
still comes from N\*.

Agents reach it as **MCP tools**, not through the shell — `harness.profile`
generates a `--mcp-config` per agent so `trace_summary`, `trace_gaps`,
`trace_gpu_idle` and `trace_slowest` appear as typed tools they can discover.

## Layout

```
src/simulator/          the product
  api.py                Simulator, EvalResult  <- start here
  stack.py              InferenceStack: diffs to sglang, carried by value
  slo.py                bounds that name their own order statistic
  costs.py              $/GPU-hour by provider
  config.py specs.py    serving config and model/hardware specs
  workload/             request loading and sanitisation (TraceLab -> market mix)
  measure/              load generation, latency, server counters, canaries
  price/                GPU-seconds -> price -> market share
  runner/               where a sweep executes (Modal, one entrypoint)
  artifacts/            report and figures written into root_dir
  conftest.py           shared fixtures for every test below
  */tests/              tests live beside the service they test
src/harness/            the auto-research harness
  contracts/            the service Protocols; no logic lives here
  memory/               SqliteMemory: experiment graph, FTS, briefs
  context/              JsonlContext: append-only agent traces
  session/              SqliteSessionStore: the fleet/TUI seam
  agent/                Workspace (the diff API), the loop, the Claude Code proposer
  orchestration/        Fleet + EvalBroker: N agents, one queue for the GPUs
  tui/                  the dashboard
  profile.py            points agents at a captured GPU profile over MCP
  daemon.py cli.py      `harness start|status|tui|scale|agent|stop|kill`
src/tracedb/            queryable GPU trace database (`tracedb`, `tracedb-mcp`)
monitor/                Modal spend monitoring
docs/methodology.md     how the method was arrived at, and every negative result
docs/example.ipynb      minimal end-to-end notebook
docs/examples/          what a finished run leaves behind
```

## Commands

```
make test                          # 112 unit tests, no GPU, no network
make lint                          # ruff

# tests sit beside the code, and every test directory is a package, so any
# of these resolve from any working directory:
pytest src/simulator/price                 # one service
pytest src/simulator/tests/test_slo.py     # one file
pytest --pyargs simulator                  # an installed copy, no repo needed
make deploy                        # push the runner
make run     ROOT=runs/baseline    # full evaluation
make submit  ROOT=runs/baseline    # start it and walk away
make collect ROOT=runs/baseline
make rescore ROOT=runs/baseline ARGS='--slo ttft:p99:1000,tpot:mean:20'
```

`rescore` is worth knowing about: every level stores its full percentile set
and raw counters, so changing the SLO, the cost basis or the utilisation
assumption costs nothing. Which order statistic the frontier is judged at is a
choice we have changed three times.

## What is assumed, and what is measured

Measured: GPU-seconds per token class, latency percentiles, cache hit rate,
running batch, market prices and volumes.

Assumed, and reported with every result because they cannot be measured from
inside the harness: the **GPU rate** and the **utilisation**. Both are fields on
`Simulator`, and `sim.assumptions()` returns the complete set.

Rates live in `simulator/costs.py`, keyed by provider — `gpu_provider=None`
gives the agreed $3.00/hr H100 default, `gpu_provider="nebius-committed"` gives
$2.50. Serverless retail (Modal's $3.95) is in the catalog but refused as a
serving basis unless you pass `allow_retail_rate=True`: it is what we pay to
run experiments, and pricing a serving business against it would flatter every
competitor by ~30%.

Utilisation is the single largest lever — at 25% every price doubles.
