# What a run leaves behind

Two kinds of run, two example trees. Both are real output, copied verbatim
except that absolute paths were made relative.

## `simulate run` -> `runs/<name>/`  (`baseline-1xh100/`)

| file | what |
|---|---|
| `config.json` | the Simulator's configuration: model, GPU, levels, SLO, cost basis |
| `stack.json` | the full stack that was measured: every modified file, patches, serving overrides, env. Load it with `InferenceStack.load(run_dir)` to re-measure exactly this |
| `sweep.json` | the raw record from the GPU: per-level latency, tokens, server counters, quality, profiles, server log tail |
| `result.json` | the priced curve: N*, $/1k, rank, share, per-level economics |
| `report.txt` | the same, as the text the CLI prints |
| `slo-*.png`, `price-vs-*.png` | the frontier plots |
| `call_id`, `pid`, `run.log` | transient: the Modal call, the local process, its log (not copied) |

## `harness start` -> `agents/<session>/`  (`fleet-build-4/`, one agent, one attempt)

| path | what |
|---|---|
| `fleet.json` | the fleet's configuration: agents, budget, baseline, levels, model |
| `timeline.md` | rewritten after every outcome: per agent, each idea with its phases, cost and results |
| `daemon.log`, `daemon.pid`, `memory.db` | the daemon's log and pid; the experiment memory (not copied) |
| `profiles/<digest>.sqlite` | GPU profiles as tracedb databases, one per stack measured (not copied) |
| `a02/runs/attempt-NNN/` | one evaluation: the same files as a `simulate run` above (`-repN` for a replicate) |
| `a02/paper/<idea>/` | the write-up: `PAPER.tex`, `paper.pdf`, figures |
| `a02/calls/<phase>-<ts>.jsonl` | every model call, one line per message: tokens, tools, timing |
| `a02/spend.jsonl` | what the agent's own GPU tool calls cost, one line each |
| `a02/candidate/sglang/` | the agent's working copy of the package it edits (not copied) |
| `a02/repo/` | the agent's git history: full package at the root commit, a commit per evaluation tagged `eval/<digest>`, `win/<digest>` on replicated wins (not copied) |
| `a02/runs/attempt-NNN/commit` | the commit that evaluation was measured from |
| `a02/workbench-N/` | each `gpu-run` / `ncu` / `equivalence` call: script, stdout, result (not copied) |
| `a02/traces/` | the agent's trace: every turn, phase-timed (not copied) |

`harness --session build-4 results`, `timeline`, `calls`, `spend`, `paper`
and `ask` all read from this tree, so it is the record of the run.
