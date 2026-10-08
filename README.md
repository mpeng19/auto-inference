# auto-inference

Price an LLM serving stack the way a marketplace would, then let a fleet of
coding agents try to lower that price.

Give the **simulator** an inference stack (stock SGLang, or stock plus a diff
and a launch line) and it answers one question: *what effective price can
this stack serve OpenRouter's traffic for this model at, under a latency SLO,
and how much of that market can one node hold?* The **harness** runs Claude
Code agents that propose kernel-scale changes from an idea bank, prices each
one with the simulator, gates it on accuracy, and remembers what happened.

```mermaid
flowchart TB
  subgraph you["operator"]
    CLI["harness start · tui · results · ask · timeline · paper"]
  end
  subgraph shared["~/.auto-inference (shared across runs)"]
    BANK[("ideas.db<br/>idea bank: book + arXiv")]
    SKILLS[("skills.db<br/>facts earlier runs established")]
    STORE[("sessions.db")]
  end
  subgraph daemon["fleet daemon (one per session, under caffeinate)"]
    FLEET["fleet<br/>slots · budget · diversity"]
    MGR["manager<br/>reviews outcomes, stashes tools, writes facts"]
    A0["agent a00<br/>claude -p in its own copy of sglang"]
    A1["agent a01"]
    A2["agent a02"]
    BROKER["eval broker<br/>queue · dedup · screen slots · replicate wins"]
    MEM[("memory.db<br/>experiments + edges")]
  end
  subgraph modal["Modal, 1×H100 per evaluation"]
    APPLY["apply stack: restore stock, write diff, launch line"]
    GATES["gates: GSM8K · LongBench F1 · MMLU"]
    SWEEP["closed-loop sweep N ∈ {4,8,12,16,24}<br/>TraceLab-shaped users, device-timer GPU-s"]
    PROF["profile at one level → tracedb"]
    WB["workbench: run a script inside the stack"]
  end
  CLI --> STORE --> FLEET
  BANK -- "claim, one per agent" --> FLEET
  FLEET --> A0 & A1 & A2
  SKILLS -- "SKILL.md" --> A0
  MGR -- "tools index" --> A0
  A0 -- "EvalRequest(stack, tier)" --> BROKER
  A0 -- "gpu-run · equivalence" --> WB
  BROKER --> APPLY --> GATES --> SWEEP --> PROF
  SWEEP -- "price · rank · share" --> BROKER --> A0
  A0 -- "verdict (3% noise floor)" --> MEM
  MEM --> MGR --> SKILLS
  PROF -- "MCP tools" --> A0
```

## Quick start

```bash
uv sync                          # everything, no GPU
uv run modal token new           # your own Modal account; nothing here is shared
make deploy                      # push the runner; `harness start` refuses a runner behind the checkout
make test                        # ~390 tests, offline
```

Then, in order, each once:

```bash
# 1. stock baselines on the fleet's grid, full and screen tier (~45 min, ~$4)
uv run simulate run --root runs/baseline        --levels 4,8,12,16,24 --seconds 120 --mkdir
uv run simulate run --root runs/baseline-screen --levels 8,12         --seconds 60  --mkdir

# 2. the token-equivalence reference for this model (~5 min, ~$0.40)
uv run simulate equivalence --root runs/equiv-ref --mkdir

# 3. fill the bank: the packaged seed set, then the arXiv feed
uv run harness ideas seed
uv run harness ideas arxiv -k 15 --model opus

# 4. a fleet
uv run harness --session build-1 start --agents 3 --evals 2 --model opus \
  --bank --manager --budget 200 --agent-budget 70 --max-attempts 4 \
  --baseline '{"bill_per_1k": <full>, "quality": {"gsm8k": <acc>, "longbench": <f1>, "mmlu": <acc>}, "screen": {"bill_per_1k": <screen>}}'
uv run harness --session build-1 tui

# 5. compound: start the next fleet from the best stack
#    (`harness results` names the run; --baseline is now that run's own report)
uv run harness --session build-2 start --agents 3 --evals 2 --model opus \
  --bank --manager --budget 200 --agent-budget 70 --max-attempts 4 \
  --base agents/build-1/a02/runs/attempt-001 \
  --baseline '{"bill_per_1k": <base full>, "quality": {...}, "screen": {"bill_per_1k": <base screen>}}'
```

Step 5 is the loop that compounds: every agent's "stock" becomes the base
(its edits, diff and no-change check are all relative to it), the stack it
evaluates is base plus its own edits, and bank claims are steered toward the
idea that produced the base. Repeat with the winner of each fleet -- or let
the campaign do it:

```bash
# 4+5, unattended: fleets in sequence, each on the last round's best
# publishable result, until the bill has halved or four rounds are spent
uv run harness --session camp-1 campaign start --rounds 4 --target 2.0 \
  --agents 3 --evals 2 --model opus --bank --manager --budget 200 \
  --agent-budget 70 --max-attempts 4 --baseline '{...}'
uv run harness campaign status --session camp-1     # rounds, bases, baselines, gain, the chain
uv run harness --session camp-1-r1 tui              # each round is an ordinary session
uv run harness campaign stop --session camp-1       # now; `harness --session camp-1-r2 stop` ends after that round
```

Round `n` runs as session `camp-1-r<n>` under `agents/camp-1/r<n>/`. When it
ends, the driver picks the round's best publishable result (falling back to
the best replicated win, and saying so), makes its run directory the next
`--base`, and sets the next `--baseline` from that run's own report -- the
full bill, the screen bill from the same stack's screen attempt (scaled by
the fleet's screen/full ratio when there was none) and its accuracy per
suite. Every idea earlier rounds tried is passed as `avoid`, and the base's
own idea seeds the claims. `agents/camp-1/campaign.json` is rewritten after
every round.

The baseline numbers come from step 1's reports. `harness start` refuses a
baseline it cannot score against, and refuses to start without a bank:
either one missing is a way the fleet runs all night and learns nothing. Step
2 is not checked, because the reference is built on first use -- running it
once up front just means no agent pays the five minutes for it.
`docs/methodology.md` has the findings that shaped these defaults and
`docs/examples/README.md` what a run leaves behind.

## How a stack is priced

1. **Sweep concurrency** with N closed-loop users replaying real coding-agent
   sessions (TraceLab) rescaled to the marketplace's token mix, 20,583 in and
   2,076 out per request. The highest N that holds the SLO (p90 TTFT ≤ 2,818
   ms, p90 TPOT ≤ 25 ms, mean TPOT ≤ 20 ms) is N\*. The report also prints
   where the binding metric crosses its limit by interpolation, because N\*
   is quantised to the grid and a level on the line flips on noise.
2. **Read GPU time by phase** at N\* from SGLang's CUDA-event device timer.
3. **Divide**: extend seconds over all input tokens, decode seconds over
   output tokens, at $3.00 per GPU-hour and 50% utilisation. No regression;
   cache hit rate is an outcome, not a control.
4. **Place it on the board**: the whole bill per 1,000 market requests
   against every OpenRouter provider for the model, and the share of daily
   demand one node serves at that price.

Stock SGLang 0.5.18 serving Qwen3.8-27B-FP8 on one H100 prices at $8.32 per
1k requests at N\*=12, rank 1 of 12 on the board, about 0.6% of the market
per GPU. The first fleet's best replicated change, an SLO-budgeted chunked
prefill sizer, took that to $6.94 with the model's outputs unchanged.

## The harness

Every service is a Protocol in `harness.contracts` and replaceable alone.
These six shape a run:

| service | question it answers |
|---|---|
| `IdeaBankService` | where ideas of the right size come from; content-addressed records, claimed one per agent, least similar first or steered by a seed |
| `AgentService` | one idea, iterated: recall → edit → check → screen → confirm → replicate |
| `EvalService` | the queue in front of the GPUs: dedup, screen slots, spend per attempt |
| `MemoryService` | every experiment with typed edges; a synthesised brief on recall |
| `SkillBankService` | facts earlier runs established; manager-written, agent-read, contradictions supersede |
| `OrchestrationService` | N agents kept diverse and inside a budget; the manager |

`ContextService` holds the transcripts behind those claims and `SessionStore`
the live snapshot a watcher reads; neither is something an agent calls.

## Reading a run

```bash
uv run harness --session S tui            # fleet tab: agents, $/1k, rank, share, time by phase, recent calls
                                          # results tab: experiments best first, diff, `a` to ask Claude about the run
uv run harness --session S results --diff
uv run harness --session S ask "which attempt touched the attention backend?"
uv run harness --session S timeline       # ideas, phases, results, cost per agent; --html for a Gantt
uv run harness --session S paper          # write-ups, one per idea that reached a full sweep
uv run harness --session S calls -v       # every model call, per-message tokens and tools
uv run harness traces show <id> --root agents/S --kind eval_submit --full
uv run harness --session S spend          # Modal dollars: evaluations, the agents' own GPU tools, orphans
git -C agents/S/a02/repo log --oneline    # the agent's history: a commit per evaluation, tags eval/<digest>, win/<digest>
uv run harness skills list                # the facts
uv run harness ideas claim --seed "..."   # one record, steered; `ideas related <id>` for its neighbours
uv run harness delete --session S         # wipe a finished fleet's directory and rows
```

Pause is `p`, resume is `r`; an agent never resumes on its own. The daemon
runs on the machine you started it from, under `caffeinate`; a closed lid still
freezes it, and the status line says so.

## Layout

```
src/simulator/    the pricing: stack, sweep runner (Modal), SLO, market, quality gates, plots
src/harness/      the loop: contracts, agent, orchestration, ideas, skills, manager, tui
src/tracedb/      GPU profiles as a queryable database, with an MCP server
docs/methodology.md   how the method was arrived at, with every negative result
src/harness/ideas/seeds/   the packaged idea-bank seed set (the book's 27 mechanisms)
docs/examples/        what a run leaves behind, one tree per kind of run
runs/, agents/        your artifacts; ignored by git, as is docs/NEXT.md (your planning notes)
```

## Assumed, not measured

$3.00 per GPU-hour, 50% utilisation, break-even pricing, the SLO above, and the
market snapshot's demand and provider prices (`make market` refreshes it).
Everything else in a report comes from a GPU timer.
