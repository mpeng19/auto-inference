# Agent trace schema (v1)

What a fleet writes while its agents work, and the contract a downstream
profile database can rely on. This is for **debugging what an agent did** —
which tools it called, in what order, how long it waited, where its tokens
went. It is not the cost/experiment record; that lives in the harness's own
memory store and is a separate concern.

## Where the files are

```
<fleet root>/traces/
  trc_<id>.jsonl        one turn per line, append-only
  trc_<id>.meta.json    summary, rewritten when the trace closes
```

One file per trace. A trace is one agent working on one idea, from the moment
it is given the idea to the moment it stops.

## The line format

Every line is a complete JSON object and **stands alone** — it carries its own
provenance, so `cat traces/*.jsonl | loader` is a valid ingest and nothing is
lost by concatenating. Provenance in the sidecar alone would mean a loader
could not tell which agent produced which turn, which is the one thing a
debugging database exists to answer.

```json
{
  "v": 1,
  "trace_id": "trc_b5b1d5aa1e93",
  "seq": 0,
  "session_id": "sess-1788316867",
  "agent_id": "a01",
  "idea_id": "idea_d82bad40e9db",
  "attempt": 2,
  "ts": 1788317959.331152,
  "kind": "prompt",
  "name": "",
  "content": "improve prefix reuse",
  "data": {},
  "tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0
}
```

| field | type | notes |
|---|---|---|
| `v` | int | schema version. **Refuse an unknown major version** rather than guessing. |
| `trace_id` | string | primary grouping key. Stable, prefixed `trc_`. |
| `seq` | int | 0-based, monotonic within a trace. Order by this, not by `ts` — two turns can share a timestamp. |
| `session_id` | string | one fleet run. Empty for traces written outside a fleet. |
| `agent_id` | string | `a00`, `a01`, … Unique within a session, **reused across sessions**. |
| `idea_id` | string | the hypothesis being worked on. |
| `attempt` | int | which attempt within the idea. |
| `ts` | float | unix seconds, wall clock. |
| `kind` | enum | see below. |
| `name` | string | tool name, or an evaluation's stack digest. |
| `content` | string | free text. May be long; truncate on display, not on ingest. |
| `data` | object | kind-specific, see below. Free-form by design. |
| `tokens_in` / `tokens_out` / `cache_read` / `cache_write` | int | may be 0 when unknown. |

### `kind`

| kind | meaning | `data` |
|---|---|---|
| `prompt` | the idea the agent was given | — |
| `thought` | reasoning, or a memory recall (`name="recall"`) or a study step (`name="study"`) | — |
| `tool_call` | the agent used a tool | tool-specific |
| `tool_result` | what came back | — |
| `message` | agent output | — |
| `eval_submit` | a diff was queued for measurement | `{tier, ticket, queued, diff}` |
| `eval_result` | the measurement came back | the metrics dict, plus `tier` |
| `error` | something failed | — |

`eval_submit.data.diff` holds the unified diff of the candidate, truncated at
20,000 characters. It is the most useful single field for reconstructing what
an agent actually changed.

### Timing (v1, from 2026-09-02)

Every turn the reference loop writes also carries, in `data`:

| key | type | notes |
|---|---|---|
| `phase` | enum | the phase this turn closes: `start`, `recall`, `propose`, `check`, `submit`, `study`, `wait`, `done` |
| `elapsed_s` | float | wall seconds that phase took |

A turn that closes a model call (`thought name=propose`, `thought name=study`)
also carries the call's own accounting: `wall_s`, `duration_ms`,
`duration_api_ms`, `num_turns`, `is_error`, `denials` (permission refusals --
non-zero means the agent asked to run a tool and was told no), `returncode`,
`cancelled`, `timed_out`. A large gap between `wall_s` and `duration_ms` is the
host, not the model: a closed laptop lid froze a fleet for five hours and this
is how the trace says so.

Since the same date a `thought name=propose` is written on the success path
too (not only when the check fails), and a successful check writes a
`tool_call name=check`.

## The sidecar

`trc_<id>.meta.json` is a summary, rewritten on close. Everything in it is
derivable from the lines except `cost_usd` and `outcome`.

```json
{"id": "trc_...", "agent_id": "a01", "idea_id": "idea_...", "attempt": 0,
 "started_at": 1788316867.42, "ended_at": 1788316887.40, "model": "sonnet",
 "harness_commit": "", "n_turns": 3, "cost_usd": 0.0, "outcome": "won"}
```

`outcome` is one of `won`, `exhausted`, `diverged`, `no_progress`, `budget`,
`error`.

## Stability

- Fields are **only added**, never removed or repurposed, within a major `v`.
- Unknown fields should be ignored by readers, so the harness can add
  instrumentation without breaking a loader.
- A trace file is append-only while its agent runs and immutable afterwards,
  so a loader may treat `(trace_id, seq)` as an idempotent primary key and
  re-ingest safely.
- **`v` may be absent.** Traces written before the envelope existed have no
  `v`, no `seq` and no ids on the line. Treat a missing `v` as `0`, take `seq`
  from the line's position in the file, and fill the ids from the sidecar.
  `harness traces show` does exactly this, so those files stay readable.

## Getting the files

```bash
harness traces list                      # what exists, by session and agent
harness traces show <trace_id>           # read one, filtered
harness traces export --out DIR          # copy into one directory for ingest
```

`export` writes the `.jsonl` and `.meta.json` files unchanged plus a
`manifest.json` naming the schema version and per-file line counts, so a loader
can verify it ingested everything.
