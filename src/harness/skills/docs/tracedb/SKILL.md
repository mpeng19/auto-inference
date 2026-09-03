---
name: tracedb
description: Query the GPU profile of your candidate stack (and stock) instead of guessing where decode time goes. Use before designing a kernel change and after a full sweep to see whether the change moved the KV-read term.
---

# tracedb: the GPU profile as a database

Every full sweep captures a kineto trace at one fixed concurrency (the fleet's
`--profile-level`, 12 by default -- the level stock prices at, not necessarily
your N*): ~20 decode steps of every CUDA kernel, its duration, its stream and
the CPU launch that issued it. The harness ingests it into a SQLite file under
the run's `profiles/<stack digest>.sqlite` and exposes it to you as MCP tools
named `trace_*`. Stock's profile, when present, is exposed under the same tools
on the server `tracedb_stock`.

## Why this exists

The price model says the decode step at 20k context is
`10.4 ms + 1.59 ms × batch`, and the per-sequence term runs at only 22-28% of
H100 memory bandwidth. That term is the money. A trace tells you *which*
kernel is that term, how long it runs, whether it overlaps anything, and
whether your change made it shorter or just moved the time somewhere else.
Without it you are optimising a number you cannot see.

## The tools

| tool | answers |
|---|---|
| `trace_summary` | overview: time span, tracks with busy fractions, top ops by total time |
| `trace_steps` | ProfilerStep boundaries, duration stats, and outlier (slow/fast) steps |
| `trace_ops_grouped pattern` | duration stats per op with templated kernel names canonicalized and grouped -- **start here** on a real GPU trace |
| `trace_ops pattern` | duration stats per op name matching a substring/glob pattern |
| `trace_slowest pattern k` | the top-k individual span instances by duration for ops matching the pattern |
| `trace_between a b` | for each op matching `a`, the latency until the NEXT op matching `b` (any track), with how many spans intervene: launch latency or cross-stream causality |
| `trace_gaps after before min_gap_us` | every instance where an op matching `after` is immediately followed on the same track by one matching `before` with idle time between; sorted by gap, with summary stats |
| `trace_overlap a b` | how much of pattern `a`'s time overlaps pattern `b` on other tracks (e.g. how well comm is hidden under compute) |
| `trace_gpu_idle min_gap_us` | gaps on GPU streams, each blamed on the CPU activity covering the gap, aggregated per blamed op |
| `trace_launches` | CPU->GPU kernel launch latency via correlation ids (p50/p99/max and the slowest pairs); high means the GPU waits on the CPU |
| `trace_step_diff idx` | per-op time in step `idx` against the MEDIAN step: why step N is slow |
| `trace_render t0 t1 tracks` | the [t0, t1] window (microseconds, from span timestamps other tools return) as a timeline PNG; returns the file path |

Patterns are glob-style on the canonical kernel name (`*attn*`, `*gemm*`,
`*rope*`, `*norm*`).

## How to use it well

1. **Before designing.** Run `trace_ops_grouped *` on stock. Sort by total
   µs. The top three groups are where the step goes; anything under 5% of
   the step cannot pay for a rewrite. Then `trace_steps` for the step
   duration stats and outliers: a few slow steps (retraction, prefill
   chunks interleaved) are a different problem from a uniformly slow one,
   and `trace_step_diff` on an outlier says which ops it spent the extra
   time in.
2. **Find the KV-read term.** The decode attention kernel (`*decode*attn*`,
   `*paged*`, `*flashinfer*decode*`) is the per-sequence term. Its mean µs
   per step divided by the KV bytes it reads is the bandwidth it achieves;
   compare with 3.35 TB/s.
3. **After your full sweep.** Run the same `trace_ops_grouped` on *your*
   profile. A real improvement shortens the kernel you targeted **and** the
   step; a fake one shortens the kernel and lengthens something else
   (`trace_step_diff`, `trace_gpu_idle`). Report both numbers in your
   write-up.
4. **Idle is a finding.** `trace_gpu_idle 50` on a decode step that is
   supposed to be bandwidth-bound: if the GPU sits idle between kernels,
   the bottleneck is launch overhead or the scheduler, and a kernel rewrite
   will not move the step.

## When the trace is not enough: `harness tool ncu`

A trace gives time. Nsight Compute gives the counters behind it: DRAM
throughput as a percent of peak, SM throughput, warp occupancy, L2 hit rate,
per kernel. `harness tool ncu bench.py --kernel "decode|attn"` runs your
script under `ncu` in the workbench and prints one row per kernel. Read
DRAM% first: a decode attention kernel that is supposed to be
bandwidth-bound and shows 25% of peak is the whole opportunity; one at 85%
is done, and a rewrite will not move it. Profile a micro-benchmark or a
handful of decode steps -- every profiled launch replays several times --
and narrow the kernel regex.

## What it cannot tell you

It is one process, one TP rank, ~20 steps at one concurrency. It says
nothing about accuracy (that is the equivalence gate and GSM8K), nothing
about other concurrency levels, and nothing about the price directly. Use it
to decide *what* to change and to confirm *how* the step changed; use the
sweep to decide whether it was worth it.
