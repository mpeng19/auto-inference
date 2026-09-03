---
name: tracedb
description: Query the GPU profile of your candidate stack (and stock) instead of guessing where decode time goes. Use before designing a kernel change and after a full sweep to see whether the change moved the KV-read term.
---

# tracedb: the GPU profile as a database

Every full sweep of a promoted candidate captures a kineto trace at the
priced concurrency level (N*): ~20 decode steps of every CUDA kernel, its
duration, its stream and the CPU launch that issued it. The harness ingests
it into a SQLite file under your agent directory (`../profiles/<stack
digest>.sqlite`) and exposes it to you as MCP tools named `trace_*`. Stock's
profile, when present, is exposed under the same tools on the server
`tracedb_stock`.

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
| `trace_summary` | how many steps, kernels, total GPU busy vs idle |
| `trace_steps` | one row per decode step: duration, kernel count, idle |
| `trace_ops_grouped pattern` | kernels grouped by canonical name with count, total and mean µs -- **start here** |
| `trace_slowest pattern k` | the k longest single kernel launches |
| `trace_ops pattern` | every launch matching a name pattern, in time order |
| `trace_between a b` | what runs between kernel a and kernel b in each step |
| `trace_gaps after before min_gap_us` | idle gaps on the GPU between two kernels |
| `trace_overlap a b` | whether two kernels overlap in time (multi-stream) |
| `trace_gpu_idle min_gap_us` | every idle gap above a threshold |
| `trace_launches` | CPU launch overhead: time from launch to kernel start |
| `trace_step_diff idx` | one step against the previous, kernel by kernel |
| `trace_render t0 t1 tracks` | an image of a window of the timeline |

Patterns are glob-style on the canonical kernel name (`*attn*`, `*gemm*`,
`*rope*`, `*norm*`).

## How to use it well

1. **Before designing.** Run `trace_ops_grouped *` on stock. Sort by total
   µs. The top three groups are where the step goes; anything under 5% of
   the step cannot pay for a rewrite. Then `trace_steps` to see whether
   steps are uniform or a few long ones dominate (retraction, prefill
   chunks interleaved).
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

## What it cannot tell you

It is one process, one TP rank, ~20 steps at one concurrency. It says
nothing about accuracy (that is the equivalence gate and GSM8K), nothing
about other concurrency levels, and nothing about the price directly. Use it
to decide *what* to change and to confirm *how* the step changed; use the
sweep to decide whether it was worth it.
