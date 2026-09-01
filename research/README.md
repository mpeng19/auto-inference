# research

How the simulator in `src/simulator/` was arrived at. Nothing here is imported
by the product.

**These files are preserved for reading, not for running.** Several were split
out of modules that stayed in the product, so their imports point at
`simulator.*` and at each other; they will not all execute as-is. The value is
the reasoning and the negative results, which `docs/HANDOFF.md` cites by name.

| | what it was, and why it is not in the product |
|---|---|
| `autoinf/simulator.py` | The **predictive** cost model — roofline, then constant-step, then affine. It is a different thing from the product's `Simulator`, which measures rather than predicts. Its calibration (`f_kv = 1.00` at 6k context) does not transfer to the 20.6k market context, where the same term measures 0.22–0.28 (HANDOFF §6c). That gap is the optimisation target. |
| `autoinf/attribution.py` | Phase B: NNLS attribution over saturated mixes. Retired §6b — it exists only to re-blend at a competitor's hit rate, and it had to run past N*, so it understated output cost 2.2×. |
| `autoinf/roofline.py` | Roofline capacity from model and hardware specs. |
| `autoinf/eval_suite.py` | The nine-workload suite that varies arrival *shape* at fixed mean rate. Right tool for scheduler stress, wrong one for pricing. Found the congestion collapse. |
| `autoinf/loadgen_openloop.py` | Open-loop replay, including the multiprocess sharded driver. |
| `autoinf/synthetic_sessions.py` | Synthetic conversations — wrong by ~250× on input length, which is what sent us to TraceLab. |
| `autoinf/workload_config.py` | `WorkloadConfig`, and the original single-percentile `SLO` that forced TTFT and TPOT onto one quantile. |
| `autoinf/gpuwatch.py` | `nvidia-smi` sampling. **A failed instrument**: it reports kernel residency, not work, and polling it 4×/s starved the load generator and moved N* from 128 to 32. |
| `autoinf/probe.py` | One-off H100 probes. Found that `--schedule-policy` takes seven values, not four. |
| `autoinf/gateway.py`, `proxy_app.py` | Recording gateway for real agent traffic. Deployed, never used for a real session. |
| `autoinf/virtual_users.py` | LLM-driven virtual users. Never run. |
| `autoinf/ledger.py` | Experiment ledger — novelty and diversity of a research loop. Comes back when the auto-research harness does. |
| `scripts/` | Report generators, the old launcher and results CLI, the plot scripts the product's `artifacts/plots.py` replaced. |
| `tests_legacy/` | The 178-test suite for the old package. Superseded by `tests/`. |
| `docs/` | `methodology.md` (how the model works and what it assumes), `capacity.md`, `checklist.md`, and the sealed prospective prediction that failed. |
| `plots/` | Earlier figures. |
| `runs/ledger.jsonl` | The experiment ledger. |

The most useful things to read first are `docs/methodology.md` §3.2–§3.3.4 —
roofline rejected, constant-step accepted, the honesty audit, and the sealed
prediction that came back FAIL — and the five failed attempts at per-token cost
in `docs/HANDOFF.md` §4, all of which produced plausible-looking numbers.
