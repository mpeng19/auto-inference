# Notes on mining `Inference Engineering` (Kiely, Baseten Books, 2026) for the idea bank

Source PDF: `docs/Inference Engineering.pdf`, 259 pages, read in full.

**Page convention.** `source` fields cite **PDF page numbers** (the unit the extraction and the
reading pass used). The book's printed page numbers run 2 lower — PDF p.138 is printed p.136.

The book is a broad survey of the whole inference stack, not a research monograph. Most of its
"techniques" chapter describes things a mature engine already implements. The records in
`book.jsonl` are therefore either (a) mechanisms the book names that stock SGLang 0.5.18 does not
do, or (b) improvements layered on top of a stock feature, stated as such in the `mechanism` field.

## Skipped as already default in stock SGLang 0.5.18

Verified by reading the installed package at
`~/.cache/auto-inference/sglang/0.5.18/sglang/` (the harness's wheel cache).

- **Radix prefix caching** (§5.3.1) — `srt/mem_cache/radix_cache.py`, on by default.
- **Chunked prefill** (§5.3.4) — default scheduling path.
- **Continuous / in-flight batching** (§7.2.1) — default.
- **PagedAttention** (§2.5, §5.3.4) — `srt/mem_cache/allocator/paged.py`.
- **FlashAttention / FlashInfer attention backends** (§2.5, §4.1.1) — full backend registry under
  `srt/layers/attention/`, including FA3 for Hopper.
- **FP8 GEMM via DeepGEMM** (§4.1.2) — `srt/layers/quantization/fp8.py`, `kernels/ops/gemm/`.
- **CUDA graphs for decode** (§4.1) — `srt/model_executor/runner/decode_cuda_graph_runner.py`.

## Skipped as already available as a stock option (a flag, not an idea)

These are levers the loop should still *sweep*, but they are not new mechanisms:

- **FP8 KV cache** (§5.1.2) — `--kv-cache-dtype fp8_e4m3 / fp8_e5m2 / mxfp8`, resolved in
  `srt/mem_cache/kv_cache_dtype.py`. The bank's KV-precision record is about **sub-8-bit** KV on
  Hopper, which stock only supports via Blackwell-only TRT-LLM MHA kernels.
- **Fused QK-norm + RoPE + KV store** (§4.1.3) — `kernels/ops/attention/fused_qknorm_rope.py`,
  `fused_qk_norm_rope_store.py`, `kernels/ops/kvcache/fused_fp8_qkv_kv_cache.py`. The generic
  "fuse norm + QKV + RoPE" idea is already done; no record written for it.
- **Flash-decoding style KV splits** (§2.5) — `get_num_kv_splits_triton` already splits the
  sequence across SMs. Only the *autotuning* of that choice is proposed (a separate record).
- **KV eviction policies** — LRU / LFU / FIFO / MRU / FILO / priority all in
  `srt/mem_cache/evict_policy.py`. A "smarter eviction policy" record was dropped as knob-tweaking.
- **Hierarchical (host) KV cache** (§5.3.2) — `--enable-hierarchical-cache`, `hiradix_cache.py`.
  The bank's record is the finer-grained *layer-pipelined prefetch during decode*, not block
  offload between requests.
- **Quest sparse attention** (§2.5) — scaffolding exists at
  `srt/mem_cache/sparsity/algorithms/quest_algorithm.py` but is off by default and its backend
  adaptor covers only fa3 and the DeepSeek DSA path. The record is about wiring and tuning it for
  the dense-GQA decode path used here.
- **EAGLE and n-gram speculation workers** (§5.2) — present under `srt/speculative/`. Both records
  are kept because neither is enabled by default and EAGLE needs heads that do not exist for this
  checkpoint; the *reason* they matter for this workload (they amortise the per-sequence KV read
  along the sequence axis) is the non-obvious part.
- **Priority scheduling and retraction** (§7.2.3) — `--enable-priority-scheduling`. The record
  proposes scheduling on *predicted service time*, not on an externally supplied priority integer.
- **`torch.compile`** (§4.2.1) — available; the book itself notes it cannot fuse plugin kernels
  like DeepGEMM or FlashAttention, which is most of what matters here.
- **Cascade attention** — present in `flashattention_backend.py` but **only** for speculative
  target-verify with `topk > 1`. It is *not* used to amortise a shared prefix across a decode
  batch, which is why that record was written.

## Chapters with nothing implementable for this workload

- **Chapter 0, Chapter 1 (Prerequisites)** — product framing, model selection, fine-tuning,
  distillation, latency percentile definitions. §1.4.1 informed the p90-focused scheduler records
  but contains no mechanism. Distillation (§1.3.3) would mean serving a different model.
- **Chapter 3 (Hardware)** — mostly SKU selection. Only §3.1.2 (cache hierarchy, L2 = 50MB) and
  §3.2.1 (Hopper async transfer features) yielded records. Explicitly unusable here:
  - §3.3.1 multi-GPU / NVLink / InfiniBand — single H100.
  - §3.3.2 Multi-Instance GPU — a 27B FP8 model is ~27GB of weights; two MIG slices would need two
    copies, over the 80GB budget.
  - §3.2.3-3.2.4 Blackwell / Rubin, §3.4 other accelerators, §3.5 local inference — wrong hardware.
    Note this also rules out FP4/NVFP4 **compute**: there are no FP4 Tensor Cores on Hopper, so
    4-bit only ever buys bandwidth and capacity here, never FLOPS.
- **Chapter 5 §5.4 (Model Parallelism)** — TP, EP, PP all require ≥2 GPUs. §5.4.3 multi-node
  likewise. The two `parallelism`-scale records in the bank are intra-GPU overlap ideas, not model
  parallelism.
- **Chapter 5 §5.5 (Disaggregation)** — needs separate prefill and decode engines on separate
  hardware, and the book's own guidance is to use it only above ~100M-1B tokens/day on 100B+
  parameter models. Its underlying *insight* (prefill compute-bound, decode memory-bound) is
  reused in the single-GPU co-execution and two-batch-overlap records.
- **Chapter 6 (Modalities)** — VLM, embedding, ASR, TTS, image and video generation. Nothing
  applies to a text LLM directly. Two ideas were *transferred* out of §6.6.1 (video attention
  quantisation): selective quantisation **by layer** (kept as a record) and **by step** (dropped —
  there is no diffusion step axis in autoregressive decode; the token axis is not analogous because
  every token's KV feeds every later token). The §6.5.2 classifier-free-guidance skip trick is
  diffusion-only.
- **Chapter 7 (Production)** — containerisation, autoscaling, multi-cloud, GPU procurement,
  reliability, canary deploys, observability, client code. All above the engine. §7.2.3 (queueing)
  and §7.2.1 (batch sizing) fed the scheduler records; the rest is infrastructure, and cold starts
  (§7.2.2) do not affect steady-state cost per output token.
- **Appendix A (Glossary)** — definitions only, no new mechanisms.
- **Appendix B (Recommended Reading)** — no prose, but the most useful section per page for this
  task. It supplied the concrete citations behind several records: AWQ, GPTQ, SmoothQuant,
  CacheBlend, EAGLE-1/2/3, Lookahead Decoding, SageAttention, Ring Attention, PagedAttention,
  FlashInfer, and tensor-decomposition KV compression.

## Deliberate omissions

- **Variants collapsed.** Medusa was dropped as a weaker variant of EAGLE (the book says it is not
  widely used in production). Draft-target speculation was dropped for the same reason — the book
  states it has the most overhead of any speculation method. MTP / frozen-KV MTP was dropped as a
  variant of the EAGLE record.
- **Content-addressed KV page dedup** was dropped as too close to the pre-RoPE storage and
  non-prefix-reuse records; it is only sound once keys are position-independent anyway.
- **Fused sampling kernels** (§4.1.1) were dropped: sampling touches a ~150k-entry logit vector
  once per step, which is negligible against a 20k-token KV read.
- **A profiling/roofline harness** was considered and dropped — it is measurement infrastructure,
  not a mechanism that lowers cost per output token. It is nonetheless the prerequisite named by
  several kernel records, because the 22-28% bandwidth figure needs to be split into
  transaction-efficiency loss versus latency-stall loss before choosing between them.
