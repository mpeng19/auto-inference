# Timeline: build-4

_rewritten by the fleet after every outcome; 58 ideas_

## a00

| when | idea | outcome | cost | edit | study | wait | results |
|---|---|---|---|---|---|---|---|
| 01:09-04:15 | Reading a P-greedy-selected nested subset of KV pages instea | error | $0.76 | 174m | 12m | 12m | screen $8.47/1k, quality 60% |
| 04:15-08:56 | Replacing per-step autoregressive/non-autoregressive draftin | error | $2.96 | 235m | 46m | 46m | screen $9.38/1k, quality 68%; screen failed: N=8: no phase-split forward time; SGLANG_ENABLE_METRICS_DEVI; screen failed: N=8: no phase-split forward time; SGLANG_ENABLE_METRICS_DEVI; screen failed: N=8: no phase-split forward time; SGLANG_ENABLE_METRICS_DEVI |
| 09:03-09:27 | Splicing cached KV for matching non-prefix chunks and recomp | error | $0.00 | 24m | 0s | 0s | - |
| 09:31-09:36 | A flattened ragged tile schedule with page-aligned TILE_KV a | error | $0.00 | 5m | 0s | 0s | - |
| 09:36-09:40 | Collapsing the ~10-20 small memory-bound kernels per transfo | error | $0.00 | 4m | 0s | 0s | - |
| 09:40-09:44 | A certified INT8 KV cache with runtime meter-driven gating w | error | $0.00 | 4m | 0s | 0s | - |
| 09:44-09:48 | Running prefill and decode as concurrently scheduled, SM-par | error | $0.00 | 4m | 0s | 0s | - |
| 09:48-09:51 | Computing prefill attention only over the top-scoring query- | error | $0.00 | 4m | 0s | 0s | - |
| 09:51-10:55 | Deep multi-stage async prefetch of KV tiles plus L2 persiste | running | $0.00 | 63m | 0s | 0s | - |

## a01

| when | idea | outcome | cost | edit | study | wait | results |
|---|---|---|---|---|---|---|---|
| 01:09-03:29 | Selecting the top-k KV pages per query with bounding-box sco | won | $3.61 | 80m | 54m | 54m | screen $8.50/1k, quality 66%; full $5.85/1k, quality 66%; full $7.70/1k, quality 66% |
| 03:29-09:25 | Spatially isolating prefill onto a disjoint SM partition — a | error | $1.25 | 331m | 25m | 25m | screen $9.42/1k, quality 64%; screen $9.40/1k, quality 66% |
| 09:25-09:29 | Caching keys pre-RoPE and applying rotation in-kernel at the | error | $0.00 | 3m | 0s | 0s | - |
| 09:29-09:33 | Storing keys at 4 bits per-channel and values at 2 bits per- | error | $0.00 | 5m | 0s | 0s | - |
| 09:33-09:37 | Replacing the fp8 KV cache with per-head residual-VQ codes f | error | $0.00 | 3m | 0s | 0s | - |
| 09:37-09:41 | Jointly choosing per-context compression level and storage t | error | $0.00 | 5m | 0s | 0s | - |
| 09:41-09:44 | Bandit-selected, load-adaptive draft length with proactive s | error | $0.00 | 3m | 0s | 0s | - |
| 09:44-09:48 | Reallocating a fixed batch-wide verify budget to the highest | error | $0.00 | 4m | 0s | 0s | - |
| 09:48-09:52 | Entropy-gated early draft stopping will lower cost per outpu | error | $0.00 | 4m | 0s | 0s | - |
| 09:52-09:55 | Splicing cached KV for matching non-prefix chunks and recomp | error | $0.00 | 3m | 0s | 0s | - |
| 09:55-09:59 | Bounding each sequence's KV to a handful of contiguous exten | error | $0.00 | 4m | 0s | 0s | - |
| 10:02-10:06 | Selecting speculation depth, KV sparsity level and cascade u | error | $0.00 | 4m | 0s | 0s | - |
| 10:06-10:09 | Evicting to a global, cross-layer/head-calibrated KV budget  | error | $0.00 | 3m | 0s | 0s | - |
| 10:13-10:16 | Jointly choosing per-context compression level and storage t | error | $0.00 | 3m | 0s | 0s | - |
| 10:16-10:19 | A certified INT8 KV cache with runtime meter-driven gating w | error | $0.00 | 3m | 0s | 0s | - |
| 10:19-10:22 | Running prefill concurrently with decode on a partitioned SM | error | $0.00 | 3m | 0s | 0s | - |
| 10:22-10:26 | Replacing the GQA KV cache with a layer-adaptive MLA latent  | error | $0.00 | 3m | 0s | 0s | - |
| 10:26-10:26 | Entropy-gated early draft stopping will lower cost per outpu | running | $0.00 | 0s | 0s | 0s | - |

## a02

| when | idea | outcome | cost | edit | study | wait | results |
|---|---|---|---|---|---|---|---|
| 01:09-01:51 | Retaining only attention-sink tokens plus a sliding window o | diverged | $0.76 | 29m | 12m | 12m | screen $8.08/1k, quality 66% |
| 01:51-05:42 | Sizing every prefill chunk against the live decode TBT slack | won | $4.27 | 162m | 65m | 65m | screen failed: no level met the SLO -- the sweep starts above the frontier;; screen $6.73/1k, quality 64%; full $6.94/1k, quality 64%; full $6.82/1k, quality 64% |
| 05:47-10:11 | Raising the decode kernel's T_dec (values/second) rather tha | won | $4.05 | 199m | 62m | 62m | screen $10.76/1k, quality 65%; screen $8.50/1k, quality 66%; full $7.74/1k, quality 66%; full $7.90/1k, quality 66% |
| 10:11-10:14 | Admitting and preempting requests by predicted prefill servi | error | $0.00 | 3m | 0s | 0s | - |
| 10:14-10:18 | Bandit-selected, load-adaptive draft length with proactive s | error | $0.00 | 3m | 0s | 0s | - |
| 10:21-10:24 | Reallocating a fixed batch-wide verify budget to the highest | error | $0.00 | 3m | 0s | 0s | - |
| 10:28-10:31 | Storing older non-anchor KV pages in packed TQ3 while keepin | error | $0.00 | 3m | 0s | 0s | - |
| 10:31-10:31 | Packing all query heads that share a KV head into one kernel | running | $0.00 | 0s | 0s | 0s | - |

## a03

| when | idea | outcome | cost | edit | study | wait | results |
|---|---|---|---|---|---|---|---|
| 01:09-04:47 | Merging aged KV pages into a smaller number of summary entri | error | $0.50 | 209m | 9m | 9m | screen $10.76/1k, quality 64% |
| 04:47-08:08 | Merging each sequence's private KV suffix down to a 30-50% b | error | $0.67 | 190m | 11m | 11m | screen $12.24/1k, quality 68% |
| 08:08-09:29 | Autotuning decode attention tile and split configuration per | error | $0.68 | 70m | 11m | 11m | screen $9.37/1k, quality 68% |
| 09:30-09:33 | Ordering admission by the F-metric with a projected peak-KV  | error | $0.00 | 3m | 0s | 0s | - |
| 09:33-09:38 | Evicting to a global, cross-layer/head-calibrated KV budget  | error | $0.00 | 5m | 0s | 0s | - |
| 09:42-09:46 | Repacking the paged KV pool so decode reads are fully coales | error | $0.00 | 4m | 0s | 0s | - |
| 09:50-09:53 | Packing all query heads that share a KV head into one kernel | error | $0.00 | 4m | 0s | 0s | - |
| 09:57-10:00 | Adding a SplitK-decomposed fused W4A16 dequant-GEMM and serv | error | $0.00 | 3m | 0s | 0s | - |
| 10:00-10:04 | Storing keys at 4 bits per-channel and values at 2 bits per- | error | $0.00 | 3m | 0s | 0s | - |
| 10:07-10:10 | Replacing the fp8 KV cache with per-head residual-VQ codes f | error | $0.00 | 3m | 0s | 0s | - |
| 10:10-10:14 | Collapsing the ~10-20 small memory-bound kernels per transfo | error | $0.00 | 3m | 0s | 0s | - |
| 10:17-10:17 | Repacking the paged KV pool so decode reads are fully coales | running | $0.00 | 0s | 0s | 0s | - |

## a04

| when | idea | outcome | cost | edit | study | wait | results |
|---|---|---|---|---|---|---|---|
| 01:09-03:57 | Caching a low-rank latent per token instead of full K and V, | diverged | $1.44 | 145m | 24m | 24m | screen $8.49/1k, quality 1%; screen $9.06/1k, quality 57% |
| 04:02-07:11 | Storing cached values at INT4 while keeping keys at FP8 will | error | $0.76 | 176m | 13m | 13m | screen $9.20/1k, quality 1% |
| 07:11-09:24 | Migrating activation outliers into the weights with per-chan | error | $1.53 | 105m | 27m | 27m | screen $9.39/1k, quality 70%; screen failed: no level met the SLO -- the sweep starts above the frontier; |
| 09:24-09:29 | Loading each shared prefix KV block once per group instead o | error | $0.00 | 5m | 0s | 0s | - |
| 09:29-09:34 | Selecting speculation depth, KV sparsity level and cascade u | error | $0.00 | 5m | 0s | 0s | - |
| 09:38-09:43 | Admitting and preempting requests by predicted prefill servi | error | $0.00 | 4m | 0s | 0s | - |
| 09:47-09:50 | Replacing the GQA KV cache with a layer-adaptive MLA latent  | error | $0.00 | 4m | 0s | 0s | - |
| 09:54-09:58 | Caching keys pre-RoPE and applying rotation in-kernel at the | error | $0.00 | 3m | 0s | 0s | - |
| 09:58-10:01 | Loading each shared prefix KV block once per group instead o | error | $0.00 | 4m | 0s | 0s | - |
| 10:01-10:05 | Ordering admission by the F-metric with a projected peak-KV  | error | $0.00 | 3m | 0s | 0s | - |
| 10:05-10:59 | A flattened ragged tile schedule with page-aligned TILE_KV a | error | $0.74 | 43m | 12m | 12m | screen $9.05/1k, quality 62% |

