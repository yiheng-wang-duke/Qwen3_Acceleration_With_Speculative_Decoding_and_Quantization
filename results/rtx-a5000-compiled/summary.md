# Benchmark results: Qwen/Qwen3-0.6B-Base -> Qwen/Qwen3-8B-Base

GPU: NVIDIA RTX A5000; torch 2.8.0+cu129, transformers 4.57.1, flash-attn 2.8.3; dtype bfloat16, attention sdpa, mode: compiled static cache (CUDA graphs) + sdpa; batch 1, greedy, max 128 new tokens. Run: `outputs/20261126T024321Z-98d277b7`.

## 1. AR baseline and forward-cost profile

| Model | Layers | Params | T(1) ms | ms/layer | T(3)/T(1) | T(5)/T(1) | T(9)/T(1) | AR tok/s (3 prompts) | Decode-only tok/s | MT-Bench AR tok/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Base (draft) | 28 | 0.60B | 5.00 | 0.179 | 1.067 | 1.083 | 1.076 | 192.29 | 203.19 | - | 1.38 |
| Qwen3-8B-Base (target) | 36 | 8.19B | 27.27 | 0.758 | 1.015 | 1.020 | 1.025 | 36.33 | 36.51 | 36.17 | 15.56 |

AR vs `transformers.generate`: 3/6 standard-prompt runs identical; differences at near ties: [1.0, 0.0, 0.0] BF16 ULP.

## 2. Acceptance (80 MT-Bench prompts) and predicted speedup

Per-token greedy agreement alpha = 0.813; cost ratio c = T_d(1)/T_t(1) = 0.183; ceiling 1/c = 5.45x.

| gamma | Prefix acceptance | Accepted / verify | E[L] tokens / verify | T_t(gamma+1)/T_t(1) | Predicted decode speedup | Predicted request speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.802 | 0.802 | 1.794 | 1.014 | 1.504x | 1.477x |
| 2 | 0.718 | 1.433 | 2.418 | 1.015 | 1.761x | 1.718x |
| 3 | 0.649 | 1.938 | 2.919 | 1.017 | 1.875x | 1.824x |
| 4 | 0.591 | 2.343 | 3.320 | 1.020 | 1.907x | 1.853x |
| 5 | 0.533 | 2.640 | 3.613 | 1.023 | 1.877x | 1.824x |
| 6 | 0.493 | 2.921 | 3.892 | 1.023 | 1.847x | 1.796x |
| 7 | 0.448 | 3.090 | 4.059 | 1.023 | 1.773x | 1.726x |
| 8 | 0.414 | 3.250 | 4.216 | 1.025 | 1.706x | 1.662x |

| Category | alpha | E[L] gamma=1 | E[L] gamma=2 | E[L] gamma=4 |
|---|---:|---:|---:|---:|
| standard | 0.875 | 1.873 | 2.552 | 3.844 |
| writing | 0.729 | 1.724 | 2.234 | 2.880 |
| roleplay | 0.739 | 1.717 | 2.246 | 2.864 |
| reasoning | 0.748 | 1.726 | 2.186 | 2.863 |
| math | 0.923 | 1.899 | 2.723 | 4.197 |
| coding | 0.867 | 1.837 | 2.577 | 3.744 |
| extraction | 0.902 | 1.868 | 2.664 | 3.908 |
| stem | 0.834 | 1.833 | 2.491 | 3.488 |
| humanities | 0.766 | 1.762 | 2.322 | 3.111 |

## 3. Paired timing (11 prompts, 1 warm-up + 3 repeats)

AR: 36.21 tok/s, 2.737 s/request. Peak memory with both models: 16.66 GiB.

| gamma | Spec tok/s | Latency speedup | E[L] measured | E[L] predicted | Greedy-identical prompts | Max divergence gap (BF16 ULP) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.05 | 1.466x | 1.803 | 1.805 | 6/11 | 1.0 |
| 2 | 62.44 | 1.726x | 2.436 | 2.449 | 6/11 | 1.0 |
| 4 | 69.61 | 1.924x | 3.403 | 3.482 | 6/11 | 1.0 |

## 4. Full MT-Bench timing (80 prompts, one pass each)

AR: 36.17 tok/s, 231.1 s total.

| gamma | Spec tok/s | Request speedup | Prefix acceptance | E[L] | Greedy-identical prompts | Max divergence gap (BF16 ULP) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.54 | 1.447x | 0.797 | 1.788 | 37/80 | 1.0 |
| 2 | 62.01 | 1.707x | 0.722 | 2.426 | 37/80 | 1.0 |
| 4 | 66.87 | 1.833x | 0.580 | 3.274 | 38/80 | 1.0 |

**Verdict:** best measured MT-Bench speedup 1.833x at gamma=4: speculative decoding is faster than the BF16 AR baseline in this configuration.
