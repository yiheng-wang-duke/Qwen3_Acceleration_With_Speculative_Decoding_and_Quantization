# Benchmark results: Qwen/Qwen3-0.6B-Base -> Qwen/Qwen3-8B-Base

GPU: NVIDIA RTX A5000; torch 2.8.0+cu129, transformers 4.57.1, flash-attn 2.8.3; dtype bfloat16, attention flash_attention_2, mode: eager dynamic cache + flash_attention_2; batch 1, greedy, max 128 new tokens. Run: `outputs/20261126T003721Z-22ffc287`.

## 1. AR baseline and forward-cost profile

| Model | Layers | Params | T(1) ms | ms/layer | T(3)/T(1) | T(5)/T(1) | T(9)/T(1) | AR tok/s (3 prompts) | Decode-only tok/s | MT-Bench AR tok/s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Base (draft) | 28 | 0.60B | 40.71 | 1.454 | 1.033 | 1.033 | 1.033 | 24.81 | 24.70 | - | 1.38 |
| Qwen3-8B-Base (target) | 36 | 8.19B | 54.65 | 1.518 | 0.990 | 0.989 | 0.991 | 18.47 | 18.49 | 18.30 | 15.57 |

AR vs `transformers.generate`: 6/6 standard-prompt runs identical.

## 2. Acceptance (80 MT-Bench prompts) and predicted speedup

Per-token greedy agreement alpha = 0.804; cost ratio c = T_d(1)/T_t(1) = 0.745; ceiling 1/c = 1.34x.

| gamma | Prefix acceptance | Accepted / verify | E[L] tokens / verify | T_t(gamma+1)/T_t(1) | Predicted decode speedup | Predicted request speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.792 | 0.792 | 1.783 | 0.987 | 1.034x | 1.011x |
| 2 | 0.709 | 1.414 | 2.401 | 0.990 | 0.973x | 0.952x |
| 3 | 0.632 | 1.885 | 2.867 | 0.993 | 0.894x | 0.874x |
| 4 | 0.569 | 2.253 | 3.230 | 0.989 | 0.820x | 0.803x |
| 5 | 0.518 | 2.554 | 3.529 | 0.989 | 0.754x | 0.739x |
| 6 | 0.471 | 2.781 | 3.754 | 0.991 | 0.692x | 0.679x |
| 7 | 0.432 | 2.962 | 3.934 | 0.990 | 0.639x | 0.627x |
| 8 | 0.399 | 3.121 | 4.091 | 0.991 | 0.593x | 0.583x |

| Category | alpha | E[L] gamma=1 | E[L] gamma=2 | E[L] gamma=4 |
|---|---:|---:|---:|---:|
| standard | 0.878 | 1.879 | 2.611 | 3.831 |
| writing | 0.724 | 1.713 | 2.228 | 2.843 |
| roleplay | 0.711 | 1.683 | 2.181 | 2.699 |
| reasoning | 0.740 | 1.712 | 2.195 | 2.779 |
| math | 0.919 | 1.899 | 2.712 | 4.129 |
| coding | 0.882 | 1.869 | 2.636 | 3.806 |
| extraction | 0.904 | 1.868 | 2.652 | 3.857 |
| stem | 0.817 | 1.822 | 2.449 | 3.351 |
| humanities | 0.754 | 1.733 | 2.298 | 3.013 |

## 3. Paired timing (11 prompts, 1 warm-up + 3 repeats)

AR: 18.43 tok/s, 5.373 s/request. Peak memory with both models: 16.42 GiB.

| gamma | Spec tok/s | Latency speedup | E[L] measured | E[L] predicted | Greedy-identical prompts | Max divergence gap (BF16 ULP) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.52 | 1.006x | 1.778 | 1.779 | 3/11 | 1.0 |
| 2 | 17.57 | 0.955x | 2.396 | 2.436 | 3/11 | 1.0 |
| 4 | 15.22 | 0.827x | 3.284 | 3.361 | 4/11 | 1.0 |

## 4. Full MT-Bench timing (80 prompts, one pass each)

AR: 18.30 tok/s, 459.2 s total.

| gamma | Spec tok/s | Request speedup | Prefix acceptance | E[L] | Greedy-identical prompts | Max divergence gap (BF16 ULP) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.53 | 1.018x | 0.787 | 1.780 | 32/80 | 2.0 |
| 2 | 17.61 | 0.963x | 0.708 | 2.395 | 31/80 | 2.0 |
| 4 | 14.91 | 0.814x | 0.567 | 3.223 | 30/80 | 2.0 |

**Verdict:** best measured MT-Bench speedup 1.018x at gamma=1: speculative decoding is faster than the BF16 AR baseline in this configuration.
