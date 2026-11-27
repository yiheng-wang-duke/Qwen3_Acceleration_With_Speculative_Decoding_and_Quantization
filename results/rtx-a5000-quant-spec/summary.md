# Weight-only quantization × speculative decoding: Qwen3-0.6B-Base → Qwen3-8B-Base

GPU: NVIDIA RTX A5000 (24 GB, 768 GB/s); torch 2.8.0+cu129, transformers 4.57.1, torchao 0.13.0,
gptqmodel 4.2.5. Compiled mode (static KV cache + `torch.compile(mode="reduce-overhead")` CUDA graphs,
SDPA), BF16 activations, batch 1, greedy, max 128 new tokens, 80 MT-Bench first-turn prompts.
Script: `scripts/quant_spec.py`, one target/draft pair per process. AR and speculative decoding of a
pair run in the same process on the same GPU. "vs BF16 AR" divides by the BF16 AR run of this study
(36.40 tok/s; the earlier compiled benchmark measured 36.17 tok/s). Different pairs ran on different
A5000 cards.

## Headline

| Target (draft: BF16 Qwen3-0.6B) | Weights GiB | PPL (BF16 9.44) | AR tok/s | AR vs BF16 | Best spec tok/s (γ) | Spec vs own AR | Spec vs BF16 AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 15.26 | 9.44 | 36.40 | 1.00× | 66.78 (5) | 1.82× | 1.83× |
| int8 weight-only (torchao) | 8.22 | 9.48 | 46.68 | 1.28× | 40.52 (5) | **0.86×** | 1.11× |
| int4 RTN g128, lm_head int4 (torchao) | 4.90 | 12.90 | 60.49 | 1.66× | **103.41 (2)** | 1.71× | **2.84×** |
| int4 RTN g128, lm_head BF16 (torchao) | 5.76 | 10.90 | 56.31 | 1.55× | 98.05 (2) | 1.71× | 2.69× |
| int4 RTN g64, lm_head BF16 (torchao) | 5.96 | 10.47 | 49.19 | 1.35× | 94.26 (2) | 1.89× | 2.59× |
| int4 HQQ g128, lm_head BF16 (torchao) | 5.76 | 10.87 | 57.02 | 1.57× | 94.07 (2) | 1.66× | 2.58× |
| **GPTQ W4 g128 + Marlin, lm_head BF16** | 5.68 | **9.72** | **74.60** | **2.05×** | 100.05 (2) | 1.34× | 2.75× |

PPL: RedPajama-V2 validation, 128 × 512 tokens. "Spec vs own AR" is lossless with respect to that
(quantized) target. Every speculative output equals the target's own AR output or differs first at
a ≤ 3 BF16-ULP near tie.

## Findings

1. **Quantization alone.** Batch-1 decode is weight-bandwidth bound, so 4-bit weights help most.
   GPTQ + Marlin gives T(1) = 12.9 ms (BF16: 26.5–27.1 ms), **2.05× AR at +3 % PPL**. torchao's
   int4 tinygemm kernel reaches only 15.5–20.6 ms, and plain round-to-nearest (RTN) costs much more
   quality. Quantizing the untied 1.2 GB lm_head to int4 alone raises PPL from 10.90 to 12.90. HQQ
   did not improve quality here, and group size 64 helps quality (10.47) but is slower (20.6 ms).
   int8 weight-only is nearly lossless (PPL 9.48) but only 1.28× faster.
2. **Joint gain exists for every int4 target.** Speculative decoding adds 1.34–1.89× on top of
   quantized AR. The best end-to-end results are **2.75× BF16 AR with GPTQ-Marlin** (PPL 9.72) and
   2.84× with torchao int4 RTN (PPL 12.90, only 3/80 outputs equal BF16's).
3. **The joint gain shrinks as the target gets faster.** The 0.6B draft costs about 5 ms per step,
   because it is bound by kernel count, not weight bytes. So c = T_d/T_t rises from 0.19 (BF16) to
   0.39 (GPTQ), and the best γ drops from 4–5 to 2. With the fastest target, speculative decoding
   adds 1.34× instead of 1.82×.
4. **Verification cost decides compatibility.** Speculative decoding needs T(γ+1) ≈ T(1). Marlin
   keeps T(5)/T(1) = 1.05 and T(9)/T(1) = 1.03. torchao int4 gives 1.09–1.13 and 1.25–1.37. torchao
   int8 weight-only only has a fused kernel for one query token: T(5)/T(1) = 3.16. So **with an
   int8 target, speculative decoding is slower than AR at every γ (0.54–0.86×)**.
5. **Quantizing the draft does not help.** An int8 0.6B draft is slower than the BF16 draft (8.2–8.4
   vs 4.5–5.1 ms) at the same acceptance (α 0.79–0.81): best 1.64× instead of 1.82× with a BF16
   target, and 1.18× instead of 1.71× with an int4 target. Using an int4 (RTN) copy of the 8B target
   as the draft for the BF16 target gives α = 0.851 but T_d = 15.8 ms (c = 0.61): 1.13× (γ=2) and
   1.17× (γ=3). That pair was cut off by the job time limit after γ=3; values are from the job log.
6. **Acceptance is robust to target quantization.** α is 0.77–0.81 for every target. The draft
   agrees with a quantized target about as often as with the BF16 target.

## Cost-model check

S(γ) = E[L]·T_t(1) / (γ·T_d(1) + T_t(γ+1)) plus measured prefill (see main README §3.3). It matches
within 1–3 % for BF16 and GPTQ-Marlin targets and for int8 drafts. For torchao int4 targets with a
BF16 draft, measured speedups are 15–30 % above the prediction. Their outputs and acceptance
lengths are correct, so the per-call timing table (one host sync per call) probably overstates the
in-loop cost of that kernel path. This was not verified further.

## Measurement notes

- Each process compiles one graph per query length and model. torchao's Inductor settings enable
  coordinate-descent autotuning, so warm-up took 20–55 min per pair. It runs outside all timings.
- A Dynamo recompile inside a timed request showed up three times (one prompt taking 30–108 s).
  Requests slower than 3× the median are re-timed (greedy decoding is deterministic, and the re-run
  must reproduce the tokens). `bf16 / int8wo` γ=2 predates this fix and is contaminated (0.85×; about
  1.43× without the outlier).
- "Identical to BF16" for quantized targets measures output drift, not a bug. BF16 vs the earlier
  BF16 run is 37–42/80, because kernels differ (autotuning) and near ties flip.

## All pairs

BF16 AR reference: 36.40 tok/s (bf16 target in this directory)

| Target | Draft | Target GiB | Draft GiB | T_t(1) ms | T_t(5)/T_t(1) | T_d(1) ms | c | PPL target | Top-1 vs BF16 | AR tok/s | AR vs BF16 AR | alpha | Best gamma | Spec tok/s | Spec vs same AR | Spec vs BF16 AR | E[L] | Identical to own AR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | bf16 | 15.26 | 1.11 | 27.08 | 1.035 | 5.06 | 0.187 | 9.435 | 0.9897 | 36.40 | 1.00x | 0.809 | 5 | 66.78 | 1.820x | 1.83x | 3.565 | 37/80 |
| bf16 | int8wo | 15.26 | 0.85 | 26.53 | 1.055 | 8.37 | 0.316 | 9.435 | 0.9897 | 37.79 | 1.04x | 0.805 | 5 | 61.24 | 1.637x | 1.68x | 3.562 | 41/80 |
| gptq-w4g128 | bf16 | 5.68 | 1.11 | 12.91 | 1.048 | 5.07 | 0.393 | 9.720 | 0.9239 | 74.60 | 2.05x | 0.803 | 2 | 100.05 | 1.344x | 2.75x | 2.390 | 33/80 |
| int4wo-g128-bf16head | bf16 | 5.76 | 1.11 | 17.21 | 1.096 | 4.47 | 0.260 | 10.898 | 0.8898 | 56.31 | 1.55x | 0.807 | 2 | 98.05 | 1.706x | 2.69x | 2.395 | 41/80 |
| int4wo-g128-hqq-bf16head | bf16 | 5.76 | 1.11 | 17.07 | 1.089 | 4.52 | 0.265 | 10.870 | 0.8957 | 57.02 | 1.57x | 0.779 | 2 | 94.07 | 1.657x | 2.58x | 2.314 | 26/80 |
| int4wo-g128-hqq | bf16 | 4.90 | 1.11 | 17.40 | 1.110 | 4.81 | 0.277 | 12.958 | 0.8491 | 54.95 | 1.51x | 0.766 | 2 | 91.79 | 1.664x | 2.52x | 2.301 | 31/80 |
| int4wo-g128 | bf16 | 4.90 | 1.11 | 15.50 | 1.134 | 4.71 | 0.304 | 12.897 | 0.8513 | 60.49 | 1.66x | 0.796 | 2 | 103.41 | 1.712x | 2.84x | 2.389 | 35/80 |
| int4wo-g128 | int8wo | 4.90 | 0.85 | 15.51 | 1.136 | 8.18 | 0.528 | 12.897 | 0.8513 | 61.29 | 1.68x | 0.790 | 3 | 71.18 | 1.182x | 1.96x | 2.774 | 37/80 |
| int4wo-g64-bf16head | bf16 | 5.96 | 1.11 | 20.55 | 1.087 | 4.67 | 0.227 | 10.469 | 0.9055 | 49.19 | 1.35x | 0.791 | 2 | 94.26 | 1.889x | 2.59x | 2.360 | 30/80 |
| int8wo | bf16 | 8.22 | 1.11 | 20.75 | 3.164 | 4.90 | 0.236 | 9.481 | 0.9781 | 46.68 | 1.28x | 0.799 | 5 | 40.52 | 0.864x | 1.11x | 3.492 | 27/80 |

## Per-gamma speculative results

| Target | Draft | gamma | tok/s | vs same AR | predicted | E[L] | acceptance | identical to own AR | max gap (ULP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | bf16 | 2 | 61.80 | 1.692x | 1.704x | 2.420 | 0.719 | 36/80 | 1.0 |
| bf16 | bf16 | 3 | 65.82 | 1.801x | 1.794x | 2.890 | 0.639 | 35/80 | 2.0 |
| bf16 | bf16 | 4 | 66.77 | 1.813x | 1.797x | 3.268 | 0.578 | 38/80 | 1.0 |
| bf16 | bf16 | 5 | 66.78 | 1.820x | 1.787x | 3.565 | 0.525 | 37/80 | 1.0 |
| bf16 | int8wo | 2 | 31.61 | 0.847x | 1.382x | 2.400 | 0.710 | 40/80 | 1.0 |
| bf16 | int8wo | 3 | 58.28 | 1.561x | 1.381x | 2.856 | 0.628 | 41/80 | 1.0 |
| bf16 | int8wo | 4 | 60.39 | 1.639x | 1.358x | 3.225 | 0.566 | 41/80 | 1.0 |
| bf16 | int8wo | 5 | 61.24 | 1.637x | 1.297x | 3.562 | 0.524 | 41/80 | 1.0 |
| gptq-w4g128 | bf16 | 1 | 93.94 | 1.262x | 1.233x | 1.784 | 0.791 | 33/80 | 1.0 |
| gptq-w4g128 | bf16 | 2 | 100.05 | 1.344x | 1.307x | 2.390 | 0.704 | 33/80 | 1.0 |
| gptq-w4g128 | bf16 | 3 | 98.17 | 1.309x | 1.269x | 2.827 | 0.619 | 33/80 | 1.0 |
| gptq-w4g128 | bf16 | 4 | 94.68 | 1.266x | 1.208x | 3.218 | 0.564 | 39/80 | 1.0 |
| int4wo-g128-bf16head | bf16 | 1 | 84.34 | 1.471x | 1.314x | 1.775 | 0.785 | 43/80 | 2.0 |
| int4wo-g128-bf16head | bf16 | 2 | 98.05 | 1.706x | 1.446x | 2.395 | 0.705 | 41/80 | 2.0 |
| int4wo-g128-bf16head | bf16 | 3 | 96.25 | 1.719x | 1.463x | 2.847 | 0.625 | 38/80 | 2.0 |
| int4wo-g128-bf16head | bf16 | 4 | 92.15 | 1.613x | 1.426x | 3.213 | 0.562 | 40/80 | 2.0 |
| int4wo-g128-hqq-bf16head | bf16 | 1 | 84.41 | 1.480x | 1.304x | 1.757 | 0.762 | 34/80 | 2.0 |
| int4wo-g128-hqq-bf16head | bf16 | 2 | 94.07 | 1.657x | 1.423x | 2.314 | 0.666 | 26/80 | 8.0 |
| int4wo-g128-hqq-bf16head | bf16 | 3 | 87.93 | 1.550x | 1.423x | 2.714 | 0.580 | 30/80 | 2.0 |
| int4wo-g128-hqq-bf16head | bf16 | 4 | 85.96 | 1.501x | 1.357x | 3.070 | 0.527 | 32/80 | 2.0 |
| int4wo-g128-hqq | bf16 | 2 | 91.79 | 1.664x | 1.408x | 2.301 | 0.659 | 31/80 | 2.0 |
| int4wo-g128-hqq | bf16 | 3 | 91.23 | 1.668x | 1.404x | 2.684 | 0.571 | 34/80 | 2.0 |
| int4wo-g128-hqq | bf16 | 4 | 84.47 | 1.537x | 1.300x | 2.941 | 0.494 | 31/80 | 3.0 |
| int4wo-g128 | bf16 | 1 | 89.05 | 1.488x | 1.283x | 1.763 | 0.770 | 35/80 | 2.0 |
| int4wo-g128 | bf16 | 2 | 103.41 | 1.712x | 1.369x | 2.389 | 0.704 | 35/80 | 1.0 |
| int4wo-g128 | bf16 | 3 | 98.97 | 1.648x | 1.338x | 2.786 | 0.605 | 38/80 | 2.0 |
| int4wo-g128 | bf16 | 4 | 93.78 | 1.524x | 1.310x | 3.163 | 0.549 | 37/80 | 1.0 |
| int4wo-g128 | bf16 | 5 | 86.89 | 1.455x | 1.233x | 3.426 | 0.495 | 33/80 | 1.0 |
| int4wo-g128 | int8wo | 1 | 66.70 | 1.093x | 1.108x | 1.768 | 0.775 | 38/80 | 1.0 |
| int4wo-g128 | int8wo | 2 | 68.46 | 1.140x | 1.098x | 2.357 | 0.688 | 43/80 | 1.0 |
| int4wo-g128 | int8wo | 3 | 71.18 | 1.182x | 1.018x | 2.774 | 0.601 | 37/80 | 1.0 |
| int4wo-g128 | int8wo | 4 | 70.69 | 1.178x | 0.945x | 3.155 | 0.548 | 41/80 | 1.0 |
| int4wo-g64-bf16head | bf16 | 1 | 74.31 | 1.495x | 1.281x | 1.762 | 0.770 | 31/80 | 2.0 |
| int4wo-g64-bf16head | bf16 | 2 | 94.26 | 1.889x | 1.426x | 2.360 | 0.689 | 30/80 | 1.0 |
| int4wo-g64-bf16head | bf16 | 3 | 92.19 | 1.824x | 1.442x | 2.766 | 0.598 | 29/80 | 2.0 |
| int4wo-g64-bf16head | bf16 | 4 | 89.67 | 1.787x | 1.435x | 3.207 | 0.564 | 33/80 | 1.0 |
| int8wo | bf16 | 1 | 25.55 | 0.539x | 0.524x | 1.785 | 0.792 | 22/80 | 2.0 |
| int8wo | bf16 | 2 | 33.05 | 0.703x | 0.654x | 2.387 | 0.702 | 27/80 | 2.0 |
| int8wo | bf16 | 3 | 36.65 | 0.782x | 0.732x | 2.833 | 0.620 | 26/80 | 2.0 |
| int8wo | bf16 | 4 | 39.35 | 0.829x | 0.774x | 3.201 | 0.560 | 28/80 | 2.0 |
| int8wo | bf16 | 5 | 40.52 | 0.864x | 0.794x | 3.492 | 0.509 | 27/80 | 2.0 |
