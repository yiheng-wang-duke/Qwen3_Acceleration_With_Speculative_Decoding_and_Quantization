# AccelerateLLM

English | [中文](README_cn.md)

**Speculative decoding for Qwen3-8B with a Qwen3-0.6B draft: 1.83× over a CUDA-graph AR
baseline, 3.65× over eager Hugging Face decoding. Outputs are the target's own greedy
outputs, up to BF16 rounding at near-ties.**

## Key results

One NVIDIA RTX A5000, BF16, batch 1, greedy decoding, 80 MT-Bench first-turn prompts.
Every speed is measured end to end per request, including prefill.

| Configuration | Throughput | vs eager AR | vs compiled AR |
| --- | ---: | ---: | ---: |
| Eager AR (FlashAttention 2 + KV cache) | 18.30 tok/s | 1.00× | 0.51× |
| Eager speculative, best γ = 1 | 18.53 tok/s | 1.02× | – |
| Compiled AR (static KV cache + CUDA graphs) | 36.17 tok/s | 1.98× | 1.00× |
| **Compiled speculative, γ = 4** | **66.87 tok/s** | **3.65×** | **1.83×** |

- **Why eager mode fails.** In the stock Hugging Face loop every layer costs about 1.5 ms
  of CPU kernel dispatch, whatever its width. The 0.6B draft therefore costs 75% of an 8B
  target step (c = 0.745), and speculative decoding only breaks even, even though 81% of
  draft tokens match.
- **What fixes it.** A static KV cache plus `torch.compile(mode="reduce-overhead")` removes
  the dispatch cost. The draft step falls 8× (40.7 → 5.0 ms) and the target step 2×
  (54.7 → 27.3 ms, about 78% of peak memory bandwidth), so c falls to 0.183.
- **The theory matches.** The cost model predicts 1.48× / 1.72× / 1.85× for γ = 1 / 2 / 4,
  and the measured values are 1.45× / 1.71× / 1.83×. Measured acceptance matches the exact
  teacher-forced simulation to within ±0.08 tokens per verification.

Details: [Section 5](#5-results), [results/rtx-a5000-compiled/summary.md](results/rtx-a5000-compiled/summary.md),
[results/rtx-a5000-eager/summary.md](results/rtx-a5000-eager/summary.md).

## What this repository does

It measures whether speculative decoding speeds up LLM inference compared with a strong
autoregressive baseline, for **Qwen3-0.6B-Base (draft) → Qwen3-8B-Base (target)** in BF16
at batch size 1, with greedy decoding on the 80 MT-Bench first-turn prompts.

- **Baseline:** autoregressive (AR) decoding of the target.
- **Acceleration:** speculative decoding (Leviathan et al., 2023) with draft lengths γ.
- **Reported:** the AR forward-cost profile, the acceptance rate and E[L] for γ = 1…8,
  the theoretical speedup, and measured speedups (paired repeats + full MT-Bench).
- **Two runtime modes**, each with its own AR baseline:
  - **eager** (default, [speculative.yaml](configs/experiments/speculative.yaml)): BF16 +
    FlashAttention 2 + dynamic KV cache.
  - **compiled** ([speculative_compiled.yaml](configs/experiments/speculative_compiled.yaml)):
    BF16 + static KV cache + `torch.compile(mode="reduce-overhead")`, i.e. CUDA graphs, with
    SDPA attention. This mode removes per-kernel CPU dispatch (Section 3.6).

> Contents: [Overview](#1-overview) · [Step-by-step run](#2-step-by-step-run-on-a-new-server) ·
> [Method](#3-method) · [Outputs](#4-outputs) · [Results](#5-results) ·
> [Troubleshooting](#6-troubleshooting) · [Repository layout](#7-repository-layout)

---

## 1. Overview

| | Draft | Target |
| --- | --- | --- |
| Model | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Layers / hidden size | 28 / 1024 | 36 / 4096 |
| Parameters (BF16 size) | 0.60 B (≈1.2 GB) | 8.19 B (≈16.4 GB) |
| Tokenizer / output vocab | identical / 151,936 | identical / 151,936 |

The models share a tokenizer and output vocabulary, so draft token IDs can be verified by
the target directly. The draft is frozen and unquantized.

**What to look for.** At batch size 1, a forward pass costs either GPU time (reading the
weights) or host time (Python + CUDA kernel-launch dispatch), whichever is larger. In the
dispatch-bound case, the draft/target cost ratio is c ≈ 28/36 = 0.78 layers, which caps
the speedup at 1/c ≈ 1.29×. In the bandwidth-bound case, c ≈ 1.2/16.4 GB ≈ 0.07 and
speedups near 2× are possible. The benchmark measures c directly (Section 3.3) and reports
honestly whether speculative decoding wins. The compiled mode exists to move the models from
the first case toward the second.

---

## 2. Step-by-step run on a new server

### Step 0: Requirements

| Item | Requirement |
| --- | --- |
| GPU | One NVIDIA GPU with **≥ 24 GB** and BF16 support (Ampere or newer: A5000/A6000, RTX 3090/4090, A100, L40S, H100, …) |
| Driver | CUDA 12.x driver |
| OS / Python | Linux x86_64, Python 3.10–3.12 (tested with 3.12) |
| Disk | ≈ 20 GB for model weights (Hugging Face cache) |
| Host RAM | ≥ 32 GB |
| Network | Needed for setup only; the benchmark itself runs offline |

The GPU must be **otherwise idle**. The benchmark refuses to time on a GPU that runs other
compute processes.

### Step 1: Clone

~~~bash
git clone <your-repo-url> AccelerateLLM
cd AccelerateLLM
~~~

### Step 2: Python environment

~~~bash
conda create -n acceleratellm python=3.12 -y
conda activate acceleratellm

# PyTorch for your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt   # required: pins transformers==4.57.1 (5.x is not supported)
pip install -e . --no-deps        # installs only this package; dependencies come from the line above
# Preinstalled images (e.g. Google Colab): torchvision/torchaudio built for another torch break
# `import transformers`; this project does not use them:
#   pip uninstall -y torchvision torchaudio

# FlashAttention 2. Option A: prebuilt wheel matching python/torch/CUDA/C++ ABI, from
# https://github.com/Dao-AILab/flash-attention/releases (v2.8.3), e.g. for py3.12 + torch2.8 + cu12:
#   pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# Check the ABI flag with:  python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# Option B: build from source (slow, needs nvcc):
#   pip install flash-attn==2.8.3 --no-build-isolation

python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
~~~

Tested versions: torch 2.8.0+cu129, transformers 4.57.1, flash-attn 2.8.3, einops 0.8.1.

### Step 3: Download models and data

~~~bash
# Optional: export HF_HOME=/path/with/20GB/free
python scripts/fetch_models.py   # Qwen3-0.6B-Base + Qwen3-8B-Base at the pinned revisions
python scripts/fetch_data.py     # MT-Bench questions -> data/mt_bench/question.jsonl (SHA256-checked)
~~~

### Step 4: Run the tests (CPU, ≈ 20 s)

~~~bash
python -m pytest -q
~~~

All tests should pass. They use tiny toy models and include an end-to-end CPU dry run of
the whole benchmark and summary pipeline (`tests/test_benchmark_pipeline.py`).

### Step 5: Check the GPU setup (≈ 2 min)

~~~bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0          # choose one idle GPU
python -m acceleratellm doctor        # versions + visible GPU
python -m acceleratellm check-models --config configs/experiments/speculative.yaml
python -m acceleratellm baseline --config configs/experiments/baseline.yaml   # one-prompt AR run of the target
~~~

`baseline` prints a run directory. Its `metrics.json` holds throughput, and
`verification.json` compares the output with `transformers.generate`.

### Step 6: Run the full benchmark (≈ 1.5 h on an RTX A5000)

~~~bash
# Plain shell, on a machine with one idle GPU:
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 bash scripts/run_benchmark.sh 2>&1 | tee outputs/benchmark.log

# or with Slurm (adjust partition / time to your cluster):
mkdir -p outputs && sbatch -p <partition> scripts/run_benchmark.sh
~~~

The script runs `doctor`, `check-models` and the tests, then `scripts/benchmark.py`, then
`scripts/summarize.py`. Progress lines to watch for:

~~~text
BENCHMARK_DIR=outputs/<timestamp>-<id>
AR draft: ...  /  AR target: ... MT-Bench ... tok/s       # phase 1
THEORY: alpha=... c=... best gamma=... decode ...x        # phase 2
PAIRED gamma=1/2/4: ... -> ...x                           # phase 3
MT-BENCH gamma=1/2/4: ... speedup ...x                    # phase 4
BENCHMARK_COMPLETE=outputs/<timestamp>-<id>
~~~

If a phase fails, `outputs/<run>/status.json` holds the error and `progress.json` holds
everything finished so far.

**Compiled mode** (optional, recommended as the second run):

~~~bash
CONFIG=configs/experiments/speculative_compiled.yaml CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_benchmark.sh 2>&1 | tee outputs/benchmark-compiled.log
~~~

The run first prints `COMPILE_PROBE={...}`, which compares compiled static-cache decoding
with eager decoding on a toy Qwen3. The first forward pass at each new query length
triggers compilation, adding a few minutes in total. Compilation happens only in untimed
warm-up and checking runs, so it does not enter the timings.

### Step 7: Summarize and bring the results back

~~~bash
python scripts/summarize.py --latest   # newest run -> results/<gpu>-<eager|compiled>/
~~~

`run_benchmark.sh` already calls this. Each GPU and mode gets its own folder (e.g.
`results/l4-eager/`, `results/rtx-a5000-compiled/`) containing `summary.md`,
`results.json` and `environment.json`. `results/` is tracked by git, so commit it to share
the numbers; `outputs/` (raw per-request records) is ignored.

---

## 3. Method

### 3.1 Decoding

- **AR baseline:** prefill the prompt, then one cached forward pass per token, taking the
  argmax at each step.
- **Speculative decoding:**
  - Each round, the draft proposes γ tokens autoregressively. The target scores the pending
    token plus all γ proposals in one forward pass.
  - Greedy mode accepts the longest prefix that matches the target argmax and appends the
    target's correction token, or a bonus token if all γ are accepted.
  - Sampling mode uses min(1, p/q) acceptance with residual resampling. It is implemented
    and tested, but not part of the benchmark.
- **KV-cache rollback:** after a rejection, both caches are cropped back to the accepted
  prefix. The device-side greedy loop makes one host transfer per round.

### 3.2 Exact greedy acceptance estimator

Under greedy decoding, the committed output is the target's AR trajectory y regardless of γ.
While the draft matches y, it extends exactly the prefix y[:j]. So a single teacher-forced
draft pass over prompt + y yields every proposal the real loop would make, d[j] =
argmax draft(y[:j]). Simulating rounds from d gives the exact accepted / proposed /
verification counts for every γ at once
([evaluation/acceptance.py](src/acceleratellm/evaluation/acceptance.py)). The tests check
this against the real decoder on hundreds of cases, and phases 3–4 compare it with measured
counts on the GPU.

Reported metrics:

- **α:** per-token greedy agreement, P(d[j] = y[j]).
- **Prefix acceptance:** accepted draft tokens / proposed draft tokens.
- **E[L]:** output tokens per target verification (accepted + 1).

### 3.3 Cost model

`T_x(q)` is the measured time of one cached forward pass with q query tokens, including
cache crop, argmax and host copy. It is averaged over contexts of 64/256/512 tokens and
20 repeats. The predicted decode speedup is

~~~text
S(γ) = E[L] · T_t(1) / (γ · T_d(1) + T_t(γ + 1)),      c = T_d(1) / T_t(1)
~~~

A request-level prediction adds the measured prefill times. The model ignores loop
orchestration (Python per round), so measured speedups are usually slightly lower.

### 3.4 Benchmark phases (`scripts/benchmark.py`)

| Phase | What | Prompts |
| --- | --- | --- |
| 0 | Tokenizer/vocab check. Eager mode: FlashAttention-2 kernel + KV-crop probe (toy Qwen3, profiler-verified). Compiled mode: compiled static-cache vs eager dynamic-cache logits and crop probe | — |
| 1 | AR per model: agreement with `transformers.generate`, timed AR (2 warm-up + 5 repeats), decode-only split, `T(1..9)`; target: one timed AR pass per MT-Bench prompt (defines trajectories) | 3 standard + 80 MT-Bench |
| 2 | Teacher-forced acceptance for γ = 1…8, by category; predicted speedups | 80 MT-Bench (+3 standard) |
| 3 | Paired AR vs speculative, 1 warm-up + 3 repeats, off-clock replay check | 3 standard + 1 per MT-Bench category |
| 4 | One timed speculative pass per prompt and γ vs the phase-1 AR pass | 80 MT-Bench |

γ values for phases 3–4 come from `speculative.draft_lengths` in
[configs/experiments/speculative.yaml](configs/experiments/speculative.yaml) (default 1, 2, 4).

### 3.5 Timing and correctness protocol

- Batch size 1, unpadded input, greedy decoding, at most 128 new tokens or EOS.
- **Prompt format:** MT-Bench first turn wrapped as `Question: {turn}\nAnswer:` (base
  models, no chat template). FastChat commit `b494d0c6…`, SHA256 `119565ad…44f7`.
- **What is timed:** CUDA is synchronized at the timing boundaries. Timing includes prefill,
  all draft and target forward passes, the accept/reject logic and cache rollback. It
  excludes model loading, tokenization and detokenization.
- **Fair comparison:** speculative and AR runs use the same GPU, the same process setup and
  the same prompts. The AR baseline is measured with only the target loaded.
- **Replay check:** timed runs in phases 1 and 3 are replayed off the clock, with
  finite-logit checks, and must reproduce identical outputs.
- **BF16 numerics:** BF16 has an 8-bit mantissa, so verifying γ+1 tokens in one pass can
  round differently from one-token decoding and flip a near-tied argmax. Timed runs use
  **no** numerical replay. Each output that differs from the AR output is recorded with the
  AR top-2 logit gap at the first differing position, measured in BF16 ULPs. Small gaps
  mean a rounding effect; a large gap would indicate a bug. `greedy_tie_fallback: true`
  enables an exact replay mode for correctness studies, and that mode is not timed.
- **Clean GPU:** a preflight before every phase requires one visible GPU, at least 20 GiB
  free and no other compute processes.

### 3.6 Compiled mode: static KV cache + CUDA graphs

At batch size 1, the eager PyTorch loop issues roughly 40–60 small CUDA kernels per decoder
layer, one at a time from Python. Each costs 20–40 µs of host time, while its GPU work often
takes under 10 µs. A forward pass therefore costs about (layers × host overhead) rather than
weight-read time. The 28-layer draft and 36-layer target then differ in cost by only
28/36, which leaves little room for speculative decoding.

`runtime.compile: true` ([hf_backend.py](src/acceleratellm/models/hf_backend.py)) changes
this as follows:

- **Static KV cache.** Each model owns one preallocated `transformers.StaticCache`
  (capacity `runtime.static_cache_len`, default 1024) at fixed addresses. The adapter
  tracks the valid length and passes explicit `cache_position`/`position_ids`. A rollback
  only lowers that length: stale slots are overwritten by later writes and hidden by the
  causal mask. Starting a new request invalidates older cache handles, and using a stale
  handle raises an error.
- **CUDA graphs.** Forward passes with at most 16 new tokens (decode = 1, draft catch-up
  = 2, verification = γ+1) run through
  `torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)`. There is one
  recorded CUDA graph per query length, replayed with a single launch. Prefill always
  runs eagerly, as recommended by transformers. Graph outputs are copied before the next
  replay.
- **Attention backend.** SDPA is used instead of FlashAttention 2, because the FA2 path in
  transformers unpads the attention mask with data-dependent shapes, which CUDA graphs
  cannot capture. At these context lengths attention is a negligible share of the time.
- **Verification.** Correctness is checked by CPU tests, where the static cache must
  equal the dynamic cache for forward, crop, AR and speculative decoding and acceptance
  counters, and by the GPU `compile_probe`. `results.json` records how many forward passes
  used the compiled and the eager paths.
- **Memory release.** Dynamo's code cache and the CUDA-graph pool keep references to a
  compiled model's weights. Between phases, `benchmark.py` therefore calls
  `torch._dynamo.reset()` and resets the CUDA-graph trees before freeing memory, so a
  deleted 8B model does not stay resident. Each phase recompiles, which takes about 30 s.

---

## 4. Outputs

`outputs/<timestamp>-<id>/` (ignored by git):

| File | Content |
| --- | --- |
| `results.json` | Complete report: model facts, AR profile, cost table, acceptance, theory, paired and full MT-Bench timing, divergences, source hashes |
| `config.yaml`, `environment.json` | Merged config; Python/package/CUDA/GPU info |
| `ar-draft.json`, `ar-target.json` | Per-request AR timings |
| `mt-bench-ar.json`, `trajectories.json` | Target AR pass over MT-Bench; greedy trajectories |
| `acceptance_per_prompt.json` | Simulated counters per prompt and γ |
| `paired-gamma{γ}.json`, `mt-bench-spec-gamma{γ}.json` | Per-request speculative timings and counters |
| `status.json`, `progress.json` | Run state; partial report on failure |

`results/<gpu>-<eager|compiled>/` (tracked by git) holds `summary.md`, `results.json` and
`environment.json`, produced by `scripts/summarize.py`.

---

## 5. Results

Measured on one **NVIDIA RTX A5000** (24 GB, 768 GB/s) with torch 2.8.0+cu129 and
transformers 4.57.1: BF16, batch 1, greedy, at most 128 new tokens, 80 MT-Bench first-turn
prompts. Full tables: [results/rtx-a5000-eager/summary.md](results/rtx-a5000-eager/summary.md)
and [results/rtx-a5000-compiled/summary.md](results/rtx-a5000-compiled/summary.md).

**Summary.** In eager mode, speculative decoding only breaks even (best 1.02×), because
the draft is dispatch-bound and costs 75% of a target step. CUDA graphs cut the draft step
8× and the target step 2×. Against the 2× faster compiled AR baseline, speculative
decoding then gives **1.83× on full MT-Bench** (γ = 4), about **3.65× end to end** over
eager AR.

### 5.1 Forward cost per decode step

| | Eager + FA2 + DynamicCache | Compiled (StaticCache + CUDA graphs) + SDPA |
| --- | ---: | ---: |
| Draft 0.6B `T(1)` | 40.71 ms (1.45 ms/layer) | **5.00 ms** (0.18 ms/layer) |
| Target 8B `T(1)` | 54.65 ms (1.52 ms/layer) | **27.27 ms** (0.76 ms/layer) |
| Target `T(5)/T(1)` (verifying 4 drafts) | 0.99 | 1.02 |
| Cost ratio c = T_d/T_t | 0.745 (ceiling 1.34×) | **0.183** (ceiling 5.45×) |
| Target AR, MT-Bench | 18.30 tok/s | **36.17 tok/s** |
| Peak memory (both models) | 16.42 GiB | 16.66 GiB |

In eager mode the per-layer cost is about 1.5 ms for both models, whatever their width, so
the forward pass is host-dispatch bound and c ≈ the layer ratio. Compiled, the target reads
16.4 GB of weights in 27.3 ms, about 78% of the A5000's peak bandwidth, so it is now
bandwidth bound. Verifying γ+1 tokens costs almost the same as decoding 1 in both modes.

### 5.2 Acceptance

Acceptance depends only on the models, so it is nearly the same in both modes. The small
differences come from BF16 near-ties.

| γ | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefix acceptance | 0.80 | 0.72 | 0.65 | 0.59 | 0.49 | 0.41 |
| E[L] tokens / verification (compiled) | 1.79 | 2.42 | 2.92 | 3.32 | 3.89 | 4.22 |

Per-token agreement α = 0.81. It varies by category, from math 0.92, extraction 0.90 and
coding 0.87 down to writing, roleplay and reasoning at about 0.73–0.75. Measured E[L]
matches the teacher-forced simulation to within ±0.08.

### 5.3 Speculative decoding vs AR (full MT-Bench, 80 prompts)

| γ | Eager: tok/s | Eager: speedup | Compiled: predicted | Compiled: tok/s | Compiled: speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 18.30 | 1.00× | – | 36.17 | 1.00× |
| 1 | 18.53 | **1.018×** | 1.48× | 52.54 | 1.447× |
| 2 | 17.61 | 0.963× | 1.72× | 62.01 | 1.707× |
| 4 | 14.91 | 0.814× | 1.85× | 66.87 | **1.833×** |

The cost model (Section 3.3) predicts the compiled speedups to within 1–2%. The paired
subset (11 prompts × 3 repeats) agrees: 1.47×, 1.73× and 1.92× for γ = 1, 2, 4. For
compiled mode, the theory puts the optimum at γ = 4 (1.85× request, 1.91× decode-only),
with γ = 3–5 within 2% of it.

**Output equality.** 37–38 of 80 compiled outputs (32 of 80 eager) are token-identical to
AR. Every divergence starts at a near-tie, where the AR top-2 logit gap is ≤ 2 BF16 ULP and
the 1-token and (γ+1)-token kernels round differently. These divergences come from BF16
arithmetic, not from the algorithm. The float32 CPU tests require exact equality.

**Evidence that CUDA graphs were used.** In the compiled run, 98.5% of draft and target
forward passes in the timed phases went through the compiled path (80,799 of 82,044). The eager remainder is
the prefills.

### 5.4 Conclusions

1. With the stock eager Hugging Face loop, speculative decoding at this model pair
   cannot help, even though α ≈ 0.8. The bottleneck is not acceptance but the draft's
   CPU dispatch cost (c = 0.75).
2. Removing dispatch overhead with a static cache and CUDA graphs makes the AR baseline
   itself 2× faster. It also makes the 0.6B draft cheap relative to the 8B target
   (c = 0.18), so speculative decoding then adds a further 1.83×.
3. Best configuration: compiled mode, γ = 4, 66.9 tok/s, versus 36.2 tok/s for compiled AR
   and 18.3 tok/s for eager AR.

---

### 5.5 Weight-only quantization × speculative decoding (draft section)

Follow-up study: `scripts/quant_spec.py` (one target/draft pair per process, compiled mode, same
80 MT-Bench prompts) and `scripts/gptq_quantize.py` (GPTQ W4 g128 checkpoint of the target, 128
RedPajama-V2 calibration sequences, run with gptqmodel's Marlin kernel). Full tables:
[results/rtx-a5000-quant-spec/summary.md](results/rtx-a5000-quant-spec/summary.md).

| Target (draft: BF16 0.6B) | PPL (BF16 9.44) | AR tok/s | AR vs BF16 | Best spec tok/s (γ) | Spec vs own AR | Spec vs BF16 AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 9.44 | 36.4 | 1.00× | 66.8 (5) | 1.82× | 1.83× |
| int8 weight-only (torchao) | 9.48 | 46.7 | 1.28× | 40.5 (5) | 0.86× | 1.11× |
| int4 RTN g128, lm_head BF16 (torchao) | 10.90 | 56.3 | 1.55× | 98.1 (2) | 1.71× | 2.69× |
| **GPTQ W4 g128 + Marlin, lm_head BF16** | **9.72** | **74.6** | **2.05×** | **100.1 (2)** | 1.34× | **2.75×** |

- **The two methods compose.** Every int4 target gains another 1.3–1.9× from speculative decoding.
  The best quality/speed point is GPTQ-Marlin + speculative decoding: 2.75× BF16 AR at +3 % PPL.
- **The extra gain shrinks with a faster target.** The 0.6B draft stays at about 5 ms per step,
  because it is kernel-count bound, so c rises from 0.19 to 0.39 and the best γ drops from 4–5 to 2.
- **Verification cost must stay flat.** Marlin's T(5)/T(1) is 1.05. torchao int8 weight-only has a
  fused kernel only for one query token (T(5)/T(1) = 3.16), so speculative decoding is slower than
  AR at every γ.
- **Do not quantize the draft.** An int8 0.6B draft is slower (8.2 vs 5 ms) at the same acceptance.
  An int4 copy of the 8B target as the draft reaches α = 0.85 but only 1.13–1.17×.
- **Quality.** Quantizing the untied lm_head to int4 raises PPL from 10.90 to 12.90. HQQ did not help.
  GPTQ calibration recovers most of the loss (9.72).

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Other compute processes on the GPU` | Pick an idle GPU with `CUDA_VISIBLE_DEVICES`. To run anyway (timings unreliable): `ACCELERATELLM_ALLOW_SHARED_GPU=1` |
| `Less than 20 GiB free on the GPU` | Use a ≥ 24 GB GPU with nothing else loaded |
| `Expose exactly one CUDA GPU` | Set `CUDA_VISIBLE_DEVICES` to a single index |
| `runtime.compile ... requires ... sdpa` | Compiled mode only supports SDPA attention; use `configs/experiments/speculative_compiled.yaml` |
| `Input exceeds runtime.static_cache_len` | Raise `runtime.static_cache_len` (prompt + new tokens + γ + 1 must fit) |
| `torch._dynamo` / Inductor errors in compiled mode | Check that torch is 2.8.0 and transformers 4.57.1; set `TORCH_LOGS=recompiles,graph_breaks` for details. The eager mode is unaffected |
| `ImportError` / undefined symbol in `flash_attn` | The wheel does not match python/torch/CUDA/ABI; reinstall the matching wheel (Step 2) |
| `No module named einops` (FA2) | `pip install einops==0.8.1` |
| `Could not import module 'Qwen3ForCausalLM'` | Run `python -c "import transformers.models.qwen3.modeling_qwen3"` to see the real error. Usual causes: transformers 5.x installed (run `pip install -r requirements.txt`), or torchvision/torchaudio that do not match torch (`pip uninstall -y torchvision torchaudio`) |
| `... is not a supported wheel on this platform` (flash-attn) | The wheel's `cpXY` tag must match your Python (e.g. `cp313` for Python 3.13), and `torchX.Y` / `cxx11abi` must match torch |
| `OSError ... local_files_only` | Run Step 3 first, or unset `HF_HUB_OFFLINE` for the download |
| `MT-Bench ... missing` / `differs from the pinned` | Run `python scripts/fetch_data.py` (it verifies SHA256) |
| Slurm job finds no `outputs/` | `mkdir -p outputs` before `sbatch` |

---

## 7. Repository layout

~~~text
configs/
  base.yaml                 runtime (BF16, FA2, cuda), generation (greedy, 128 tokens), timing repeats
  models.yaml               pinned draft/target revisions
  experiments/baseline.yaml       AR baseline
  experiments/speculative.yaml    speculative settings; also the benchmark configuration (eager)
  experiments/speculative_compiled.yaml   same study, static KV cache + torch.compile (CUDA graphs)
src/acceleratellm/
  config.py                 YAML inheritance + strict validation
  models/                   HF backend (BF16, FA2/SDPA, dynamic or static+compiled KV cache), tokenizer checks
  decoding/                 AR, speculative (reference + device greedy), sampling, near-tie replay
  evaluation/               timing harness, exact acceptance simulator, speedup cost model
  experiments/              CLI runners (baseline, speculative) + artifact/environment writers
  data/mt_bench.py          pinned MT-Bench loader and prompt format
scripts/
  fetch_models.py, fetch_data.py   downloads (pinned, verified)
  benchmark.py              full 5-phase benchmark
  run_benchmark.sh          bash/sbatch wrapper: checks + tests + benchmark + summary
  summarize.py              results.json -> results/<gpu>-<mode>/summary.md
tests/                      CPU tests with toy models, incl. end-to-end benchmark dry run
~~~

CLI: `python -m acceleratellm {doctor, validate, check-models, baseline, speculative} --config <yaml>`.
`speculative` runs a small paired AR-vs-speculative comparison on the configured prompts,
with greedy and sampling checks.

The proposal ([nlp_project_proposal.pdf](nlp_project_proposal.pdf)) also covered weight
quantization (GPTQ/QAT). The current scope is BF16 AR vs speculative decoding with Qwen3.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Dao. *FlashAttention-2.* ICLR 2024.
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Qwen Team. *Qwen3 Technical Report.* 2025.
