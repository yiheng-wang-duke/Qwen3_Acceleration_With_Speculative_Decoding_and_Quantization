# AccelerateLLM

Does speculative decoding speed up LLM inference compared with a strong autoregressive
baseline? This repository measures it for **Qwen3-0.6B-Base (draft) → Qwen3-8B-Base
(target)** with **BF16 + FlashAttention 2 + KV cache**, batch size 1, greedy decoding, on
the 80 MT-Bench first-turn prompts.

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

*To be filled from `results/<gpu>-<mode>/summary.md` after Step 7.* Each summary contains:

1. **AR baseline and forward-cost profile:** per-model `T(1)`, ms/layer, `T(q)/T(1)`, AR
   throughput (standard prompts and MT-Bench), decode-only throughput, peak memory.
2. **Acceptance and predicted speedup:** α, prefix acceptance, E[L] and predicted speedup
   for γ = 1…8; E[L] per MT-Bench category.
3. **Paired timing:** measured speedup vs AR, measured vs predicted E[L], greedy agreement.
4. **Full MT-Bench timing:** request-level speedup per γ, and the verdict on whether
   speculative decoding beats the BF16 AR baseline.

---

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
