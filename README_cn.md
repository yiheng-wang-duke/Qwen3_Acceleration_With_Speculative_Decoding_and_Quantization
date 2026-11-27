# AccelerateLLM

[English](README.md) | 中文

**以 Qwen3-0.6B 为 draft、Qwen3-8B 为 target 做投机解码（speculative decoding）：相对 CUDA Graph
自回归基线加速 1.83×，相对 Hugging Face eager 解码加速 3.65×。输出就是 target 自身的贪心输出，
仅在近似平局处可能因 BF16 舍入而不同。**

## 核心结果

单张 NVIDIA RTX A5000，BF16，batch 1，贪心解码，80 条 MT-Bench 第一轮问题。所有速度均为
按请求端到端测量（含 prefill）。

| 配置 | 吞吐 | 相对 eager AR | 相对编译 AR |
| --- | ---: | ---: | ---: |
| Eager AR（FlashAttention 2 + KV cache） | 18.30 tok/s | 1.00× | 0.51× |
| Eager 投机解码，最佳 γ = 1 | 18.53 tok/s | 1.02× | – |
| 编译 AR（静态 KV cache + CUDA Graph） | 36.17 tok/s | 1.98× | 1.00× |
| **编译投机解码，γ = 4** | **66.87 tok/s** | **3.65×** | **1.83×** |

- **eager 模式为什么不行。** 原生 Hugging Face 循环里，每层大约要花 1.5 ms 在 CPU 下发 kernel
  上，与层宽无关。于是 0.6B draft 的一步耗时达到 8B target 一步的 75%（c = 0.745）。即便 81%
  的 draft token 与 target 一致，投机解码也只能打平。
- **怎么解决。** 静态 KV cache 加 `torch.compile(mode="reduce-overhead")` 消除了下发开销。draft
  单步降低 8×（40.7 → 5.0 ms），target 单步降低 2×（54.7 → 27.3 ms，约为显存带宽峰值的 78%），
  c 降到 0.183。
- **理论与实测吻合。** 代价模型预测 γ = 1 / 2 / 4 时为 1.48× / 1.72× / 1.85×，实测为
  1.45× / 1.71× / 1.83×。实测接受长度与精确的 teacher-forced 模拟相差不超过每次验证 ±0.08 个 token。

详细内容：[第 5 节](#5-结果)、[results/rtx-a5000-compiled/summary.md](results/rtx-a5000-compiled/summary.md)、
[results/rtx-a5000-eager/summary.md](results/rtx-a5000-eager/summary.md)。

## 本仓库做什么

测量投机解码相对强自回归基线能否加速 LLM 推理。模型为 **Qwen3-0.6B-Base（draft）→
Qwen3-8B-Base（target）**，BF16，batch size 1，在 80 条 MT-Bench 第一轮问题上贪心解码。

- **基线：** target 的自回归（AR）解码。
- **加速手段：** 投机解码（Leviathan et al., 2023），draft 长度为 γ。
- **报告内容：** AR 前向代价剖析、γ = 1…8 的接受率与 E[L]、理论加速比，以及实测加速比
  （配对重复测量 + 完整 MT-Bench）。
- **两种运行模式**，各自有独立的 AR 基线：
  - **eager**（默认，[speculative.yaml](configs/experiments/speculative.yaml)）：BF16 +
    FlashAttention 2 + 动态 KV cache。
  - **compiled**（[speculative_compiled.yaml](configs/experiments/speculative_compiled.yaml)）：
    BF16 + 静态 KV cache + `torch.compile(mode="reduce-overhead")`（即 CUDA Graph），注意力用 SDPA。
    该模式消除了逐 kernel 的 CPU 下发开销（第 3.6 节）。

> 目录：[概览](#1-概览) · [分步运行](#2-在新服务器上分步运行) · [方法](#3-方法) ·
> [输出](#4-输出) · [结果](#5-结果) · [常见问题](#6-常见问题) · [仓库结构](#7-仓库结构)

---

## 1. 概览

| | Draft | Target |
| --- | --- | --- |
| 模型 | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| Revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| 层数 / 隐藏维度 | 28 / 1024 | 36 / 4096 |
| 参数量（BF16 大小） | 0.60 B（≈1.2 GB） | 8.19 B（≈16.4 GB） |
| Tokenizer / 输出词表 | 相同 / 151,936 | 相同 / 151,936 |

两个模型共享 tokenizer 和输出词表，target 可直接验证 draft 的 token ID。draft 冻结、未量化。

**关注点。** batch size 1 时，一次前向的耗时取 GPU 时间（读权重）和主机时间（Python + CUDA
kernel 下发）中的较大者。在下发受限的情况下，draft/target 代价比 c ≈ 层数比 28/36 = 0.78，
加速上限为 1/c ≈ 1.29×。在带宽受限的情况下，c ≈ 1.2/16.4 GB ≈ 0.07，有可能接近 2× 加速。
benchmark 直接测量 c（第 3.3 节），并如实报告投机解码是否更快。编译模式的作用就是把模型从
前一种情况推向后一种。

---

## 2. 在新服务器上分步运行

### Step 0：环境要求

| 项目 | 要求 |
| --- | --- |
| GPU | 一张 **≥ 24 GB**、支持 BF16 的 NVIDIA GPU（Ampere 及更新：A5000/A6000、RTX 3090/4090、A100、L40S、H100 等） |
| 驱动 | CUDA 12.x 驱动 |
| 系统 / Python | Linux x86_64，Python 3.10–3.12（在 3.12 上测试） |
| 磁盘 | 约 20 GB 存放模型权重（Hugging Face 缓存） |
| 内存 | ≥ 32 GB |
| 网络 | 仅安装阶段需要，benchmark 本身离线运行 |

GPU 必须**空闲**。如果 GPU 上有其他计算进程，benchmark 会拒绝计时。

### Step 1：克隆

~~~bash
git clone <your-repo-url> AccelerateLLM
cd AccelerateLLM
~~~

### Step 2：Python 环境

~~~bash
conda create -n acceleratellm python=3.12 -y
conda activate acceleratellm

# 按你的 CUDA 版本安装 PyTorch（见 https://pytorch.org/get-started/locally/）
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt   # 必需：固定 transformers==4.57.1（不支持 5.x）
pip install -e . --no-deps        # 只安装本包；依赖来自上一行
# 预装镜像（如 Google Colab）：为其他 torch 版本编译的 torchvision/torchaudio 会导致
# `import transformers` 失败；本项目不需要它们：
#   pip uninstall -y torchvision torchaudio

# FlashAttention 2。方式 A：与 python/torch/CUDA/C++ ABI 匹配的预编译 wheel，来自
# https://github.com/Dao-AILab/flash-attention/releases (v2.8.3)，例如 py3.12 + torch2.8 + cu12：
#   pip install --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
# 查看 ABI 标志：  python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# 方式 B：从源码编译（慢，需要 nvcc）：
#   pip install flash-attn==2.8.3 --no-build-isolation

python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.is_available(), flash_attn.__version__)"
~~~

测试过的版本：torch 2.8.0+cu129、transformers 4.57.1、flash-attn 2.8.3、einops 0.8.1。

### Step 3：下载模型和数据

~~~bash
# 可选：export HF_HOME=/path/with/20GB/free
python scripts/fetch_models.py   # 按固定 revision 下载 Qwen3-0.6B-Base + Qwen3-8B-Base
python scripts/fetch_data.py     # MT-Bench 问题 -> data/mt_bench/question.jsonl（校验 SHA256）
~~~

### Step 4：运行测试（CPU，约 20 秒）

~~~bash
python -m pytest -q
~~~

所有测试都应通过。测试使用小型玩具模型，并包含整个 benchmark 和汇总流程的端到端 CPU 演练
（`tests/test_benchmark_pipeline.py`）。

### Step 5：检查 GPU 环境（约 2 分钟）

~~~bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0          # 选一张空闲 GPU
python -m acceleratellm doctor        # 版本 + 可见 GPU
python -m acceleratellm check-models --config configs/experiments/speculative.yaml
python -m acceleratellm baseline --config configs/experiments/baseline.yaml   # target 单条 prompt 的 AR 运行
~~~

`baseline` 会打印一个运行目录。其中 `metrics.json` 是吞吐，`verification.json` 是与
`transformers.generate` 输出的比对。

### Step 6：运行完整 benchmark（RTX A5000 上约 1.5 小时）

~~~bash
# 普通 shell，机器上有一张空闲 GPU：
mkdir -p outputs
CUDA_VISIBLE_DEVICES=0 bash scripts/run_benchmark.sh 2>&1 | tee outputs/benchmark.log

# 或使用 Slurm（按你的集群调整 partition / time）：
mkdir -p outputs && sbatch -p <partition> scripts/run_benchmark.sh
~~~

脚本依次运行 `doctor`、`check-models` 和测试，然后是 `scripts/benchmark.py`，最后是
`scripts/summarize.py`。需要关注的进度行：

~~~text
BENCHMARK_DIR=outputs/<timestamp>-<id>
AR draft: ...  /  AR target: ... MT-Bench ... tok/s       # 阶段 1
THEORY: alpha=... c=... best gamma=... decode ...x        # 阶段 2
PAIRED gamma=1/2/4: ... -> ...x                           # 阶段 3
MT-BENCH gamma=1/2/4: ... speedup ...x                    # 阶段 4
BENCHMARK_COMPLETE=outputs/<timestamp>-<id>
~~~

如果某个阶段失败，`outputs/<run>/status.json` 记录错误，`progress.json` 保存已完成的部分。

**编译模式**（可选，推荐作为第二次运行）：

~~~bash
CONFIG=configs/experiments/speculative_compiled.yaml CUDA_VISIBLE_DEVICES=0 \
  bash scripts/run_benchmark.sh 2>&1 | tee outputs/benchmark-compiled.log
~~~

运行开始时会打印 `COMPILE_PROBE={...}`，在玩具 Qwen3 上比较编译后的静态 cache 解码与 eager
解码。每遇到一个新的 query 长度，第一次前向会触发编译，总共多花几分钟。编译只发生在不计时的
预热和检查运行中，不计入计时结果。

### Step 7：汇总并取回结果

~~~bash
python scripts/summarize.py --latest   # 最新一次运行 -> results/<gpu>-<eager|compiled>/
~~~

`run_benchmark.sh` 已经会调用这一步。每种 GPU 和模式各有一个目录（如 `results/l4-eager/`、
`results/rtx-a5000-compiled/`），包含 `summary.md`、`results.json` 和 `environment.json`。
`results/` 由 git 跟踪，提交它即可分享结果；`outputs/`（逐请求原始记录）被 git 忽略。

---

## 3. 方法

### 3.1 解码

- **AR 基线：** 先 prefill prompt，之后每个 token 做一次带 cache 的前向，每步取 argmax。
- **投机解码：**
  - 每一轮，draft 自回归地提出 γ 个 token。target 在一次前向中为待定 token 和全部 γ 个提议打分。
  - 贪心模式接受与 target argmax 一致的最长前缀，并追加 target 的纠正 token；若 γ 个全部接受，
    则追加一个 bonus token。
  - 采样模式使用 min(1, p/q) 接受和残差重采样。已实现并测试，但不在 benchmark 中。
- **KV cache 回滚：** 发生拒绝后，两个 cache 都裁剪回已接受的前缀。设备端贪心循环每轮只做一次
  主机传输。

### 3.2 精确的贪心接受率估计

在贪心解码下，无论 γ 取多少，最终输出都是 target 的 AR 轨迹 y。只要 draft 与 y 一致，它扩展的
就恰好是前缀 y[:j]。因此，对 prompt + y 做一次 teacher-forced 的 draft 前向，就能得到真实循环中
每一个提议 d[j] = argmax draft(y[:j])。基于 d 模拟各轮，即可一次得到所有 γ 下精确的接受 / 提议 /
验证次数（[evaluation/acceptance.py](src/acceleratellm/evaluation/acceptance.py)）。测试在数百个
用例上把它与真实解码器对照，阶段 3–4 在 GPU 上把它与实测计数对照。

报告的指标：

- **α：** 逐 token 的贪心一致率，P(d[j] = y[j])。
- **前缀接受率：** 被接受的 draft token 数 / 提议的 draft token 数。
- **E[L]：** 每次 target 验证产出的 token 数（接受数 + 1）。

### 3.3 代价模型

`T_x(q)` 为一次带 cache、q 个 query token 的前向的实测时间，包括 cache 裁剪、argmax 和主机拷贝，
在 64/256/512 token 上下文、各 20 次重复上取平均。预测的解码加速比为

~~~text
S(γ) = E[L] · T_t(1) / (γ · T_d(1) + T_t(γ + 1)),      c = T_d(1) / T_t(1)
~~~

请求级预测再加上实测的 prefill 时间。模型忽略了循环编排开销（每轮的 Python 开销），因此实测
加速比通常略低。

### 3.4 Benchmark 阶段（`scripts/benchmark.py`）

| 阶段 | 内容 | Prompts |
| --- | --- | --- |
| 0 | Tokenizer/词表检查。eager 模式：FlashAttention-2 kernel + KV 裁剪探针（玩具 Qwen3，经 profiler 验证）。编译模式：编译静态 cache 与 eager 动态 cache 的 logits 及裁剪探针 | — |
| 1 | 每个模型的 AR：与 `transformers.generate` 的一致性、计时 AR（2 次预热 + 5 次重复）、纯解码拆分、`T(1..9)`；target：每条 MT-Bench prompt 一次计时 AR（确定轨迹） | 3 条标准 + 80 条 MT-Bench |
| 2 | γ = 1…8 的 teacher-forced 接受率，按类别；预测加速比 | 80 条 MT-Bench（+3 条标准） |
| 3 | 配对 AR 与投机解码，1 次预热 + 3 次重复，计时外回放检查 | 3 条标准 + 每个 MT-Bench 类别 1 条 |
| 4 | 每条 prompt、每个 γ 一次计时投机解码，与阶段 1 的 AR 对比 | 80 条 MT-Bench |

阶段 3–4 的 γ 取自 [configs/experiments/speculative.yaml](configs/experiments/speculative.yaml)
中的 `speculative.draft_lengths`（默认 1、2、4）。

### 3.5 计时与正确性协议

- batch size 1，无 padding，贪心解码，最多 128 个新 token 或遇到 EOS。
- **Prompt 格式：** MT-Bench 第一轮包装为 `Question: {turn}\nAnswer:`（base 模型，不用 chat
  template）。FastChat commit `b494d0c6…`，SHA256 `119565ad…44f7`。
- **计时范围：** 在计时边界同步 CUDA。计时包含 prefill、所有 draft 与 target 前向、接受/拒绝逻辑
  和 cache 回滚；不含模型加载、分词和反分词。
- **公平比较：** 投机解码与 AR 使用相同 GPU、相同进程设置和相同 prompts。AR 基线测量时只加载 target。
- **回放检查：** 阶段 1 和 3 的计时运行会在计时外回放，检查 logits 有限，且必须复现相同输出。
- **BF16 数值：** BF16 只有 8 位尾数，一次验证 γ+1 个 token 的舍入可能与单 token 解码不同，从而
  翻转近似平局的 argmax。计时运行**不**做数值回放。每个与 AR 输出不同的结果都会被记录，并给出
  第一个不同位置上 AR top-2 logit 的差距（以 BF16 ULP 计）。差距小说明是舍入效应；差距大则意味着
  bug。`greedy_tie_fallback: true` 启用精确回放模式用于正确性研究，该模式不计时。
- **干净的 GPU：** 每个阶段前的预检要求只有一张可见 GPU、至少 20 GiB 空闲、没有其他计算进程。

### 3.6 编译模式：静态 KV cache + CUDA Graph

batch size 1 时，eager PyTorch 循环每个 decoder 层要从 Python 逐个下发大约 40–60 个小 CUDA kernel。
每个 kernel 花 20–40 µs 主机时间，而其 GPU 计算常常不到 10 µs。于是一次前向的耗时约为
（层数 × 主机开销），而不是读权重的时间。28 层的 draft 和 36 层的 target 代价之比只有 28/36，
留给投机解码的空间很小。

`runtime.compile: true`（[hf_backend.py](src/acceleratellm/models/hf_backend.py)）做了如下改变：

- **静态 KV cache。** 每个模型拥有一个预分配、地址固定的 `transformers.StaticCache`（容量为
  `runtime.static_cache_len`，默认 1024）。adapter 自己记录有效长度，并显式传入
  `cache_position`/`position_ids`。回滚只是把长度调低：失效的槽位会被后续写入覆盖，并被因果
  mask 屏蔽。开始新请求会使旧的 cache 句柄失效，使用失效句柄会报错。
- **CUDA Graph。** 新 token 数不超过 16 的前向（解码 = 1，draft 追赶 = 2，验证 = γ+1）走
  `torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)`。每个 query 长度录制一个
  CUDA Graph，回放只需一次启动。按 transformers 的建议，prefill 始终 eager 运行。下一次回放前会
  拷贝 graph 的输出。
- **注意力后端。** 使用 SDPA 而非 FlashAttention 2，因为 transformers 中的 FA2 路径会按数据相关的
  形状对 attention mask 做 unpad，CUDA Graph 无法捕获。在这些上下文长度下，注意力占用的时间可以忽略。
- **验证。** 正确性由 CPU 测试检查：静态 cache 在前向、裁剪、AR 和投机解码以及接受计数上都必须与
  动态 cache 一致；另有 GPU 上的 `compile_probe`。`results.json` 记录了有多少次前向走了编译路径、
  多少次走了 eager 路径。
- **显存释放。** Dynamo 的代码缓存和 CUDA Graph 内存池会持有已编译模型权重的引用。因此在阶段之间，
  `benchmark.py` 会先调用 `torch._dynamo.reset()` 并重置 CUDA Graph 树，再释放显存，避免已删除的
  8B 模型仍驻留在显存中。每个阶段会重新编译，约需 30 秒。

---

## 4. 输出

`outputs/<timestamp>-<id>/`（被 git 忽略）：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 完整报告：模型信息、AR 剖析、代价表、接受率、理论值、配对和完整 MT-Bench 计时、分歧记录、源码哈希 |
| `config.yaml`、`environment.json` | 合并后的配置；Python/包/CUDA/GPU 信息 |
| `ar-draft.json`、`ar-target.json` | 逐请求的 AR 计时 |
| `mt-bench-ar.json`、`trajectories.json` | target 在 MT-Bench 上的 AR 运行；贪心轨迹 |
| `acceptance_per_prompt.json` | 每条 prompt、每个 γ 的模拟计数 |
| `paired-gamma{γ}.json`、`mt-bench-spec-gamma{γ}.json` | 逐请求的投机解码计时和计数 |
| `status.json`、`progress.json` | 运行状态；失败时的部分报告 |

`results/<gpu>-<eager|compiled>/`（由 git 跟踪）包含 `scripts/summarize.py` 生成的 `summary.md`、
`results.json` 和 `environment.json`。

---

## 5. 结果

在一张 **NVIDIA RTX A5000**（24 GB，768 GB/s）上测量，torch 2.8.0+cu129，transformers 4.57.1：
BF16，batch 1，贪心，最多 128 个新 token，80 条 MT-Bench 第一轮问题。完整表格见
[results/rtx-a5000-eager/summary.md](results/rtx-a5000-eager/summary.md) 和
[results/rtx-a5000-compiled/summary.md](results/rtx-a5000-compiled/summary.md)。

**总结。** eager 模式下投机解码只能打平（最好 1.02×），因为 draft 受下发开销限制，耗时达到 target
一步的 75%。CUDA Graph 把 draft 单步降低 8×、target 单步降低 2×。相对快了 2× 的编译 AR 基线，
投机解码在**完整 MT-Bench 上达到 1.83×**（γ = 4），相对 eager AR 端到端约 **3.65×**。

### 5.1 每个解码步的前向代价

| | Eager + FA2 + DynamicCache | 编译（StaticCache + CUDA Graph）+ SDPA |
| --- | ---: | ---: |
| Draft 0.6B `T(1)` | 40.71 ms（1.45 ms/层） | **5.00 ms**（0.18 ms/层） |
| Target 8B `T(1)` | 54.65 ms（1.52 ms/层） | **27.27 ms**（0.76 ms/层） |
| Target `T(5)/T(1)`（验证 4 个 draft） | 0.99 | 1.02 |
| 代价比 c = T_d/T_t | 0.745（上限 1.34×） | **0.183**（上限 5.45×） |
| Target AR，MT-Bench | 18.30 tok/s | **36.17 tok/s** |
| 峰值显存（两个模型） | 16.42 GiB | 16.66 GiB |

eager 模式下，两个模型每层耗时都约 1.5 ms，与宽度无关，说明前向受主机下发限制，c ≈ 层数比。
编译后，target 在 27.3 ms 内读完 16.4 GB 权重，约为 A5000 峰值带宽的 78%，已变为带宽受限。
两种模式下，验证 γ+1 个 token 的代价都与解码 1 个几乎相同。

### 5.2 接受率

接受率只取决于模型，因此两种模式下几乎相同，细微差别来自 BF16 近似平局。

| γ | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 前缀接受率 | 0.80 | 0.72 | 0.65 | 0.59 | 0.49 | 0.41 |
| E[L] 每次验证产出 token 数（编译） | 1.79 | 2.42 | 2.92 | 3.32 | 3.89 | 4.22 |

逐 token 一致率 α = 0.81，随类别变化：数学 0.92、信息抽取 0.90、编程 0.87，写作、角色扮演和
推理约 0.73–0.75。实测 E[L] 与 teacher-forced 模拟相差不超过 ±0.08。

### 5.3 投机解码 vs AR（完整 MT-Bench，80 条 prompt）

| γ | Eager：tok/s | Eager：加速比 | 编译：预测 | 编译：tok/s | 编译：加速比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 18.30 | 1.00× | – | 36.17 | 1.00× |
| 1 | 18.53 | **1.018×** | 1.48× | 52.54 | 1.447× |
| 2 | 17.61 | 0.963× | 1.72× | 62.01 | 1.707× |
| 4 | 14.91 | 0.814× | 1.85× | 66.87 | **1.833×** |

代价模型（第 3.3 节）对编译模式加速比的预测误差在 1–2% 以内。配对子集（11 条 prompt × 3 次重复）
结果一致：γ = 1、2、4 时分别为 1.47×、1.73×、1.92×。对编译模式，理论最优为 γ = 4（请求级 1.85×，
纯解码 1.91×），γ = 3–5 都与之相差不到 2%。

**输出一致性。** 编译模式 80 条中有 37–38 条（eager 为 32 条）与 AR 逐 token 完全一致。每一处分歧
都始于近似平局：AR top-2 logit 差距 ≤ 2 BF16 ULP，单 token 和 (γ+1)-token kernel 的舍入不同。
这些分歧来自 BF16 运算而非算法；float32 的 CPU 测试要求完全一致。

**CUDA Graph 确实被使用的证据。** 编译运行中，计时阶段 98.5% 的 draft 和 target 前向走了编译路径
（82,044 次中的 80,799 次），其余 eager 前向是 prefill。

### 5.4 结论

1. 使用原生 eager Hugging Face 循环时，这对模型上的投机解码无法加速，尽管 α ≈ 0.8。瓶颈不是接受率，
   而是 draft 的 CPU 下发开销（c = 0.75）。
2. 用静态 cache 和 CUDA Graph 消除下发开销后，AR 基线本身快了 2×；同时 0.6B draft 相对 8B target
   变得很便宜（c = 0.18），投机解码在此之上再带来 1.83×。
3. 最佳配置：编译模式、γ = 4，66.9 tok/s；编译 AR 为 36.2 tok/s，eager AR 为 18.3 tok/s。

---

### 5.5 仅权重量化 × 投机解码（草稿）

后续实验：`scripts/quant_spec.py`（每个进程跑一组 target/draft，编译模式，同样 80 条 MT-Bench）和
`scripts/gptq_quantize.py`（target 的 GPTQ W4 g128 checkpoint，128 条 RedPajama-V2 校准序列，用 gptqmodel
的 Marlin kernel 推理）。完整表格见 [results/rtx-a5000-quant-spec/summary.md](results/rtx-a5000-quant-spec/summary.md)。

| Target（draft：BF16 0.6B） | 困惑度（BF16 9.44） | AR tok/s | AR 相对 BF16 | 最佳投机 tok/s（γ） | 相对自身 AR | 相对 BF16 AR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 9.44 | 36.4 | 1.00× | 66.8 (5) | 1.82× | 1.83× |
| int8 仅权重（torchao） | 9.48 | 46.7 | 1.28× | 40.5 (5) | 0.86× | 1.11× |
| int4 RTN g128，lm_head 保留 BF16（torchao） | 10.90 | 56.3 | 1.55× | 98.1 (2) | 1.71× | 2.69× |
| **GPTQ W4 g128 + Marlin，lm_head 保留 BF16** | **9.72** | **74.6** | **2.05×** | **100.1 (2)** | 1.34× | **2.75×** |

- **两种方法可以叠加。** 所有 int4 target 在量化 AR 之上，投机解码还能再加速 1.3–1.9×。质量和速度兼顾的最佳点是
  GPTQ-Marlin + 投机解码：相对 BF16 AR 2.75×，困惑度只上升 3%。
- **target 越快，额外收益越小。** 0.6B draft 受 kernel 数量限制，每步始终约 5 ms，所以 c 从 0.19 升到 0.39，
  最佳 γ 从 4–5 降到 2。
- **验证代价必须保持平坦。** Marlin 的 T(5)/T(1) 是 1.05。torchao int8 仅权重量化只在单个 query token 时有融合
  kernel（T(5)/T(1) = 3.16），所以投机解码在所有 γ 下都比 AR 慢。
- **不要量化 draft。** int8 的 0.6B draft 在接受率相同的情况下更慢（8.2 vs 5 ms）。用 8B 自身的 int4 副本作
  draft，接受率达到 0.85，但加速只有 1.13–1.17×。
- **质量。** 把未共享权重的 lm_head 也量化成 int4，会让困惑度从 10.90 升到 12.90。HQQ 没有帮助。GPTQ 校准挽回了
  大部分损失（9.72）。

## 6. 常见问题

| 现象 | 解决办法 |
| --- | --- |
| `Other compute processes on the GPU` | 用 `CUDA_VISIBLE_DEVICES` 选一张空闲 GPU。若仍要运行（计时不可靠）：`ACCELERATELLM_ALLOW_SHARED_GPU=1` |
| `Less than 20 GiB free on the GPU` | 使用 ≥ 24 GB 且没有加载其他内容的 GPU |
| `Expose exactly one CUDA GPU` | 把 `CUDA_VISIBLE_DEVICES` 设为单个编号 |
| `runtime.compile ... requires ... sdpa` | 编译模式只支持 SDPA 注意力；使用 `configs/experiments/speculative_compiled.yaml` |
| `Input exceeds runtime.static_cache_len` | 调大 `runtime.static_cache_len`（prompt + 新 token + γ + 1 必须放得下） |
| 编译模式下的 `torch._dynamo` / Inductor 报错 | 确认 torch 为 2.8.0、transformers 为 4.57.1；设置 `TORCH_LOGS=recompiles,graph_breaks` 查看细节。eager 模式不受影响 |
| `flash_attn` 的 `ImportError` / undefined symbol | wheel 与 python/torch/CUDA/ABI 不匹配；重新安装匹配的 wheel（Step 2） |
| `No module named einops`（FA2） | `pip install einops==0.8.1` |
| `Could not import module 'Qwen3ForCausalLM'` | 运行 `python -c "import transformers.models.qwen3.modeling_qwen3"` 查看真实错误。常见原因：装了 transformers 5.x（运行 `pip install -r requirements.txt`），或 torchvision/torchaudio 与 torch 不匹配（`pip uninstall -y torchvision torchaudio`） |
| `... is not a supported wheel on this platform`（flash-attn） | wheel 的 `cpXY` 标签必须匹配你的 Python（如 Python 3.13 用 `cp313`），`torchX.Y` / `cxx11abi` 必须匹配 torch |
| `OSError ... local_files_only` | 先完成 Step 3，或下载时取消 `HF_HUB_OFFLINE` |
| `MT-Bench ... missing` / `differs from the pinned` | 运行 `python scripts/fetch_data.py`（会校验 SHA256） |
| Slurm 作业找不到 `outputs/` | `sbatch` 前先 `mkdir -p outputs` |

---

## 7. 仓库结构

~~~text
configs/
  base.yaml                 运行时（BF16、FA2、cuda）、生成（贪心、128 token）、计时重复次数
  models.yaml               固定的 draft/target revision
  experiments/baseline.yaml       AR 基线
  experiments/speculative.yaml    投机解码设置；也是 benchmark 配置（eager）
  experiments/speculative_compiled.yaml   同一实验，静态 KV cache + torch.compile（CUDA Graph）
src/acceleratellm/
  config.py                 YAML 继承 + 严格校验
  models/                   HF 后端（BF16、FA2/SDPA、动态或静态+编译 KV cache）、tokenizer 检查
  decoding/                 AR、投机解码（参考实现 + 设备端贪心）、采样、近似平局回放
  evaluation/               计时工具、精确接受率模拟器、加速比代价模型
  experiments/              CLI 运行器（baseline、speculative）+ 结果/环境写出
  data/mt_bench.py          固定版本的 MT-Bench 加载器和 prompt 格式
scripts/
  fetch_models.py, fetch_data.py   下载（固定版本、校验）
  benchmark.py              完整的 5 阶段 benchmark
  run_benchmark.sh          bash/sbatch 包装：检查 + 测试 + benchmark + 汇总
  summarize.py              results.json -> results/<gpu>-<mode>/summary.md
tests/                      使用玩具模型的 CPU 测试，含端到端 benchmark 演练
~~~

CLI：`python -m acceleratellm {doctor, validate, check-models, baseline, speculative} --config <yaml>`。
`speculative` 在配置的 prompts 上做一个小规模的 AR 与投机解码配对比较，并包含贪心和采样检查。

项目提案（[nlp_project_proposal.pdf](nlp_project_proposal.pdf)）还涵盖了权重量化（GPTQ/QAT）。
当前范围是基于 Qwen3 的 BF16 AR 与投机解码对比。

## 参考文献

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023.
- Dao. *FlashAttention-2.* ICLR 2024.
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
- Qwen Team. *Qwen3 Technical Report.* 2025.
