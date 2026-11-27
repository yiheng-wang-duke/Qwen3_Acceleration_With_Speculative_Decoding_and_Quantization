#!/usr/bin/env python3
"""Weight-only quantization x speculative decoding (compiled mode), one pair per process.

  python scripts/quant_spec.py --target int4wo-g128 --draft bf16 --gammas 1 2 3 4 5 --out outputs/quant-spec
  python scripts/quant_spec.py --target bf16 --draft self:int4wo-g128 --gammas 2 4 6 --out outputs/quant-spec
  python scripts/quant_spec.py --summarize outputs/quant-spec

Draft: a scheme name applies to Qwen3-0.6B-Base; ``self:<scheme>`` drafts with a quantized
copy of the target model itself (its embedding table is shared with the target).
Each process uses one scheme pair: torchao's set_inductor_config changes global Inductor
settings, so pairs must not share a process.

All timings use the compiled static-cache path of configs/experiments/speculative_compiled.yaml
(BF16 activations, SDPA, CUDA graphs). Per pair, on the same GPU in the same process:
  * resident weight bytes, cached forward cost T(q) for target and draft;
  * target AR over the 80 MT-Bench prompts (its own greedy trajectories, the lossless
    reference for speculative decoding with this target);
  * target quality vs the BF16 target: teacher-forced top-1 agreement / NLL on the BF16
    trajectories, identical outputs, and RedPajama-V2 validation perplexity (128 x 512);
  * exact greedy acceptance for gamma 1..8 (teacher-forced draft) and the cost-model prediction;
  * one timed speculative pass per prompt for each requested gamma, speedup vs this AR.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def bind_idle_slurm_gpu(max_wait=3600, poll=60):
    """Cluster fallback, before CUDA initialises.

    On shared nodes the GPUs Slurm allocates can be busy with processes started outside
    Slurm. A wrapper exporting ACCELERATELLM_GPU_MINOR=auto that found no idle allocated
    GPU leaves CUDA_VISIBLE_DEVICES empty. Then wait for an allocated GPU with no compute
    process and >= 20 GiB free, bind it by UUID, and requeue the job if none frees up.

    athena's gres.conf lists /dev/nvidia[0-7], but device cgroups are not enforced and
    other jobs simply use Slurm's CUDA_VISIBLE_DEVICES=N, i.e. the N-th GPU in PCI order
    (minor numbers differ from that order). The allocated index is the PCI-order index.
    """
    allocation = os.environ.get('SLURM_JOB_GPUS')
    # The old wrapper resolver binds by minor number, i.e. possibly another job's GPU
    # (job 252203 lost all pairs to that): always re-bind here, by PCI-order index.
    if os.environ.get('ACCELERATELLM_GPU_MINOR') != 'auto' or not allocation:
        return
    allocated = set()
    for part in allocation.split(','):
        lo, _, hi = part.partition('-')
        allocated.update(str(i) for i in range(int(lo), int(hi or lo) + 1))
    deadline = time.time() + max_wait
    while True:
        root = ET.fromstring(subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True))
        gpus = sorted(root.findall('gpu'), key=lambda g: g.get('id', ''))  # PCI bus id order
        # Only the PCI-order index: the same-numbered minor is, by convention, another job's GPU.
        for gpu in [g for i, g in enumerate(gpus) if str(i) in allocated]:
            processes = gpu.find('processes')
            free = gpu.findtext('fb_memory_usage/free', '0 MiB').split()[0]
            # Graphics-only contexts (type G, e.g. Xorg holding 4 MiB) do not compete for compute.
            compute = [p for p in processes.findall('process_info') if 'C' in p.findtext('type', 'C')] \
                if processes is not None else [None]
            if not compute and int(free) >= 20 * 1024:
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu.findtext('uuid')
                print(f'BOUND_GPU pci_index={gpus.index(gpu)} minor={gpu.findtext("minor_number")} '
                      f'uuid={gpu.findtext("uuid")}', flush=True)
                return
        if time.time() > deadline:
            print('NO_IDLE_GPU: requeueing job', flush=True)
            subprocess.run(['scontrol', 'requeue', os.environ['SLURM_JOB_ID']], check=False)
            sys.exit(3)
        print(f'WAITING_FOR_IDLE_GPU allocation={allocation}', flush=True)
        time.sleep(poll)


def claim_pair(argv):
    """Several jobs may carry the same pair: skip it if done or held by a live job."""
    def arg(name, default):
        return argv[argv.index(name) + 1] if name in argv else default
    if '--summarize' in argv:
        return
    out = Path(arg('--out', 'outputs/quant-spec'))
    tag = f't-{arg("--target", "bf16")}__d-{arg("--draft", "bf16").replace(":", "-")}'
    if (out / f'{tag}.json').exists():
        print(f'SKIP_DONE {tag}', flush=True)
        sys.exit(0)
    lock = out / f'{tag}.lock'
    job = os.environ.get('SLURM_JOB_ID', '')
    if lock.exists():
        owner = lock.read_text().strip()
        live = subprocess.run(['squeue', '-h', '-j', owner, '-t', 'R'], capture_output=True, text=True).stdout.strip()
        if owner and owner != job and live:
            print(f'SKIP_HELD {tag} by job {owner}', flush=True)
            sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    lock.write_text(job)


if __name__ == '__main__':
    claim_pair(sys.argv)
    bind_idle_slurm_gpu()
    # Share Inductor's compiled kernels and coordinate-descent autotuning results across
    # processes, jobs and nodes (torchao enables per-kernel tuning; it dominates warm-up).
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(Path.home() / '.cache' / 'torchinductor-acceleratellm'))

import torch  # noqa: E402  (after GPU binding)
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402  (shared timing harness, preflight, cleanup)

from acceleratellm.config import load_config  # noqa: E402
from acceleratellm.data.mt_bench import load_mt_bench  # noqa: E402
from acceleratellm.decoding.speculative import SpeculativeDecoder  # noqa: E402
from acceleratellm.evaluation.acceptance import aggregate, simulate_greedy_rounds, teacher_forced_choices  # noqa: E402
from acceleratellm.evaluation.speculative_theory import theory  # noqa: E402
from acceleratellm.experiments.baseline import _eos_ids  # noqa: E402
from acceleratellm.experiments.speculative import _prompt_ids  # noqa: E402
from acceleratellm.models.compatibility import load_tokenizer  # noqa: E402
from acceleratellm.models.hf_backend import HFModelAdapter  # noqa: E402

CONFIG = 'configs/experiments/speculative_compiled.yaml'
REFERENCE = 'outputs/20261126T024321Z-98d277b7/trajectories.json'  # BF16 compiled target, MT-Bench
PPL_DATA = 'data/redpajama_v2/validation.jsonl'  # 128 x 512 Qwen2.5/Qwen3 (shared BPE) token ids
THEORY_GAMMAS = list(range(1, 9))
OUTLIER = 3.0  # re-time requests slower than OUTLIER x the median (recompile inside the clock)
SCHEMES = {
    'bf16': None,
    'int8wo': ('int8', {}),
    'int4wo-g128': ('int4', {'group_size': 128}),
    'int4wo-g64': ('int4', {'group_size': 64}),
    'int4wo-g128-hqq': ('int4', {'group_size': 128, 'use_hqq': True}),
    # '-bf16head': lm_head (untied in Qwen3-8B, 1.2 GB) stays BF16; only decoder linears quantized.
    'int4wo-g128-bf16head': ('int4', {'group_size': 128}),
    'int4wo-g64-bf16head': ('int4', {'group_size': 64}),
    'int4wo-g128-hqq-bf16head': ('int4', {'group_size': 128, 'use_hqq': True}),
    # GPTQ W4 g128 checkpoint (scripts/gptq_quantize.py, lm_head BF16) run with gptqmodel's Marlin kernel.
    'gptq-w4g128': ('gptq', {'path': 'checkpoints/qwen3-8b-base-gptq-w4g128'}),
}


def register_marlin_op():
    """Expose gptqmodel's pybind Marlin GEMM as a torch custom op.

    torch.compile(fullgraph=True) cannot trace an opaque pybind call; as a custom op with
    a shape-only fake implementation it becomes one node, captured by the CUDA graph.
    """
    if hasattr(torch.ops.acceleratellm, 'gptq_marlin_gemm'):
        return
    import gptqmodel_marlin_kernels as kernels
    import gptqmodel.nn_modules.qlinear.marlin as marlin

    @torch.library.custom_op('acceleratellm::gptq_marlin_gemm', mutates_args=('workspace',))
    def gemm(a: torch.Tensor, b_q: torch.Tensor, scales: torch.Tensor, zp: torch.Tensor, g_idx: torch.Tensor,
             perm: torch.Tensor, workspace: torch.Tensor, num_bits: int, size_m: int, size_n: int, size_k: int,
             is_k_full: bool, use_atomic_add: bool, use_fp32_reduce: bool) -> torch.Tensor:
        return kernels.gptq_marlin_gemm(a, b_q, scales, zp, g_idx, perm, workspace, num_bits, size_m, size_n,
                                        size_k, is_k_full, use_atomic_add, use_fp32_reduce)

    @gemm.register_fake
    def _(a, b_q, scales, zp, g_idx, perm, workspace, num_bits, size_m, size_n, size_k, is_k_full,
          use_atomic_add, use_fp32_reduce):
        return a.new_empty((size_m, size_n))

    class Shim:  # only the GEMM is rerouted; repack etc. still come from the real extension
        gptq_marlin_gemm = staticmethod(torch.ops.acceleratellm.gptq_marlin_gemm)

        def __getattr__(self, name):
            return getattr(kernels, name)
    marlin.gptqmodel_marlin_kernels = Shim()


def load_gptq_marlin(path):
    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
    from gptqmodel.utils import BACKEND
    from gptq_quantize import preserve_initializers
    if not (Path(path) / 'artifact.json').exists():
        raise FileNotFoundError(f'{path} missing: run scripts/gptq_quantize.py first')
    register_marlin_op()
    with preserve_initializers():
        wrapper = GPTQModel.load(str(path), backend=BACKEND.MARLIN, device='cuda:0', torch_dtype=torch.bfloat16,
                                 attn_implementation='sdpa', trust_remote_code=False)
    model = wrapper.model
    layers = [m for m in model.modules() if isinstance(m, MarlinQuantLinear)]
    if len(layers) != 7 * model.config.num_hidden_layers:
        raise RuntimeError(f'Expected {7 * model.config.num_hidden_layers} Marlin layers, found {len(layers)}')
    for m in layers:  # forward() would otherwise cast scales (a module-state mutation) inside the graph
        if m.scales.dtype != torch.bfloat16:
            m.scales = m.scales.to(torch.bfloat16)
    return model.eval()


def quantize(model, scheme, device=None):
    if SCHEMES[scheme] is None:
        return
    if SCHEMES[scheme][0] == 'gptq':
        raise ValueError('GPTQ checkpoints are loaded, not quantized in place')
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    kind, kwargs = SCHEMES[scheme]
    config = Int8WeightOnlyConfig(**kwargs) if kind == 'int8' else Int4WeightOnlyConfig(**kwargs)
    keep_head = scheme.endswith('-bf16head')
    quantize_(model, config, device=device, filter_fn=lambda m, name: isinstance(m, torch.nn.Linear)
              and not (keep_head and name == 'lm_head'))


def load(spec, device):
    return AutoModelForCausalLM.from_pretrained(
        spec['model_id'], revision=spec['revision'], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': device}, trust_remote_code=False)


def weight_bytes(model):
    """Bytes of all parameters and buffers, counting tensor-subclass inner tensors."""
    def size(t):
        if hasattr(t, '__tensor_flatten__'):
            names, _ = t.__tensor_flatten__()
            return sum(size(getattr(t, n)) for n in names)
        return t.numel() * t.element_size()
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            total += size(t)
    return total


def linear_types(model):
    kinds = {type(m.weight).__name__ for m in model.modules() if isinstance(m, torch.nn.Linear)}
    kinds |= {type(m).__name__ for m in model.modules() if hasattr(m, 'qweight')}
    return sorted(kinds)


@torch.inference_mode()
def target_quality(adapter, tokenizer, mt_bench, reference):
    """Top-1 agreement and NLL of the BF16 reference trajectories under this model."""
    agree = positions = 0
    nll = 0.
    for p in mt_bench:
        ids = _prompt_ids(tokenizer, p['text'], False, adapter.device)
        ref = reference[p['id']]
        tail = torch.tensor([ref[:-1]], device=ids.device, dtype=ids.dtype)
        logits = adapter.forward(torch.cat((ids, tail), dim=1)).logits[0, ids.shape[1] - 1:].float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Non-finite logits in teacher-forced pass')
        target = torch.tensor(ref, device=logits.device)
        agree += int((logits.argmax(-1) == target).sum())
        positions += len(ref)
        nll += float(torch.nn.functional.cross_entropy(logits, target, reduction='sum'))
    return {'teacher_forced_top1_agreement_vs_bf16': agree / positions,
            'nll_per_token_on_bf16_trajectories': nll / positions, 'positions': positions}


@torch.inference_mode()
def perplexity(model):
    rows = [json.loads(line)['input_ids'] for line in Path(PPL_DATA).read_text().splitlines()]
    device = next(model.parameters()).device
    nll, count = 0., 0
    for ids in rows:
        x = torch.tensor([ids], device=device)
        logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        nll += float(torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction='sum'))
        count += x.shape[1] - 1
    return {'sequences': len(rows), 'tokens': count, 'nll_per_token': nll / count, 'perplexity': math.exp(nll / count)}


def run(target_scheme, draft_name, gammas, out):
    config = load_config(CONFIG)
    runtime, settings = config['runtime'], config['generation']
    tspec, dspec = config['models']['target'], config['models']['draft']
    self_draft = draft_name.startswith('self:')
    draft_scheme = draft_name.removeprefix('self:')
    if target_scheme not in SCHEMES or draft_scheme not in SCHEMES:
        raise ValueError(f'Unknown scheme in {target_scheme} / {draft_name}')
    tag = f't-{target_scheme}__d-{draft_name.replace(":", "-")}'
    reference = json.loads(Path(REFERENCE).read_text())
    tokenizer = load_tokenizer(tspec, True)
    mt_bench = load_mt_bench()
    bm.preflight()
    torch.cuda.reset_peak_memory_stats()
    report = {'tag': tag, 'target_scheme': target_scheme, 'draft': draft_name,
              'target_torchao_config': repr(SCHEMES[target_scheme]),
              'draft_torchao_config': repr(SCHEMES[draft_scheme]),
              'models': {'target': tspec, 'draft': tspec if self_draft else dspec}}

    started = time.perf_counter()
    if SCHEMES[target_scheme] and SCHEMES[target_scheme][0] == 'gptq':
        target_model = load_gptq_marlin(SCHEMES[target_scheme][1]['path'])
    elif SCHEMES[target_scheme] is None:
        target_model = load(tspec, 'cuda')
    else:
        # Quantize layer by layer from a host copy: the GPU only ever holds the quantized
        # weights plus one layer's temporaries (HQQ fragmented ~9 GiB quantizing in place).
        target_model = load(tspec, 'cpu')
        quantize(target_model, target_scheme, device='cuda')
        target_model.to('cuda')
        bm.clean()
    if self_draft:
        # Second copy of the 8B weights: load on the host, share the (unquantized) embedding
        # table with the target, quantize each linear layer straight onto the GPU.
        draft_model = load(tspec, 'cpu')
        draft_model.model.embed_tokens = target_model.model.embed_tokens
        if draft_scheme.endswith('-bf16head') or SCHEMES[draft_scheme] is None:
            draft_model.lm_head = target_model.lm_head  # identical BF16 weights: share them
        else:
            # The 1.2 GB lm_head has the largest quantization temporaries: do it first,
            # while the GPU still has room (it OOMed last, after the decoder layers).
            quantize(draft_model.lm_head, draft_scheme, device='cuda')
            bm.clean()
        quantize(draft_model.model, draft_scheme, device='cuda')
        draft_model.to('cuda')
    else:
        draft_model = load(dspec, 'cuda')
        quantize(draft_model, draft_scheme)
    bm.sync()
    bm.clean()
    report['load_and_quantize_seconds'] = time.perf_counter() - started
    report['resident_allocated_bytes'] = torch.cuda.memory_allocated()
    report['weight_bytes'] = {'target': weight_bytes(target_model), 'draft': weight_bytes(draft_model)}
    report['linear_weight_types'] = {'target': linear_types(target_model), 'draft': linear_types(draft_model)}
    print(f'[{tag}] resident {report["resident_allocated_bytes"] / 2**30:.2f} GiB, weights target '
          f'{report["weight_bytes"]["target"] / 2**30:.2f} / draft {report["weight_bytes"]["draft"] / 2**30:.2f} GiB, '
          f'types {report["linear_weight_types"]}', flush=True)

    # Perplexity before compilation (plain eager forward, no cache).
    report['perplexity'] = {'target': perplexity(target_model), 'draft': perplexity(draft_model)}
    print(f'[{tag}] PPL target {report["perplexity"]["target"]["perplexity"]:.3f} draft '
          f'{report["perplexity"]["draft"]["perplexity"]:.3f}', flush=True)

    target = HFModelAdapter(target_model, static_cache_len=runtime['static_cache_len'], compile_mode='reduce-overhead')
    draft = HFModelAdapter(draft_model, static_cache_len=runtime['static_cache_len'], compile_mode='reduce-overhead')
    eos = _eos_ids(target, tokenizer)
    ids_by = {p['id']: _prompt_ids(tokenizer, p['text'], False, target.device) for p in mt_bench}

    # Cost tables (also compile every query length 1..9 for both models, off the clock).
    started = time.perf_counter()
    bm.split_ar(target, ids_by[mt_bench[0]['id']], settings, eos)
    report['components'] = {'target': bm.component_costs(target, tokenizer),
                            'draft': bm.component_costs(draft, tokenizer)}
    report['warmup_and_cost_seconds'] = time.perf_counter() - started
    tq = report['components']['target']['mean_seconds_by_query']
    dq = report['components']['draft']['mean_seconds_by_query']
    print(f'[{tag}] target T1 {tq["1"] * 1e3:.2f} ms T5/T1 {tq["5"] / tq["1"]:.3f} T9/T1 {tq["9"] / tq["1"]:.3f}; '
          f'draft T1 {dq["1"] * 1e3:.2f} ms; c {dq["1"] / tq["1"]:.3f}', flush=True)

    # Target AR over MT-Bench: this target's own greedy trajectories.
    rows, trajectories = [], {}
    for p in mt_bench:
        row = bm.split_ar(target, ids_by[p['id']], settings, eos)
        ref = reference[p['id']]
        prefix = next((i for i, (a, b) in enumerate(zip(row['token_ids'], ref)) if a != b),
                      min(len(ref), len(row['token_ids'])))
        row.update(prompt_id=p['id'], category=p['category'], identical_to_bf16=row['token_ids'] == ref,
                   common_prefix_with_bf16=prefix)
        rows.append(row)
        trajectories[p['id']] = row['token_ids']
    # A Dynamo recompile inside a timed request (minutes with torchao's autotuning) would
    # dominate the total: re-time such outliers. Greedy decoding is deterministic, so the
    # re-run does identical work; it must reproduce the same tokens.
    median = statistics.median(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in rows)
    for r in rows:
        if r['prefill_to_first_token_seconds'] + r['decode_seconds'] > OUTLIER * median:
            again = bm.split_ar(target, ids_by[r['prompt_id']], settings, eos)
            if again['token_ids'] != r['token_ids']:
                raise RuntimeError(f'AR re-timing of {r["prompt_id"]} changed the output')
            print(f'[{tag}] RETIMED AR {r["prompt_id"]}: '
                  f'{r["prefill_to_first_token_seconds"] + r["decode_seconds"]:.2f} s -> '
                  f'{again["prefill_to_first_token_seconds"] + again["decode_seconds"]:.2f} s', flush=True)
            r.update(retimed_from_seconds=r['prefill_to_first_token_seconds'] + r['decode_seconds'],
                     prefill_to_first_token_seconds=again['prefill_to_first_token_seconds'],
                     decode_seconds=again['decode_seconds'])
    report['ar_retimed_prompts'] = [r['prompt_id'] for r in rows if 'retimed_from_seconds' in r]
    ar_seconds = sum(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in rows)
    tokens = sum(len(r['token_ids']) for r in rows)
    report['mt_bench_ar'] = {
        'generated_tokens': tokens, 'total_seconds': ar_seconds, 'request_tokens_per_second': tokens / ar_seconds,
        'decode_only_tokens_per_second': sum(r['decode_tokens'] for r in rows) / sum(r['decode_seconds'] for r in rows),
        'identical_to_bf16_outputs': sum(r['identical_to_bf16'] for r in rows),
        'mean_common_prefix_with_bf16': statistics.mean(r['common_prefix_with_bf16'] for r in rows)}
    print(f'[{tag}] AR {report["mt_bench_ar"]["request_tokens_per_second"]:.2f} tok/s '
          f'(decode {report["mt_bench_ar"]["decode_only_tokens_per_second"]:.2f}); identical to BF16 '
          f'{report["mt_bench_ar"]["identical_to_bf16_outputs"]}/80', flush=True)

    report['quality_vs_bf16'] = target_quality(target, tokenizer, mt_bench, reference)

    # Exact greedy acceptance against this target's trajectories, and the cost-model prediction.
    budget = settings['max_new_tokens']
    per_prompt, agree, positions = {}, 0, 0
    for p in mt_bench:
        traj = trajectories[p['id']]
        choices = teacher_forced_choices(draft, ids_by[p['id']], traj)
        agree += sum(a == b for a, b in zip(choices, traj))
        positions += len(traj)
        per_prompt[p['id']] = {str(g): simulate_greedy_rounds(choices, traj, g, budget, eos) for g in THEORY_GAMMAS}
    acceptance = {str(g): aggregate([per_prompt[i][str(g)] for i in per_prompt]) for g in THEORY_GAMMAS}
    th = theory(acceptance, {'mean_seconds_by_query': dq}, {'mean_seconds_by_query': tq})
    prefill = sum(r['prefill_to_first_token_seconds'] for r in rows) + len(rows) * dq['9']
    for g in THEORY_GAMMAS:
        th[str(g)]['predicted_mt_bench_request_speedup'] = ar_seconds / (
            prefill + acceptance[str(g)]['verification_steps'] * (g * dq['1'] + tq[str(g + 1)]))
    report['acceptance'] = {'per_token_agreement': agree / positions, 'mt_bench': acceptance}
    report['theory'] = th
    best = max(THEORY_GAMMAS, key=lambda g: th[str(g)]['predicted_mt_bench_request_speedup'])
    print(f'[{tag}] alpha {agree / positions:.3f}; predicted best gamma {best}: request '
          f'{th[str(best)]["predicted_mt_bench_request_speedup"]:.3f}x', flush=True)

    # Timed speculative decoding, one pass per prompt per gamma, vs the AR above.
    spec = {}
    for g in gammas:
        decoder = SpeculativeDecoder(g, validate_logits=False, greedy_implementation='device', greedy_tie_fallback=False)
        decoder.generate(target, ids_by[mt_bench[0]['id']], settings, eos, draft)  # warm-up
        srows, divergences = [], []
        for p in mt_bench:
            bm.sync()
            started = time.perf_counter()
            res = decoder.generate(target, ids_by[p['id']], settings, eos, draft)
            bm.sync()
            elapsed = time.perf_counter() - started
            srows.append({'prompt_id': p['id'], 'category': p['category'], 'seconds': elapsed,
                          'generated_tokens': len(res.token_ids), 'proposed_tokens': res.proposed_tokens,
                          'accepted_tokens': res.accepted_tokens, 'verification_steps': res.verification_steps,
                          'token_ids': res.token_ids})
            if res.token_ids != trajectories[p['id']]:
                divergences.append({'prompt_id': p['id'],
                                    **bm.divergence(target, ids_by[p['id']], trajectories[p['id']], res.token_ids)})
        median = statistics.median(r['seconds'] for r in srows)
        for r in srows:
            if r['seconds'] > OUTLIER * median:
                bm.sync()
                started = time.perf_counter()
                res = decoder.generate(target, ids_by[r['prompt_id']], settings, eos, draft)
                bm.sync()
                elapsed = time.perf_counter() - started
                if res.token_ids != r['token_ids']:
                    raise RuntimeError(f'Speculative re-timing of {r["prompt_id"]} changed the output')
                print(f'[{tag}] RETIMED SPEC gamma={g} {r["prompt_id"]}: {r["seconds"]:.2f} s -> {elapsed:.2f} s', flush=True)
                r.update(retimed_from_seconds=r['seconds'], seconds=elapsed)
        seconds = sum(r['seconds'] for r in srows)
        tok = sum(r['generated_tokens'] for r in srows)
        counts = aggregate(srows)
        spec[str(g)] = {'request_tokens_per_second': tok / seconds, 'total_seconds': seconds,
                        'request_speedup_vs_same_target_ar': ar_seconds / seconds, **counts,
                        'predicted_request_speedup': th[str(g)]['predicted_mt_bench_request_speedup'],
                        'retimed_prompts': [r['prompt_id'] for r in srows if 'retimed_from_seconds' in r],
                        'identical_to_same_target_ar': 80 - len(divergences),
                        'identical_to_bf16_outputs': sum(r['token_ids'] == reference[r['prompt_id']] for r in srows),
                        'max_divergence_gap_ulps': max((d['gap_in_ulps'] for d in divergences), default=None),
                        'divergences': divergences}
        (out / f'{tag}-spec-gamma{g}.json').parent.mkdir(parents=True, exist_ok=True)
        (out / f'{tag}-spec-gamma{g}.json').write_text(json.dumps(srows))
        print(f'[{tag}] SPEC gamma={g}: {tok / seconds:.2f} tok/s, {ar_seconds / seconds:.3f}x vs this AR '
              f'(pred {th[str(g)]["predicted_mt_bench_request_speedup"]:.3f}x), E[L] '
              f'{counts["mean_output_tokens_per_verification"]:.3f}, identical {80 - len(divergences)}/80', flush=True)
    report['speculative'] = spec
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['call_counts'] = {'target': dict(target.call_counts), 'draft': dict(draft.call_counts)}
    report['gpu'] = bm.GPU_OBSERVATIONS
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{tag}.json').write_text(json.dumps(report, indent=2))
    (out / f'{tag}-mt-bench-ar.json').write_text(json.dumps(rows))
    print(f'[{tag}] DONE peak {report["peak_allocated_bytes"] / 2**30:.2f} GiB', flush=True)


def summarize(out):
    reports = sorted((json.loads(p.read_text()) for p in out.glob('t-*.json') if '-spec-' not in p.name
                      and '-mt-bench-ar' not in p.name), key=lambda r: r['tag'])
    base = next((r for r in reports if r['target_scheme'] == 'bf16'), None)
    bf16_ar = base['mt_bench_ar']['request_tokens_per_second'] if base else float('nan')
    lines = [f'BF16 AR reference: {bf16_ar:.2f} tok/s (bf16 target in this directory)', '',
             '| Target | Draft | Target GiB | Draft GiB | T_t(1) ms | T_t(5)/T_t(1) | T_d(1) ms | c | PPL target | '
             'Top-1 vs BF16 | AR tok/s | AR vs BF16 AR | alpha | Best gamma | Spec tok/s | Spec vs same AR | '
             'Spec vs BF16 AR | E[L] | Identical to own AR |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        tq = r['components']['target']['mean_seconds_by_query']
        dq = r['components']['draft']['mean_seconds_by_query']
        ar = r['mt_bench_ar']['request_tokens_per_second']
        spec = r.get('speculative', {})
        g, s = max(spec.items(), key=lambda kv: kv[1]['request_tokens_per_second']) if spec else ('-', None)
        cells = [r['target_scheme'], r['draft'], f"{r['weight_bytes']['target'] / 2**30:.2f}",
                 f"{r['weight_bytes']['draft'] / 2**30:.2f}", f"{tq['1'] * 1e3:.2f}", f"{tq['5'] / tq['1']:.3f}",
                 f"{dq['1'] * 1e3:.2f}", f"{dq['1'] / tq['1']:.3f}", f"{r['perplexity']['target']['perplexity']:.3f}",
                 f"{r['quality_vs_bf16']['teacher_forced_top1_agreement_vs_bf16']:.4f}", f'{ar:.2f}',
                 f'{ar / bf16_ar:.2f}x', f"{r['acceptance']['per_token_agreement']:.3f}", str(g)]
        if s:
            cells += [f"{s['request_tokens_per_second']:.2f}", f"{s['request_speedup_vs_same_target_ar']:.3f}x",
                      f"{s['request_tokens_per_second'] / bf16_ar:.2f}x", f"{s['mean_output_tokens_per_verification']:.3f}",
                      f"{s['identical_to_same_target_ar']}/80"]
        else:
            cells += ['-'] * 5
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '## Per-gamma speculative results', '',
              '| Target | Draft | gamma | tok/s | vs same AR | predicted | E[L] | acceptance | identical to own AR | max gap (ULP) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        for g, s in r.get('speculative', {}).items():
            gap = s['max_divergence_gap_ulps']
            lines.append(f"| {r['target_scheme']} | {r['draft']} | {g} | {s['request_tokens_per_second']:.2f} | "
                         f"{s['request_speedup_vs_same_target_ar']:.3f}x | {s['predicted_request_speedup']:.3f}x | "
                         f"{s['mean_output_tokens_per_verification']:.3f} | {s['token_acceptance_rate']:.3f} | "
                         f"{s['identical_to_same_target_ar']}/80 | {'-' if gap is None else f'{gap:.1f}'} |")
    text = '\n'.join(lines)
    print(text)
    (out / 'summary.md').write_text(text + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target', default='bf16')
    parser.add_argument('--draft', default='bf16')
    parser.add_argument('--gammas', type=int, nargs='*', default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', type=Path, default=Path('outputs/quant-spec'))
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        torch.set_num_threads(4)
        torch._logging.set_logs(recompiles=True)  # explain any recompile in the job log
        run(args.target, args.draft, args.gammas, args.out)


if __name__ == '__main__':
    main()
