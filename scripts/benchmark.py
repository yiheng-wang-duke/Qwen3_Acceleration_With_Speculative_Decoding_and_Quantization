#!/usr/bin/env python3
"""BF16 + FlashAttention 2 + KV cache benchmark: AR baseline vs speculative decoding.

Draft and target come from configs/experiments/speculative.yaml (Qwen3-0.6B-Base ->
Qwen3-8B-Base). One process, one GPU; every phase runs on the same device.
Run through scripts/run_benchmark.sh (plain bash or sbatch); see README.

 0. Tokenizer/vocabulary checks; FA2 kernel + KV-cache crop probe on a toy Qwen3.
 1. AR profiling per model: agreement with transformers.generate, timed AR on the
    3 standard prompts, decode-only split, cached forward cost for 1..9 query tokens.
    The target also runs one timed greedy pass over the 80 MT-Bench prompts; those
    trajectories are the committed greedy outputs for all later phases.
 2. Acceptance + theory: one teacher-forced draft pass per trajectory gives the exact
    greedy acceptance counters for every gamma in 1..8 (evaluation/acceptance.py);
    plug-in cost model gives the predicted speedup.
 3. Paired timing on a stratified subset (3 standard + 1 MT-Bench prompt per category):
    AR vs speculative for each configured gamma, warm-up + repeats + off-clock replay.
 4. Full MT-Bench timing: one timed speculative pass per prompt per gamma, compared
    with the phase-1 AR pass of the same target on the same prompts.

BF16 keeps 8 mantissa bits: verifying gamma+1 tokens at once and decoding one token
at a time can round differently and flip a near-tied argmax. Timed runs therefore use
no numerical replay; each divergence from the AR output is recorded together with the
AR top-2 logit gap at the first differing position (large gaps would indicate a bug).
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from acceleratellm.config import load_config
from acceleratellm.data.mt_bench import load_mt_bench, stratified_subset
from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.evaluation.acceptance import aggregate, simulate_greedy_rounds, teacher_forced_choices
from acceleratellm.evaluation.generation_benchmark import measure_generation, seed_all, summarize
from acceleratellm.evaluation.speculative_theory import theory
from acceleratellm.experiments.artifacts import create_run, write_json
from acceleratellm.experiments.baseline import _eos_ids
from acceleratellm.experiments.speculative import _prompt_ids, _reference
from acceleratellm.models.compatibility import check_model_pair, load_tokenizer
from acceleratellm.models.hf_backend import HFModelAdapter

THEORY_GAMMAS = list(range(1, 9))
QUERIES = list(range(1, 10))
CONTEXTS = [64, 256, 512]
PAIRED_BENCH = {'warmup': 1, 'repeats': 3}
GPU_OBSERVATIONS = []


def sync():
    torch.cuda.synchronize()


def clean():
    # Compiled mode: dynamo's code cache and the CUDA-graph tree pool keep references to
    # the compiled model's weights; drop them, or a deleted 8B model stays resident.
    torch._dynamo.reset()
    from torch._inductor import cudagraph_trees
    cudagraph_trees.reset_cudagraph_trees()
    gc.collect()
    torch.cuda.empty_cache()


MIN_FREE_GIB = 20  # Qwen3-8B (16.4 GB BF16) + Qwen3-0.6B (1.2 GB) + caches


def preflight():
    """Timing is only meaningful on an otherwise idle GPU: check before every phase.

    Exactly one visible GPU is required (select it with CUDA_VISIBLE_DEVICES).
    Set ACCELERATELLM_ALLOW_SHARED_GPU=1 to proceed despite other compute processes.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0')
    properties = torch.cuda.get_device_properties(0)
    uuid = 'GPU-' + str(properties.uuid).removeprefix('GPU-')
    try:
        apps = subprocess.check_output(['nvidia-smi', '--id=' + uuid, '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True)
        foreign = [line.strip() for line in apps.splitlines() if line.strip() and line.strip() != str(os.getpid())]
    except (OSError, subprocess.CalledProcessError):
        foreign = None  # nvidia-smi unavailable (e.g. some containers): cannot check
    free, total = torch.cuda.mem_get_info()
    observation = {'gpu_uuid': uuid, 'gpu_name': properties.name, 'free_bytes': free,
                   'total_bytes': total, 'other_processes': foreign, 'time_unix': time.time()}
    GPU_OBSERVATIONS.append(observation)
    print('GPU_PREFLIGHT=' + json.dumps(observation), flush=True)
    if free < MIN_FREE_GIB * 2**30:
        raise RuntimeError(f'Less than {MIN_FREE_GIB} GiB free on the GPU')
    if foreign and os.environ.get('ACCELERATELLM_ALLOW_SHARED_GPU') != '1':
        raise RuntimeError(f'Other compute processes on the GPU {foreign}; timings would be invalid')


def bf16_ulp(x):
    """Spacing of BF16 values at x; x may be a Python float or a (CUDA) tensor scalar."""
    x = torch.tensor(float(x), dtype=torch.bfloat16)  # CPU scalar: never mixes devices
    return float(torch.nextafter(x, torch.tensor(float('inf'), dtype=torch.bfloat16)) - x)


@torch.inference_mode()
def split_ar(adapter, prompt, settings, eos):
    """Greedy AR with prefill (incl. first token) and decode timed separately."""
    sync()
    started = time.perf_counter()
    result = adapter.forward(prompt)
    token = result.logits[:, -1, :].argmax(-1, keepdim=True)
    output = [int(token.item())]
    sync()
    prefill = time.perf_counter() - started
    started = time.perf_counter()
    while len(output) < settings['max_new_tokens'] and output[-1] not in eos:
        result = adapter.forward(token, result.cache)
        token = result.logits[:, -1, :].argmax(-1, keepdim=True)
        output.append(int(token.item()))
    sync()
    return {'prefill_to_first_token_seconds': prefill, 'decode_seconds': time.perf_counter() - started,
            'decode_tokens': len(output) - 1, 'token_ids': output}


@torch.inference_mode()
def divergence(adapter, ids, expected, actual):
    """First differing position and the AR top-2 logit gap there (diagnostic only)."""
    if expected == actual:
        return None
    first = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
    current = adapter.forward(ids)
    for token in expected[:first]:  # same one-token path that produced `expected`
        current = adapter.forward(torch.tensor([[token]], device=ids.device), current.cache)
    top = current.logits[0, -1].float().topk(2)
    gap = float(top.values[0] - top.values[1])
    ulp = bf16_ulp(top.values[0])
    return {'position': first, 'expected_token': expected[first] if first < len(expected) else None,
            'actual_token': actual[first] if first < len(actual) else None,
            'ar_top2_ids': top.indices.tolist(), 'ar_top2_gap': gap, 'bf16_ulp_at_top': ulp,
            'gap_in_ulps': gap / ulp, 'numerically_explained': gap <= 8 * ulp}


@torch.inference_mode()
def kernel_probe(dtype):
    seed_all(42)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, attention_dropout=0.)
    cfg._attn_implementation = 'flash_attention_2'
    flash = Qwen3ForCausalLM(cfg).to(dtype=dtype, device='cuda').eval()
    cfg2 = copy.deepcopy(cfg)
    cfg2._attn_implementation = 'sdpa'
    sdpa = Qwen3ForCausalLM(cfg2).to(dtype=dtype, device='cuda').eval()
    sdpa.load_state_dict(flash.state_dict())
    ids = torch.tensor([[3, 8, 12, 19, 5, 20, 7, 6, 21]], device='cuda')
    fa, ref = HFModelAdapter(flash), HFModelAdapter(sdpa)
    a, b = fa.forward(ids[:, :5]), ref.forward(ids[:, :5])
    errors = []
    for tokens, keep in [(ids[:, 5:8], 6), (ids[:, 6:9], 7), (ids[:, 7:8], 8)]:
        a, b = fa.forward(tokens, a.cache), ref.forward(tokens, b.cache)
        if not bool(torch.isfinite(a.logits).all()):
            raise FloatingPointError('Non-finite FA2 probe logits')
        torch.testing.assert_close(a.logits.float(), b.logits.float(), atol=.03, rtol=.05)
        errors.append(float((a.logits.float() - b.logits.float()).abs().max()))
        a.cache, b.cache = fa.crop_cache(a.cache, keep), ref.crop_cache(b.cache, keep)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        fa.forward(ids)
        sync()
    names = [event.key for event in prof.key_averages()]
    kernels = [n for n in names if 'flash' in n.lower()]
    if not kernels or any('scaled_dot_product' in n for n in names):
        raise RuntimeError(f'Expected the external FlashAttention kernel: {kernels}')
    return {'backend': 'flash_attention_2', 'dtype': str(dtype), 'flash_events': kernels,
            'cached_crop_vs_sdpa_max_abs_errors': errors}


@torch.inference_mode()
def compile_probe(dtype):
    """Compiled static-cache decoding (CUDA graphs) vs eager dynamic cache on a toy Qwen3."""
    seed_all(42)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256, attention_dropout=0.)
    cfg._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(cfg).to(dtype=dtype, device='cuda').eval()
    eager = HFModelAdapter(model)
    compiled = HFModelAdapter(model, static_cache_len=64, compile_mode='reduce-overhead')
    ids = torch.tensor([[3, 8, 12, 19, 5, 20, 7, 6, 21, 9, 30, 2]], device='cuda')
    started = time.perf_counter()
    errors = []
    for _ in range(2):  # second pass replays the recorded graphs
        a, b = eager.forward(ids[:, :5]), compiled.forward(ids[:, :5])
        for tokens, keep in [(ids[:, 5:8], 6), (ids[:, 6:8], 8), (ids[:, 8:9], 9), (ids[:, 9:12], 10)]:
            a, b = eager.forward(tokens, a.cache), compiled.forward(tokens, b.cache)
            if not bool(torch.isfinite(b.logits).all()):
                raise FloatingPointError('Non-finite compiled probe logits')
            torch.testing.assert_close(b.logits.float(), a.logits.float(), atol=.05, rtol=.05)
            errors.append(float((a.logits.float() - b.logits.float()).abs().max()))
            a.cache, b.cache = eager.crop_cache(a.cache, keep), compiled.crop_cache(b.cache, keep)
    if compiled.call_counts['compiled'] == 0:
        raise RuntimeError('Compiled decode path was never used')
    return {'mode': 'static cache + torch.compile(reduce-overhead)', 'dtype': str(dtype),
            'max_abs_error_vs_eager_dynamic': max(errors), 'call_counts': compiled.call_counts,
            'seconds_including_compilation': time.perf_counter() - started}


@torch.inference_mode()
def component_costs(adapter, tokenizer):
    source = tokenizer('The following is a detailed explanation of a scientific experiment. ' * 120,
                       return_tensors='pt', add_special_tokens=False).input_ids.to(adapter.device)
    records = []
    for context in CONTEXTS:
        for query in QUERIES:
            cache = adapter.forward(source[:, :context]).cache
            tokens = source[:, context:context + query]
            for repeat in range(23):  # 3 warm-ups, 20 timed
                if repeat == 3:
                    sync()
                    started = time.perf_counter()
                cache = adapter.crop_cache(cache, context)
                result = adapter.forward(tokens, cache)
                result.logits[0].argmax(-1).tolist()
                cache = result.cache
            sync()
            records.append({'context_tokens': context, 'query_tokens': query,
                            'seconds': (time.perf_counter() - started) / 20})
    return {'records': records,
            'mean_seconds_by_query': {str(q): statistics.mean(r['seconds'] for r in records if r['query_tokens'] == q)
                                      for q in QUERIES},
            'scope': 'cached forward + cache crop + argmax of each position + host transfer; 3 warm-ups/20 repeats; contexts 64/256/512'}


def model_facts(adapter):
    cfg = adapter.model.config
    params = list(adapter.model.parameters())
    return {'parameters': sum(p.numel() for p in params),
            'parameter_bytes': sum(p.numel() * p.element_size() for p in params),
            'num_hidden_layers': cfg.num_hidden_layers, 'hidden_size': cfg.hidden_size,
            'vocab_size': adapter.vocab_size, 'dtype': str(params[0].dtype),
            'attn_implementation': cfg._attn_implementation,
            'compile_mode': adapter.compile_mode, 'static_cache_len': adapter.static_cache_len}


def profile_model(role, spec, root, config, standard, mt_bench, tokenizer):
    settings = config['generation']
    preflight()
    torch.cuda.reset_peak_memory_stats()
    adapter = HFModelAdapter.load(spec, config['runtime'])
    eos = _eos_ids(adapter, tokenizer)
    entry = {'model': spec, 'facts': model_facts(adapter), 'eos': sorted(eos), 'hf_agreement': []}
    records, split_records, trajectories = [], [], {}
    for p in standard:
        ids = _prompt_ids(tokenizer, p['text'], False, adapter.device)
        reference = _reference(adapter, ids, settings, tokenizer, eos)
        checked = AutoregressiveDecoder().generate(adapter, ids, settings, eos).token_ids
        # transformers.generate runs eagerly with an attention mask (FA2: varlen kernel);
        # our loop may use other kernels/graphs. BF16 near ties can flip: diagnosed, not asserted.
        entry['hf_agreement'].append({'prompt': p['id'], 'equal': checked == reference,
                                      'divergence': divergence(adapter, ids, checked, reference)})
        trajectories[p['id']] = checked
        rows = measure_generation(AutoregressiveDecoder(), adapter, ids, settings, eos,
                                  config['benchmark'], config['seed'])
        if any(row['token_ids'] != checked for row in rows):
            raise RuntimeError('Timed AR differs from the checked AR path')
        for row in rows:
            row['prompt_id'] = p['id']
        records.extend(rows)
        for repeat in range(4):  # 1 warm-up + 3 timed
            row = split_ar(adapter, ids, settings, eos)
            if row['token_ids'] != checked:
                raise RuntimeError('Split AR differs from the checked AR path')
            if repeat:
                row.update(prompt_id=p['id'], repeat=repeat - 1)
                split_records.append(row)
    summary = summarize(records)
    summary['decode_only_tokens_per_second'] = (sum(r['decode_tokens'] for r in split_records)
                                                / sum(r['decode_seconds'] for r in split_records))
    summary['mean_prefill_to_first_token_seconds'] = statistics.mean(
        r['prefill_to_first_token_seconds'] for r in split_records)
    summary['output_lengths'] = [len(trajectories[p['id']]) for p in standard]
    summary['components'] = component_costs(adapter, tokenizer)
    entry['standard_prompts'] = summary
    if role == 'target':
        mt_rows = []
        split_ar(adapter, _prompt_ids(tokenizer, mt_bench[0]['text'], False, adapter.device), settings, eos)
        for p in mt_bench:
            ids = _prompt_ids(tokenizer, p['text'], False, adapter.device)
            row = split_ar(adapter, ids, settings, eos)
            row.update(prompt_id=p['id'], category=p['category'], prompt_tokens=ids.shape[1])
            mt_rows.append(row)
            trajectories[p['id']] = row['token_ids']
        total = sum(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in mt_rows)
        tokens = sum(len(r['token_ids']) for r in mt_rows)
        entry['mt_bench'] = {
            'prompts': len(mt_rows), 'generated_tokens': tokens,
            'eos_stops': sum(r['token_ids'][-1] in eos for r in mt_rows),
            'total_seconds': total, 'request_tokens_per_second': tokens / total,
            'decode_only_tokens_per_second': sum(r['decode_tokens'] for r in mt_rows) / sum(r['decode_seconds'] for r in mt_rows),
            'mean_prefill_to_first_token_seconds': statistics.mean(r['prefill_to_first_token_seconds'] for r in mt_rows),
            'mean_prompt_tokens': statistics.mean(r['prompt_tokens'] for r in mt_rows),
            'scope': 'one timed greedy pass per prompt after one warm-up request'}
        write_json(root / 'mt-bench-ar.json', mt_rows)
    entry['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    write_json(root / f'ar-{role}.json', {'requests': records, 'split': split_records})
    q = summary['components']['mean_seconds_by_query']
    print(f'AR {role}: std {summary["aggregate_tokens_per_second"]:.2f} tok/s, decode '
          f'{summary["decode_only_tokens_per_second"]:.2f} tok/s, T1 {q["1"]*1e3:.2f} ms'
          + (f', MT-Bench {entry["mt_bench"]["request_tokens_per_second"]:.2f} tok/s' if role == 'target' else ''),
          flush=True)
    entry['call_counts'] = dict(adapter.call_counts)
    del adapter
    clean()
    return entry, trajectories


def acceptance_and_theory(report, root, config, prompts, tokenizer, trajectories):
    budget = config['generation']['max_new_tokens']
    preflight()
    draft = HFModelAdapter.load(config['models']['draft'], config['runtime'])
    eos = set(report['models']['target']['eos'])
    per_prompt, by_category = {}, {}
    for p in prompts:
        ids = _prompt_ids(tokenizer, p['text'], False, draft.device)
        traj = trajectories[p['id']]
        choices = teacher_forced_choices(draft, ids, traj)
        cat = by_category.setdefault(p['category'], {'agree': 0, 'positions': 0, 'ids': []})
        cat['agree'] += sum(a == b for a, b in zip(choices, traj))
        cat['positions'] += len(traj)
        cat['ids'].append(p['id'])
        per_prompt[p['id']] = {str(g): simulate_greedy_rounds(choices, traj, g, budget, eos) for g in THEORY_GAMMAS}
    del draft
    clean()
    mt_ids = [p['id'] for p in prompts if p['category'] != 'standard']
    acceptance = {str(g): aggregate([per_prompt[i][str(g)] for i in mt_ids]) for g in THEORY_GAMMAS}
    dcost = report['models']['draft']['standard_prompts']['components']['mean_seconds_by_query']
    tcost = report['models']['target']['standard_prompts']['components']['mean_seconds_by_query']
    th = theory(acceptance, {'mean_seconds_by_query': dcost}, {'mean_seconds_by_query': tcost})
    # Request-level prediction on MT-Bench: measured AR (prefill + decode) per prompt vs
    # target prefill + draft prefill (approximated by its 9-token forward) + rounds.
    mt_ar = json.loads((root / 'mt-bench-ar.json').read_text())
    ar_seconds = sum(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in mt_ar)
    prefill = sum(r['prefill_to_first_token_seconds'] for r in mt_ar) + len(mt_ar) * dcost['9']
    for g in THEORY_GAMMAS:
        cycle = g * dcost['1'] + tcost[str(g + 1)]
        th[str(g)]['predicted_mt_bench_request_speedup'] = ar_seconds / (
            prefill + acceptance[str(g)]['verification_steps'] * cycle)
    mt_agree = sum(v['agree'] for c, v in by_category.items() if c != 'standard')
    mt_pos = sum(v['positions'] for c, v in by_category.items() if c != 'standard')
    report['acceptance'] = {
        'method': 'teacher-forced draft argmax on the target greedy AR trajectory; exact greedy round simulation',
        'per_token_agreement_mt_bench': mt_agree / mt_pos,
        'mt_bench': acceptance,
        'by_category': {c: {'per_token_agreement': v['agree'] / v['positions'],
                            **{f'gamma{g}': aggregate([per_prompt[i][str(g)] for i in v['ids']]) for g in (1, 2, 4)}}
                        for c, v in by_category.items()}}
    report['theory'] = th
    write_json(root / 'acceptance_per_prompt.json', per_prompt)
    best = max(THEORY_GAMMAS, key=lambda g: th[str(g)]['predicted_decode_speedup'])
    print(f'THEORY: alpha={mt_agree / mt_pos:.3f} c={th["1"]["draft_target_cost_ratio"]:.3f} best gamma={best} '
          f'decode {th[str(best)]["predicted_decode_speedup"]:.3f}x request '
          f'{th[str(best)]["predicted_mt_bench_request_speedup"]:.3f}x', flush=True)
    return per_prompt


def paired_and_full(report, root, config, prompts, subset, tokenizer, trajectories, per_prompt):
    settings = config['generation']
    by_id = {p['id']: p for p in prompts}
    preflight()
    torch.cuda.reset_peak_memory_stats()
    target = HFModelAdapter.load(config['models']['target'], config['runtime'])
    eos = set(report['models']['target']['eos'])
    ids_by = {p['id']: _prompt_ids(tokenizer, p['text'], False, target.device) for p in prompts}
    ar_records = []
    for pid in subset:
        rows = measure_generation(AutoregressiveDecoder(), target, ids_by[pid], settings, eos, PAIRED_BENCH, config['seed'])
        if any(r['token_ids'] != trajectories[pid] for r in rows):
            raise RuntimeError(f'Paired AR differs from the phase-1 trajectory on {pid}')
        for r in rows:
            r['prompt_id'] = pid
        ar_records.extend(rows)
    ar = summarize(ar_records)
    draft = HFModelAdapter.load(config['models']['draft'], config['runtime'])
    paired = {}
    for g in config['speculative']['draft_lengths']:
        decoder = SpeculativeDecoder(g, greedy_implementation='device', greedy_tie_fallback=False)
        records, divergences, predicted = [], [], []
        for pid in subset:
            rows = measure_generation(decoder, target, ids_by[pid], settings, eos, PAIRED_BENCH, config['seed'], draft)
            out = rows[0]['token_ids']
            if any(r['token_ids'] != out for r in rows):
                raise RuntimeError('Speculative output changed across timed repeats')
            if out != trajectories[pid]:
                divergences.append({'prompt_id': pid, **divergence(target, ids_by[pid], trajectories[pid], out)})
            for r in rows:
                r['prompt_id'] = pid
            records.extend(rows)
            predicted.append(per_prompt[pid][str(g)])
        s = summarize(records)
        s['latency_speedup_vs_ar'] = ar['mean_latency_seconds'] / s['mean_latency_seconds']
        s['throughput_speedup_vs_ar'] = s['aggregate_tokens_per_second'] / ar['aggregate_tokens_per_second']
        s['predicted_counts_same_prompts'] = aggregate(predicted)
        s['greedy_agreement_prompts'] = len(subset) - len(divergences)
        s['divergences'] = divergences
        paired[str(g)] = s
        write_json(root / f'paired-gamma{g}.json', records)
        print(f'PAIRED gamma={g}: {s["aggregate_tokens_per_second"]:.2f} vs AR {ar["aggregate_tokens_per_second"]:.2f} tok/s '
              f'-> {s["latency_speedup_vs_ar"]:.3f}x; E[L] {s["mean_output_tokens_per_verification"]:.3f} '
              f'(pred {s["predicted_counts_same_prompts"]["mean_output_tokens_per_verification"]:.3f}); '
              f'agree {s["greedy_agreement_prompts"]}/{len(subset)}', flush=True)
    report['paired'] = {'prompts': subset, 'benchmark': PAIRED_BENCH, 'autoregressive': ar, 'speculative': paired,
                        'peak_allocated_bytes_both_models': torch.cuda.max_memory_allocated(),
                        'timing_scope': 'prefill + draft + verification + rollback; no numerical replay'}
    write_json(root / 'progress.json', report)
    # Phase 4: full MT-Bench, one timed pass per prompt and gamma.
    mt = [p for p in prompts if p['category'] != 'standard']
    mt_ar = {r['prompt_id']: r for r in json.loads((root / 'mt-bench-ar.json').read_text())}
    ar_seconds = sum(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in mt_ar.values())
    full = {}
    for g in config['speculative']['draft_lengths']:
        decoder = SpeculativeDecoder(g, validate_logits=False, greedy_implementation='device', greedy_tie_fallback=False)
        decoder.generate(target, ids_by[mt[0]['id']], settings, eos, draft)  # warm-up
        rows, divergences = [], []
        for p in mt:
            sync()
            started = time.perf_counter()
            res = decoder.generate(target, ids_by[p['id']], settings, eos, draft)
            sync()
            elapsed = time.perf_counter() - started
            rows.append({'prompt_id': p['id'], 'category': p['category'], 'seconds': elapsed,
                         'generated_tokens': len(res.token_ids), 'proposed_tokens': res.proposed_tokens,
                         'accepted_tokens': res.accepted_tokens, 'verification_steps': res.verification_steps,
                         'token_ids': res.token_ids})
            if res.token_ids != trajectories[p['id']]:
                divergences.append({'prompt_id': p['id'],
                                    **divergence(target, ids_by[p['id']], trajectories[p['id']], res.token_ids)})
        spec_seconds = sum(r['seconds'] for r in rows)
        tokens = sum(r['generated_tokens'] for r in rows)
        counts = aggregate(rows)
        full[str(g)] = {'request_tokens_per_second': tokens / spec_seconds, 'total_seconds': spec_seconds,
                        'request_speedup_vs_ar': ar_seconds / spec_seconds, **counts,
                        'greedy_agreement_prompts': len(mt) - len(divergences), 'divergences': divergences}
        write_json(root / f'mt-bench-spec-gamma{g}.json', rows)
        print(f'MT-BENCH gamma={g}: {tokens / spec_seconds:.2f} tok/s, speedup {ar_seconds / spec_seconds:.3f}x, '
              f'E[L] {counts["mean_output_tokens_per_verification"]:.3f}, agree {len(mt) - len(divergences)}/{len(mt)}',
              flush=True)
    report['mt_bench_full'] = {'autoregressive_total_seconds': ar_seconds,
                               'autoregressive_tokens_per_second': report['models']['target']['mt_bench']['request_tokens_per_second'],
                               'speculative': full,
                               'scope': 'one timed pass per prompt after one warm-up request; AR from phase 1 (target only loaded)'}
    report['call_counts_paired_and_full'] = {'target': dict(target.call_counts), 'draft': dict(draft.call_counts)}
    del target, draft
    clean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/speculative.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config['runtime']['threads'])
    seed_all(config['seed'])
    root = create_run(config)
    print(f'BENCHMARK_DIR={root}', flush=True)
    write_json(root / 'status.json', {'status': 'running'})
    standard = [{'id': f'std-{i}', 'category': 'standard', 'text': t}
                for i, t in enumerate([config['prompt']['text'], *config['speculative']['validation_prompts']])]
    mt_bench = load_mt_bench()
    prompts = standard + mt_bench
    subset = [p['id'] for p in standard] + stratified_subset(mt_bench)
    report = {'schema_version': 1, 'run_dir': str(root), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'models': {}, 'config': config, 'gpu_observations': GPU_OBSERVATIONS, 'paired_subset': subset,
              'runtime_mode': ('compiled static cache (CUDA graphs)' if config['runtime']['compile']
                               else 'eager dynamic cache') + f" + {config['runtime']['attn_implementation']}",
              'prompt_format': "MT-Bench first turn as 'Question: {turn}\\nAnswer:'; standard prompts raw; no chat template",
              'theory_formula': 'S(gamma) = E[L] * T_t(1) / (gamma * T_d(1) + T_t(gamma+1)), E[L] = 1 + accepted/verification'}
    try:
        tokenizer = load_tokenizer(config['models']['target'], True)
        report['compatibility'] = check_model_pair(config['models'], True)
        preflight()
        dtype = getattr(torch, config['runtime']['dtype'])
        if config['runtime']['attn_implementation'] == 'flash_attention_2':
            report['kernel_probe'] = kernel_probe(dtype)
        if config['runtime']['compile']:
            report['compile_probe'] = compile_probe(dtype)
            print('COMPILE_PROBE=' + json.dumps(report['compile_probe']), flush=True)
        clean()
        trajectories = {}
        for role in ('draft', 'target'):
            entry, traj = profile_model(role, config['models'][role], root, config, standard, mt_bench, tokenizer)
            report['models'][role] = entry
            if role == 'target':
                trajectories = traj
            write_json(root / 'progress.json', report)
        write_json(root / 'trajectories.json', trajectories)
        per_prompt = acceptance_and_theory(report, root, config, prompts, tokenizer, trajectories)
        write_json(root / 'progress.json', report)
        paired_and_full(report, root, config, prompts, subset, tokenizer, trajectories, per_prompt)
        report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), *sorted(Path('src/acceleratellm').rglob('*.py'))]}
        write_json(root / 'results.json', report)
        write_json(root / 'status.json', {'status': 'complete'})
        print(f'BENCHMARK_COMPLETE={root}', flush=True)
    except Exception as exc:
        write_json(root / 'progress.json', report)
        write_json(root / 'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
