#!/usr/bin/env python3
"""Feasibility probe: weight-only quantization of the target in compiled mode (AR only).

One scheme per process (torchao's set_inductor_config changes global Inductor settings,
so schemes must not share a process):

  python scripts/quant_probe.py --scheme int4wo-g128 --out outputs/quant-probe
  python scripts/quant_probe.py --summarize outputs/quant-probe

For one scheme it measures, on the target from configs/experiments/speculative_compiled.yaml
(static KV cache + torch.compile(reduce-overhead), SDPA, BF16 activations):
  * resident weight memory and peak memory;
  * cached forward cost T(q), q = 1..9 (same harness as benchmark.py phase 1);
  * greedy AR over the 80 MT-Bench prompts (request and decode-only tok/s);
  * quality vs the BF16 target: teacher-forced top-1 agreement and NLL on the BF16
    reference trajectories, and exact / common-prefix match of the generated outputs.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as bm  # noqa: E402  (shared timing harness, preflight, cleanup)

from acceleratellm.config import load_config  # noqa: E402
from acceleratellm.data.mt_bench import load_mt_bench  # noqa: E402
from acceleratellm.experiments.baseline import _eos_ids  # noqa: E402
from acceleratellm.experiments.speculative import _prompt_ids  # noqa: E402
from acceleratellm.models.compatibility import load_tokenizer  # noqa: E402
from acceleratellm.models.hf_backend import HFModelAdapter  # noqa: E402

CONFIG = 'configs/experiments/speculative_compiled.yaml'
REFERENCE = 'outputs/20261126T024321Z-98d277b7/trajectories.json'  # BF16 compiled target, MT-Bench
SCHEMES = {
    'bf16': None,
    'int8wo': ('int8', {}),
    'int4wo-g128': ('int4', {'group_size': 128}),
    'int4wo-g128-hqq': ('int4', {'group_size': 128, 'use_hqq': True}),
    'int4wo-g128-bf16head': ('int4', {'group_size': 128}),  # lm_head kept in BF16
}


def quantize(model, scheme):
    if SCHEMES[scheme] is None:
        return
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    kind, kwargs = SCHEMES[scheme]
    config = Int8WeightOnlyConfig(**kwargs) if kind == 'int8' else Int4WeightOnlyConfig(**kwargs)
    keep_head = scheme.endswith('bf16head')
    quantize_(model, config, filter_fn=lambda m, name: isinstance(m, torch.nn.Linear)
              and not (keep_head and name == 'lm_head'))


@torch.inference_mode()
def teacher_forced(adapter, ids, trajectory):
    """Top-1 choices and NLL of the reference trajectory under this model."""
    tail = torch.tensor([trajectory[:-1]], device=ids.device, dtype=ids.dtype)
    full = torch.cat((ids, tail), dim=1) if len(trajectory) > 1 else ids
    logits = adapter.forward(full).logits[0, ids.shape[1] - 1:].float()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Non-finite logits in teacher-forced pass')
    target = torch.tensor(trajectory, device=logits.device)
    nll = torch.nn.functional.cross_entropy(logits, target, reduction='sum')
    return logits.argmax(-1).tolist(), float(nll)


def run(scheme, out):
    config = load_config(CONFIG)
    spec, runtime = config['models']['target'], config['runtime']
    settings = config['generation']
    reference = json.loads(Path(REFERENCE).read_text())
    tokenizer = load_tokenizer(spec, True)
    mt_bench = load_mt_bench()
    bm.preflight()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        spec['model_id'], revision=spec['revision'], local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, trust_remote_code=False)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    quantize(model, scheme)
    bm.sync()
    quantize_seconds = time.perf_counter() - started
    bm.clean()
    resident = torch.cuda.memory_allocated()
    adapter = HFModelAdapter(model, static_cache_len=runtime['static_cache_len'], compile_mode='reduce-overhead')
    eos = _eos_ids(adapter, tokenizer)
    report = {'scheme': scheme, 'torchao_config': repr(SCHEMES[scheme]), 'model': spec,
              'load_seconds': load_seconds, 'quantize_seconds': quantize_seconds,
              'resident_weight_bytes': resident,
              'linear_weight_types': sorted({type(m.weight).__name__ for m in model.modules()
                                             if isinstance(m, torch.nn.Linear)})}
    print(f'[{scheme}] resident {resident / 2**30:.2f} GiB, quantize {quantize_seconds:.1f} s, '
          f'weights {report["linear_weight_types"]}', flush=True)

    # Warm-up compiles every query length used below (outside all timings).
    started = time.perf_counter()
    bm.split_ar(adapter, _prompt_ids(tokenizer, mt_bench[0]['text'], False, adapter.device), settings, eos)
    report['components'] = bm.component_costs(adapter, tokenizer)
    report['warmup_and_cost_seconds'] = time.perf_counter() - started
    q = report['components']['mean_seconds_by_query']
    print(f'[{scheme}] T1 {q["1"] * 1e3:.2f} ms, T5/T1 {q["5"] / q["1"]:.3f}, T9/T1 {q["9"] / q["1"]:.3f}', flush=True)

    rows = []
    for p in mt_bench:
        ids = _prompt_ids(tokenizer, p['text'], False, adapter.device)
        row = bm.split_ar(adapter, ids, settings, eos)
        ref = reference[p['id']]
        prefix = next((i for i, (a, b) in enumerate(zip(row['token_ids'], ref)) if a != b), min(len(ref), len(row['token_ids'])))
        row.update(prompt_id=p['id'], category=p['category'], identical_to_bf16=row['token_ids'] == ref,
                   common_prefix_with_bf16=prefix)
        rows.append(row)
    total = sum(r['prefill_to_first_token_seconds'] + r['decode_seconds'] for r in rows)
    tokens = sum(len(r['token_ids']) for r in rows)
    report['mt_bench_ar'] = {
        'generated_tokens': tokens, 'total_seconds': total, 'request_tokens_per_second': tokens / total,
        'decode_only_tokens_per_second': sum(r['decode_tokens'] for r in rows) / sum(r['decode_seconds'] for r in rows),
        'identical_to_bf16_outputs': sum(r['identical_to_bf16'] for r in rows),
        'mean_common_prefix_with_bf16': statistics.mean(r['common_prefix_with_bf16'] for r in rows)}

    agree = positions = 0
    nll = 0.
    by_category = {}
    for p in mt_bench:
        ids = _prompt_ids(tokenizer, p['text'], False, adapter.device)
        ref = reference[p['id']]
        choices, prompt_nll = teacher_forced(adapter, ids, ref)
        hits = sum(a == b for a, b in zip(choices, ref))
        agree, positions, nll = agree + hits, positions + len(ref), nll + prompt_nll
        c = by_category.setdefault(p['category'], [0, 0])
        c[0], c[1] = c[0] + hits, c[1] + len(ref)
    report['quality_vs_bf16'] = {
        'teacher_forced_top1_agreement': agree / positions, 'positions': positions,
        'nll_per_token_on_bf16_trajectories': nll / positions,
        'by_category_top1_agreement': {k: v[0] / v[1] for k, v in by_category.items()}}
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['call_counts'] = dict(adapter.call_counts)
    report['gpu'] = bm.GPU_OBSERVATIONS
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{scheme}.json').write_text(json.dumps(report, indent=2))
    (out / f'{scheme}-mt-bench-ar.json').write_text(json.dumps(rows))
    m, qv = report['mt_bench_ar'], report['quality_vs_bf16']
    print(f'[{scheme}] MT-Bench AR {m["request_tokens_per_second"]:.2f} tok/s (decode {m["decode_only_tokens_per_second"]:.2f}); '
          f'top1 agreement {qv["teacher_forced_top1_agreement"]:.4f}, NLL {qv["nll_per_token_on_bf16_trajectories"]:.4f}; '
          f'identical outputs {m["identical_to_bf16_outputs"]}/80', flush=True)


def summarize(out):
    reports = [json.loads((out / f'{s}.json').read_text()) for s in SCHEMES if (out / f'{s}.json').exists()]
    base = next((r for r in reports if r['scheme'] == 'bf16'), None)
    lines = ['| Scheme | Weights GiB | Peak GiB | T(1) ms | T(5)/T(1) | T(9)/T(1) | MT-Bench AR tok/s | vs BF16 | '
             'Top-1 agreement | NLL / token | Identical outputs | Mean common prefix |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        q, m, qv = r['components']['mean_seconds_by_query'], r['mt_bench_ar'], r['quality_vs_bf16']
        speed = m['request_tokens_per_second'] / base['mt_bench_ar']['request_tokens_per_second'] if base else float('nan')
        lines.append(f"| {r['scheme']} | {r['resident_weight_bytes'] / 2**30:.2f} | {r['peak_allocated_bytes'] / 2**30:.2f} | "
                     f"{q['1'] * 1e3:.2f} | {q['5'] / q['1']:.3f} | {q['9'] / q['1']:.3f} | {m['request_tokens_per_second']:.2f} | "
                     f"{speed:.2f}x | {qv['teacher_forced_top1_agreement']:.4f} | {qv['nll_per_token_on_bf16_trajectories']:.4f} | "
                     f"{m['identical_to_bf16_outputs']}/80 | {m['mean_common_prefix_with_bf16']:.1f} |")
    text = '\n'.join(lines)
    print(text)
    (out / 'summary.md').write_text(text + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scheme', choices=list(SCHEMES))
    parser.add_argument('--out', type=Path, default=Path('outputs/quant-probe'))
    parser.add_argument('--summarize', type=Path)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.scheme:
        run(args.scheme, args.out)
    else:
        parser.error('give --scheme or --summarize')


if __name__ == '__main__':
    main()
