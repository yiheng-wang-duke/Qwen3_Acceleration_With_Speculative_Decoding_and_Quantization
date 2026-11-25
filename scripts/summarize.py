#!/usr/bin/env python3
"""Render a finished benchmark run as Markdown and copy it to results/.

  python scripts/summarize.py --latest                 # newest outputs/*/results.json
  python scripts/summarize.py outputs/<run>/results.json
"""
import argparse
import json
import shutil
from pathlib import Path


def latest() -> Path:
    # Run directories start with a UTC timestamp; skip results written by other tools.
    for path in sorted(Path('outputs').glob('*/results.json'), reverse=True):
        if 'mt_bench_full' in json.loads(path.read_text()):
            return path
    raise FileNotFoundError('No finished benchmark run (outputs/*/results.json); run scripts/run_benchmark.sh')


def render(r: dict, env: dict) -> str:
    gpu = r['gpu_observations'][0]['gpu_name'] if r['gpu_observations'] else 'unknown GPU'
    pk = env.get('packages', {})
    d, t = r['models']['draft'], r['models']['target']
    out = [f"# Benchmark results: {d['model']['model_id']} -> {t['model']['model_id']}", '',
           f"GPU: {gpu}; torch {pk.get('torch')}, transformers {pk.get('transformers')}, flash-attn {pk.get('flash-attn')}; "
           f"dtype {r['config']['runtime']['dtype']}, attention {r['config']['runtime']['attn_implementation']}, "
           f"batch 1, greedy, max {r['config']['generation']['max_new_tokens']} new tokens. Run: `{r['run_dir']}`.", '',
           '## 1. AR baseline and forward-cost profile', '',
           '| Model | Layers | Params | T(1) ms | ms/layer | T(3)/T(1) | T(5)/T(1) | T(9)/T(1) | AR tok/s (3 prompts) | Decode-only tok/s | MT-Bench AR tok/s | Peak GiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for role, m in (('draft', d), ('target', t)):
        f, s = m['facts'], m['standard_prompts']
        q = s['components']['mean_seconds_by_query']
        mt = f"{m['mt_bench']['request_tokens_per_second']:.2f}" if 'mt_bench' in m else '-'
        out.append(f"| {m['model']['model_id'].split('/')[-1]} ({role}) | {f['num_hidden_layers']} | {f['parameters']/1e9:.2f}B | "
                   f"{q['1']*1e3:.2f} | {q['1']*1e3/f['num_hidden_layers']:.3f} | {q['3']/q['1']:.3f} | {q['5']/q['1']:.3f} | "
                   f"{q['9']/q['1']:.3f} | {s['aggregate_tokens_per_second']:.2f} | {s['decode_only_tokens_per_second']:.2f} | "
                   f"{mt} | {m['peak_allocated_bytes']/2**30:.2f} |")
    hf = [a for m in (d, t) for a in m['hf_agreement'] if not a['equal']]
    out += ['', f"AR vs `transformers.generate`: {6 - len(hf)}/6 standard-prompt runs identical"
            + (f"; differences at near ties: {[round(a['divergence']['gap_in_ulps'], 1) for a in hf]} BF16 ULP" if hf else '') + '.']
    th, acc = r['theory'], r['acceptance']
    c = th['1']['draft_target_cost_ratio']
    out += ['', '## 2. Acceptance (80 MT-Bench prompts) and predicted speedup', '',
            f"Per-token greedy agreement alpha = {acc['per_token_agreement_mt_bench']:.3f}; "
            f"cost ratio c = T_d(1)/T_t(1) = {c:.3f}; ceiling 1/c = {1/c:.2f}x.", '',
            '| gamma | Prefix acceptance | Accepted / verify | E[L] tokens / verify | T_t(gamma+1)/T_t(1) | Predicted decode speedup | Predicted request speedup |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for g, a in acc['mt_bench'].items():
        out.append(f"| {g} | {a['token_acceptance_rate']:.3f} | {a['mean_accepted_tokens_per_verification']:.3f} | "
                   f"{a['mean_output_tokens_per_verification']:.3f} | {th[g]['verify_target_cost_ratio']:.3f} | "
                   f"{th[g]['predicted_decode_speedup']:.3f}x | {th[g]['predicted_mt_bench_request_speedup']:.3f}x |")
    out += ['', '| Category | alpha | E[L] gamma=1 | E[L] gamma=2 | E[L] gamma=4 |', '|---|---:|---:|---:|---:|']
    for cat, v in acc['by_category'].items():
        out.append(f"| {cat} | {v['per_token_agreement']:.3f} | " + ' | '.join(
            f"{v[f'gamma{g}']['mean_output_tokens_per_verification']:.3f}" for g in (1, 2, 4)) + ' |')
    p = r['paired']
    out += ['', f"## 3. Paired timing ({len(p['prompts'])} prompts, {p['benchmark']['warmup']} warm-up + {p['benchmark']['repeats']} repeats)", '',
            f"AR: {p['autoregressive']['aggregate_tokens_per_second']:.2f} tok/s, {p['autoregressive']['mean_latency_seconds']:.3f} s/request. "
            f"Peak memory with both models: {p['peak_allocated_bytes_both_models']/2**30:.2f} GiB.", '',
            '| gamma | Spec tok/s | Latency speedup | E[L] measured | E[L] predicted | Greedy-identical prompts | Max divergence gap (BF16 ULP) |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    for g, s in p['speculative'].items():
        gaps = [x['gap_in_ulps'] for x in s['divergences']]
        out.append(f"| {g} | {s['aggregate_tokens_per_second']:.2f} | {s['latency_speedup_vs_ar']:.3f}x | "
                   f"{s['mean_output_tokens_per_verification']:.3f} | {s['predicted_counts_same_prompts']['mean_output_tokens_per_verification']:.3f} | "
                   f"{s['greedy_agreement_prompts']}/{len(p['prompts'])} | {max(gaps) if gaps else 0:.1f} |")
    f4 = r['mt_bench_full']
    out += ['', '## 4. Full MT-Bench timing (80 prompts, one pass each)', '',
            f"AR: {f4['autoregressive_tokens_per_second']:.2f} tok/s, {f4['autoregressive_total_seconds']:.1f} s total.", '',
            '| gamma | Spec tok/s | Request speedup | Prefix acceptance | E[L] | Greedy-identical prompts | Max divergence gap (BF16 ULP) |',
            '|---:|---:|---:|---:|---:|---:|---:|']
    best = None
    for g, s in f4['speculative'].items():
        gaps = [x['gap_in_ulps'] for x in s['divergences']]
        best = max(best or (0, g), (s['request_speedup_vs_ar'], g))
        out.append(f"| {g} | {s['request_tokens_per_second']:.2f} | {s['request_speedup_vs_ar']:.3f}x | "
                   f"{s['token_acceptance_rate']:.3f} | {s['mean_output_tokens_per_verification']:.3f} | "
                   f"{s['greedy_agreement_prompts']}/80 | {max(gaps) if gaps else 0:.1f} |")
    verdict = 'faster than' if best[0] > 1 else 'not faster than'
    out += ['', f"**Verdict:** best measured MT-Bench speedup {best[0]:.3f}x at gamma={best[1]}: speculative decoding is "
            f"{verdict} the BF16 AR baseline in this configuration.", '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path)
    parser.add_argument('--latest', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    path = latest() if args.latest or args.results is None else args.results
    r = json.loads(path.read_text())
    env_path = path.parent / 'environment.json'
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    text = render(r, env)
    print(text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.md').write_text(text)
    shutil.copyfile(path, args.output_dir / 'results.json')
    if env_path.exists():
        shutil.copyfile(env_path, args.output_dir / 'environment.json')
    print(f'\nwrote {args.output_dir}/summary.md, results.json, environment.json')


if __name__ == '__main__':
    main()
