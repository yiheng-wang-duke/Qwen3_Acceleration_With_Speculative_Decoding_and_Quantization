"""Plug-in steady-state speculative cost model, with explicit boundary yield.

The observed accepted prefix per round is not the conditional per-token alpha.
Using E[K]+1 avoids assuming IID token acceptance. The costs exclude prefills,
cache catch-up differences and speculative loop orchestration.
"""

def theory(acceptance, draft_cost, target_cost):
    rows = {}
    td, tt = draft_cost['mean_seconds_by_query']['1'], target_cost['mean_seconds_by_query']['1']
    for gamma, stats in acceptance.items():
        g = int(gamma)
        tv = target_cost['mean_seconds_by_query'][str(g + 1)]
        yield_ideal = 1 + stats['accepted_tokens'] / stats['verification_steps']
        yield_bounded = stats['generated_tokens'] / stats['verification_steps']
        cycle = g * td + tv
        rows[gamma] = {
            'draft_token_seconds': td, 'target_token_seconds': tt,
            'target_verify_seconds': tv, 'draft_target_cost_ratio': td / tt,
            'verify_target_cost_ratio': tv / tt,
            'expected_output_per_round': yield_ideal,
            'observed_output_per_round_with_eos_budget': yield_bounded,
            'classic_speedup_assuming_one_target_cost': yield_ideal / (g * td / tt + 1),
            'predicted_decode_tokens_per_second': yield_ideal / cycle,
            'predicted_decode_speedup': yield_ideal * tt / cycle,
            'boundary_adjusted_decode_tokens_per_second': yield_bounded / cycle,
            'boundary_adjusted_decode_speedup': yield_bounded * tt / cycle,
        }
    return rows

