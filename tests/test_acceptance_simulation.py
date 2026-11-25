import copy

import pytest
import torch
from transformers import Qwen3ForCausalLM

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.evaluation.acceptance import (
    aggregate, match_lengths, simulate_greedy_rounds, teacher_forced_choices)
from acceleratellm.models.hf_backend import HFModelAdapter
from toy_models import TableModel

KEYS = ("proposed_tokens", "accepted_tokens", "verification_steps")


def _settings(budget):
    return {"strategy": "greedy", "max_new_tokens": budget, "temperature": 1., "top_p": 1.}


def test_match_lengths_counts_consecutive_agreement():
    assert match_lengths([1, 2, 9, 4, 5], [1, 2, 3, 4, 5]) == [2, 1, 0, 2, 1]
    with pytest.raises(ValueError):
        match_lengths([1], [1, 2])


@pytest.mark.parametrize("gamma", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("budget", [1, 2, 5, 17, 40])
@pytest.mark.parametrize("eos", [set(), {2}])
@pytest.mark.parametrize("prompt", [[[0]], [[1, 2, 0]], [[3, 3, 1, 0]]])
@pytest.mark.parametrize("skew", [0, 1, 2])
def test_simulation_reproduces_real_greedy_counters(gamma, budget, eos, prompt, skew):
    choose = lambda p: (sum((i + 1) * x for i, x in enumerate(p)) + len(p)) % 4
    draft_choose = lambda p: (choose(p) + (len(p) % (3 + skew) == 0)) % 4
    target, draft = TableModel(choose=choose), TableModel(choose=draft_choose)
    ids = torch.tensor(prompt)
    trajectory = AutoregressiveDecoder().generate(target, ids, _settings(budget), eos).token_ids
    choices = teacher_forced_choices(draft, ids, trajectory)
    predicted = simulate_greedy_rounds(choices, trajectory, gamma, budget, eos)
    for impl in ("device", "reference"):
        real = SpeculativeDecoder(gamma, greedy_implementation=impl).generate(
            target, ids, _settings(budget), eos, draft)
        assert real.token_ids == trajectory
        assert predicted["accepted_tokens"] == real.accepted_tokens
        assert predicted["verification_steps"] == real.verification_steps
        if impl == "device":
            # The reference loop stops drafting at a proposed EOS; device drafts all.
            assert predicted["proposed_tokens"] == real.proposed_tokens


@pytest.mark.parametrize("case", ["accepted_eos", "bonus_eos", "rejected_eos", "all_accepted"])
def test_simulation_eos_cases(case):
    target_choose = draft_choose = lambda p: 1
    if case == "accepted_eos":
        target_choose = draft_choose = lambda p: 2 if len(p) > 3 else 1
    elif case == "bonus_eos":
        target_choose = lambda p: 1 if len(p) < 5 else 2
    elif case == "rejected_eos":
        draft_choose = lambda p: 2
    target, draft = TableModel(choose=target_choose), TableModel(choose=draft_choose)
    ids = torch.tensor([[0]])
    trajectory = AutoregressiveDecoder().generate(target, ids, _settings(13), {2}).token_ids
    for gamma in (1, 2, 4, 8):
        real = SpeculativeDecoder(gamma, greedy_implementation="device").generate(
            target, ids, _settings(13), {2}, draft)
        predicted = simulate_greedy_rounds(teacher_forced_choices(draft, ids, trajectory),
                                           trajectory, gamma, 13, {2})
        assert all(predicted[k] == getattr(real, k) for k in KEYS)


@pytest.mark.parametrize("gamma", [1, 2, 4])
def test_simulation_matches_real_transformer_cache(tiny_model, gamma):
    torch.manual_seed(53)
    draft_model = Qwen3ForCausalLM(copy.deepcopy(tiny_model.config)).eval()
    # Make the draft agree often: perturb a copy of the target instead of random init.
    draft_model.load_state_dict(tiny_model.state_dict())
    with torch.no_grad():
        for p in draft_model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    target, draft = HFModelAdapter(tiny_model.double()), HFModelAdapter(draft_model.double())
    ids = torch.tensor([[4, 8, 6]])
    trajectory = AutoregressiveDecoder().generate(target, ids, _settings(30), set()).token_ids
    predicted = simulate_greedy_rounds(teacher_forced_choices(draft, ids, trajectory),
                                       trajectory, gamma, 30, set())
    real = SpeculativeDecoder(gamma, greedy_implementation="device").generate(
        target, ids, _settings(30), set(), draft)
    assert real.token_ids == trajectory
    assert all(predicted[k] == getattr(real, k) for k in KEYS)
    assert 0 < predicted["accepted_tokens"]


def test_simulation_rejects_inconsistent_trajectories():
    with pytest.raises(ValueError):
        simulate_greedy_rounds([1, 1], [1, 1], 2, 5, set())  # no EOS but short
    with pytest.raises(ValueError):
        simulate_greedy_rounds([2, 1], [2, 1], 2, 5, {2})  # continues past EOS


def test_aggregate_ratios():
    total = aggregate([{"proposed_tokens": 4, "accepted_tokens": 2, "verification_steps": 2, "generated_tokens": 4},
                       {"proposed_tokens": 6, "accepted_tokens": 3, "verification_steps": 3, "generated_tokens": 6}])
    assert total["token_acceptance_rate"] == .5
    assert total["mean_output_tokens_per_verification"] == 2
