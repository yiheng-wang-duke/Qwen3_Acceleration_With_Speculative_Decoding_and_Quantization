import json
import pytest
import torch
from acceleratellm.decoding.sampling import probabilities, rejection_distribution
from acceleratellm.decoding.speculative import SpeculativeDecoder
from toy_models import TableModel


@pytest.mark.parametrize("p,q", [
    ([.1, .7, .2], [.8, .1, .1]),
    ([0., .3, .7], [.8, .2, 0.]),
    ([1., 0., 0.], [0., .5, .5]),
])
def test_acceptance_plus_residual_equals_target_exactly(p, q):
    p, q = torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
    accepted_mass = torch.minimum(p, q)
    rejection_mass = 1 - accepted_mass.sum()
    residual = rejection_distribution(p, q)
    torch.testing.assert_close(accepted_mass + rejection_mass * residual, p)


def test_impossible_rejection_is_not_silently_replaced():
    p = torch.tensor([.2, .3, .5])
    with pytest.raises(FloatingPointError, match="correction mass"):
        rejection_distribution(p, p)


@pytest.mark.parametrize("length,temperature,top_p", [(1, 1., 1.), (3, .7, .85)])
def test_conditional_two_token_distribution(length, temperature, top_p, record_property):
    # Check all nine joint output probabilities, including history after rejection.
    p = torch.tensor([[.15, .55, .30], [.45, .10, .45], [.65, .25, .10]])
    q = torch.tensor([[.75, .15, .10], [.10, .75, .15], [.20, .10, .70]])
    target, draft = TableModel(p), TableModel(q)
    decoder = SpeculativeDecoder(length)
    opts = {"strategy": "sampling", "max_new_tokens": 2,
            "temperature": temperature, "top_p": top_p}
    transformed = probabilities(p.log(), temperature, top_p)
    expected = transformed[0, :, None] * transformed
    counts = torch.zeros(3, 3)
    torch.manual_seed(314159)
    trials = 4000
    for _ in range(trials):
        result = decoder.generate(target, torch.tensor([[0]]), opts, set(), draft)
        assert len(result.token_ids) == 2
        counts[tuple(result.token_ids)] += 1
    observed = counts / trials
    tv_distance = 0.5 * (observed - expected).abs().sum()
    record_property("trials", trials)
    record_property("total_variation_distance", tv_distance.item())
    record_property("threshold", .04)
    record_property("observed_joint_probabilities", json.dumps(observed.tolist()))
    record_property("target_joint_probabilities", json.dumps(expected.tolist()))
    assert tv_distance < .04, (tv_distance.item(), observed, expected)


def test_sampling_identical_models_never_reject_and_replays():
    transitions = [[.2, .5, .3], [.6, .1, .3], [.1, .2, .7]]
    target, draft = TableModel(transitions), TableModel(transitions)
    opts = {"strategy": "sampling", "max_new_tokens": 12, "temperature": .8, "top_p": .9}
    torch.manual_seed(81)
    first = SpeculativeDecoder(4).generate(target, torch.tensor([[0]]), opts, set(), draft)
    torch.manual_seed(81)
    second = SpeculativeDecoder(4).generate(target, torch.tensor([[0]]), opts, set(), draft)
    assert first == second
    assert first.accepted_tokens == first.proposed_tokens
