"""Shape-dependent FP16 ties must not change the ordinary AR output."""
import pytest
import torch
from acceleratellm.schemas import ForwardResult
from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from toy_models import TableModel


class ShapeSensitiveTarget(TableModel):
    def forward(self, tokens, cache=None):
        result = super().forward(tokens, cache)
        logits = torch.full_like(result.logits, -1000., dtype=torch.float16)
        for i, row in enumerate(result.logits[0]):
            winner = int(row.argmax())
            alternate = (winner + 1) % self.vocab_size
            logits[0, i, winner] = 16.
            logits[0, i, alternate] = 15.984375
            if cache is not None and tokens.shape[1] > 1:
                logits[0, i, alternate] = 16.015625
        return ForwardResult(logits, result.cache)


@pytest.mark.parametrize('gamma', [1, 2, 4, 8])
@pytest.mark.parametrize('budget', [1, 2, 9])
@pytest.mark.parametrize('eos', [set(), {1}, {2}])
def test_replay_restores_ar_across_shape_dependent_ties_eos_budget(gamma, budget, eos):
    choose = lambda p: (sum(p) + len(p)) % 3
    target = ShapeSensitiveTarget(choose=choose)
    draft = TableModel(choose=choose)
    ids = torch.tensor([[0, 2]])
    settings = {'strategy':'greedy', 'max_new_tokens':budget, 'temperature':1., 'top_p':1.}
    expected = AutoregressiveDecoder().generate(target, ids, settings, eos)
    result = SpeculativeDecoder(gamma, greedy_implementation='device', greedy_tie_fallback=True).generate(
        target, ids, settings, eos, draft)
    assert result.token_ids == expected.token_ids
    assert result.stop_reason == expected.stop_reason
    assert result.numerical_fallback_steps > 0
    assert 0 <= result.accepted_tokens <= result.proposed_tokens


def test_control_without_replay_exposes_shape_disagreement():
    target = ShapeSensitiveTarget(choose=lambda p: 1)
    draft = TableModel(choose=lambda p: 1)
    ids = torch.tensor([[0, 2]])
    settings = {'strategy':'greedy', 'max_new_tokens':3, 'temperature':1., 'top_p':1.}
    expected = AutoregressiveDecoder().generate(target, ids, settings, set())
    result = SpeculativeDecoder(2, greedy_implementation='device').generate(target, ids, settings, set(), draft)
    assert result.token_ids != expected.token_ids


def test_confident_logits_do_not_trigger_replay():
    target = TableModel(choose=lambda p: 1)
    draft = TableModel(choose=lambda p: 2)
    settings = {'strategy':'greedy', 'max_new_tokens':9, 'temperature':1., 'top_p':1.}
    result = SpeculativeDecoder(4, greedy_implementation='device', greedy_tie_fallback=True).generate(
        target, torch.tensor([[0]]), settings, set(), draft)
    assert result.token_ids == [1] * 9
    assert result.numerical_fallback_steps == 0
