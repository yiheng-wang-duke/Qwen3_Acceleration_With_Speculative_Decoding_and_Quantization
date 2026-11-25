import copy
import pytest
import torch
from transformers import Qwen3ForCausalLM

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.models.hf_backend import HFModelAdapter
from toy_models import TableModel


def settings(budget=12, strategy="greedy"):
    return {"max_new_tokens": budget, "strategy": strategy, "temperature": 1., "top_p": 1.}


def test_all_accepted_bonus_and_fewer_target_calls():
    choose = lambda prefix: (prefix[-1] + 1) % 4
    target, draft = TableModel(choose=choose), TableModel(choose=choose)
    result = SpeculativeDecoder(2).generate(target, torch.tensor([[0]]), settings(7), set(), draft)
    assert result.token_ids == [1, 2, 3, 0, 1, 2, 3]
    assert result.accepted_tokens == result.proposed_tokens == 5
    assert result.verification_steps == target.calls == 3


@pytest.mark.parametrize("length", [1, 2, 4, 8])
@pytest.mark.parametrize("prompt", [[[0]], [[1, 2, 0]]])
def test_first_and_middle_rejection_restore_history(length, prompt):
    def target_choice(prefix):
        return (sum((i + 1) * t for i, t in enumerate(prefix)) + len(prefix)) % 4
    def draft_choice(prefix):
        answer = target_choice(prefix)
        return (answer + 1) % 4 if len(prefix) % 2 else answer
    target, draft = TableModel(choose=target_choice), TableModel(choose=draft_choice)
    ids = torch.tensor(prompt)
    expected = AutoregressiveDecoder().generate(target, ids, settings(15), set())
    target.calls = 0
    result = SpeculativeDecoder(length).generate(target, ids, settings(15), set(), draft)
    assert result.token_ids == expected.token_ids
    assert result.accepted_tokens < result.proposed_tokens
    assert target.calls == result.verification_steps + (ids.shape[1] > 1)


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 5])
def test_budget_is_never_exceeded(budget):
    target = TableModel(choose=lambda prefix: 1)
    draft = TableModel(choose=lambda prefix: 1)
    result = SpeculativeDecoder(8).generate(target, torch.tensor([[0]]), settings(budget), set(), draft)
    assert result.token_ids == [1] * budget
    if budget == 0:
        assert target.calls == draft.calls == result.verification_steps == 0


@pytest.mark.parametrize("case", ["accepted", "correction", "bonus", "rejected_draft_eos"])
def test_eos_handling(case):
    if case == "accepted":
        target = TableModel(choose=lambda prefix: 2)
        draft = TableModel(choose=lambda prefix: 2)
        expected = [2]
    elif case == "correction":
        target = TableModel(choose=lambda prefix: 2)
        draft = TableModel(choose=lambda prefix: 1)
        expected = [2]
    elif case == "bonus":
        target = TableModel(choose=lambda prefix: 1 if prefix[-1] == 0 else 2)
        draft = TableModel(choose=lambda prefix: 1)
        expected = [1, 2]
    else:
        target = TableModel(choose=lambda prefix: 1)
        draft = TableModel(choose=lambda prefix: 2)
        expected = [1] * 5
    result = SpeculativeDecoder(1 if case == "bonus" else 4).generate(
        target, torch.tensor([[0]]), settings(5), {2}, draft)
    assert result.token_ids == expected
    assert result.stop_reason == ("length" if case == "rejected_draft_eos" else "eos")


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [2, 4, 8])
def test_real_transformer_greedy_and_cache_positions(tiny_model, device, length):
    torch.manual_seed(53)
    draft_model = Qwen3ForCausalLM(copy.deepcopy(tiny_model.config)).eval()
    dtype = torch.float32
    target = HFModelAdapter(tiny_model.to(device=device, dtype=dtype))
    draft = HFModelAdapter(draft_model.to(device=device, dtype=dtype))
    ids = torch.tensor([[4, 8, 6]], device=device)
    expected = AutoregressiveDecoder().generate(target, ids, settings(17), set())
    actual = SpeculativeDecoder(length).generate(target, ids, settings(17), set(), draft)
    assert actual.token_ids == expected.token_ids
    assert 0 <= actual.accepted_tokens <= actual.proposed_tokens
    assert actual.verification_steps > 0


@pytest.mark.parametrize("role", ["target", "draft"])
def test_nonfinite_logits_fail(role):
    target = TableModel([[.5, .5], [.5, .5]])
    draft = TableModel([[.5, .5], [.5, .5]])
    (target if role == "target" else draft).logits.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match=role):
        SpeculativeDecoder().generate(target, torch.tensor([[0]]), settings(), set(), draft)


def test_invalid_draft_and_context():
    target = TableModel(choose=lambda prefix: 1)
    with pytest.raises(ValueError, match="requires"):
        SpeculativeDecoder().generate(target, torch.tensor([[0]]), settings(), set())
    draft = TableModel(choose=lambda prefix: 1)
    draft.context_length = 3
    with pytest.raises(ValueError, match="context"):
        SpeculativeDecoder().generate(target, torch.tensor([[0]]), settings(3), set(), draft)
    with pytest.raises(ValueError, match="positive"):
        SpeculativeDecoder(0)
