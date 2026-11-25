import pytest
import torch
from transformers import GenerationConfig

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.sampling import probabilities
from acceleratellm.models.hf_backend import HFModelAdapter

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_greedy_matches_transformers(tiny_model, device, attention):
    tiny_model.config._attn_implementation = attention
    dtype = torch.float32
    adapter = HFModelAdapter(tiny_model.to(device=device, dtype=dtype))
    prompt = torch.tensor([[4, 5, 6]], device=device)
    settings = {"strategy": "greedy", "max_new_tokens": 8, "temperature": 1., "top_p": 1.}
    actual = AutoregressiveDecoder().generate(adapter, prompt, settings, {2})
    with torch.inference_mode():
        expected = adapter.model.generate(
            prompt, attention_mask=torch.ones_like(prompt),
            generation_config=GenerationConfig(
                do_sample=False, max_new_tokens=8, eos_token_id=2, pad_token_id=0))
    assert actual.token_ids == expected[0, prompt.shape[1]:].tolist()
    assert adapter.model.training is False


def test_eos_and_zero_budget(tiny_model):
    adapter = HFModelAdapter(tiny_model)
    prompt = torch.tensor([[4, 5]])
    first = adapter.forward(prompt).logits[0, -1].argmax().item()
    settings = {"strategy": "greedy", "max_new_tokens": 8}
    result = AutoregressiveDecoder().generate(adapter, prompt, settings, {first})
    assert result.token_ids == [first]
    assert result.stop_reason == "eos"
    settings["max_new_tokens"] = 0
    assert AutoregressiveDecoder().generate(adapter, prompt, settings, {first}).token_ids == []


def test_context_budget_rejected(tiny_model):
    adapter = HFModelAdapter(tiny_model)
    with pytest.raises(ValueError, match="context"):
        AutoregressiveDecoder().generate(
            adapter, torch.tensor([[4]]), {"strategy": "greedy", "max_new_tokens": 128}, set())


def test_sampling_distribution_and_reproducibility(tiny_model):
    probs = probabilities(torch.tensor([[3., 2., 1.]]), 1., 0.7)
    assert probs[0, 2] == 0
    torch.testing.assert_close(probs.sum(-1), torch.ones(1))
    adapter = HFModelAdapter(tiny_model)
    settings = {"strategy": "sampling", "max_new_tokens": 6, "temperature": .8, "top_p": .9}
    torch.manual_seed(19)
    a = AutoregressiveDecoder().generate(adapter, torch.tensor([[4, 5]]), settings, set())
    torch.manual_seed(19)
    b = AutoregressiveDecoder().generate(adapter, torch.tensor([[4, 5]]), settings, set())
    assert a.token_ids == b.token_ids
    assert len(a.token_ids) == 6


def test_nonfinite_logits_are_rejected(tiny_model, monkeypatch):
    from acceleratellm.schemas import ForwardResult
    adapter = HFModelAdapter(tiny_model)
    monkeypatch.setattr(adapter, "forward", lambda tokens, cache=None: ForwardResult(
        torch.full((1, tokens.shape[1], 32), float("nan")), None))
    with pytest.raises(FloatingPointError, match="Non-finite"):
        AutoregressiveDecoder().generate(adapter, torch.tensor([[4, 5]]),
            {"strategy": "greedy", "max_new_tokens": 2}, set())
