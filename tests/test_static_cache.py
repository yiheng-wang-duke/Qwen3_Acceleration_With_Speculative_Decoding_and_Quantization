"""Static KV-cache mode (used by torch.compile decoding) must equal the dynamic cache.

CUDA-graph compilation itself is GPU-only and is checked by the benchmark's
compile probe; here the static-cache bookkeeping (explicit positions, length-only
crop, stale-handle protection) is verified on CPU in float32.
"""
import copy
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3ForCausalLM

from acceleratellm.config import load_config, validate_config
from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.models.hf_backend import HFModelAdapter, StaticKV

ROOT = Path(__file__).resolve().parents[1]
GREEDY = {"strategy": "greedy", "max_new_tokens": 20, "temperature": 1., "top_p": 1.}


def _draft_of(model):
    torch.manual_seed(11)
    draft = Qwen3ForCausalLM(copy.deepcopy(model.config)).eval()
    draft.load_state_dict(model.state_dict())
    with torch.no_grad():
        for p in draft.parameters():
            p.add_(0.03 * torch.randn_like(p))
    return draft


def test_static_forward_and_crop_match_dynamic(tiny_model):
    dynamic, static = HFModelAdapter(tiny_model), HFModelAdapter(tiny_model, static_cache_len=32)
    ids = torch.tensor([[4, 9, 6, 12, 7, 5, 21, 8, 3, 30]])
    a, b = dynamic.forward(ids[:, :4]), static.forward(ids[:, :4])
    torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-5)
    assert isinstance(b.cache, StaticKV) and static.cache_length(b.cache) == 4
    for tokens, keep in [(ids[:, 4:7], 5), (ids[:, 5:9], 7), (ids[:, 7:8], 8), (ids[:, 8:10], 10)]:
        a, b = dynamic.forward(tokens, a.cache), static.forward(tokens, b.cache)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-5)
        assert dynamic.cache_length(a.cache) == static.cache_length(b.cache)
        a.cache, b.cache = dynamic.crop_cache(a.cache, keep), static.crop_cache(b.cache, keep)
        assert static.cache_length(b.cache) == keep


@pytest.mark.parametrize("gamma", [1, 2, 4])
@pytest.mark.parametrize("impl", ["device", "reference"])
def test_static_decoding_equals_dynamic(tiny_model, gamma, impl):
    draft_model = _draft_of(tiny_model)
    dyn_t, dyn_d = HFModelAdapter(tiny_model), HFModelAdapter(draft_model)
    st_t, st_d = HFModelAdapter(tiny_model, static_cache_len=64), HFModelAdapter(draft_model, static_cache_len=64)
    ids = torch.tensor([[4, 8, 6]])
    ar_dyn = AutoregressiveDecoder().generate(dyn_t, ids, GREEDY, set())
    assert AutoregressiveDecoder().generate(st_t, ids, GREEDY, set()).token_ids == ar_dyn.token_ids
    decoder = SpeculativeDecoder(gamma, greedy_implementation=impl)
    spec_dyn = decoder.generate(dyn_t, ids, GREEDY, set(), dyn_d)
    spec_st = decoder.generate(st_t, ids, GREEDY, set(), st_d)
    assert spec_st.token_ids == spec_dyn.token_ids == ar_dyn.token_ids
    assert (spec_st.accepted_tokens, spec_st.verification_steps) == (spec_dyn.accepted_tokens, spec_dyn.verification_steps)
    # The shared buffer is reused across requests without leaking state.
    assert decoder.generate(st_t, ids, GREEDY, set(), st_d).token_ids == ar_dyn.token_ids


def test_stale_handle_and_capacity_are_rejected(tiny_model):
    static = HFModelAdapter(tiny_model, static_cache_len=8)
    old = static.forward(torch.tensor([[4, 5]])).cache
    static.forward(torch.tensor([[6]]))  # new request: invalidates `old`
    with pytest.raises(RuntimeError, match="Stale"):
        static.forward(torch.tensor([[7]]), old)
    with pytest.raises(RuntimeError, match="Stale"):
        static.crop_cache(old, 1)
    with pytest.raises(ValueError, match="static_cache_len"):
        static.forward(torch.arange(4, 13).unsqueeze(0))


def test_compile_requires_static_cache_and_sdpa(tiny_model):
    with pytest.raises(ValueError, match="static"):
        HFModelAdapter(tiny_model, compile_mode="reduce-overhead")
    with pytest.raises(ValueError, match="sdpa"):
        HFModelAdapter(tiny_model, static_cache_len=16, compile_mode="reduce-overhead")  # fixture uses eager


def test_compiled_config_is_valid_and_guarded():
    config = load_config(ROOT / "configs/experiments/speculative_compiled.yaml")
    assert config["runtime"]["compile"] is True and config["runtime"]["attn_implementation"] == "sdpa"
    assert load_config(ROOT / "configs/experiments/speculative.yaml")["runtime"]["compile"] is False
    bad = deepcopy(config)
    bad["runtime"]["attn_implementation"] = "flash_attention_2"
    with pytest.raises(ValueError, match="sdpa"):
        validate_config(bad)
    bad = deepcopy(config)
    bad["runtime"]["static_cache_len"] = 4
    with pytest.raises(ValueError):
        validate_config(bad)
