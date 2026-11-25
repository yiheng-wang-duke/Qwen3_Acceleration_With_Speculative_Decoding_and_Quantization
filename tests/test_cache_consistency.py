import pytest
import torch

from acceleratellm.models.hf_backend import HFModelAdapter


def test_cache_rollback_matches_fresh_forward(tiny_model):
    adapter = HFModelAdapter(tiny_model)
    prompt = torch.tensor([[4, 5, 6]])
    prefix = adapter.forward(prompt)
    rejected = adapter.forward(torch.tensor([[10, 11, 12]]), prefix.cache)
    assert adapter.cache_length(rejected.cache) == 6
    cropped = adapter.crop_cache(rejected.cache, 3)
    replay = adapter.forward(torch.tensor([[7, 8]]), cropped)
    fresh = adapter.forward(torch.tensor([[4, 5, 6, 7, 8]]))
    torch.testing.assert_close(replay.logits, fresh.logits[:, -2:, :], rtol=1e-4, atol=1e-5)
    assert adapter.cache_length(replay.cache) == 5
    with pytest.raises(ValueError, match="bounds"):
        adapter.crop_cache(replay.cache, 6)


def test_cuda_request_never_silently_uses_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        HFModelAdapter.load({"model_id": "unused", "revision": "unused"},
                            {"device": "cuda", "dtype": "bfloat16"})
