"""FP16/BF16/FP32 Hugging Face backend. CPU fallback is always explicit."""
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from acceleratellm.schemas import ForwardResult


class HFModelAdapter:
    def __init__(self, model: Any):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.context_length = model.config.max_position_embeddings
        self.model.requires_grad_(False)

    @classmethod
    def load(cls, spec: dict, runtime: dict) -> "HFModelAdapter":
        device = runtime["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Run with Slurm; no CPU fallback is applied.")
        flash = runtime["attn_implementation"] == "flash_attention_2"
        if flash and (device != "cuda" or runtime["dtype"] not in {"float16", "bfloat16"}):
            raise ValueError("FlashAttention 2 experiments require CUDA and float16/bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], revision=spec["revision"],
            local_files_only=runtime["local_files_only"],
            dtype=getattr(torch, runtime["dtype"]),
            attn_implementation=runtime["attn_implementation"],
            **({"device_map": {"": device}} if flash else {}),
            trust_remote_code=False,
        )
        if model.config._attn_implementation != runtime["attn_implementation"]:
            raise RuntimeError("Loaded attention implementation differs from the requested backend")
        return cls(model.to(device))

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, cache: Any = None) -> ForwardResult:
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Expected a nonempty, unpadded [1, length] token tensor")
        if self.cache_length(cache) + tokens.shape[1] > self.context_length:
            raise ValueError("Input exceeds the model context length")
        result = self.model(
            input_ids=tokens.to(self.device), past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        return ForwardResult(result.logits, result.past_key_values)

    @staticmethod
    def cache_length(cache: Any) -> int:
        if cache is None:
            return 0
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        return int(cache[0][0].shape[-2])

    def crop_cache(self, cache: Any, length: int) -> Any:
        if not 0 <= length <= self.cache_length(cache):
            raise ValueError("Cache crop length is out of bounds")
        if cache is None:
            return None
        if hasattr(cache, "crop"):
            cache.crop(length)
            if self.cache_length(cache) != length:
                raise RuntimeError("Backend could not restore the requested cache length")
            return cache
        return tuple(tuple(tensor[..., :length, :] for tensor in layer) for layer in cache)
