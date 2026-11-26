"""FP16/BF16/FP32 Hugging Face backend. CPU fallback is always explicit.

Two cache modes share one interface (forward / cache_length / crop_cache):

* dynamic (default): transformers DynamicCache, cropped in place.
* static: one preallocated transformers StaticCache per adapter. The adapter tracks
  the valid length itself and passes explicit cache_position / position_ids, because
  StaticCache.get_seq_length() counts non-zero slots. Cropping only lowers that length:
  slots beyond it are overwritten by later writes and are hidden by the causal mask
  (built from cache_position). Optionally the short decode/verification forwards are
  compiled with torch.compile(mode="reduce-overhead"), i.e. replayed as CUDA graphs,
  which removes per-kernel CPU dispatch; prefill always runs eagerly.

The static buffer is shared by all requests of one adapter: starting a new request
(forward with cache=None) invalidates every older handle, and using one raises.
"""
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from acceleratellm.schemas import ForwardResult

COMPILE_MAX_QUERY = 16  # decode (1), draft catch-up (2) and verification (gamma+1) steps


@dataclass
class StaticKV:
    cache: Any
    length: int
    epoch: int


class HFModelAdapter:
    def __init__(self, model: Any, static_cache_len: int | None = None, compile_mode: str | None = None):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.vocab_size = model.get_output_embeddings().weight.shape[0]
        self.context_length = model.config.max_position_embeddings
        self.model.requires_grad_(False)
        if compile_mode is not None and static_cache_len is None:
            raise ValueError("torch.compile decoding requires the static KV cache")
        self.static_cache_len = static_cache_len
        self.compile_mode = compile_mode
        self._static = None
        self._epoch = 0
        self._compiled = None
        self.call_counts = {"compiled": 0, "eager": 0}  # static mode only; evidence for reports
        if compile_mode is not None:
            if model.config._attn_implementation != "sdpa":
                raise ValueError("Compiled static-cache decoding requires attn_implementation=sdpa")
            # One graph per query length (1..COMPILE_MAX_QUERY) and model; keep them all.
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            torch._dynamo.config.accumulated_cache_size_limit = max(
                torch._dynamo.config.accumulated_cache_size_limit, 512)
            self._compiled = torch.compile(self.model.forward, mode=compile_mode, fullgraph=True, dynamic=False)

    @classmethod
    def load(cls, spec: dict, runtime: dict) -> "HFModelAdapter":
        device = runtime["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback is applied.")
        flash = runtime["attn_implementation"] == "flash_attention_2"
        if flash and (device != "cuda" or runtime["dtype"] not in {"float16", "bfloat16"}):
            raise ValueError("FlashAttention 2 experiments require CUDA and float16/bfloat16")
        compile_enabled = bool(runtime.get("compile", False))
        if compile_enabled and (device != "cuda" or runtime["attn_implementation"] != "sdpa"):
            raise ValueError("runtime.compile requires device=cuda and attn_implementation=sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], revision=spec["revision"],
            local_files_only=runtime["local_files_only"],
            dtype=getattr(torch, runtime["dtype"]),
            attn_implementation=runtime["attn_implementation"],
            # Load weights straight to the GPU: avoids a full host-RAM copy (8B = 16 GB).
            **({"device_map": {"": device}} if device == "cuda" else {}),
            trust_remote_code=False,
        )
        if model.config._attn_implementation != runtime["attn_implementation"]:
            raise RuntimeError("Loaded attention implementation differs from the requested backend")
        return cls(model.to(device),
                   static_cache_len=runtime.get("static_cache_len") if compile_enabled else None,
                   compile_mode="reduce-overhead" if compile_enabled else None)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, cache: Any = None) -> ForwardResult:
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("Expected a nonempty, unpadded [1, length] token tensor")
        if self.cache_length(cache) + tokens.shape[1] > self.context_length:
            raise ValueError("Input exceeds the model context length")
        if self.static_cache_len is not None:
            return self._static_forward(tokens.to(self.device), cache)
        result = self.model(
            input_ids=tokens.to(self.device), past_key_values=cache,
            use_cache=True, return_dict=True,
        )
        return ForwardResult(result.logits, result.past_key_values)

    def _static_forward(self, tokens: torch.Tensor, cache: Any) -> ForwardResult:
        if cache is None:
            if self._static is None:
                from transformers import StaticCache
                self._static = StaticCache(config=self.model.config, max_cache_len=self.static_cache_len)
            self._static.reset()  # no-op before lazy allocation; buffer addresses stay fixed
            self._epoch += 1
            cache = StaticKV(self._static, 0, self._epoch)
        elif not isinstance(cache, StaticKV) or cache.epoch != self._epoch:
            raise RuntimeError("Stale static-cache handle: a newer request reused this adapter's buffer")
        start, count = cache.length, tokens.shape[1]
        if start + count > self.static_cache_len:
            raise ValueError("Input exceeds runtime.static_cache_len")
        position = torch.arange(start, start + count, device=self.device)
        kwargs = dict(input_ids=tokens, past_key_values=cache.cache, cache_position=position,
                      position_ids=position.unsqueeze(0), use_cache=True, return_dict=True)
        if self._compiled is not None and start > 0 and count <= COMPILE_MAX_QUERY:
            torch.compiler.cudagraph_mark_step_begin()
            # CUDA-graph outputs are overwritten by the next replay: keep a private copy.
            logits = self._compiled(**kwargs).logits.clone()
            self.call_counts["compiled"] += 1
        else:
            logits = self.model(**kwargs).logits
            self.call_counts["eager"] += 1
        cache.length = start + count
        return ForwardResult(logits, cache)

    @staticmethod
    def cache_length(cache: Any) -> int:
        if cache is None:
            return 0
        if isinstance(cache, StaticKV):
            return cache.length
        if hasattr(cache, "get_seq_length"):
            return int(cache.get_seq_length())
        return int(cache[0][0].shape[-2])

    def crop_cache(self, cache: Any, length: int) -> Any:
        if not 0 <= length <= self.cache_length(cache):
            raise ValueError("Cache crop length is out of bounds")
        if cache is None:
            return None
        if isinstance(cache, StaticKV):
            if cache.epoch != self._epoch:
                raise RuntimeError("Stale static-cache handle")
            cache.length = length
            return cache
        if hasattr(cache, "crop"):
            cache.crop(length)
            if self.cache_length(cache) != length:
                raise RuntimeError("Backend could not restore the requested cache length")
            return cache
        return tuple(tuple(tensor[..., :length, :] for tensor in layer) for layer in cache)
