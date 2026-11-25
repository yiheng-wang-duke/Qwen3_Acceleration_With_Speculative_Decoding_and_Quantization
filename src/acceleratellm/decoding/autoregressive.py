"""Batch-one decoding with prompt prefill followed by cached one-token forwards."""
import torch

from acceleratellm.models.interface import ModelAdapter
from acceleratellm.schemas import GenerationResult
from .sampling import probabilities


class AutoregressiveDecoder:
    def __init__(self, validate_logits: bool = True):
        self.validate_logits = validate_logits

    @torch.inference_mode()
    def generate(self, target: ModelAdapter, prompt: torch.Tensor, settings: dict,
                 eos_token_ids: set[int], draft: ModelAdapter | None = None) -> GenerationResult:
        if draft is not None:
            raise ValueError("Autoregressive decoding does not use a draft model")
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError("Expected a nonempty, unpadded [1, length] prompt")
        budget = settings["max_new_tokens"]
        if type(budget) is not int or budget < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        strategy = settings["strategy"]
        if strategy not in {"greedy", "sampling"}:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")
        if prompt.shape[1] + budget > target.context_length:
            raise ValueError("Prompt plus token budget exceeds context length")
        if budget == 0:
            return GenerationResult([])
        result = target.forward(prompt)
        generated = []
        stop_reason = "length"
        for index in range(budget):
            if self.validate_logits and not bool(torch.isfinite(result.logits).all()):
                raise FloatingPointError("Non-finite target logits; verify the attention backend, dtype and weights")
            logits = result.logits[:, -1, :]
            if strategy == "greedy":
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = probabilities(logits, settings["temperature"], settings["top_p"])
                next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "eos"
                break
            if index + 1 < budget:
                result = target.forward(next_token, result.cache)
        return GenerationResult(generated, stop_reason=stop_reason)
