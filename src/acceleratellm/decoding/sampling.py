"""Shared normalized sampling distribution for future AR/speculative parity."""
import math

import torch


def probabilities(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be finite and positive; top_p must be in (0, 1]")
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1:
        sorted_probs, indices = probs.sort(descending=True, dim=-1)
        remove = sorted_probs.cumsum(-1) - sorted_probs >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0)
        probs = torch.zeros_like(probs).scatter(-1, indices, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def rejection_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """Normalize (p-q)+ at the first rejected proposal. There is no p fallback."""
    residual = (target_probs - draft_probs).clamp_min(0)
    mass = residual.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(mass).all()) or bool((mass <= 0).any()):
        raise FloatingPointError("Rejected proposal has no finite positive correction mass")
    return residual / mass
