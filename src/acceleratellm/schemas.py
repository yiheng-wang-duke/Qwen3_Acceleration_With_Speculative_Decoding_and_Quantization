"""Shared contracts; algorithms depend on these rather than concrete backends."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ForwardResult:
    # logits: [batch=1, input_length, vocabulary]; cache includes the input.
    logits: Any
    cache: Any


@dataclass
class GenerationResult:
    # Completion tokens only. Cache ownership remains internal to the decoder.
    token_ids: list[int]
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    verification_steps: int = 0
    stop_reason: str = "length"
    numerical_fallback_steps: int = 0

