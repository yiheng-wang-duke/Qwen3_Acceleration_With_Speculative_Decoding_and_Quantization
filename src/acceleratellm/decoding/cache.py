"""Explicit cache invariants for single-sequence speculative generation."""
from typing import Any

from acceleratellm.models.interface import ModelAdapter


def require_cache_length(model: ModelAdapter, cache: Any, expected: int, role: str) -> None:
    actual = model.cache_length(cache)
    if actual != expected:
        raise RuntimeError(f"{role} cache has {actual} tokens; expected {expected}")
