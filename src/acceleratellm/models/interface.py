"""A forward consumes only tokens not already represented in the cache."""
from typing import Any, Protocol

from acceleratellm.schemas import ForwardResult


class ModelAdapter(Protocol):
    device: Any
    context_length: int
    vocab_size: int

    def forward(self, tokens: Any, cache: Any = None) -> ForwardResult: ...

    def cache_length(self, cache: Any) -> int: ...

    def crop_cache(self, cache: Any, length: int) -> Any:
        """Return a cache representing exactly the first length tokens."""
        ...
