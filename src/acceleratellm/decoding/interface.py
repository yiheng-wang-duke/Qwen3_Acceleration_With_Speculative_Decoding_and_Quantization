from typing import Protocol, Any

from acceleratellm.models.interface import ModelAdapter
from acceleratellm.schemas import GenerationResult


class Decoder(Protocol):
    def generate(self, target: ModelAdapter, prompt: Any, settings: dict,
                 eos_token_ids: set[int], draft: ModelAdapter | None = None) -> GenerationResult: ...
