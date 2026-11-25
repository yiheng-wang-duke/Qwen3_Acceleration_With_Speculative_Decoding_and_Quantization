"""Exact toy autoregressive models with inspectable token caches."""
import torch
from acceleratellm.schemas import ForwardResult


class TableModel:
    def __init__(self, transitions=None, choose=None, vocab_size=4):
        self.device = torch.device("cpu")
        self.context_length = 128
        self.choose = choose
        if transitions is not None:
            values = torch.as_tensor(transitions, dtype=torch.float32)
            self.logits = torch.where(values > 0, values.log(), torch.full_like(values, -1000))
            self.vocab_size = values.shape[-1]
        else:
            self.logits = None
            self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, tokens, cache=None):
        prefix = list(cache) if cache is not None else []
        outputs = []
        for token in tokens[0].tolist():
            prefix.append(token)
            if self.choose is None:
                outputs.append(self.logits[token])
            else:
                logits = torch.full((self.vocab_size,), -1000.)
                logits[self.choose(prefix)] = 0.
                outputs.append(logits)
        self.calls += 1
        return ForwardResult(torch.stack(outputs).unsqueeze(0), prefix)

    @staticmethod
    def cache_length(cache):
        return len(cache) if cache is not None else 0

    @staticmethod
    def crop_cache(cache, length):
        if not 0 <= length <= len(cache):
            raise ValueError("Invalid cache crop")
        return cache[:length]
