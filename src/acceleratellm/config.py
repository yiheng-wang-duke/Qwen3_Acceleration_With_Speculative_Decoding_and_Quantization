"""Composable YAML configuration with strict validation for executable settings."""
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(left: dict, right: dict) -> dict:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    with path.open() as stream:
        current = yaml.safe_load(stream)
    if not isinstance(current, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parents = current.pop("extends", [])
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError("extends must be a list of relative YAML paths")
    combined: dict[str, Any] = {}
    for parent in parents:
        combined = _merge(combined, _read(path.parent / parent, (*stack, path)))
    return _merge(combined, current)


def _integer(value: Any, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: dict) -> None:
    required = {"seed", "output_dir", "runtime", "models", "generation", "benchmark", "prompt"}
    allowed = required | {"speculative"}
    if missing := required - config.keys():
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if unknown := config.keys() - allowed:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    expected = {
        "runtime": {"device", "dtype", "attn_implementation", "local_files_only", "threads",
                    "compile", "static_cache_len"},
        "models": {"target", "draft"},
        "generation": {"strategy", "max_new_tokens", "temperature", "top_p"},
        "benchmark": {"warmup", "repeats"},
        "prompt": {"text", "use_chat_template"},
    }
    for section, keys in expected.items():
        value = config[section]
        if not isinstance(value, dict) or value.keys() != keys:
            raise ValueError(f"{section} must contain exactly {sorted(keys)}")
    _integer(config["seed"], "seed", 0)
    runtime = config["runtime"]
    if runtime["device"] not in {"cuda", "cpu"}:
        raise ValueError("runtime.device must be cuda or cpu")
    if runtime["dtype"] not in {"float16", "bfloat16", "float32"}:
        raise ValueError("runtime.dtype must be float16, bfloat16 or float32")
    if runtime["attn_implementation"] not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError("runtime.attn_implementation must be eager, sdpa or flash_attention_2")
    if type(runtime["local_files_only"]) is not bool:
        raise ValueError("runtime.local_files_only must be boolean")
    _integer(runtime["threads"], "runtime.threads", 1)
    if type(runtime["compile"]) is not bool:
        raise ValueError("runtime.compile must be boolean")
    _integer(runtime["static_cache_len"], "runtime.static_cache_len", 16)
    if runtime["compile"] and (runtime["device"] != "cuda" or runtime["attn_implementation"] != "sdpa"):
        raise ValueError("runtime.compile (static cache + CUDA graphs) requires device=cuda and attn_implementation=sdpa")
    for role in ("target", "draft"):
        model = config["models"][role]
        if not isinstance(model, dict) or model.keys() != {"model_id", "revision"}:
            raise ValueError(f"models.{role} requires model_id and revision")
        if not all(isinstance(v, str) and v.strip() for v in model.values()):
            raise ValueError(f"models.{role} values must be nonempty strings")
    gen = config["generation"]
    if gen["strategy"] not in {"greedy", "sampling"}:
        raise ValueError("generation.strategy must be greedy or sampling")
    _integer(gen["max_new_tokens"], "generation.max_new_tokens", 0)
    for name in ("temperature", "top_p"):
        if type(gen[name]) not in (int, float) or not 0 < gen[name] < float("inf"):
            raise ValueError(f"generation.{name} must be finite and positive")
    if gen["top_p"] > 1:
        raise ValueError("generation.top_p must be <= 1")
    _integer(config["benchmark"]["warmup"], "benchmark.warmup", 0)
    _integer(config["benchmark"]["repeats"], "benchmark.repeats", 1)
    prompt = config["prompt"]
    if not isinstance(prompt["text"], str) or not prompt["text"].strip():
        raise ValueError("prompt.text must be a nonempty string")
    if type(prompt["use_chat_template"]) is not bool:
        raise ValueError("prompt.use_chat_template must be boolean")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise ValueError("output_dir must be a nonempty path string")
    if "speculative" in config:
        spec = config["speculative"]
        required_spec = {"draft_lengths", "validation_prompts", "sampling_seeds"}
        if (not isinstance(spec, dict) or not required_spec <= spec.keys()
                or spec.keys() - required_spec - {"greedy_implementation", "greedy_tie_fallback"}):
            raise ValueError("speculative requires draft_lengths, validation_prompts, sampling_seeds")
        if spec.get("greedy_implementation", "reference") not in {"reference", "device"}:
            raise ValueError("Unknown greedy speculative implementation")
        if type(spec.get("greedy_tie_fallback", False)) is not bool:
            raise ValueError("greedy_tie_fallback must be boolean")
        if spec.get("greedy_tie_fallback", False) and spec.get("greedy_implementation") != "device":
            raise ValueError("greedy_tie_fallback requires device greedy")
        lengths = spec["draft_lengths"]
        if not isinstance(lengths, list) or not lengths:
            raise ValueError("speculative.draft_lengths must be a nonempty list")
        for length in lengths:
            _integer(length, "speculative.draft_lengths", 1)
        if len(lengths) != len(set(lengths)):
            raise ValueError("speculative.draft_lengths must be unique")
        prompts = spec["validation_prompts"]
        if not isinstance(prompts, list) or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("speculative.validation_prompts must contain nonempty strings")
        seeds = spec["sampling_seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("speculative.sampling_seeds must be a nonempty list")
        for seed in seeds:
            _integer(seed, "speculative.sampling_seeds", 0)


def load_config(path: str | Path) -> dict:
    config = _read(Path(path))
    validate_config(config)
    return config
