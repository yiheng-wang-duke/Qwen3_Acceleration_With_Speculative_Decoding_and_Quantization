from copy import deepcopy
from pathlib import Path

import pytest

from acceleratellm.config import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_inheritance_merges_base_models_and_speculative():
    config = load_config(ROOT / "configs/experiments/speculative.yaml")
    assert config["runtime"]["dtype"] == "bfloat16"
    assert config["generation"]["max_new_tokens"] == 128
    assert config["models"]["draft"]["model_id"] == "Qwen/Qwen3-0.6B-Base"
    assert config["speculative"]["draft_lengths"] == [1, 2, 4]
    assert "speculative" not in load_config(ROOT / "configs/experiments/baseline.yaml")


@pytest.mark.parametrize("section,key,value", [
    ("runtime", "device", "automatic"),
    ("runtime", "dtype", "int4"),
    ("runtime", "attn_implementation", "flash_attention_3"),
    ("generation", "max_new_tokens", -1),
    ("generation", "max_new_tokens", True),
    ("generation", "temperature", float("nan")),
    ("generation", "top_p", 1.1),
    ("benchmark", "repeats", 0),
])
def test_reject_invalid_settings(section, key, value):
    config = deepcopy(load_config(ROOT / "configs/experiments/baseline.yaml"))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_reject_unknown_sections_and_bad_draft_lengths():
    config = deepcopy(load_config(ROOT / "configs/experiments/speculative.yaml"))
    with pytest.raises(ValueError, match="Unknown"):
        validate_config({**config, "quantization": {}})
    config["speculative"]["draft_lengths"] = [2, 2]
    with pytest.raises(ValueError, match="unique"):
        validate_config(config)


def test_inheritance_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: [b.yaml]")
    (tmp_path / "b.yaml").write_text("extends: [a.yaml]")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")
