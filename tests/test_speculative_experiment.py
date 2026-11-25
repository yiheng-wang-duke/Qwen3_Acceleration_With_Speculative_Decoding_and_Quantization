import json
from pathlib import Path

import pytest

from acceleratellm.config import load_config, validate_config
from acceleratellm.experiments.speculative import run_speculative

ROOT = Path(__file__).resolve().parents[1]


def local_config(tmp_path, tiny_model, tokenizer):
    model_dir = tmp_path / "model"
    tiny_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = load_config(ROOT / "configs/experiments/speculative.yaml")
    config["models"] = {role: {"model_id": str(model_dir), "revision": "fixture"}
                        for role in ("target", "draft")}
    config["runtime"].update(device="cpu", dtype="float32", attn_implementation="sdpa", threads=2)
    config["generation"]["max_new_tokens"] = 6
    config["benchmark"].update(warmup=1, repeats=2)
    config["prompt"]["text"] = "word4 word5"
    config["speculative"] = {"draft_lengths": [2, 4],
                             "validation_prompts": ["word7 word8"],
                             "sampling_seeds": [15]}
    config["output_dir"] = str(tmp_path / "outputs")
    return config


def test_paired_experiment_writes_validated_metrics(tmp_path, tiny_model, tokenizer):
    config = local_config(tmp_path, tiny_model, tokenizer)
    run_dir = run_speculative(config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    verification = json.loads((run_dir / "verification.json").read_text())
    assert metrics["valid_for_benchmark"]
    assert metrics["device"] == "cpu" and metrics["dtype"] == "float32"
    assert metrics["autoregressive"]["requests"] == 4
    assert metrics["autoregressive"]["peak_allocated_bytes"] is None
    for length in ("2", "4"):
        summary = metrics["speculative"][length]
        assert summary["requests"] == 4
        assert summary["logits_finite"] and summary["replay_validated"]
        assert 0 <= summary["token_acceptance_rate"] <= 1
        assert summary["latency_speedup_vs_ar"] > 0
    assert verification["greedy_agreement"]
    assert len(verification["greedy_checks"]) == 4
    assert len(verification["sampling_checks"]) == 2
    assert verification["draft_frozen"]
    assert len((run_dir / "generations.jsonl").read_text().splitlines()) == 12
    assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"


def test_greedy_mismatch_is_a_failed_run(tmp_path, tiny_model, tokenizer, monkeypatch):
    from acceleratellm.decoding.speculative import SpeculativeDecoder
    config = local_config(tmp_path, tiny_model, tokenizer)
    original = SpeculativeDecoder.generate
    def wrong(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.token_ids[0] = (result.token_ids[0] + 1) % 32
        return result
    monkeypatch.setattr(SpeculativeDecoder, "generate", wrong)
    with pytest.raises(RuntimeError, match="greedy mismatch"):
        run_speculative(config)
    run_dir = next(Path(config["output_dir"]).iterdir())
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert (run_dir / "greedy_mismatch.json").is_file()
    assert not (run_dir / "metrics.json").exists()


@pytest.mark.parametrize("key,value", [
    ("draft_lengths", [2, 2]), ("draft_lengths", [0]), ("draft_lengths", [True]),
    ("sampling_seeds", []), ("validation_prompts", [""]),
])
def test_speculative_configuration_validation(key, value):
    config = load_config(ROOT / "configs/experiments/speculative.yaml")
    config["speculative"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)
