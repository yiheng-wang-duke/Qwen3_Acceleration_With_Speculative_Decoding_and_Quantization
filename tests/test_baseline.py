import json
from pathlib import Path

from acceleratellm.config import load_config
from acceleratellm.experiments.baseline import run_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_real_local_model_baseline_artifacts(tmp_path, tiny_model, tokenizer):
    model_dir = tmp_path / "model"
    tiny_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    config = load_config(ROOT / "configs/experiments/baseline.yaml")
    config["models"] = {
        role: {"model_id": str(model_dir), "revision": "local-fixture"} for role in ("target", "draft")}
    config["runtime"].update(device="cpu", dtype="float32", attn_implementation="sdpa", threads=2)
    config["generation"]["max_new_tokens"] = 4
    config["benchmark"].update(warmup=1, repeats=2)
    config["prompt"]["text"] = "word4 word5 word6"
    config["output_dir"] = str(tmp_path / "outputs")
    run_dir = run_baseline(config)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["greedy_agreement"] is True
    assert metrics["logits_finite"] is True
    assert metrics["device"] == "cpu" and metrics["dtype"] == "float32"
    assert metrics["peak_allocated_bytes"] is None
    assert metrics["repetitions"] == 2
    assert metrics["mean_latency_seconds"] > 0
    assert len((run_dir / "generations.jsonl").read_text().splitlines()) == 2
    assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"
    for name in ("config.yaml", "environment.json", "compatibility.json", "verification.json", "run.log"):
        assert (run_dir / name).is_file()
