"""End-to-end CPU dry run of scripts/benchmark.py + scripts/summarize.py on toy Qwen3 models.

GPU-only pieces (preflight, FA2 kernel probe, CUDA memory stats) are stubbed; every
phase, artifact and summary table is exercised so a GPU run cannot fail on plumbing.
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_and_summary_end_to_end(tmp_path, tiny_model, tokenizer, monkeypatch):
    benchmark, summarize = _load("benchmark"), _load("summarize")
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    tiny_model.save_pretrained(target_dir)
    torch.manual_seed(3)
    draft = Qwen3ForCausalLM(copy.deepcopy(tiny_model.config)).eval()
    draft.load_state_dict(tiny_model.state_dict())
    with torch.no_grad():
        for p in draft.parameters():
            p.add_(0.05 * torch.randn_like(p))
    draft.save_pretrained(draft_dir)
    for d in (target_dir, draft_dir):
        tokenizer.save_pretrained(d)

    config = yaml.safe_load((ROOT / "configs/base.yaml").read_text())
    config.update(output_dir=str(tmp_path / "outputs"),
                  models={"target": {"model_id": str(target_dir), "revision": "fixture"},
                          "draft": {"model_id": str(draft_dir), "revision": "fixture"}},
                  speculative={"draft_lengths": [1, 2], "validation_prompts": ["word7 word8", "word9"],
                               "sampling_seeds": [1], "greedy_implementation": "device",
                               "greedy_tie_fallback": False})
    config["runtime"].update(device="cpu", dtype="float32", attn_implementation="sdpa", threads=2)
    config["generation"]["max_new_tokens"] = 6
    config["benchmark"].update(warmup=1, repeats=1)
    config["prompt"]["text"] = "word4 word5"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    mt = [{"id": f"mt-{i}", "category": cat, "text": f"word{4 + i} word{10 + i}"}
          for i, cat in enumerate(["writing", "writing", "math", "coding"])]
    monkeypatch.setattr(benchmark, "load_mt_bench", lambda: mt)
    monkeypatch.setattr(benchmark, "preflight", lambda: None)
    monkeypatch.setattr(benchmark, "kernel_probe", lambda dtype: {"stubbed": True})
    monkeypatch.setattr(benchmark, "sync", lambda: None)
    monkeypatch.setattr(benchmark, "clean", lambda: None)
    monkeypatch.setattr(benchmark, "CONTEXTS", [8, 16])
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--config", str(config_path)])
    benchmark.main()

    (run,) = (tmp_path / "outputs").iterdir()
    assert json.loads((run / "status.json").read_text())["status"] == "complete"
    results = json.loads((run / "results.json").read_text())
    assert set(results["theory"]) == {str(g) for g in range(1, 9)}
    assert results["paired"]["prompts"] == ["std-0", "std-1", "std-2", "mt-0", "mt-2", "mt-3"]
    for g in ("1", "2"):
        paired = results["paired"]["speculative"][g]
        # CPU float32: speculative output equals AR and measured counts equal the simulation.
        assert paired["greedy_agreement_prompts"] == 6
        repeats = results["paired"]["benchmark"]["repeats"]
        same = paired["predicted_counts_same_prompts"]
        assert (paired["accepted_tokens"], paired["verification_steps"]) == (
            same["accepted_tokens"] * repeats, same["verification_steps"] * repeats)
        full = results["mt_bench_full"]["speculative"][g]
        assert full["greedy_agreement_prompts"] == 4
        predicted = results["acceptance"]["mt_bench"][g]
        assert (full["accepted_tokens"], full["verification_steps"]) == (
            predicted["accepted_tokens"], predicted["verification_steps"])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize.py", str(run / "results.json"), "--output-dir", str(tmp_path / "results")])
    summarize.main()
    text = (tmp_path / "results/summary.md").read_text()
    for heading in ("## 1. AR baseline", "## 2. Acceptance", "## 3. Paired timing", "## 4. Full MT-Bench", "**Verdict:**"):
        assert heading in text
    assert (tmp_path / "results/results.json").is_file()


def test_bf16_ulp_accepts_floats_and_tensor_scalars():
    benchmark = _load("benchmark")
    assert benchmark.bf16_ulp(25.0) == 0.125
    assert benchmark.bf16_ulp(torch.tensor(12.3)) == 0.0625
    assert benchmark.bf16_ulp(torch.tensor(25.0, dtype=torch.bfloat16)) == 0.125
    if torch.cuda.is_available():  # regression: CUDA logits once crashed the divergence diagnostic
        assert benchmark.bf16_ulp(torch.tensor([25.0], device="cuda").topk(1).values[0]) == 0.125


def test_divergence_reports_first_difference(tiny_model):
    from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
    from acceleratellm.models.hf_backend import HFModelAdapter
    benchmark = _load("benchmark")
    adapter = HFModelAdapter(tiny_model)
    ids = torch.tensor([[4, 8, 6]])
    settings = {"strategy": "greedy", "max_new_tokens": 8, "temperature": 1., "top_p": 1.}
    expected = AutoregressiveDecoder().generate(adapter, ids, settings, set()).token_ids
    assert benchmark.divergence(adapter, ids, expected, list(expected)) is None
    actual = expected[:3] + [(expected[3] + 1) % 32] + expected[4:]
    report = benchmark.divergence(adapter, ids, expected, actual)
    assert report["position"] == 3 and report["expected_token"] == expected[3]
    assert report["ar_top2_ids"][0] == expected[3] and report["ar_top2_gap"] >= 0
    assert report["gap_in_ulps"] == report["ar_top2_gap"] / report["bf16_ulp_at_top"]
