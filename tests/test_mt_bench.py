import json

import pytest

from acceleratellm.data import mt_bench


def test_pinned_mt_bench_file_loads_80_prompts_when_present():
    if not mt_bench.DEFAULT_PATH.exists():
        pytest.skip("run scripts/fetch_data.py to download MT-Bench")
    prompts = mt_bench.load_mt_bench()
    assert len(prompts) == 80 and len({p["category"] for p in prompts}) == 8
    assert prompts[0]["text"].startswith("Question: ") and prompts[0]["text"].endswith("\nAnswer:")
    subset = mt_bench.stratified_subset(prompts)
    assert len(subset) == 8 and subset[0] == prompts[0]["id"]


def test_modified_file_is_rejected(tmp_path):
    path = tmp_path / "question.jsonl"
    path.write_text(json.dumps({"question_id": 1, "category": "x", "turns": ["hi"]}) + "\n")
    with pytest.raises(RuntimeError, match="pinned"):
        mt_bench.load_mt_bench(path)
    with pytest.raises(FileNotFoundError):
        mt_bench.verify(tmp_path / "missing.jsonl")
