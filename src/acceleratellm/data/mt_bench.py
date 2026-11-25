"""Pinned MT-Bench first-turn prompts (FastChat llm_judge question set)."""
import hashlib
import json
from pathlib import Path

COMMIT = "b494d0c6b4e7935f1764f8439e75da3e66beccc7"
URL = f"https://raw.githubusercontent.com/lm-sys/FastChat/{COMMIT}/fastchat/llm_judge/data/mt_bench/question.jsonl"
SHA256 = "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7"
DEFAULT_PATH = Path("data/mt_bench/question.jsonl")
# Base (non-chat) models: a minimal question/answer frame, no chat template.
PROMPT_FORMAT = "Question: {turn}\nAnswer:"


def verify(path: Path = DEFAULT_PATH) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_data.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(f"{path} differs from the pinned MT-Bench question file")


def load_mt_bench(path: Path = DEFAULT_PATH) -> list[dict]:
    """80 prompts: {'id', 'category', 'text'} with the base-model prompt format applied."""
    verify(path)
    prompts = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        prompts.append({"id": f"mt-{row['question_id']}", "category": row["category"],
                        "text": PROMPT_FORMAT.format(turn=row["turns"][0])})
    if len(prompts) != 80:
        raise RuntimeError("Expected 80 MT-Bench questions")
    return prompts


def stratified_subset(prompts: list[dict]) -> list[str]:
    """First question of every category (8 prompts)."""
    seen, ids = set(), []
    for p in prompts:
        if p["category"] not in seen:
            seen.add(p["category"])
            ids.append(p["id"])
    return ids
