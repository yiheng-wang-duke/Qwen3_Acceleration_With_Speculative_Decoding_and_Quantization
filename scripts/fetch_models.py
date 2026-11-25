"""Download pinned draft and target snapshots (weights + tokenizer). Network use is explicit."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from acceleratellm.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/speculative.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    for role, spec in config["models"].items():
        if not Path(spec["model_id"]).is_dir() and (
            len(spec["revision"]) != 40 or any(c not in "0123456789abcdef" for c in spec["revision"])
        ):
            raise ValueError(f"{role} must use a pinned 40-character commit before download")
        patterns = ["config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "vocab.json", "merges.txt",
                    "*.safetensors", "*.safetensors.index.json"]
        path = snapshot_download(spec["model_id"], revision=spec["revision"],
                                 allow_patterns=patterns, max_workers=2)
        print(role, path, flush=True)


if __name__ == "__main__":
    main()
