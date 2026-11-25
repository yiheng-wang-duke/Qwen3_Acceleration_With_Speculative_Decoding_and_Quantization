"""Small JSON/YAML artifacts with explicit environment provenance."""
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
import yaml


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def environment_report() -> dict:
    packages = {}
    for name in ("torch", "transformers", "PyYAML", "huggingface-hub", "safetensors", "triton", "flash-attn", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "uuid": str(torch.cuda.get_device_properties(i).uuid),
             "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
             "compute_capability": list(torch.cuda.get_device_capability(i))}
            for i in range(torch.cuda.device_count())
        ],
        "slurm": {key: os.environ[key] for key in (
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "SLURM_JOB_GPUS",
            "ACCELERATELLM_ORIGINAL_CUDA_VISIBLE_DEVICES") if key in os.environ},
    }


def create_run(config: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(config["output_dir"]) / f"{stamp}-{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(path / "environment.json", environment_report())
    return path
