"""Small real-model baseline, with separately recorded correctness verification."""
import logging
import random
import time
from pathlib import Path

import torch

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.models.compatibility import check_model_pair, load_tokenizer
from acceleratellm.models.hf_backend import HFModelAdapter
from .artifacts import create_run, write_json


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _eos_ids(adapter: HFModelAdapter, tokenizer) -> set[int]:
    value = adapter.model.generation_config.eos_token_id
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        return set()
    return set(value if isinstance(value, list) else [value])


def run_baseline(config: dict) -> Path:
    # A matrix is deliberately not executable until all backends exist.
    if "speculative" in config:
        raise ValueError("Use configs/experiments/baseline.yaml; speculative runs use the speculative command")
    runtime = config["runtime"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Submit scripts/slurm_baseline.sh using sbatch.")
    torch.set_num_threads(runtime["threads"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if runtime["device"] == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    run_dir = create_run(config)
    logger = logging.getLogger(f"acceleratellm.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Checking model/tokenizer pair")
        compatibility = check_model_pair(config["models"], runtime["local_files_only"])
        write_json(run_dir / "compatibility.json", compatibility)
        logger.info("Loading target model")
        adapter = HFModelAdapter.load(config["models"]["target"], runtime)
        tokenizer = load_tokenizer(config["models"]["target"], runtime["local_files_only"])
        text = config["prompt"]["text"]
        if config["prompt"]["use_chat_template"]:
            if not tokenizer.chat_template:
                raise ValueError("Requested chat template but tokenizer has no chat template")
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
        else:
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        ids = ids.to(adapter.device)
        checked_decoder = AutoregressiveDecoder(validate_logits=True)
        decoder = AutoregressiveDecoder(validate_logits=False)
        settings = config["generation"]
        eos = _eos_ids(adapter, tokenizer)
        logger.info("Validating finite logits before timing")
        preflight = checked_decoder.generate(adapter, ids, settings, eos)
        for _ in range(config["benchmark"]["warmup"]):
            decoder.generate(adapter, ids, settings, eos)
        _sync(adapter.device)
        records = []
        logger.info("Starting %s measured repetitions", config["benchmark"]["repeats"])
        for repeat in range(config["benchmark"]["repeats"]):
            # Seed outside timing, and use independent, reproducible sample streams.
            torch.manual_seed(config["seed"] + repeat)
            if adapter.device.type == "cuda":
                torch.cuda.manual_seed_all(config["seed"] + repeat)
                torch.cuda.reset_peak_memory_stats(adapter.device)
            _sync(adapter.device)
            started = time.perf_counter()
            result = decoder.generate(adapter, ids, settings, eos)
            _sync(adapter.device)
            elapsed = time.perf_counter() - started
            records.append({
                "repeat": repeat, "seed": config["seed"] + repeat,
                "latency_seconds": elapsed, "generated_tokens": len(result.token_ids),
                "tokens_per_second": len(result.token_ids) / elapsed,
                "peak_allocated_bytes": (torch.cuda.max_memory_allocated(adapter.device)
                                         if adapter.device.type == "cuda" else None),
                "peak_reserved_bytes": (torch.cuda.max_memory_reserved(adapter.device)
                                        if adapter.device.type == "cuda" else None),
                "stop_reason": result.stop_reason, "token_ids": result.token_ids,
                "text": tokenizer.decode(result.token_ids, skip_special_tokens=True),
            })
        # Correctness work is outside measured regions; match the same target and settings.
        # Replay sampled streams with finite checks outside timing. Greedy streams
        # must match the already checked preflight path exactly.
        for record in records:
            if settings["strategy"] == "sampling":
                torch.manual_seed(record["seed"])
                if adapter.device.type == "cuda":
                    torch.cuda.manual_seed_all(record["seed"])
                checked = checked_decoder.generate(adapter, ids, settings, eos)
            else:
                checked = preflight
            if checked.token_ids != record["token_ids"]:
                raise RuntimeError("Measured output differs from the numerically validated replay")
        verification = {"greedy_agreement": None, "reference": "transformers.generate",
                        "logits_finite": True, "measured_outputs_match_validated_replay": True}
        if settings["strategy"] == "greedy" and settings["max_new_tokens"] > 0:
            from transformers import GenerationConfig
            reference_settings = GenerationConfig(
                do_sample=False, num_beams=1, max_new_tokens=settings["max_new_tokens"],
                eos_token_id=sorted(eos) or None,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
                    min(eos) if eos else 0),
                bos_token_id=tokenizer.bos_token_id,
                use_cache=True, repetition_penalty=1.0, no_repeat_ngram_size=0,
            )
            with torch.inference_mode():
                reference = adapter.model.generate(
                    input_ids=ids, generation_config=reference_settings,
                    attention_mask=torch.ones_like(ids))
            expected = reference[0, ids.shape[1]:].tolist()
            agreement = all(record["token_ids"] == expected for record in records)
            verification.update(greedy_agreement=agreement, reference_token_ids=expected)
        write_json(run_dir / "verification.json", verification)
        with (run_dir / "generations.jsonl").open("w") as stream:
            import json
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        total_time = sum(record["latency_seconds"] for record in records)
        metrics = {
            "schema_version": 1,
            "experiment": "autoregressive",
            "device": adapter.device.type, "dtype": runtime["dtype"],
            "attn_implementation": runtime["attn_implementation"],
            "scope": "single_prompt_smoke_test",
            "valid_for_benchmark": True,
            "batch_size": 1, "prompt_tokens": ids.shape[1],
            "repetitions": len(records), "warmup": config["benchmark"]["warmup"],
            "mean_latency_seconds": total_time / len(records),
            "aggregate_tokens_per_second": sum(r["generated_tokens"] for r in records) / total_time,
            "peak_allocated_bytes": max((r["peak_allocated_bytes"] or 0) for r in records) or None,
            "peak_reserved_bytes": max((r["peak_reserved_bytes"] or 0) for r in records) or None,
            "greedy_agreement": verification["greedy_agreement"],
            "logits_finite": verification["logits_finite"],
            "numerical_check_scope": "unmeasured preflight and matching replay of measured token sequences",
            "timing_scope": "prefill + cached decoding; excludes loading, tokenization, text decoding, correctness check",
            "records": [{k: v for k, v in r.items() if k not in {"text", "token_ids"}} for r in records],
        }
        write_json(run_dir / "metrics.json", metrics)
        if verification["greedy_agreement"] is False:
            raise RuntimeError("Greedy baseline differs from Transformers.generate; inspect verification.json")
        write_json(run_dir / "status.json", {"status": "complete"})
        logger.info("Baseline complete")
        return run_dir
    except Exception as exc:
        logger.exception("Baseline failed")
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
