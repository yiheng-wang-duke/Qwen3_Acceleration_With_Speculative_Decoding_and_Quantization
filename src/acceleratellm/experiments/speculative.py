"""Paired AR/speculative comparison with independent correctness checks."""
import json
import logging
from pathlib import Path

import torch
from transformers import GenerationConfig

from acceleratellm.decoding.autoregressive import AutoregressiveDecoder
from acceleratellm.decoding.speculative import SpeculativeDecoder
from acceleratellm.evaluation.generation_benchmark import measure_generation, seed_all, summarize
from acceleratellm.models.compatibility import check_model_pair, load_tokenizer
from acceleratellm.models.hf_backend import HFModelAdapter
from .artifacts import create_run, write_json


def _prompt_ids(tokenizer, text: str, chat: bool, device):
    if chat:
        if not tokenizer.chat_template:
            raise ValueError("Requested chat template but tokenizer has none")
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                            add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device)


def _reference(target, ids, settings, tokenizer, eos):
    cfg = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=settings["max_new_tokens"],
        eos_token_id=sorted(eos) or None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
            min(eos) if eos else 0),
        bos_token_id=tokenizer.bos_token_id, use_cache=True,
        repetition_penalty=1.0, no_repeat_ngram_size=0)
    with torch.inference_mode():
        output = target.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=cfg)
    return output[0, ids.shape[1]:].tolist()


def run_speculative(config: dict) -> Path:
    if "speculative" not in config or "matrix" in config:
        raise ValueError("The paired benchmark requires a speculative configuration")
    runtime = config["runtime"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device. Submit scripts/slurm_speculative.sh with sbatch.")
    if config["generation"]["max_new_tokens"] == 0:
        raise ValueError("The paired benchmark requires a positive generation budget")
    torch.set_num_threads(runtime["threads"])
    seed_all(config["seed"])
    run_dir = create_run(config)
    logger = logging.getLogger(f"acceleratellm.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(run_dir / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        compatibility = check_model_pair(config["models"], runtime["local_files_only"])
        write_json(run_dir / "compatibility.json", compatibility)
        logger.info("Loading target for standalone autoregressive measurements")
        target = HFModelAdapter.load(config["models"]["target"], runtime)
        tokenizer = load_tokenizer(config["models"]["target"], runtime["local_files_only"])
        eos_value = target.model.generation_config.eos_token_id
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos = set(eos_value if isinstance(eos_value, list) else [eos_value]) if eos_value is not None else set()
        texts = [config["prompt"]["text"], *config["speculative"]["validation_prompts"]]
        prompts = [_prompt_ids(tokenizer, text, config["prompt"]["use_chat_template"], target.device)
                   for text in texts]
        # Fail before timing if the draft cannot hold a full request.
        common_context = min(m["context_length"] for m in compatibility["models"].values())
        if any(ids.shape[1] + config["generation"]["max_new_tokens"] > common_context for ids in prompts):
            raise ValueError("A request exceeds the common target/draft context length")
        settings = config["generation"]
        greedy_settings = {**settings, "strategy": "greedy"}
        references = []
        records = []
        greedy_checks = []
        for prompt_id, ids in enumerate(prompts):
            logger.info("AR reference and measurement: prompt %s", prompt_id)
            reference = _reference(target, ids, greedy_settings, tokenizer, eos)
            checked = AutoregressiveDecoder().generate(target, ids, greedy_settings, eos)
            if checked.token_ids != reference:
                raise RuntimeError(f"AR differs from Transformers.generate on prompt {prompt_id}")
            references.append(reference)
            rows = measure_generation(AutoregressiveDecoder(), target, ids, settings, eos,
                                      config["benchmark"], config["seed"] + prompt_id * 100)
            for row in rows:
                row.update(decoder="autoregressive", draft_length=None, prompt_id=prompt_id)
                if settings["strategy"] == "greedy" and row["token_ids"] != reference:
                    raise RuntimeError("Measured AR differs from its greedy reference")
            records.extend(rows)
        ar_summary = summarize(records)
        # AR peak allocated memory was captured before loading the draft.
        logger.info("Loading frozen draft")
        draft = HFModelAdapter.load(config["models"]["draft"], runtime)
        if any(parameter.requires_grad for parameter in draft.model.parameters()):
            raise RuntimeError("Draft parameters must remain frozen")
        by_length = {}
        for length in config["speculative"]["draft_lengths"]:
            decoder = SpeculativeDecoder(length,greedy_implementation=config["speculative"].get("greedy_implementation","reference"),
                                         greedy_tie_fallback=config["speculative"].get("greedy_tie_fallback",False))
            length_records = []
            for prompt_id, ids in enumerate(prompts):
                logger.info("Speculative reference and measurement: gamma=%s prompt=%s", length, prompt_id)
                checked = decoder.generate(target, ids, greedy_settings, eos, draft)
                agreement = checked.token_ids == references[prompt_id]
                greedy_checks.append({"draft_length": length, "prompt_id": prompt_id,
                                      "agreement": agreement, "generated_tokens": len(checked.token_ids)})
                if not agreement:
                    write_json(run_dir / "greedy_mismatch.json", {
                        "draft_length": length, "prompt_id": prompt_id,
                        "reference_token_ids": references[prompt_id],
                        "speculative_token_ids": checked.token_ids})
                    raise RuntimeError(f"Speculative greedy mismatch: gamma={length}, prompt={prompt_id}")
                rows = measure_generation(decoder, target, ids, settings, eos,
                                          config["benchmark"], config["seed"] + prompt_id * 100, draft)
                for row in rows:
                    row.update(decoder="speculative", draft_length=length, prompt_id=prompt_id)
                    if settings["strategy"] == "greedy" and row["token_ids"] != references[prompt_id]:
                        raise RuntimeError("Measured speculative output differs from the AR reference")
                length_records.extend(rows)
            summary = summarize(length_records)
            summary["latency_speedup_vs_ar"] = ar_summary["mean_latency_seconds"] / summary["mean_latency_seconds"]
            summary["throughput_speedup_vs_ar"] = summary["aggregate_tokens_per_second"] / ar_summary["aggregate_tokens_per_second"]
            by_length[str(length)] = summary
            records.extend(length_records)
        sampling_checks = []
        sample_settings = {**settings, "strategy": "sampling", "temperature": .8, "top_p": .9,
                           "max_new_tokens": min(16, settings["max_new_tokens"])}
        for length in config["speculative"]["draft_lengths"]:
            for seed in config["speculative"]["sampling_seeds"]:
                logger.info("Sampling replay check: gamma=%s seed=%s", length, seed)
                decoder = SpeculativeDecoder(length,greedy_implementation=config["speculative"].get("greedy_implementation","reference"),
                                         greedy_tie_fallback=config["speculative"].get("greedy_tie_fallback",False))
                seed_all(seed)
                first = decoder.generate(target, prompts[0], sample_settings, eos, draft)
                seed_all(seed)
                replay = decoder.generate(target, prompts[0], sample_settings, eos, draft)
                if first != replay:
                    raise RuntimeError("Speculative sampling is not reproducible with the same seed")
                sampling_checks.append({
                    "draft_length": length, "seed": seed, "reproducible": True, "logits_finite": True,
                    "token_ids": first.token_ids, "text": tokenizer.decode(first.token_ids, skip_special_tokens=True),
                    "proposed_tokens": first.proposed_tokens, "accepted_tokens": first.accepted_tokens,
                    "verification_steps": first.verification_steps})
        verification = {
            "greedy_agreement": all(row["agreement"] for row in greedy_checks),
            "reference": "standalone AR and Transformers.generate with the same configured target",
            "greedy_checks": greedy_checks, "reference_token_ids": references,
            "sampling_checks": sampling_checks, "sampling_settings": sample_settings,
            "sampling_scope": "full-model finite-logit/reproducibility sanity checks; distribution tested separately on exact toy models",
            "draft_frozen": True, "logits_finite": True,
        }
        write_json(run_dir / "verification.json", verification)
        with (run_dir / "generations.jsonl").open("w") as stream:
            for row in records:
                row["text"] = tokenizer.decode(row["token_ids"], skip_special_tokens=True)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        metrics = {
            "greedy_implementation":config["speculative"].get("greedy_implementation","reference"),
            "greedy_tie_fallback":config["speculative"].get("greedy_tie_fallback",False),
            "model_parameters":{"target":sum(p.numel() for p in target.model.parameters()),
                                "draft":sum(p.numel() for p in draft.model.parameters())},
            "schema_version": 1, "experiment": "autoregressive_vs_speculative",
            "scope": "small_prompt_comparison", "valid_for_benchmark": True,
            "device": target.device.type, "dtype": runtime["dtype"],
            "attn_implementation": runtime["attn_implementation"],
            "batch_size": 1, "prompt_count": len(prompts), "prompts": texts,
            "generation_settings": settings, "benchmark_settings": config["benchmark"],
            "autoregressive": ar_summary, "speculative": by_length,
            "greedy_agreement": verification["greedy_agreement"], "logits_finite": True,
            "timing_scope": "prefill + draft + target verification + sampling + cache rollback (+ numerical replay if greedy_tie_fallback); excludes loading/tokenization/text decoding",
            "memory_scope": "AR measured before draft load; speculative peak allocated includes both models and caches",
            "acceptance_definition": "accepted draft prefix tokens / all proposed tokens, including proposals discarded after first rejection",
            "sampling_note": "AR and speculative sampling preserve the same target law; identical seeds do not imply identical sequences",
        }
        write_json(run_dir / "metrics.json", metrics)
        write_json(run_dir / "status.json", {"status": "complete"})
        logger.info("Speculative validation and comparison complete")
        return run_dir
    except Exception as exc:
        logger.exception("Speculative run failed")
        write_json(run_dir / "status.json", {"status": "failed", "error": str(exc)})
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
