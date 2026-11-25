"""Measure complete requests, then numerically validate identical replays off clock."""
import copy
import time

import torch


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_generation(decoder, target, prompt, settings, eos, benchmark, seed, draft=None):
    checked, timed = copy.copy(decoder), copy.copy(decoder)
    checked.validate_logits = True
    timed.validate_logits = False
    seed_all(seed)
    checked.generate(target, prompt, settings, eos, draft)
    for _ in range(benchmark["warmup"]):
        timed.generate(target, prompt, settings, eos, draft)
    records = []
    results = []
    for repeat in range(benchmark["repeats"]):
        run_seed = seed + repeat
        seed_all(run_seed)
        if target.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target.device)
        synchronize(target.device)
        started = time.perf_counter()
        result = timed.generate(target, prompt, settings, eos, draft)
        synchronize(target.device)
        elapsed = time.perf_counter() - started
        results.append(result)
        records.append({
            "repeat": repeat, "seed": run_seed, "prompt_tokens": prompt.shape[1],
            "token_ids": result.token_ids, "stop_reason": result.stop_reason,
            "generated_tokens": len(result.token_ids), "latency_seconds": elapsed,
            "tokens_per_second": len(result.token_ids) / elapsed,
            "proposed_tokens": result.proposed_tokens, "accepted_tokens": result.accepted_tokens,
            "verification_steps": result.verification_steps,
            "numerical_fallback_steps": result.numerical_fallback_steps,
            "peak_allocated_bytes": (torch.cuda.max_memory_allocated(target.device)
                                     if target.device.type == "cuda" else None),
            "peak_reserved_bytes": (torch.cuda.max_memory_reserved(target.device)
                                    if target.device.type == "cuda" else None),
        })
    for record, expected in zip(records, results):
        seed_all(record["seed"])
        replay = checked.generate(target, prompt, settings, eos, draft)
        if replay != expected:
            raise RuntimeError("Measured generation differs from its numerically checked replay")
        record["logits_finite"] = True
        record["replay_validated"] = True
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty benchmark")
    duration = sum(r["latency_seconds"] for r in records)
    tokens = sum(r["generated_tokens"] for r in records)
    proposed = sum(r["proposed_tokens"] for r in records)
    accepted = sum(r["accepted_tokens"] for r in records)
    steps = sum(r["verification_steps"] for r in records)
    return {
        "requests": len(records),
        "numerical_fallback_steps": sum(r.get("numerical_fallback_steps",0) for r in records),
        "mean_latency_seconds": duration / len(records),
        "aggregate_tokens_per_second": tokens / duration,
        "generated_tokens": tokens,
        "proposed_tokens": proposed, "accepted_tokens": accepted, "verification_steps": steps,
        "token_acceptance_rate": accepted / proposed if proposed else None,
        "mean_accepted_tokens_per_verification": accepted / steps if steps else None,
        "mean_output_tokens_per_verification": tokens / steps if steps else None,
        "peak_allocated_bytes": max(r["peak_allocated_bytes"] or 0 for r in records) or None,
        "peak_reserved_bytes": max(r["peak_reserved_bytes"] or 0 for r in records) or None,
        "logits_finite": all(r["logits_finite"] for r in records),
        "replay_validated": all(r["replay_validated"] for r in records),
    }
