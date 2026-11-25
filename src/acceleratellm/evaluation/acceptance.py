"""Exact greedy speculative acceptance from one target trajectory and one draft pass.

Under greedy decoding the committed output equals the target's AR trajectory y,
independently of the draft length. At round start i, the draft has the committed
prefix y[:i]; while its proposals match y they extend that same prefix, so its
proposal at position j is d[j] = argmax draft(prompt + y[:j]). Therefore one
teacher-forced draft forward gives every proposal the real loop can make, and the
round structure for any gamma follows by simulation.

Exactness assumes teacher-forced draft logits select the same argmax as the
incremental draft cache; this is exact in exact arithmetic and validated against
real speculative runs on GPU (low-precision near ties can differ).
"""
from __future__ import annotations


def match_lengths(draft_choices: list[int], trajectory: list[int]) -> list[int]:
    """m[i] = number of consecutive positions j >= i with draft_choices[j] == trajectory[j]."""
    if len(draft_choices) != len(trajectory):
        raise ValueError("Draft choices and trajectory must align position by position")
    lengths = [0] * (len(trajectory) + 1)
    for i in range(len(trajectory) - 1, -1, -1):
        lengths[i] = lengths[i + 1] + 1 if draft_choices[i] == trajectory[i] else 0
    return lengths[:-1]


def simulate_greedy_rounds(draft_choices: list[int], trajectory: list[int], gamma: int,
                           budget: int, eos: set[int]) -> dict:
    """Replicates SpeculativeDecoder greedy counters (device implementation).

    `trajectory` is the full greedy AR output for this prompt and budget, i.e. it
    either ends with the first EOS or has exactly `budget` tokens.
    """
    if type(gamma) is not int or gamma < 1:
        raise ValueError("gamma must be a positive integer")
    n = len(trajectory)
    if n == 0 or n > budget:
        raise ValueError("Trajectory must be nonempty and within budget")
    if any(t in eos for t in trajectory[:-1]):
        raise ValueError("Trajectory continues past EOS")
    if trajectory[-1] not in eos and n != budget:
        raise ValueError("Trajectory without EOS must use the full budget")
    m = match_lengths(draft_choices, trajectory)
    i = proposed = accepted_total = steps = 0
    while True:
        count = min(gamma, budget - i)
        # Proposals beyond the trajectory end can only follow an accepted EOS.
        accepted = min(m[i], count, n - i)
        proposed += count
        steps += 1
        eos_offset = next((k for k in range(accepted) if trajectory[i + k] in eos), None)
        if eos_offset is not None:
            accepted_total += eos_offset + 1
            i += eos_offset + 1
            break
        accepted_total += accepted
        i += accepted
        if i == budget:
            break
        # Correction or bonus token from the target, which is trajectory[i].
        token = trajectory[i]
        i += 1
        if token in eos or i == budget:
            break
    if i != n:
        raise RuntimeError("Simulation did not reproduce the trajectory length")
    return {"proposed_tokens": proposed, "accepted_tokens": accepted_total,
            "verification_steps": steps, "generated_tokens": n}


def teacher_forced_choices(draft, prompt, trajectory: list[int]) -> list[int]:
    """d[j] = argmax draft(prompt + trajectory[:j]) for every trajectory position j."""
    import torch
    with torch.inference_mode():
        tail = torch.tensor([trajectory[:-1]], device=prompt.device, dtype=prompt.dtype)
        ids = torch.cat((prompt, tail), dim=1) if len(trajectory) > 1 else prompt
        logits = draft.forward(ids).logits[0, prompt.shape[1] - 1:]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Non-finite draft logits in teacher-forced pass")
        choices = logits.argmax(-1).tolist()
    if len(choices) != len(trajectory):
        raise RuntimeError("Teacher-forced pass did not score every trajectory position")
    return choices


def aggregate(rows: list[dict]) -> dict:
    keys = ("proposed_tokens", "accepted_tokens", "verification_steps", "generated_tokens")
    total = {k: sum(r[k] for r in rows) for k in keys}
    total["token_acceptance_rate"] = total["accepted_tokens"] / total["proposed_tokens"]
    total["mean_accepted_tokens_per_verification"] = total["accepted_tokens"] / total["verification_steps"]
    total["mean_output_tokens_per_verification"] = total["generated_tokens"] / total["verification_steps"]
    return total
