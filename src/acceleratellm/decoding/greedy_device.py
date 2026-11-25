"""Greedy speculative decoding with one host decision transfer per iteration."""
import torch
from acceleratellm.schemas import GenerationResult
from .cache import require_cache_length


@torch.inference_mode()
def replay_greedy_block(decoder, target, prompt, generated, proposals):
    """Rebuild the exact AR cache before resolving a low-precision near-tie decision.

    Rechecking only the final query is insufficient: earlier parallel verification
    calls can already have introduced rounding differences into the KV cache.
    """
    current = target.forward(prompt)
    decoder._check(current.logits, "target replay")
    for token_id in generated:
        current = target.forward(torch.tensor([[token_id]], device=target.device), current.cache)
        decoder._check(current.logits, "target replay")
    choices = []
    for token_id in proposals:
        chosen = int(current.logits[0, -1].argmax().item())
        choices.append(chosen)
        if token_id != chosen:
            break
        current = target.forward(torch.tensor([[token_id]], device=target.device), current.cache)
        decoder._check(current.logits, "target replay")
    if len(choices) == len(proposals) and choices[-1] == proposals[-1]:
        choices.append(int(current.logits[0, -1].argmax().item()))
    # Entries beyond the first rejection are never read by the caller.
    choices.extend([0] * (len(proposals) + 1 - len(choices)))
    return current.cache, torch.tensor(choices, device=target.device)


@torch.inference_mode()
def generate_device_greedy(decoder, target, draft, prompt, settings, eos):
    budget = settings["max_new_tokens"]
    prefix_length = prompt.shape[1]
    target_cache = None
    if prefix_length > 1:
        prefill = target.forward(prompt[:,:-1])
        decoder._check(prefill.logits,"target")
        target_cache = prefill.cache
        del prefill
    result = draft.forward(prompt)
    decoder._check(result.logits,"draft")
    draft_cache, draft_logits = result.cache, result.logits[:,-1,:]
    del result
    pending = prompt[:,-1:]
    generated = []
    proposed_total = accepted_total = steps = numerical_fallback_steps = 0
    stop_reason = "length"
    while len(generated) < budget:
        if decoder.validate_logits:
            require_cache_length(target,target_cache,prefix_length-1,"target")
            require_cache_length(draft,draft_cache,prefix_length,"draft")
        count = min(decoder.draft_length,budget-len(generated))
        candidates = []
        for index in range(count):
            decoder._check(draft_logits,"draft")
            candidate = draft_logits.argmax(-1,keepdim=True)
            candidates.append(candidate)
            # Keep candidate IDs on device. An EOS proposal is handled when the
            # target decision is copied, instead of synchronizing each draft step.
            if index+1 < count:
                result = draft.forward(candidate,draft_cache)
                decoder._check(result.logits,"draft")
                draft_cache, draft_logits = result.cache, result.logits[:,-1,:]
        proposals = torch.cat(candidates,dim=1)
        verified = target.forward(torch.cat((pending,proposals),dim=1),target_cache)
        decoder._check(verified.logits,"target")
        choices = verified.logits[0].argmax(-1)
        # Include tie flags in the existing single host transfer. Two adjacent
        # representable output-logit spacings define the numerical sensitivity band.
        if decoder.greedy_tie_fallback and verified.logits.shape[-1] > 1:
            top = verified.logits[0].topk(2, dim=-1).values
            spacing = torch.nextafter(top[:,0], torch.full_like(top[:,0], float("inf"))) - top[:,0]
            near_ties = ((top[:,0] - top[:,1]) <= 2 * spacing).long()
            decision = torch.cat((proposals[0], choices, near_ties)).tolist()
        else:
            decision = torch.cat((proposals[0], choices)).tolist()
        proposal_ids = decision[:count]
        target_ids = decision[count:2*count+1]
        accepted = next((i for i in range(count) if proposal_ids[i]!=target_ids[i]),count)
        # Ignore sensitivity after the first rejected candidate: those logits do
        # not affect this iteration's output. No alternate tie-breaking is applied.
        if decoder.greedy_tie_fallback and any(decision[2*count+1:2*count+2+accepted]):
            verified.cache, choices = replay_greedy_block(decoder, target, prompt, generated, proposal_ids)
            target_ids = choices.tolist()
            accepted = next((i for i in range(count) if proposal_ids[i]!=target_ids[i]),count)
            numerical_fallback_steps += 1
        # Accepted tokens beyond the first EOS are never committed or counted.
        eos_offset = next((i for i in range(accepted) if proposal_ids[i] in eos),None)
        if eos_offset is not None:
            accepted = eos_offset+1
        proposed_total += count
        accepted_total += accepted
        steps += 1
        generated.extend(proposal_ids[:accepted])
        if eos_offset is not None:
            stop_reason = "eos"
            break
        if len(generated)==budget:
            break
        token_id = target_ids[accepted]
        correction = choices[accepted].reshape(1,1)
        generated.append(token_id)
        if token_id in eos:
            stop_reason = "eos"
            break
        if len(generated)==budget:
            break
        target_cache = target.crop_cache(verified.cache,prefix_length+accepted)
        if accepted<count:
            draft_cache = draft.crop_cache(draft_cache,prefix_length+accepted)
            draft_input = correction
        else:
            draft_input = torch.cat((proposals[:,-1:],correction),dim=1)
        result = draft.forward(draft_input,draft_cache)
        decoder._check(result.logits,"draft")
        draft_cache, draft_logits = result.cache,result.logits[:,-1,:]
        pending = correction
        prefix_length += accepted+1
    return GenerationResult(generated,proposed_tokens=proposed_total,
                            accepted_tokens=accepted_total,verification_steps=steps,
                            stop_reason=stop_reason, numerical_fallback_steps=numerical_fallback_steps)
