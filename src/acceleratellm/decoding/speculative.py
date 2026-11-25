"""Single-draft speculative decoding (Leviathan et al., ICML 2023).

At iteration entry, target cache excludes the final committed token ('pending'),
whereas draft cache contains the entire committed prefix. Target verifies pending
+ proposals in one forward, producing all proposal scores and the bonus score.
"""
import torch

from acceleratellm.models.interface import ModelAdapter
from acceleratellm.schemas import GenerationResult
from .cache import require_cache_length
from .sampling import probabilities, rejection_distribution


class SpeculativeDecoder:
    def __init__(self, draft_length: int = 4, validate_logits: bool = True, greedy_implementation: str = "reference", greedy_tie_fallback: bool = False):
        if type(draft_length) is not int or draft_length < 1:
            raise ValueError("draft_length must be a positive integer")
        if greedy_implementation not in {"reference", "device"}:
            raise ValueError("Unknown greedy speculative implementation")
        if type(greedy_tie_fallback) is not bool:
            raise ValueError("greedy_tie_fallback must be boolean")
        self.greedy_tie_fallback = greedy_tie_fallback
        self.greedy_implementation = greedy_implementation
        self.draft_length = draft_length
        self.validate_logits = validate_logits

    def _check(self, logits: torch.Tensor, role: str) -> None:
        if self.validate_logits and not bool(torch.isfinite(logits).all()):
            raise FloatingPointError(f"Non-finite {role} logits in speculative decoding")

    @torch.inference_mode()
    def generate(self, target: ModelAdapter, prompt: torch.Tensor, settings: dict,
                 eos_token_ids: set[int], draft: ModelAdapter | None = None) -> GenerationResult:
        if draft is None:
            raise ValueError("Speculative decoding requires a frozen draft model")
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError("Expected a nonempty, unpadded [1, length] prompt")
        if torch.device(target.device) != torch.device(draft.device):
            raise ValueError("Target and draft must use the same device")
        if target.vocab_size != draft.vocab_size:
            raise ValueError("Target and draft output vocab sizes must match")
        budget = settings["max_new_tokens"]
        if type(budget) is not int or budget < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        strategy = settings["strategy"]
        if strategy not in {"greedy", "sampling"}:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")
        if prompt.shape[1] + budget > min(target.context_length, draft.context_length):
            raise ValueError("Prompt plus token budget exceeds target or draft context length")
        if any(type(i) is not int or not 0 <= i < target.vocab_size for i in eos_token_ids):
            raise ValueError("EOS token IDs must be within the shared vocabulary")
        if budget == 0:
            return GenerationResult([])
        prompt = prompt.to(target.device)
        if strategy == "greedy" and self.greedy_implementation == "device":
            from .greedy_device import generate_device_greedy
            return generate_device_greedy(self,target,draft,prompt,settings,eos_token_ids)
        prefix_length = prompt.shape[1]
        target_cache = None
        if prefix_length > 1:
            prefill = target.forward(prompt[:, :-1])
            self._check(prefill.logits, "target")
            target_cache = prefill.cache
            del prefill
        draft_result = draft.forward(prompt)
        self._check(draft_result.logits, "draft")
        draft_cache = draft_result.cache
        draft_logits = draft_result.logits[:, -1, :]
        del draft_result
        pending = prompt[:, -1:]
        generated: list[int] = []
        proposed_total = accepted_total = steps = 0
        stop_reason = "length"

        while len(generated) < budget:
            if self.validate_logits:
                require_cache_length(target, target_cache, prefix_length - 1, "target")
                require_cache_length(draft, draft_cache, prefix_length, "draft")
            proposal_ids = []
            draft_distributions = []
            count = min(self.draft_length, budget - len(generated))
            for index in range(count):
                self._check(draft_logits, "draft")
                if strategy == "greedy":
                    candidate = draft_logits.argmax(-1, keepdim=True)
                else:
                    q = probabilities(draft_logits, settings["temperature"], settings["top_p"])
                    draft_distributions.append(q[0])
                    candidate = torch.multinomial(q, num_samples=1)
                token_id = int(candidate.item())
                proposal_ids.append(token_id)
                # The last proposed token is deliberately left out of draft cache.
                if token_id in eos_token_ids or index + 1 == count:
                    break
                draft_result = draft.forward(candidate, draft_cache)
                self._check(draft_result.logits, "draft")
                draft_cache = draft_result.cache
                draft_logits = draft_result.logits[:, -1, :]
            proposals = torch.tensor([proposal_ids], device=target.device, dtype=torch.long)
            count = len(proposal_ids)
            proposed_total += count
            verified = target.forward(torch.cat((pending, proposals), dim=1), target_cache)
            self._check(verified.logits, "target")
            steps += 1
            # Position i scores proposal i; the final position scores a bonus.
            logits = verified.logits[0]
            if strategy == "greedy":
                target_choices = logits.argmax(-1)
                accepted_mask = target_choices[:count] == proposals[0]
            else:
                p = probabilities(logits, settings["temperature"], settings["top_p"])
                q = torch.stack(draft_distributions)
                indices = proposals[0, :, None]
                p_candidate = p[:count].gather(1, indices).squeeze(1)
                q_candidate = q.gather(1, indices).squeeze(1)
                ratios = (p_candidate / q_candidate).clamp(max=1)
                # '<' ensures zero-probability tokens cannot be accepted even if U=0.
                accepted_mask = torch.rand_like(ratios) < ratios
            rejected = (~accepted_mask).nonzero(as_tuple=False)
            accepted = int(rejected[0, 0].item()) if rejected.numel() else count
            accepted_total += accepted
            generated.extend(proposal_ids[:accepted])
            if generated and generated[-1] in eos_token_ids:
                stop_reason = "eos"
                break
            if len(generated) == budget:
                break
            if strategy == "greedy":
                correction = target_choices[accepted].reshape(1, 1)
            else:
                distribution = (rejection_distribution(p[accepted], q[accepted])
                                if accepted < count else p[count])
                correction = torch.multinomial(distribution, num_samples=1).reshape(1, 1)
            token_id = int(correction.item())
            generated.append(token_id)
            if token_id in eos_token_ids:
                stop_reason = "eos"
                break
            if len(generated) == budget:
                break

            # Keep only accepted proposals in target cache; correction stays pending.
            target_cache = target.crop_cache(verified.cache, prefix_length + accepted)
            if accepted < count:
                draft_cache = draft.crop_cache(draft_cache, prefix_length + accepted)
                draft_input = correction
            else:
                # Draft cache lacks the final accepted proposal. Catch up that token
                # and the bonus together, rather than making two model calls.
                draft_input = torch.cat((proposals[:, -1:], correction), dim=1)
            draft_result = draft.forward(draft_input, draft_cache)
            self._check(draft_result.logits, "draft")
            draft_cache = draft_result.cache
            draft_logits = draft_result.logits[:, -1, :]
            pending = correction
            prefix_length += accepted + 1

        return GenerationResult(generated, proposed_tokens=proposed_total,
                                accepted_tokens=accepted_total, verification_steps=steps,
                                stop_reason=stop_reason)
