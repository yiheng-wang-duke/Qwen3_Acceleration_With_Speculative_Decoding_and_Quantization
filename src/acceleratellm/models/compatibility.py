"""Check tokenizer semantics and model output ranges before sharing token IDs."""
import hashlib
import json
from typing import Any

from transformers import AutoConfig, AutoTokenizer


def tokenizer_fingerprint(tokenizer: Any) -> str:
    # The serialized backend captures normalization, pre-tokenization, BPE merges
    # and post-processing: identical vocabularies alone are not sufficient.
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("Compatibility verification requires fast tokenizers")
    payload = {
        "backend": json.loads(backend.to_str()),
        "special_tokens_map": tokenizer.special_tokens_map,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "chat_template": tokenizer.chat_template,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def check_tokenizers(target: Any, draft: Any, target_vocab: int, draft_vocab: int) -> dict:
    target_hash = tokenizer_fingerprint(target)
    draft_hash = tokenizer_fingerprint(draft)
    if target_hash != draft_hash:
        raise ValueError("Target and draft tokenizer semantics differ")
    if target_vocab != draft_vocab:
        raise ValueError("Target/draft output vocab sizes differ; speculative decoding needs a shared vocabulary")
    token_ids = list(target.get_vocab().values())
    if not token_ids or min(token_ids) < 0 or max(token_ids) >= min(target_vocab, draft_vocab):
        raise ValueError("Tokenizer IDs fall outside the model vocabulary")
    return {"compatible": True, "tokenizer_sha256": target_hash,
            "tokenizer_size": len(target), "output_vocab_size": target_vocab}


def load_tokenizer(spec: dict, local_files_only: bool = True):
    return AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"],
        local_files_only=local_files_only, use_fast=True, trust_remote_code=False,
    )


def check_model_pair(models: dict, local_files_only: bool = True) -> dict:
    configs = {}
    tokenizers = {}
    for role, spec in models.items():
        configs[role] = AutoConfig.from_pretrained(
            spec["model_id"], revision=spec["revision"],
            local_files_only=local_files_only, trust_remote_code=False,
        )
        tokenizers[role] = load_tokenizer(spec, local_files_only)
    report = check_tokenizers(
        tokenizers["target"], tokenizers["draft"],
        configs["target"].vocab_size, configs["draft"].vocab_size,
    )
    report["models"] = {
        role: {"model_id": models[role]["model_id"],
               "requested_revision": models[role]["revision"],
               "resolved_revision": getattr(cfg, "_commit_hash", None),
               "model_type": cfg.model_type,
               "context_length": cfg.max_position_embeddings}
        for role, cfg in configs.items()
    }
    return report
