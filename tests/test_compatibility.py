import pytest

from acceleratellm.models.compatibility import check_tokenizers, tokenizer_fingerprint


def test_identical_tokenizers(tokenizer):
    report = check_tokenizers(tokenizer, tokenizer, 32, 32)
    assert report["compatible"]
    assert len(report["tokenizer_sha256"]) == 64


def test_detect_normalization_difference(tokenizer):
    import copy
    from tokenizers.normalizers import Lowercase
    draft = copy.deepcopy(tokenizer)
    draft.backend_tokenizer.normalizer = Lowercase()
    assert tokenizer.get_vocab() == draft.get_vocab()
    assert tokenizer_fingerprint(tokenizer) != tokenizer_fingerprint(draft)
    with pytest.raises(ValueError, match="semantics"):
        check_tokenizers(tokenizer, draft, 32, 32)


def test_detect_output_vocab_mismatch(tokenizer):
    with pytest.raises(ValueError, match="output vocab"):
        check_tokenizers(tokenizer, tokenizer, 32, 64)
