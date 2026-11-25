import pytest

from acceleratellm.config import load_config
from acceleratellm.evaluation.speculative_theory import theory


def test_theory_uses_prefix_yield_and_measured_verification_cost():
    # Two rounds, accepted lengths 3 and 0, four committed outputs at EOS/budget.
    # E[K]+1=2.5. A round costs 4*1ms + 6ms = 10ms, vs 4ms per AR token.
    acceptance = {"4": {"accepted_tokens": 3, "verification_steps": 2,
                         "generated_tokens": 4, "proposed_tokens": 8}}
    result = theory(acceptance, {"mean_seconds_by_query": {"1": .001}},
                    {"mean_seconds_by_query": {"1": .004, "5": .006}})["4"]
    assert result["predicted_decode_tokens_per_second"] == pytest.approx(250)
    assert result["predicted_decode_speedup"] == pytest.approx(1)
    assert result["boundary_adjusted_decode_tokens_per_second"] == pytest.approx(200)
    # Treating 3/8 as IID alpha would produce a different and incorrect E[K].
    assert result["expected_output_per_round"] == 2.5
    assert result["classic_speedup_assuming_one_target_cost"] == pytest.approx(1.25)


def test_zero_acceptance_still_emits_target_correction():
    result = theory({"2": {"accepted_tokens": 0, "verification_steps": 10,
                            "generated_tokens": 10}},
                    {"mean_seconds_by_query": {"1": .001}},
                    {"mean_seconds_by_query": {"1": .004, "3": .004}})["2"]
    assert result["predicted_decode_speedup"] == pytest.approx(2/3)
    assert result["expected_output_per_round"] == 1


def test_speculative_config_is_bf16_fa2_with_pinned_qwen3_models():
    config = load_config("configs/experiments/speculative.yaml")
    assert config["runtime"]["attn_implementation"] == "flash_attention_2"
    assert config["runtime"]["dtype"] == "bfloat16"
    assert {spec["model_id"] for spec in config["models"].values()} == {"Qwen/Qwen3-8B-Base", "Qwen/Qwen3-0.6B-Base"}
    assert config["speculative"]["greedy_implementation"] == "device"
    assert all(len(spec["revision"]) == 40 for spec in config["models"].values())
