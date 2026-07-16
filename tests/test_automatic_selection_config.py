from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_primary_config_has_no_hardcoded_digital_projection():
    with (REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    selection = config["digital_selection"]
    greedy = selection["greedy_marginal"]
    assert config["profiling"]["include_lm_head"] is True
    assert selection["forced_digital"] == []
    assert greedy["forced_digital"] == []
    assert selection["explicit_sets"] == {}
    assert greedy["enabled"] is True
    assert greedy["candidate_pool_size"] is None


def test_primary_hardware_can_hold_initial_all_analog_candidate_set():
    with (REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    available_tiers = (
        int(config["hardware"]["num_tiles"])
        * int(config["hardware"]["tiers_per_tile"])
    )
    # GPT-2 Small: 480 transformer shards + 198 tied LM-head shards.
    assert available_tiers >= 678


def test_final_evaluation_consumes_automatic_greedy_points():
    with (REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    expected = "greedy_measured_gain_per_cost_per_macs_per_token"
    assert config["phase4"]["evaluate_budget_types"] == ["greedy_step"]
    assert config["phase4"]["evaluate_selection_methods"] == [expected]
    assert config["phase5"]["evaluate_budget_types"] == ["greedy_step"]
    assert config["phase5"]["evaluate_selection_methods"] == [expected]
