"""Invariants of every shipped full-pipeline configuration."""
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mapping.sharding import count_projection_shards

# Architecture geometry for capacity arithmetic, keyed by model.name:
# (n_layer, n_embd, vocab_size).
MODEL_GEOMETRY = {
    "gpt2": (12, 768, 50257),
    "gpt2-medium": (24, 1024, 50257),
}


def full_pipeline_configs() -> list[Path]:
    paths = sorted(
        path
        for path in (REPO_ROOT / "configs/full_pipeline").glob("*.yaml")
        if "smoke" not in path.name
    )
    assert paths, "No full-pipeline configurations found."
    return paths


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize("config_path", full_pipeline_configs(), ids=lambda p: p.stem)
def test_config_has_no_hardcoded_digital_projection(config_path: Path):
    config = load(config_path)
    selection = config["digital_selection"]
    greedy = selection["greedy_marginal"]
    assert config["profiling"]["include_lm_head"] is True
    assert selection["forced_digital"] == []
    assert greedy["forced_digital"] == []
    assert selection["explicit_sets"] == {}
    assert greedy["enabled"] is True
    assert greedy["candidate_pool_size"] is None


@pytest.mark.parametrize("config_path", full_pipeline_configs(), ids=lambda p: p.stem)
def test_hardware_can_hold_initial_all_analog_candidate_set(config_path: Path):
    config = load(config_path)
    model_name = str(config["model"]["name"])
    assert model_name in MODEL_GEOMETRY, f"Add {model_name} to MODEL_GEOMETRY."
    n_layer, n_embd, vocab_size = MODEL_GEOMETRY[model_name]

    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    available_tiers = (
        int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    )

    # Per-block projections in canonical [out, in] coordinates.
    per_block = {
        "attn.c_attn": (3 * n_embd, n_embd),
        "attn.c_proj": (n_embd, n_embd),
        "mlp.c_fc": (4 * n_embd, n_embd),
        "mlp.c_proj": (n_embd, 4 * n_embd),
    }
    required = 0
    for block_index in range(n_layer):
        for role, (out_features, in_features) in per_block.items():
            required += count_projection_shards(
                f"block_{block_index}/{role}", out_features, in_features, tier_rows, tier_cols
            )
    if bool(config["profiling"]["include_lm_head"]):
        required += count_projection_shards(
            "lm_head", vocab_size, n_embd, tier_rows, tier_cols
        )
    assert required <= available_tiers, (
        f"{config_path.name}: all-analog candidate set needs {required} tiers "
        f"but the substrate provides {available_tiers}."
    )


@pytest.mark.parametrize("config_path", full_pipeline_configs(), ids=lambda p: p.stem)
def test_final_evaluation_consumes_automatic_greedy_points(config_path: Path):
    config = load(config_path)
    expected = "greedy_measured_gain_per_cost_per_macs_per_token"
    assert "greedy_step" in config["phase4"]["evaluate_budget_types"]
    assert expected in config["phase4"]["evaluate_selection_methods"]
