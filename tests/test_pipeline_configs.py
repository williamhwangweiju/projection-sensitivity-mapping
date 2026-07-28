"""Invariants of the shipped full-pipeline configurations."""
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
}


def full_pipeline_configs() -> list[Path]:
    paths = sorted(
        path
        for path in (REPO_ROOT / "configs/full_pipeline").glob("*.yaml")
        if "smoke" not in path.name
    )
    assert paths, "No full-pipeline configurations found."
    return paths


def all_configs() -> list[Path]:
    return sorted((REPO_ROOT / "configs/full_pipeline").glob("*.yaml"))


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.mark.parametrize("config_path", full_pipeline_configs(), ids=lambda p: p.stem)
def test_hardware_holds_the_all_analog_candidate_set(config_path: Path):
    config = load(config_path)
    model_name = str(config["model"]["name"])
    assert model_name in MODEL_GEOMETRY, f"Add {model_name} to MODEL_GEOMETRY."
    n_layer, n_embd, vocab_size = MODEL_GEOMETRY[model_name]

    tier_rows = int(config["hardware"]["tier_shape"]["rows"])
    tier_cols = int(config["hardware"]["tier_shape"]["cols"])
    available_tiers = (
        int(config["hardware"]["num_tiles"]) * int(config["hardware"]["tiers_per_tile"])
    )

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
        f"{config_path.name}: all-analog deployment needs {required} tiers "
        f"but the substrate provides {available_tiers}."
    )


@pytest.mark.parametrize("config_path", full_pipeline_configs(), ids=lambda p: p.stem)
def test_primary_config_profiles_the_full_candidate_universe(config_path: Path):
    config = load(config_path)
    assert config["profiling"]["include_lm_head"] is True
    n_layer = MODEL_GEOMETRY[str(config["model"]["name"])][0]
    assert sorted(config["profiling"]["profile_blocks"]) in ([], list(range(n_layer)))


@pytest.mark.parametrize("config_path", all_configs(), ids=lambda p: p.stem)
def test_policy_lists_are_coherent(config_path: Path):
    config = load(config_path)
    phase3_policies = [str(value) for value in config["phase3"]["policies"]]
    phase4_policies = [str(value) for value in config["phase4"]["policies"]]
    assert phase3_policies == phase4_policies
    methods = set(config["phase4"].get("method_policies", []))
    baselines = set(config["phase4"].get("baseline_policies", []))
    assert methods <= set(phase4_policies)
    assert baselines <= set(phase4_policies)
    assert not (methods & baselines)
    if "static_fisher" in phase3_policies:
        assert bool(config["profiling"]["proxy"]["enabled"]), (
            "static_fisher requires profiling.proxy.enabled"
        )
