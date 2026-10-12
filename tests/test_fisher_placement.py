"""static_fisher placement and the additive Fisher-proxy driver."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.phase3_baselines.run_fisher_placement import (
    POLICY,
    place_static_fisher,
    proxy_profile_rows,
)
from scripts.run_fisher_proxy_analysis import FISHER_POLICY, derive_config, record_run
from src.evaluation.noise_materialization import read_placement_csv
from src.mapping.placement import place_shards
from src.mapping.sharding import build_shards
from src.simulators.tile_fidelity import TileFidelityTrace, save_trace


def profile_rows():
    return [
        {"projection_id": "p_sensitive", "out_features": 6, "in_features": 6, "sensitivity_score_for_mapping": 10.0},
        {"projection_id": "p_robust", "out_features": 6, "in_features": 6, "sensitivity_score_for_mapping": 1.0},
    ]


def proxy_rows():
    # The proxy inverts the measured ranking so the two placements must differ.
    return [
        {"projection_id": "p_sensitive", "fisher_score": 0.5},
        {"projection_id": "p_robust", "fisher_score": 20.0},
    ]


GEOMETRY = dict(tier_rows=4, tier_cols=4)
NOISE = [0.01, 0.02, 0.04, 0.08]


def test_proxy_profile_rows_swap_the_score_and_keep_geometry():
    rows = profile_rows()
    swapped = proxy_profile_rows(rows, proxy_rows())
    assert [row["projection_id"] for row in swapped] == ["p_sensitive", "p_robust"]
    assert [row["sensitivity_score_for_mapping"] for row in swapped] == [0.5, 20.0]
    assert [(row["out_features"], row["in_features"]) for row in swapped] == [(6, 6), (6, 6)]
    # The measured rows are copied, never mutated.
    assert [row["sensitivity_score_for_mapping"] for row in rows] == [10.0, 1.0]


def test_proxy_profile_rows_require_every_analog_projection():
    with pytest.raises(ValueError, match="p_robust"):
        proxy_profile_rows(profile_rows(), proxy_rows()[:1])


def test_static_fisher_places_by_proxy_importance_over_identical_shards():
    records = place_static_fisher(
        profile_rows(), proxy_rows(), noise=NOISE, available=[True] * 4, tiers_per_tile=2, **GEOMETRY
    )
    measured = build_shards(profile_rows(), digital_projection_ids=[], **GEOMETRY)
    assert {record.shard_id for record in records} == {shard.shard_id for shard in measured}
    assert {record.policy for record in records} == {POLICY}
    quietest = min(records, key=lambda r: (r.tile_noise_std, r.tile_id, r.tier_id))
    assert quietest.projection_id == "p_robust"
    assert quietest.sensitivity == 20.0
    # The measured-sensitivity placement makes the opposite choice.
    proposed = place_shards(
        measured, noise=NOISE, available=[True] * 4, tiers_per_tile=2,
        policy="static_sensitivity", timestep=0, seed=42,
    )
    assert min(proposed, key=lambda r: (r.tile_noise_std, r.tile_id, r.tier_id)).projection_id == "p_sensitive"


def test_derive_config_redirects_phases_and_pairs_against_static_fisher(tmp_path):
    base = {
        "experiment": {"seed": 42, "placement_seed": 42},
        "model": {"name": "gpt2"},
        "phase3": {"output_root": "data/results/phase3", "policies": ["random", "static_sensitivity"], "mapping_timestep": 0},
        "phase4": {"output_root": "data/results/phase4", "policies": ["random", "static_sensitivity"],
                   "method_policies": ["static_sensitivity"], "baseline_policies": ["random"], "timesteps": [0]},
    }
    derived = derive_config(
        base, output_root=tmp_path, phase3_policies=["static_sensitivity", FISHER_POLICY],
        phase4_policies=["static_sensitivity", FISHER_POLICY], trace_seed=43,
    )
    assert derived["phase3"]["policies"] == ["static_sensitivity", FISHER_POLICY]
    assert derived["phase3"]["output_root"] == str(tmp_path / "phase3")
    assert derived["phase4"]["output_root"] == str(tmp_path / "phase4")
    assert derived["phase4"]["policies"] == ["static_sensitivity", FISHER_POLICY]
    assert derived["phase4"]["method_policies"] == ["static_sensitivity"]
    assert derived["phase4"]["baseline_policies"] == [FISHER_POLICY]
    assert derived["experiment"] == {"seed": 43, "placement_seed": 43}
    assert derived["phase4"]["timesteps"] == [0] and derived["model"] == {"name": "gpt2"}
    # The base configuration is left untouched.
    assert base["phase3"]["policies"] == ["random", "static_sensitivity"]
    assert base["experiment"]["seed"] == 42


def test_record_run_merges_by_trace_seed(tmp_path):
    manifest = tmp_path / "runs.yaml"
    record_run(manifest, {"trace_seed": 43, "output_root": "b"})
    record_run(manifest, {"trace_seed": 41, "output_root": "a"})
    record_run(manifest, {"trace_seed": 43, "output_root": "c"})
    runs = yaml.safe_load(manifest.read_text())["runs"]
    assert [(run["trace_seed"], run["output_root"]) for run in runs] == [(41, "a"), (43, "c")]


def _synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    profile = tmp_path / "phase1.json"
    profile.write_text(json.dumps({
        "mapping_sensitivity_unit": "delta_nll_noise",
        "projections": profile_rows(),
    }))
    proxy = tmp_path / "proxy.json"
    proxy.write_text(json.dumps({"projections": proxy_rows(), "rank_correlation": {}}))
    noise = np.array([NOISE, NOISE], dtype=np.float32)
    trace = tmp_path / "trace.npz"
    save_trace(str(trace), TileFidelityTrace(
        noise_std=noise,
        fidelity_score=1.0 - noise,
        available=np.ones_like(noise, dtype=bool),
        faulted=np.zeros_like(noise, dtype=bool),
        class_code=np.zeros(4, dtype=np.int64),
        thermal_zone=np.zeros(4, dtype=np.int64),
        fault_onset=np.full(4, -1, dtype=np.int64),
    ))
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "experiment": {"seed": 7, "placement_seed": 7},
        "hardware": {"num_tiles": 4, "tiers_per_tile": 2, "tier_shape": {"rows": 4, "cols": 4}, "num_thermal_zones": 1},
        "phase2": {"fidelity_model": {"max_noise_std": 0.08}},
        "phase3": {"sensitivity_floor": 0.0, "output_root": str(tmp_path / "unused_phase3"),
                   "mapping_timestep": 0, "policies": ["random", "sequential", "hardware_only", "static_sensitivity"]},
        "phase4": {"output_root": str(tmp_path / "unused_phase4"), "policies": ["random", "static_sensitivity"],
                   "method_policies": ["static_sensitivity"], "baseline_policies": ["random"],
                   "timesteps": [0], "num_realizations": 1, "unavailable_noise_std": 0.08},
    }, sort_keys=False))
    return {"profile": profile, "proxy": proxy, "trace": trace, "config": config}


def _run_driver(inputs: dict[str, Path], output_root: Path, *extra: str) -> subprocess.CompletedProcess:
    command = [
        sys.executable, str(REPO_ROOT / "scripts/run_fisher_proxy_analysis.py"),
        "--config", str(inputs["config"]), "--phase1", str(inputs["profile"]),
        "--trace", str(inputs["trace"]), "--proxy-artifact", str(inputs["proxy"]),
        "--output-root", str(output_root), "--skip-phase4", *extra,
    ]
    return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)


def test_driver_places_validates_and_leaves_the_base_config_alone(tmp_path):
    inputs = _synthetic_inputs(tmp_path)
    before = inputs["config"].read_text()
    result = _run_driver(inputs, tmp_path / "run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Pipeline contracts validated successfully." in result.stdout
    assert inputs["config"].read_text() == before

    manifest = json.loads((tmp_path / "run/phase3/phase3_manifest_fisher.json").read_text())
    assert [item["policy"] for item in manifest["placements"]] == ["static_sensitivity", FISHER_POLICY]
    derived = yaml.safe_load((tmp_path / "run/fisher_runtime_config.yaml").read_text())
    assert derived["phase3"]["policies"] == ["static_sensitivity", FISHER_POLICY]
    assert derived["phase4"]["method_policies"] == ["static_sensitivity"]
    assert derived["phase4"]["baseline_policies"] == [FISHER_POLICY]

    fisher = read_placement_csv(tmp_path / "run/phase3/placement_static_fisher.csv")
    proposed = read_placement_csv(tmp_path / "run/phase3/placement_static_sensitivity.csv")
    assert {row["shard_id"] for row in fisher} == {row["shard_id"] for row in proposed}
    assert {row["policy"] for row in fisher} == {FISHER_POLICY}
    on_quietest = lambda rows: {row["projection_id"] for row in rows if int(row["tile_id"]) == 0}
    assert on_quietest(fisher) == {"p_robust"} and on_quietest(proposed) == {"p_sensitive"}

    # A frozen manifest can be extended instead of re-placing the base policies.
    reuse = _run_driver(
        inputs, tmp_path / "reuse", "--phase3-manifest", str(tmp_path / "run/phase3/phase3_manifest.json")
    )
    assert reuse.returncode == 0, reuse.stdout + reuse.stderr
    reused = json.loads((tmp_path / "reuse/phase3/phase3_manifest_fisher.json").read_text())
    assert [item["policy"] for item in reused["placements"]] == ["static_sensitivity", FISHER_POLICY]
    assert reused["base_phase3_manifest"].endswith("run/phase3/phase3_manifest.json")
