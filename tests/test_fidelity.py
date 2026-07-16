from __future__ import annotations

from pathlib import Path

import yaml

from src.simulators.tile_fidelity import generate_fidelity_trace


ROOT = Path(__file__).resolve().parents[1]


def test_fidelity_trace_is_deterministic_and_bounded() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/full_pipeline/gpt2_3dcim.yaml").read_text()
    )
    first = generate_fidelity_trace(config, 42)
    second = generate_fidelity_trace(config, 42)
    assert first.noise_std.shape == (120, 72)
    assert (first.noise_std == second.noise_std).all()
    assert first.noise_std.min() >= config["fidelity_model"]["min_noise_std"]
    assert first.noise_std.max() <= config["fidelity_model"]["max_noise_std"]
    assert len(first.fault_events) == 6
