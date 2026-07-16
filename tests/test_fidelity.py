import numpy as np

from src.simulators.hardware import HardwareConfig
from src.simulators.tile_fidelity import TileFidelityModel


def config():
    return {
        "fidelity_model": {
            "num_timesteps": 8,
            "reference_noise_std": 0.023,
            "min_noise_std": 0.005,
            "max_noise_std": 0.080,
            "fidelity_classes": {
                "high": {"fraction": 0.25, "noise_multiplier_range": [0.7, 0.8]},
                "medium": {"fraction": 0.5, "noise_multiplier_range": [0.9, 1.1]},
                "low": {"fraction": 0.25, "noise_multiplier_range": [1.3, 1.5]},
            },
            "degradation": {
                "gradual_drift": {"enabled": True, "total_increase_range": [0.1, 0.2]},
                "thermal_variation": {"enabled": True, "correlation": 0.9, "standard_deviation_fraction": 0.03},
                "localized_fault": {
                    "enabled": True,
                    "num_affected_tiles": 1,
                    "onset_timestep_range": [3, 3],
                    "noise_increase_range": [0.4, 0.4],
                    "candidate_classes": ["high", "medium", "low"],
                    "make_unavailable": False,
                },
            },
        }
    }


def test_trace_is_deterministic_and_bounded():
    hardware = HardwareConfig(8, 2, 4, 4, 2)
    first = TileFidelityModel(hardware, config(), 42).generate()
    second = TileFidelityModel(hardware, config(), 42).generate()
    assert np.array_equal(first.noise_std, second.noise_std)
    assert first.noise_std.shape == (8, 8)
    assert first.noise_std.min() >= 0.005
    assert first.noise_std.max() <= 0.080
    assert first.faulted[-1].sum() == 1
