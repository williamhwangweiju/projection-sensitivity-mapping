import pytest

from src.mapping.digital_selection import (
    DigitalCandidate,
    candidates_from_profile,
    operating_point_record,
    select_by_count,
    select_by_fraction,
)


def candidates():
    return [
        DigitalCandidate("a", sensitivity=10.0, parameter_count=100, macs_per_token=100),
        DigitalCandidate("b", sensitivity=8.0, parameter_count=10, macs_per_token=10),
        DigitalCandidate("lm_head", sensitivity=2.0, parameter_count=1000, macs_per_token=1000, tied_to_embedding=True),
    ]


def proxy_candidates():
    return [
        DigitalCandidate("a", sensitivity=10.0, parameter_count=100, macs_per_token=100, fisher_score=1.0, magnitude_score=0.5),
        DigitalCandidate("b", sensitivity=8.0, parameter_count=10, macs_per_token=10, fisher_score=3.0, magnitude_score=0.1),
        DigitalCandidate("c", sensitivity=2.0, parameter_count=1000, macs_per_token=1000, fisher_score=2.0, magnitude_score=0.9),
    ]


def test_forced_digital_anchor_is_preserved():
    selected = select_by_count(candidates(), method="sensitivity_rank", count=2, forced=["lm_head"])
    assert selected == ["a", "lm_head"]


def test_cost_normalized_selection_changes_choice():
    selected = select_by_count(candidates(), method="sensitivity_per_parameter", count=1)
    assert selected == ["b"]


def test_operating_point_reports_cost_fractions():
    point = operating_point_record(
        candidates(),
        method="sensitivity_rank",
        budget_type="projection_count",
        budget_value=1,
        digital_projection_ids=["a"],
    )
    assert point["digital_projection_count"] == 1
    assert point["analog_projection_count"] == 2
    assert abs(point["digital_parameter_fraction"] - 100 / 1110) < 1e-12
    assert point["digital_set_id"].startswith("digital_")


def test_fraction_budget_is_monotonic():
    small = select_by_fraction(candidates(), method="sensitivity_rank", fraction=0.05, cost_field="parameter_count")
    large = select_by_fraction(candidates(), method="sensitivity_rank", fraction=0.50, cost_field="parameter_count")
    assert set(small).issubset(set(large))


def test_fisher_and_magnitude_rankings_differ_from_measured():
    assert select_by_count(proxy_candidates(), method="fisher_rank", count=1) == ["b"]
    assert select_by_count(proxy_candidates(), method="magnitude_rank", count=1) == ["c"]
    assert select_by_count(proxy_candidates(), method="sensitivity_rank", count=1) == ["a"]


def test_fisher_per_mac_normalizes_by_cost():
    # c has fisher 2.0 over 1000 MACs; b has 3.0 over 10 MACs.
    assert select_by_count(proxy_candidates(), method="fisher_per_mac", count=1) == ["b"]


def test_proxy_methods_require_proxy_artifact():
    with pytest.raises(ValueError, match="proxy"):
        candidates()[0].score("fisher_rank")
    with pytest.raises(ValueError, match="proxy"):
        candidates()[0].score("magnitude_rank")


def test_candidates_from_profile_merges_proxy_sidecar():
    payload = {
        "projections": [
            {
                "projection_id": "a",
                "sensitivity_score_for_mapping": 1.5,
                "parameter_count": 10,
                "macs_per_token": 10,
            }
        ]
    }
    proxies = {"a": {"fisher_score": 4.0, "magnitude_score": 0.25}}
    merged = candidates_from_profile(payload, proxies)
    assert merged[0].fisher_score == 4.0
    assert merged[0].magnitude_score == 0.25
    without = candidates_from_profile(payload)
    assert without[0].fisher_score is None
