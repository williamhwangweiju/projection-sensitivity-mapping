from src.mapping.digital_selection import (
    DigitalCandidate,
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
