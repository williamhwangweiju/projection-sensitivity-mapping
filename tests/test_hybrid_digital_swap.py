"""Unit tests for HybridAnalogModel's temporary digital-swap machinery."""
from types import SimpleNamespace

import pytest


def _make_hybrid():
    pytest.importorskip("torch")
    pytest.importorskip("aihwkit")
    from src.evaluation.aihwkit_gpt2 import HybridAnalogModel

    parent = SimpleNamespace()
    digital_module = object()
    analog_module = object()
    parent.proj = analog_module
    handle = SimpleNamespace(parent=parent, attribute="proj")

    hybrid = HybridAnalogModel.__new__(HybridAnalogModel)
    hybrid.digital_projection_ids = frozenset({"perma_digital"})
    hybrid.states = {"p": SimpleNamespace(handle=handle, analog_module=analog_module)}
    hybrid.original_modules = {"p": digital_module}
    return hybrid, parent, digital_module, analog_module


def test_temporarily_digital_swaps_and_restores():
    hybrid, parent, digital_module, analog_module = _make_hybrid()
    with hybrid.temporarily_digital(["p", "perma_digital"]) as swapped:
        assert swapped == ["p"]
        assert parent.proj is digital_module
    assert parent.proj is analog_module


def test_temporarily_digital_restores_on_exception():
    hybrid, parent, _, analog_module = _make_hybrid()
    with pytest.raises(RuntimeError):
        with hybrid.temporarily_digital(["p"]):
            raise RuntimeError("evaluation failed")
    assert parent.proj is analog_module


def test_swap_to_digital_rejects_unknown_projection():
    hybrid, _, _, _ = _make_hybrid()
    with pytest.raises(KeyError):
        hybrid.swap_to_digital(["not_a_projection"])
