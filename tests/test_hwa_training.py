"""Noise-wrapper semantics for Phase 0 hardware-aware training."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel

from src.training.hwa import (
    HWANoiseSettings,
    NoisyProjection,
    restore_generator_states,
    set_noise_enabled,
    snapshot_generator_states,
    unwrap_analog_candidates,
    wrap_analog_candidates,
)


def tiny_model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=128)
    model = GPT2LMHeadModel(config)
    model.config.use_cache = False
    return model.eval()


def settings_from(noise_std_range=(0.023, 0.023), clip_in_forward=True, exclude=()):
    return HWANoiseSettings.from_config(
        {
            "analog": {"clip_sigma": 2.5, "range_mode": "peak_to_peak"},
            "hwa_training": {
                "noise": {
                    "noise_std_range": list(noise_std_range),
                    "clip_in_forward": clip_in_forward,
                    "include_lm_head": True,
                    "exclude_projection_ids": list(exclude),
                }
            },
        }
    )


def batch(model, seed=1):
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0, model.config.vocab_size, (2, 16), generator=generator
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def test_noise_perturbs_forward_and_zero_noise_is_identity():
    model = tiny_model()
    inputs = batch(model)
    with torch.no_grad():
        clean_logits = model(**inputs).logits.clone()

    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    assert len(wrapped) == 2 * 4 + 1  # four roles per block plus lm_head
    with torch.no_grad():
        noisy_logits = model(**inputs).logits
    assert not torch.allclose(noisy_logits, clean_logits)
    unwrap_analog_candidates(model, wrapped)

    wrapped = wrap_analog_candidates(
        model, settings_from(noise_std_range=(0.0, 0.0), clip_in_forward=False), seed=42
    )
    with torch.no_grad():
        identity_logits = model(**inputs).logits
    assert torch.allclose(identity_logits, clean_logits, atol=1e-6)
    unwrap_analog_candidates(model, wrapped)


def test_noise_disabled_bypasses_wrapper():
    model = tiny_model()
    inputs = batch(model)
    with torch.no_grad():
        clean_logits = model(**inputs).logits.clone()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    set_noise_enabled(wrapped, False)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.allclose(logits, clean_logits, atol=0.0)
    unwrap_analog_candidates(model, wrapped)


def test_parameters_unchanged_by_noisy_forward_backward():
    model = tiny_model()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    snapshot = {name: param.detach().clone() for name, param in model.named_parameters()}
    loss = model(**batch(model)).loss
    loss.backward()
    for name, param in model.named_parameters():
        assert torch.equal(param.detach(), snapshot[name]), name
    unwrap_analog_candidates(model, wrapped)


def test_straight_through_clip_gradients_reach_clipped_weights():
    model = tiny_model()
    projection = model.transformer.h[0].attn.c_attn
    with torch.no_grad():
        # Push one weight far past the 2.5-sigma clip threshold.
        projection.weight[0, 0] = 100.0
    wrapped = wrap_analog_candidates(
        model, settings_from(noise_std_range=(0.0, 0.0)), seed=42
    )
    loss = model(**batch(model)).loss
    loss.backward()
    grad = projection.weight.grad
    assert grad is not None
    # A hard clamp would zero this coordinate's gradient; STE keeps it alive.
    assert float(grad[0, 0].abs()) > 0.0
    unwrap_analog_candidates(model, wrapped)


def test_tied_embedding_stays_clean_under_lm_head_noise():
    model = tiny_model()
    assert (
        model.lm_head.weight.data_ptr()
        == model.transformer.wte.weight.data_ptr()
    )
    inputs = batch(model)
    with torch.no_grad():
        clean_embeddings = model.transformer.wte(inputs["input_ids"]).clone()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    with torch.no_grad():
        model(**inputs)
        noisy_run_embeddings = model.transformer.wte(inputs["input_ids"])
    assert torch.equal(noisy_run_embeddings, clean_embeddings)
    unwrap_analog_candidates(model, wrapped)
    assert (
        model.lm_head.weight.data_ptr()
        == model.transformer.wte.weight.data_ptr()
    )


def test_short_training_loop_updates_clean_weights_and_unwraps():
    model = tiny_model().train()
    before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    original_classes = {
        projection_id: type(wrapper.wrapped)
        for projection_id, wrapper in wrapped.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch(model, seed=step)).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    unwrap_analog_candidates(model, wrapped)
    after = model.transformer.h[0].attn.c_attn.weight.detach()
    assert not torch.equal(before, after)
    for projection_id, wrapper in wrapped.items():
        restored = getattr(wrapper.unwrap_parent, wrapper.unwrap_attribute)
        assert isinstance(restored, original_classes[projection_id])
        assert not isinstance(restored, NoisyProjection)
    assert (
        model.lm_head.weight.data_ptr()
        == model.transformer.wte.weight.data_ptr()
    )


def test_generator_state_roundtrip_reproduces_noise():
    model = tiny_model()
    inputs = batch(model)
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    with torch.no_grad():
        model(**inputs)  # advance the generators
    states = {
        projection_id: wrapper.generator_state()
        for projection_id, wrapper in wrapped.items()
    }
    with torch.no_grad():
        first = model(**inputs).logits.clone()
    for projection_id, wrapper in wrapped.items():
        wrapper.set_generator_state(states[projection_id], torch.device("cpu"))
    with torch.no_grad():
        second = model(**inputs).logits
    assert torch.equal(first, second)
    unwrap_analog_candidates(model, wrapped)


def test_eval_passes_do_not_advance_training_noise_stream():
    inputs_model = tiny_model()
    inputs = batch(inputs_model)

    # Reference: two consecutive noisy forwards with no eval in between.
    reference = tiny_model()
    ref_wrapped = wrap_analog_candidates(reference, settings_from(), seed=42)
    with torch.no_grad():
        reference(**inputs)
        expected_second = reference(**inputs).logits.clone()
    unwrap_analog_candidates(reference, ref_wrapped)

    # Same model/seed, but an "eval" (extra noisy forwards) happens between
    # the two training forwards, bracketed by snapshot/restore.
    model = tiny_model()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    with torch.no_grad():
        model(**inputs)
        snapshot = snapshot_generator_states(wrapped)
        model(**inputs)  # eval-like noisy pass that would advance generators
        model(**inputs)
        restore_generator_states(wrapped, snapshot, torch.device("cpu"))
        second = model(**inputs).logits
    assert torch.equal(second, expected_second)
    unwrap_analog_candidates(model, wrapped)


def test_snapshot_before_first_forward_restores_fresh_generators():
    inputs_model = tiny_model()
    inputs = batch(inputs_model)

    reference = tiny_model()
    ref_wrapped = wrap_analog_candidates(reference, settings_from(), seed=42)
    with torch.no_grad():
        expected_first = reference(**inputs).logits.clone()
    unwrap_analog_candidates(reference, ref_wrapped)

    model = tiny_model()
    wrapped = wrap_analog_candidates(model, settings_from(), seed=42)
    snapshot = snapshot_generator_states(wrapped)  # all None: no forward yet
    with torch.no_grad():
        model(**inputs)  # advances the lazily created generators
        restore_generator_states(wrapped, snapshot, torch.device("cpu"))
        first = model(**inputs).logits
    assert torch.equal(first, expected_first)
    unwrap_analog_candidates(model, wrapped)


def test_exclude_projection_ids_are_not_wrapped():
    model = tiny_model()
    wrapped = wrap_analog_candidates(
        model, settings_from(exclude=("lm_head",)), seed=42
    )
    assert "lm_head" not in wrapped
    assert not isinstance(model.lm_head, NoisyProjection)
    unwrap_analog_candidates(model, wrapped)
