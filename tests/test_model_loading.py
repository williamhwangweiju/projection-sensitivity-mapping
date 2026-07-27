"""Checkpoint resolution and canonical loading for the shared model loader."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from src.common.model_loading import load_model_and_tokenizer, resolve_model_source


def tiny_checkpoint(directory: Path) -> Path:
    config = GPT2Config(
        n_layer=2, n_embd=32, n_head=2, n_positions=64, vocab_size=256
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(directory)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.save_pretrained(directory)
    return directory


def test_resolve_falls_back_to_name_without_checkpoint():
    source, metadata = resolve_model_source({"model": {"name": "gpt2", "checkpoint": None}})
    assert source == "gpt2"
    assert metadata == {"model_name": "gpt2", "checkpoint": None, "loaded_from": "gpt2"}


def test_resolve_missing_checkpoint_is_an_error(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="model.checkpoint"):
        resolve_model_source({"model": {"name": "gpt2", "checkpoint": str(missing)}})


def test_load_from_checkpoint_directory(tmp_path):
    checkpoint = tiny_checkpoint(tmp_path / "checkpoint_final")
    config = {"model": {"name": "gpt2", "checkpoint": str(checkpoint), "device": "cpu"}}
    model, tokenizer, metadata = load_model_and_tokenizer(config, device="cpu")
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["model_name"] == "gpt2"
    assert model.config.n_layer == 2
    assert not model.training
    assert model.config.use_cache is False
    assert model.config.pad_token_id == tokenizer.pad_token_id
    assert next(model.parameters()).dtype == torch.float32
