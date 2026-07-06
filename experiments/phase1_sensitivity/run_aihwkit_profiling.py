#!/usr/bin/env python3
"""Run GPT-2 projection programming-noise sensitivity profiling.

The runner profiles only the blocks selected in YAML and writes compact,
plot-ready projection sensitivity results aligned with the IBM Unified Workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import datasets
import numpy as np
import torch
import transformers
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_repo_root(script_path: Path) -> Path:
    for candidate in (script_path.resolve().parent, *script_path.resolve().parents):
        if (candidate / "src" / "profilers" / "aihwkit_profiler.py").is_file():
            return candidate
    raise RuntimeError("Could not find src/profilers/aihwkit_profiler.py")

REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.profilers.aihwkit_profiler import AIHWKITSensitivityProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULTS = {
    "model_name": "gpt2",
    "device": "cpu",
    "dataset_name": "Salesforce/wikitext",
    "dataset_config": "wikitext-103-raw-v1",
    "dataset_split": "test",
    "sequence_length": 1024,
    "stride": 512,
    "batch_size": 1,
    "max_tokens": None,
    "document_separator": "\n\n",
    "drop_incomplete_final_sequence": True,
    "seed": 42,
    "save_dir": "./data/results",
    "results_prefix": "lammie_2026_aihwkit_stage1_2",
}

def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

def parse_max_tokens(value: Any) -> Optional[int]:
    if value is None or value == 0:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return int(value)

def resolve_device(value: Any) -> str:
    device = str(value or DEFAULTS["device"]).lower()
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; using CPU")
        return "cpu"
    if device.startswith("mps"):
        logger.warning("AIHWKIT does not use MPS tiles natively; using CPU")
        return "cpu"
    return device

def make_window(
    token_ids: Sequence[int],
    start: int,
    sequence_length: int,
    previous_end: int,
    pad_token_id: int,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    end = min(start + sequence_length, len(token_ids))
    actual = list(token_ids[start:end])
    target_length = min(end - previous_end, len(actual))
    padding = sequence_length - len(actual)

    input_ids = actual + [pad_token_id] * padding
    attention_mask = [1] * len(actual) + [0] * padding
    labels = list(input_ids)

    # Ignore overlap already scored by the previous window.
    for index in range(len(actual) - target_length):
        labels[index] = -100
    for index in range(len(actual), sequence_length):
        labels[index] = -100

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    valid_targets = int((batch["labels"][1:] != -100).sum().item())
    return batch, end, valid_targets

def build_batches(
    config: Mapping[str, Any],
    tokenizer: Any,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})

    dataset_name = dataset_cfg.get("name", DEFAULTS["dataset_name"])
    dataset_config = dataset_cfg.get("config", DEFAULTS["dataset_config"])
    dataset_split = dataset_cfg.get("split", DEFAULTS["dataset_split"])
    sequence_length = int(dataset_cfg.get("sequence_length", DEFAULTS["sequence_length"]))
    stride = int(dataset_cfg.get("stride", DEFAULTS["stride"]))
    batch_size = int(dataset_cfg.get("batch_size", DEFAULTS["batch_size"]))
    max_tokens = parse_max_tokens(dataset_cfg.get("max_tokens", DEFAULTS["max_tokens"]))
    separator = str(dataset_cfg.get("document_separator", DEFAULTS["document_separator"]))
    drop_incomplete = bool(dataset_cfg.get("drop_incomplete_final_sequence", DEFAULTS["drop_incomplete_final_sequence"]))

    logger.info("Loading %s/%s split=%s", dataset_name, dataset_config, dataset_split)
    raw_dataset = load_dataset(str(dataset_name), str(dataset_config), split=str(dataset_split))

    token_ids: List[int] = []
    for sample in raw_dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text + separator, add_special_tokens=False))
        if max_tokens is not None and len(token_ids) >= max_tokens:
            del token_ids[max_tokens:]
            break

    windows: List[Dict[str, torch.Tensor]] = []
    predicted_tokens = 0
    previous_end = 0
    start = 0

    while start < len(token_ids):
        remaining = len(token_ids) - start
        if remaining < sequence_length and drop_incomplete:
            break
        if remaining < 2:
            break

        window, end, valid_targets = make_window(
            token_ids, start, sequence_length, previous_end, int(tokenizer.pad_token_id)
        )
        windows.append(window)
        predicted_tokens += valid_targets
        previous_end = end

        if end >= len(token_ids):
            break
        start += stride

    batches = [
        {
            key: torch.stack([item[key] for item in windows[index:index + batch_size]])
            for key in ("input_ids", "attention_mask", "labels")
        }
        for index in range(0, len(windows), batch_size)
    ]

    metadata = {
        "name": dataset_name,
        "config": dataset_config,
        "split": dataset_split,
        "sequence_length": sequence_length,
        "stride": stride,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "collected_tokens": len(token_ids),
        "num_windows": len(windows),
        "num_batches": len(batches),
        "predicted_tokens_per_pass": predicted_tokens,
    }
    
    logger.info("Prepared %d tokens, %d windows, %d batches", len(token_ids), len(windows), len(batches))
    return batches, metadata

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Preserve sensitivity values and profiler diagnostics."""
    if not results:
        raise ValueError("The profiler returned no projection results.")

    return {
        "digital_perplexity": float(results[0]["ppl_clean"]),
        "projections": [dict(result) for result in results],
    }

def package_versions() -> Dict[str, Any]:
    import aihwkit
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "aihwkit": getattr(aihwkit, "__version__", None),
    }

def save_results(
    config: Mapping[str, Any],
    model: Any,
    dataset_metadata: Mapping[str, Any],
    profiler: AIHWKITSensitivityProfiler,
    results: Mapping[str, Any],
) -> Path:
    experiment_cfg = config.get("experiment", {})
    save_dir = Path(experiment_cfg.get("save_dir", DEFAULTS["save_dir"]))
    if not save_dir.is_absolute():
        save_dir = REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = experiment_cfg.get("results_filename_prefix", DEFAULTS["results_prefix"])
    output_path = save_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    payload = {
        "metadata": {
            "paper": "Heterogeneous Mapping for Analog In-Memory Computing Accelerators: A Unified Workflow",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software_versions": package_versions(),
            "model": {
                "name": model.config._name_or_path,
                "n_layer": model.config.n_layer,
                "n_embd": model.config.n_embd,
                "vocab_size": model.config.vocab_size,
                "dtype": str(next(model.parameters()).dtype),
            },
            "dataset": dict(dataset_metadata),
            "analog_configuration": profiler.analog_configuration(),
        },
        "requested_config": dict(config),
        "results": dict(results),
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return output_path

def main(config_path: Path) -> Path:
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    
    model_name = str(model_cfg.get("name", DEFAULTS["model_name"]))
    device = resolve_device(model_cfg.get("device", DEFAULTS["device"]))

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    batches, dataset_metadata = build_batches(config, tokenizer)

    profiler = AIHWKITSensitivityProfiler(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    projection_results = profiler.profile_all(batches)
    results = aggregate(projection_results)

    output_path = save_results(
        config,
        model,
        dataset_metadata,
        profiler,
        results,
    )

    print("\nAIHWKIT PROFILING COMPLETE")
    print(f"Digital FP32 perplexity: {results['digital_perplexity']:.6f}")
    print(f"Profiled projections: {len(results['projections'])}")
    print(f"Results saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "lammie_2026.yaml",
    )
    args = parser.parse_args()
    main(args.config)