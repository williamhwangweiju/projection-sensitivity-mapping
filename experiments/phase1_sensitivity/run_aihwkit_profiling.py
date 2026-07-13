#!/usr/bin/env python3
"""Run GPT-2 AIHWKit projection-sensitivity profiling.

The runner builds fixed-length WikiText batches, executes the AIHWKit profiler,
and saves a compact JSON profile for Phase 2/3 mapping experiments.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import datasets
import torch
import transformers
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_repo_root(script_path: Path) -> Path:
    """Find the repository root from this script location."""
    for candidate in (script_path.resolve().parent, *script_path.resolve().parents):
        if (candidate / "src" / "profilers" / "aihwkit_profiler.py").is_file():
            return candidate
    raise RuntimeError("Could not find src/profilers/aihwkit_profiler.py.")


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.profilers.aihwkit_profiler import AIHWKITSensitivityProfiler


def load_config(path: Path) -> Dict[str, Any]:
    """Load the experiment YAML configuration."""
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML dictionary.")
    return config


def parse_max_tokens(value: Any) -> Optional[int]:
    """Return None for full-dataset evaluation, otherwise an integer cap."""
    return None if value is None else int(value)


def make_window(
    token_ids: Sequence[int],
    start: int,
    sequence_length: int,
    previous_end: int,
    pad_token_id: int,
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """Create one fixed-length causal-LM evaluation window."""
    end = min(start + sequence_length, len(token_ids))
    tokens = list(token_ids[start:end])
    target_length = min(end - previous_end, len(tokens))
    padding = sequence_length - len(tokens)

    input_ids = tokens + [pad_token_id] * padding
    attention_mask = [1] * len(tokens) + [0] * padding
    labels = list(input_ids)

    # Ignore overlapping context and padding so each token is scored once.
    for index in range(len(tokens) - target_length):
        labels[index] = -100
    for index in range(len(tokens), sequence_length):
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
    """Tokenize the configured dataset into fixed-length evaluation batches."""
    dataset_cfg = config["dataset"]

    dataset_name = str(dataset_cfg["name"])
    dataset_config = str(dataset_cfg["config"])
    dataset_split = str(dataset_cfg["split"])
    sequence_length = int(dataset_cfg["sequence_length"])
    stride = int(dataset_cfg["stride"])
    batch_size = int(dataset_cfg["batch_size"])
    max_tokens = parse_max_tokens(dataset_cfg["max_tokens"])
    separator = str(dataset_cfg["document_separator"])
    drop_incomplete = bool(dataset_cfg["drop_incomplete_final_sequence"])

    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    raw_dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    token_ids: List[int] = []
    for sample in raw_dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids.extend(tokenizer.encode(text + separator, add_special_tokens=False))
        if max_tokens is not None and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
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
            token_ids=token_ids,
            start=start,
            sequence_length=sequence_length,
            previous_end=previous_end,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        windows.append(window)
        predicted_tokens += valid_targets
        previous_end = end

        if end >= len(token_ids):
            break
        start += stride

    if not windows:
        raise ValueError("Dataset preprocessing produced no evaluation windows.")

    batches = [
        {
            key: torch.stack(
                [window[key] for window in windows[index : index + batch_size]]
            )
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
        "document_separator": separator,
        "drop_incomplete_final_sequence": drop_incomplete,
        "collected_tokens": len(token_ids),
        "num_windows": len(windows),
        "num_batches": len(batches),
        "predicted_tokens_per_pass": predicted_tokens,
    }
    return batches, metadata


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Package profiler outputs in the JSON structure used downstream."""
    if not results:
        raise ValueError("The profiler returned no projection results.")
    return {
        "digital_perplexity": float(results[0]["ppl_clean"]),
        "projections": [dict(result) for result in results],
    }


def package_versions() -> Dict[str, Any]:
    """Record software versions needed to reproduce the run."""
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
    """Write the sensitivity profile JSON."""
    experiment_cfg = config["experiment"]
    save_dir = Path(experiment_cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = str(experiment_cfg["results_filename_prefix"])
    output_path = save_dir / f"{prefix}_{timestamp}.json"

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


def run_phase1_analysis(results_file: Path) -> None:
    """Run analyze_phase1.py for the newly created profiling result."""
    analyze_script = REPO_ROOT / "experiments" / "phase1_sensitivity" / "analyze_phase1.py"
    if not analyze_script.is_file():
        raise FileNotFoundError(f"Phase 1 analyzer script not found: {analyze_script}")

    command = [
        sys.executable,
        str(analyze_script),
        "--results-file",
        str(results_file),
        "--output-dir",
        str(results_file.parent),
    ]
    subprocess.run(command, check=True)


def main(config_path: Path) -> Path:
    """Run the full Phase 1 sensitivity profiling workflow."""
    config = load_config(config_path)
    model_cfg = config["model"]

    model_name = str(model_cfg["name"])
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
        config=config,
        model=model,
        dataset_metadata=dataset_metadata,
        profiler=profiler,
        results=results,
    )

    print("\nAIHWKIT PROFILING COMPLETE")
    print(f"Digital FP32 perplexity: {results['digital_perplexity']:.6f}")
    print(f"Profiled projections: {len(results['projections'])}")
    print(f"Results saved to: {output_path}")
    print("\nRunning Phase 1 analysis...")
    run_phase1_analysis(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "phase1_sensitivity" / "lammie_2026.yaml",
    )
    args = parser.parse_args()
    main(args.config)
