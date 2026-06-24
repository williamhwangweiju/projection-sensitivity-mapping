#!/usr/bin/env python3
"""Run the Lammie 2026 Stage 1-2 AIHWKIT GPT-2 experiment.

This driver is designed for ``AIHWKITLammieSensitivityProfiler`` from
``src/profilers/aihwkit_profiler.py``.

Experiment:
- GPT-2-small (12 decoder blocks)
- WikiText-103 test split
- 49 weight projections
- exactly one analog projection at a time
- all remaining projections in FP32
- 10 fixed programming-noise realizations per projection
- sensitivity = mean analog perplexity - digital perplexity

The profiler owns all analog behavior: clipping, 512 x 512 physical tiling,
ADC/DAC resolution, programming noise, module replacement, weight restoration,
and token-weighted perplexity. This driver only loads data/model, invokes the
profiler, aggregates results, and saves experiment metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import yaml
import datasets
import transformers
from datasets import load_dataset as hf_load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def find_repo_root(script_path: Path) -> Path:
    """Find the repository containing src/profilers/aihwkit_profiler.py."""
    resolved = script_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        profiler_path = (
            candidate / "src" / "profilers" / "aihwkit_profiler.py"
        )
        if profiler_path.is_file():
            return candidate

    raise RuntimeError(
        "Could not locate the repository root. Place this script somewhere "
        "inside the repository containing "
        "'src/profilers/aihwkit_profiler.py'."
    )


REPO_ROOT = find_repo_root(Path(__file__))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.profilers.aihwkit_profiler import (  # noqa: E402
    AIHWKITLammieSensitivityProfiler,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the YAML root object."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    if loaded is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    if not isinstance(loaded, dict):
        raise TypeError("The YAML root must be a mapping.")

    return loaded


def set_global_seed(seed: int) -> None:
    """Seed model construction and all host-side preprocessing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    """Resolve the requested device to an AIHWKIT-supported device."""
    requested = str(requested_device).strip().lower()

    if requested == "mps" or requested.startswith("mps:"):
        raise ValueError(
            "AIHWKIT analog tiles do not support Apple's MPS backend. "
            "Use model.device='cpu' on macOS."
        )

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA was requested but PyTorch cannot access CUDA; using CPU."
            )
            return "cpu"
        return requested

    if requested != "cpu":
        raise ValueError(
            f"Unsupported device {requested_device!r}. Use 'cpu' or 'cuda'."
        )

    return "cpu"


def normalize_wikitext_config(
    config: Mapping[str, Any],
) -> Tuple[str, str, str]:
    """Support both the old and corrected YAML dataset layouts.

    Correct layout:
        name: Salesforce/wikitext
        config: wikitext-103-raw-v1
        split: test

    Old layout supported for compatibility:
        name: wikitext
        split: wikitext-103

    The old ``split: wikitext-103`` value is interpreted as the dataset
    configuration, while the actual evaluation split becomes ``test``.
    """
    dataset_cfg = config.get("dataset", {})
    if not isinstance(dataset_cfg, Mapping):
        raise TypeError("config['dataset'] must be a mapping.")

    dataset_name = str(
        dataset_cfg.get("name", "Salesforce/wikitext")
    ).strip()
    dataset_config = dataset_cfg.get("config")
    dataset_split = str(dataset_cfg.get("split", "test")).strip()

    if dataset_name == "wikitext":
        dataset_name = "Salesforce/wikitext"

    legacy_config_names = {
        "wikitext-103": "wikitext-103-raw-v1",
        "wikitext-103-v1": "wikitext-103-v1",
        "wikitext-103-raw-v1": "wikitext-103-raw-v1",
    }

    if dataset_config is None and dataset_split in legacy_config_names:
        dataset_config = legacy_config_names[dataset_split]
        dataset_split = str(
            dataset_cfg.get("eval_split", "test")
        ).strip()

    if dataset_config is None:
        dataset_config = "wikitext-103-raw-v1"
    else:
        dataset_config = str(dataset_config).strip()

    if dataset_name not in {"Salesforce/wikitext"}:
        raise ValueError(
            "This paper reconstruction requires WikiText-103. "
            f"Received dataset.name={dataset_name!r}."
        )

    allowed_configs = {
        "wikitext-103-raw-v1",
        "wikitext-103-v1",
    }
    if dataset_config not in allowed_configs:
        raise ValueError(
            "Expected WikiText-103 configuration "
            f"{sorted(allowed_configs)}, received {dataset_config!r}."
        )

    if dataset_split not in {"train", "validation", "test"}:
        raise ValueError(
            "dataset.split must be 'train', 'validation', or 'test'. "
            f"Received {dataset_split!r}."
        )

    return dataset_name, dataset_config, dataset_split


def parse_max_tokens(value: Any) -> Optional[int]:
    """Parse max_tokens, where null/0/'all' means no artificial cap."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "all",
        "none",
        "null",
    }:
        return None

    parsed = int(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError("dataset.max_tokens cannot be negative.")
    return parsed


def load_wikitext103_batches(
    config: Mapping[str, Any],
    tokenizer,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Tokenize WikiText-103 into reusable, contiguous fixed-length batches.

    A Python list is returned deliberately: the profiler performs hundreds of
    complete dataset passes and rejects one-shot generators.
    """
    dataset_cfg = config.get("dataset", {})
    dataset_name, dataset_config, dataset_split = (
        normalize_wikitext_config(config)
    )

    sequence_length = int(dataset_cfg.get("sequence_length", 1024))
    batch_size = int(dataset_cfg.get("batch_size", 1))
    max_tokens = parse_max_tokens(dataset_cfg.get("max_tokens"))

    if sequence_length <= 1:
        raise ValueError("dataset.sequence_length must be at least 2.")
    if batch_size <= 0:
        raise ValueError("dataset.batch_size must be positive.")
    if max_tokens is not None and max_tokens < sequence_length:
        raise ValueError(
            "dataset.max_tokens must be at least dataset.sequence_length, "
            "or null/'all' for the complete selected split."
        )

    if max_tokens is not None:
        logger.warning(
            "WikiText-103 is capped at %d tokens. With sequence_length=%d, "
            "this provides at most %d full sequences.",
            max_tokens,
            sequence_length,
            max_tokens // sequence_length,
        )

    logger.info(
        "Loading %s, configuration=%s, split=%s",
        dataset_name,
        dataset_config,
        dataset_split,
    )
    raw_dataset = hf_load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
    )

    token_ids: List[int] = []
    for sample in raw_dataset:
        text = sample.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        encoded = tokenizer.encode(
            text + "\n\n",
            add_special_tokens=False,
        )
        token_ids.extend(encoded)

        if max_tokens is not None and len(token_ids) >= max_tokens:
            del token_ids[max_tokens:]
            break

    usable_tokens = (
        len(token_ids) // sequence_length
    ) * sequence_length
    if usable_tokens == 0:
        raise ValueError(
            "Tokenization produced no complete evaluation sequence."
        )

    dropped_tokens = len(token_ids) - usable_tokens
    if dropped_tokens:
        logger.info(
            "Dropping %d trailing token(s) to form fixed %d-token sequences.",
            dropped_tokens,
            sequence_length,
        )

    tokens = torch.tensor(
        token_ids[:usable_tokens],
        dtype=torch.long,
    )
    blocks = tokens.reshape(-1, sequence_length)
    masks = torch.ones_like(blocks, dtype=torch.long)

    batches: List[Dict[str, torch.Tensor]] = []
    for start in range(0, blocks.shape[0], batch_size):
        stop = min(start + batch_size, blocks.shape[0])
        batches.append(
            {
                "input_ids": blocks[start:stop],
                "attention_mask": masks[start:stop],
            }
        )

    num_sequences = int(blocks.shape[0])
    predicted_tokens = num_sequences * (sequence_length - 1)

    metadata: Dict[str, Any] = {
        "hf_dataset": dataset_name,
        "hf_config": dataset_config,
        "split": dataset_split,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "requested_max_tokens": max_tokens,
        "evaluated_tokens": int(usable_tokens),
        "predicted_tokens_per_full_pass": int(predicted_tokens),
        "num_sequences": num_sequences,
        "num_batches": len(batches),
        "dropped_trailing_tokens": int(dropped_tokens),
    }

    logger.info(
        "Prepared WikiText-103: %d tokens, %d sequences, %d batches, "
        "%d next-token targets per complete pass.",
        metadata["evaluated_tokens"],
        metadata["num_sequences"],
        metadata["num_batches"],
        metadata["predicted_tokens_per_full_pass"],
    )
    return batches, metadata


def validate_gpt2_small(model, sequence_length: int) -> None:
    """Ensure that the loaded model matches the paper's GPT-2-small target."""
    num_layers = int(getattr(model.config, "n_layer", -1))
    hidden_size = int(getattr(model.config, "n_embd", -1))
    vocab_size = int(getattr(model.config, "vocab_size", -1))
    max_positions = int(getattr(model.config, "n_positions", -1))

    expected = {
        "n_layer": (num_layers, 12),
        "n_embd": (hidden_size, 768),
        "vocab_size": (vocab_size, 50257),
    }
    mismatches = [
        f"{name}={actual}, expected {target}"
        for name, (actual, target) in expected.items()
        if actual != target
    ]
    if mismatches:
        raise ValueError(
            "The loaded model is not standard GPT-2-small:\n- "
            + "\n- ".join(mismatches)
        )

    if sequence_length > max_positions:
        raise ValueError(
            f"dataset.sequence_length={sequence_length} exceeds GPT-2's "
            f"context limit of {max_positions}."
        )


def aggregate_projection_results(
    projection_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build block summaries and a global sensitivity ranking."""
    if len(projection_results) != 49:
        raise RuntimeError(
            f"Expected 49 projection results, received "
            f"{len(projection_results)}."
        )

    clean_values = {
        float(result["ppl_clean"])
        for result in projection_results
    }
    if len(clean_values) != 1:
        raise RuntimeError(
            "The profiler returned inconsistent digital baselines."
        )
    digital_perplexity = next(iter(clean_values))

    block_values: Dict[str, List[float]] = {}
    for result in projection_results:
        block_id = str(result["block_id"])
        sensitivity = float(result["sensitivity_mean"])
        block_values.setdefault(block_id, []).append(sensitivity)

    block_averages: Dict[str, Dict[str, Any]] = {}
    for block_id, values in block_values.items():
        values_array = np.asarray(values, dtype=np.float64)
        block_averages[block_id] = {
            "mean_delta_ppl": float(values_array.mean()),
            "std_delta_ppl": float(values_array.std(ddof=0)),
            "num_projections": int(values_array.size),
        }

    ranking = sorted(
        (
            {
                "rank": 0,
                "block_id": str(result["block_id"]),
                "proj_name": str(result["proj_name"]),
                "sensitivity_mean": float(
                    result["sensitivity_mean"]
                ),
                "sensitivity_std": float(
                    result["sensitivity_std"]
                ),
            }
            for result in projection_results
        ),
        key=lambda item: item["sensitivity_mean"],
        reverse=True,
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    return {
        "digital_perplexity": digital_perplexity,
        "projections": projection_results,
        "block_averages": block_averages,
        "sensitivity_ranking": ranking,
    }


def run_profiling(
    model,
    tokenizer,
    batches: List[Dict[str, torch.Tensor]],
    config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Invoke the profiler exactly once for the complete 49-projection run."""
    profiler = AIHWKITLammieSensitivityProfiler(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        seed=int(config.get("experiment", {}).get("seed", 42)),
    )

    logger.info(
        "Starting 49 projections x %d programming realizations.",
        profiler.num_seeds,
    )

    # Critical: profile_all() computes the digital FP32 baseline once and passes
    # it into every profile_projection() call.
    projection_results = profiler.profile_all(dataset=batches)

    return aggregate_projection_results(projection_results)


def package_versions() -> Dict[str, Optional[str]]:
    """Record software versions that materially affect reproducibility."""
    try:
        import aihwkit
        aihwkit_version = getattr(aihwkit, "__version__", None)
    except Exception:
        aihwkit_version = None

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "aihwkit": aihwkit_version,
    }


def build_metadata(
    config: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    device: str,
    model,
) -> Dict[str, Any]:
    """Build a self-contained record of the executed methodology."""
    profiling_cfg = config.get("profiling", {})
    model_cfg = config.get("model", {})

    return {
        "paper": (
            "Lammie 2026, Heterogeneous Mapping for Analog In-Memory "
            "Computing Accelerators: A Unified Workflow"
        ),
        "stage": "Stages 1-2 projection precision sensitivity profiling",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": package_versions(),
        "model": {
            "requested_name": model_cfg.get("name", "gpt2"),
            "model_type": getattr(model.config, "model_type", None),
            "n_layer": int(model.config.n_layer),
            "n_embd": int(model.config.n_embd),
            "vocab_size": int(model.config.vocab_size),
            "n_positions": int(model.config.n_positions),
            "dtype": str(next(model.parameters()).dtype),
        },
        "device": device,
        "dataset": dict(dataset_metadata),
        "projection_count": 49,
        "stage1_aimc_config": {
            "weight_clipping_std_multiple": float(
                profiling_cfg.get(
                    "weight_clipping_std_multiple",
                    2.5,
                )
            ),
            "crossbar_rows": int(
                profiling_cfg.get("crossbar_rows", 512)
            ),
            "crossbar_cols": int(
                profiling_cfg.get("crossbar_cols", 512)
            ),
            "adc_dac_bits": int(
                profiling_cfg.get("adc_dac_bits", 8)
            ),
            "programming_noise_std": float(
                profiling_cfg.get("programming_noise_std", 0.023)
            ),
            "programming_noise_range_mode": str(
                profiling_cfg.get(
                    "programming_noise_range_mode",
                    "absmax",
                )
            ),
            "t_inference_seconds": float(
                profiling_cfg.get("t_inference_seconds", 0.0)
            ),
            "input_bound": float(
                profiling_cfg.get("input_bound", 1.0)
            ),
            "output_bound": float(
                profiling_cfg.get("output_bound", 12.0)
            ),
            "weight_scaling_omega": float(
                profiling_cfg.get("weight_scaling_omega", 1.0)
            ),
            "read_noise_enabled": False,
            "drift_enabled": False,
            "short_term_weight_noise_enabled": False,
        },
        "stage2_config": {
            "num_noise_realizations": int(
                profiling_cfg.get("num_seeds", 10)
            ),
            "metric": "delta perplexity",
            "one_projection_analog_at_a_time": True,
            "all_other_projections_fp32": True,
            "digital_baseline_computed_once": True,
        },
    }


def save_results(
    results: Mapping[str, Any],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Save metadata, resolved YAML, and experiment results as JSON."""
    save_dir = Path(
        config.get("experiment", {}).get(
            "save_dir",
            "./results/lammie_2026",
        )
    )
    if not save_dir.is_absolute():
        save_dir = REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = (
        save_dir
        / f"lammie_2026_aihwkit_stage1_2_{timestamp}.json"
    )

    payload = {
        "metadata": dict(metadata),
        "resolved_config": dict(config),
        "results": dict(results),
    }

    with results_file.open("w", encoding="utf-8") as stream:
        json.dump(
            payload,
            stream,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )

    return results_file


def block_sort_key(block_id: str) -> Tuple[int, int]:
    """Sort block_0...block_11 numerically and place head last."""
    if block_id.startswith("block_"):
        return (0, int(block_id.split("_", maxsplit=1)[1]))
    return (1, 0)


def print_header(
    config: Mapping[str, Any],
    device: str,
    dataset_spec: Tuple[str, str, str],
) -> None:
    """Print the effective experiment configuration."""
    profiling_cfg = config.get("profiling", {})
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    dataset_name, dataset_config, dataset_split = dataset_spec

    print("\n" + "=" * 78)
    print("LAMMIE 2026 STAGES 1-2: AIHWKIT GPT-2 SENSITIVITY PROFILING")
    print("=" * 78)
    print(
        f"Model: {model_cfg.get('name', 'gpt2')} "
        "(GPT-2-small, 12 decoder blocks)"
    )
    print(
        f"Dataset: {dataset_name}/{dataset_config}, "
        f"split={dataset_split}"
    )
    print(f"Device: {device}")
    print()
    print("Stage 1 AIMC configuration")
    print(
        "  Weight clipping: +/-"
        f"{profiling_cfg.get('weight_clipping_std_multiple', 2.5)} "
        "x layer RMS"
    )
    print(
        "  Crossbar tiling: "
        f"{profiling_cfg.get('crossbar_rows', 512)} x "
        f"{profiling_cfg.get('crossbar_cols', 512)}"
    )
    print(
        "  ADC/DAC quantization: approximately "
        f"{profiling_cfg.get('adc_dac_bits', 8)} bit"
    )
    print(
        "  Programming noise: sigma_w="
        f"{profiling_cfg.get('programming_noise_std', 0.023)}, "
        "range_mode="
        f"{profiling_cfg.get('programming_noise_range_mode', 'absmax')}"
    )
    print(
        "  t_inference: "
        f"{profiling_cfg.get('t_inference_seconds', 0.0)} seconds"
    )
    print()
    print("Stage 2 sensitivity analysis")
    print(
        f"  Sequence length: "
        f"{dataset_cfg.get('sequence_length', 1024)}"
    )
    print(
        f"  Batch size: {dataset_cfg.get('batch_size', 1)}"
    )
    print(
        f"  Max tokens: {dataset_cfg.get('max_tokens', 'all')}"
    )
    print(
        f"  Noise realizations: "
        f"{profiling_cfg.get('num_seeds', 10)}"
    )
    print("  Projections: 49 (12 blocks x 4 + lm_head)")
    print("  Clean FP32 perplexity: computed once")
    print("=" * 78 + "\n")


def main(config_path: Path) -> Path:
    """Execute the complete experiment and return the result path."""
    config = load_config(config_path)

    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "gpt2"))
    if model_name != "gpt2":
        raise ValueError(
            "This reconstruction targets GPT-2-small ('gpt2'). "
            f"Received model.name={model_name!r}."
        )

    seed = int(config.get("experiment", {}).get("seed", 42))
    set_global_seed(seed)

    device = resolve_device(model_cfg.get("device", "cpu"))
    dataset_spec = normalize_wikitext_config(config)
    print_header(config, device, dataset_spec)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model in FP32: %s", model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    sequence_length = int(
        config.get("dataset", {}).get("sequence_length", 1024)
    )
    validate_gpt2_small(model, sequence_length)

    batches, dataset_metadata = load_wikitext103_batches(
        config,
        tokenizer,
    )

    # The profiler moves the model to the selected device in FP32.
    results = run_profiling(
        model=model,
        tokenizer=tokenizer,
        batches=batches,
        config=config,
        device=device,
    )

    metadata = build_metadata(
        config=config,
        dataset_metadata=dataset_metadata,
        device=device,
        model=model,
    )
    results_file = save_results(
        results=results,
        metadata=metadata,
        config=config,
    )

    print("\n" + "=" * 78)
    print("LAMMIE 2026 AIHWKIT PROFILING COMPLETE")
    print("=" * 78)
    print(
        f"Digital FP32 perplexity: "
        f"{results['digital_perplexity']:.6f}"
    )
    print(f"Results: {results_file}")
    print()
    print("Block-level mean delta PPL")
    for block_id in sorted(
        results["block_averages"],
        key=block_sort_key,
    ):
        stats = results["block_averages"][block_id]
        print(
            f"  {block_id:>8s}: "
            f"{stats['mean_delta_ppl']:.6f} +/- "
            f"{stats['std_delta_ppl']:.6f}"
        )

    print()
    print("Five most sensitive projections")
    for item in results["sensitivity_ranking"][:5]:
        print(
            f"  {item['rank']:>2d}. "
            f"{item['block_id']}/{item['proj_name']}: "
            f"{item['sensitivity_mean']:.6f} +/- "
            f"{item['sensitivity_std']:.6f}"
        )
    print("=" * 78 + "\n")

    return results_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Lammie 2026 Stage 1-2 GPT-2 AIHWKIT sensitivity "
            "profiling"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "lammie_2026.yaml",
        help="Path to the Lammie 2026 YAML configuration.",
    )
    arguments = parser.parse_args()

    try:
        main(arguments.config)
    except KeyboardInterrupt:
        logger.error("Experiment interrupted by the user.")
        raise SystemExit(130)
    except Exception:
        logger.exception("Experiment failed.")
        raise SystemExit(1)
