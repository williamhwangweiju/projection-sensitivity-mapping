#!/usr/bin/env python3
"""Run Phase 1 projection sensitivity profiling on calibration data."""
from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import file_sha256, git_commit, load_yaml, resolve_path, save_json
from src.common.dataset import build_causal_lm_batches
from src.profilers.aihwkit_profiler import AIHWKITSensitivityProfiler


def package_versions() -> dict[str, Any]:
    import aihwkit
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "aihwkit": getattr(aihwkit, "__version__", None),
    }


def main(config_path: Path) -> Path:
    config = load_yaml(config_path)
    model_name = str(config["model"]["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.float()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    batches, dataset_metadata = build_causal_lm_batches(config, tokenizer)
    profiler = AIHWKITSensitivityProfiler(model, config)
    projections = profiler.profile_all(batches)
    first = projections[0]
    payload = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": git_commit(REPO_ROOT),
            "config_path": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "software_versions": package_versions(),
            "model": {
                "name": model_name,
                "n_layer": int(model.config.n_layer),
                "n_embd": int(model.config.n_embd),
                "vocab_size": int(model.config.vocab_size),
            },
            "dataset": dataset_metadata,
            "analog_configuration": profiler.analog_configuration(),
        },
        "baseline": {"clean_nll": first["nll_clean"], "clean_ppl": first["ppl_clean"]},
        "mapping_sensitivity_field": "sensitivity_score_for_mapping",
        "mapping_sensitivity_unit": "delta_nll_noise",
        "projections": projections,
        "requested_config": config,
    }
    phase = config["phase1"]
    output_root = resolve_path(phase["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_root / f"{phase.get('results_filename_prefix', 'gpt2_hybrid_sensitivity')}_{timestamp}.json"
    save_json(output_path, payload)
    print(f"Phase 1 complete: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    args = parser.parse_args()
    main(args.config)
