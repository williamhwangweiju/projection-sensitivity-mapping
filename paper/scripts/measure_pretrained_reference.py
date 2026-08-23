#!/usr/bin/env python3
"""Measure the pretrained (Hugging Face) GPT-2 digital reference on the exact
Phase-4 evaluation windows (WikiText-103 test, seq 1024, stride 1024).

Uses the repo's own dataset builder and metric so the windowing is identical to
the archived Phase-4 runs (expected: 279 windows, 285,417 predicted tokens).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common.config import load_yaml  # noqa: E402
from src.common.dataset import build_causal_lm_batches  # noqa: E402
from src.common.metrics import evaluate_nll_ppl  # noqa: E402
from src.common.model_loading import load_model_and_tokenizer  # noqa: E402

device_name = sys.argv[1] if len(sys.argv) > 1 else ("mps" if torch.backends.mps.is_available() else "cpu")
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "paper/data/pretrained_digital_reference.json"

config = load_yaml(REPO / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
config["model"] = dict(config["model"])
config["model"]["checkpoint"] = None  # pretrained HF weights
config["model"]["device"] = device_name
device = torch.device(device_name)

t0 = time.time()
model, tokenizer, model_meta = load_model_and_tokenizer(config, device=device)
eval_config = dict(config)
eval_config["dataset"] = dict(config["evaluation_dataset"])
batches, dataset_meta = build_causal_lm_batches(eval_config, tokenizer)
print(f"windows={len(batches)} dataset_meta={dataset_meta}", flush=True)
nll, ppl, tokens = evaluate_nll_ppl(model, batches, device)
elapsed = time.time() - t0
result = {
    "model": "gpt2 (pretrained Hugging Face weights, no checkpoint)",
    "model_metadata": model_meta,
    "dataset": dataset_meta,
    "device": device_name,
    "nll": nll,
    "ppl": ppl,
    "predicted_tokens": tokens,
    "elapsed_s": elapsed,
}
out_path.write_text(json.dumps(result, indent=2, default=str))
print(json.dumps(result, indent=2, default=str))
