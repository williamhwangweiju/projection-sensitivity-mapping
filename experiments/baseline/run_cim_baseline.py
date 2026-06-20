"""
End-to-end IBM 3D-CIM baseline for a GPT-style decoder-only transformer.

This script:
1. Builds a decoder-only transformer.
2. Maps it onto the IBM 3D-CIM accelerator.
3. Traces autoregressive decoding.
4. Schedules execution.
5. Reports latency, energy, memory, and FLOPs.
6. Saves results to JSON.

Important:
- This is a CIM execution/scheduling baseline.
- With device="meta", this does not generate real GPT-2 text.
- Use --preset gpt2_small to run a GPT-2-shaped model.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import schedule_execution, fast_trace_decoder
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
    make_use_linear,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy


def make_json_serializable(obj: Any) -> Any:
    """
    Convert objects returned by threedsim into JSON-safe values.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    try:
        return float(obj)
    except Exception:
        return str(obj)


def get_preset_config(preset: str) -> Dict[str, int]:
    """
    Model presets.

    toy:
        Small README-style model for sanity checking.

    gpt2_small:
        GPT-2 small-shaped model.
        This is architecture-shaped like GPT-2 small, but not pretrained HF GPT-2.
    """
    if preset == "toy":
        return {
            "num_layers": 3,
            "d_model": 512,
            "nhead": 8,
            "d_ff": 4 * 512,
            "vocab_size": 1024,
        }

    if preset == "gpt2_small":
        return {
            "num_layers": 12,
            "d_model": 768,
            "nhead": 12,
            "d_ff": 4 * 768,
            "vocab_size": 50257,
        }

    raise ValueError(f"Unknown preset: {preset}")


def run_cim_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run one end-to-end CIM baseline experiment.
    """

    preset_cfg = get_preset_config(args.preset)

    num_layers = args.num_layers or preset_cfg["num_layers"]
    d_model = args.d_model or preset_cfg["d_model"]
    nhead = args.nhead or preset_cfg["nhead"]
    d_ff = args.d_ff or preset_cfg["d_ff"]
    vocab_size = args.vocab_size or preset_cfg["vocab_size"]

    print("=" * 80)
    print("IBM 3D-CIM Decoder-Only Transformer Baseline")
    print("=" * 80)
    print(f"Preset:       {args.preset}")
    print(f"Device:       {args.device}")
    print(f"Layers:       {num_layers}")
    print(f"d_model:      {d_model}")
    print(f"nhead:        {nhead}")
    print(f"d_ff:         {d_ff}")
    print(f"vocab_size:   {vocab_size}")
    print(f"start_len:    {args.start_len}")
    print(f"target_len:   {args.target_len}")
    print(f"batch size:   {args.batch_size}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Configure accelerator
    # -------------------------------------------------------------------------
    accelerator_config = AcceleratorConfig(
        tiles=args.tiles,
        tiers=args.tiers,
        tier_shape=(args.tier_rows, args.tier_cols),
        kv_caching=args.kv_caching,
    )

    acc = Accelerator(accelerator_config, device=args.device)

    # -------------------------------------------------------------------------
    # 2. Build decoder-only transformer
    # -------------------------------------------------------------------------
    decoder_layer_kwargs = {
        "d_model": d_model,
        "nhead": nhead,
        "dim_feedforward": d_ff,
    }

    embedding_layer_kwargs = {
        "vocab_size": vocab_size,
        "embedding_dim": d_model,
        "max_seq_length": args.target_len,
    }

    model = DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=num_layers,
        decoder_layer_kwargs=decoder_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device=args.device,
    )

    # -------------------------------------------------------------------------
    # 3. Assign accelerator to model
    # -------------------------------------------------------------------------
    assign_acc(model, acc)

    # -------------------------------------------------------------------------
    # 4. Map model to CIM accelerator
    # -------------------------------------------------------------------------
    mapper = Mapper(
        accelerator=acc,
        model=model,
        map_strategy=MapStrategy(
            strategy=Strategy.GREEDY_IN_ORDER,
            split_ffn=args.split_ffn,
            stack_embedding=args.stack_embedding,
        ),
    )

    print("[1/4] Mapping model to CIM accelerator...")
    mapper.map_network()

    fill_name_fields(model)
    make_traceable(model, is_traceable=True)

    # -------------------------------------------------------------------------
    # 5. Trace autoregressive decoding
    # -------------------------------------------------------------------------
    print("[2/4] Tracing decoder execution...")
    make_use_linear(model, use_linear=True)

    fast_traced = fast_trace_decoder(
        model,
        start_len=args.start_len,
        target_len=args.target_len,
        bsz=args.batch_size,
    )

    # -------------------------------------------------------------------------
    # 6. Schedule execution
    # -------------------------------------------------------------------------
    print("[3/4] Scheduling execution graph...")

    (
        execution_time,
        memory,
        peak_memory,
        energy,
        flops,
        energy_breakdown,
        latency_breakdown,
    ) = schedule_execution(
        fast_traced.graph,
        accelerator=model.accelerator,
        copy_and_cleanup_graph=False,
        communication=args.communication,
    )

    # -------------------------------------------------------------------------
    # 7. Collect results
    # -------------------------------------------------------------------------
    results = {
        "experiment": "ibm_3d_cim_decoder_baseline",
        "preset": args.preset,
        "model": {
            "num_layers": num_layers,
            "d_model": d_model,
            "nhead": nhead,
            "d_ff": d_ff,
            "vocab_size": vocab_size,
        },
        "inference": {
            "batch_size": args.batch_size,
            "start_len": args.start_len,
            "target_len": args.target_len,
            "kv_caching": args.kv_caching,
        },
        "accelerator": {
            "tiles": args.tiles,
            "tiers": args.tiers,
            "tier_shape": [args.tier_rows, args.tier_cols],
        },
        "mapping": {
            "strategy": "GREEDY_IN_ORDER",
            "split_ffn": args.split_ffn,
            "stack_embedding": args.stack_embedding,
        },
        "results": {
            "execution_time_ns": make_json_serializable(execution_time),
            "scratchpad_memory_bytes": make_json_serializable(memory),
            "peak_memory_bytes": make_json_serializable(peak_memory),
            "energy_nj": make_json_serializable(energy),
            "flops": make_json_serializable(flops),
            "energy_breakdown": make_json_serializable(energy_breakdown),
            "latency_breakdown": make_json_serializable(latency_breakdown),
        },
    }

    print("[4/4] Done.")
    print("=" * 80)
    print(f"Execution time: {execution_time} ns")
    print(f"Peak memory:    {peak_memory} bytes")
    print(f"Energy:         {energy} nJ")
    print(f"FLOPs:          {flops}")
    print("=" * 80)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IBM 3D-CIM end-to-end baseline for GPT-style transformer."
    )

    # Model preset
    parser.add_argument(
        "--preset",
        type=str,
        default="toy",
        choices=["toy", "gpt2_small"],
        help="Model preset to use.",
    )

    # Optional model overrides
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)

    # Inference settings
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-len", type=int, default=1)
    parser.add_argument("--target-len", type=int, default=12)

    # Accelerator settings
    parser.add_argument("--tiles", type=int, default=100)
    parser.add_argument("--tiers", type=int, default=1024)
    parser.add_argument("--tier-rows", type=int, default=512)
    parser.add_argument("--tier-cols", type=int, default=512)

    # Execution settings
    parser.add_argument(
        "--device",
        type=str,
        default="meta",
        help='Use "meta" for shape-only tracing.',
    )

    parser.add_argument(
        "--kv-caching",
        action="store_true",
        help="Enable KV caching in accelerator config.",
    )

    parser.add_argument(
        "--no-communication",
        action="store_true",
        help="Disable communication modeling.",
    )

    parser.add_argument(
        "--no-split-ffn",
        action="store_true",
        help="Disable FFN splitting in mapper.",
    )

    parser.add_argument(
        "--no-stack-embedding",
        action="store_true",
        help="Disable embedding stacking in mapper.",
    )

    # Output
    parser.add_argument(
        "--out",
        type=str,
        default="results/cim_baseline.json",
        help="Path to save JSON results.",
    )

    args = parser.parse_args()

    args.communication = not args.no_communication
    args.split_ffn = not args.no_split_ffn
    args.stack_embedding = not args.no_stack_embedding

    return args


def main() -> None:
    args = parse_args()

    results = run_cim_baseline(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()