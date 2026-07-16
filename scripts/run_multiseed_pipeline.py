#!/usr/bin/env python3
"""Run Phases 2–5 across independent hardware trace seeds.

Phase 1 sensitivity and Phase 1.5 digital operating points are intentionally
reused across traces. Each trace receives an isolated runtime YAML and output
root, preventing accidental overwrites while preserving paired policy noise
within that trace.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_yaml

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency error is explicit
    raise RuntimeError("PyYAML is required.") from exc


def _redirect_outputs(config: dict[str, Any], root: Path) -> None:
    config["phase2"]["output_root"] = str(root / "phase2")
    config["phase3"]["output_root"] = str(root / "phase3")
    config["phase4"]["output_root"] = str(root / "phase4")
    config["phase5"]["output_root"] = str(root / "phase5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml",
    )
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--operating-points", type=Path, required=True)
    parser.add_argument("--trace-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data/results/multiseed",
    )
    parser.add_argument("--skip-phase4", action="store_true")
    parser.add_argument("--skip-phase5", action="store_true")
    parser.add_argument("--skip-adaptive-quality", action="store_true")
    parser.add_argument(
        "--vary-placement-seed",
        action="store_true",
        help="Use each trace seed for the random placement baseline as well.",
    )
    args = parser.parse_args()

    base = load_yaml(args.config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for trace_seed in args.trace_seeds:
        trace_root = (args.output_root / f"trace_seed_{trace_seed}").resolve()
        trace_root.mkdir(parents=True, exist_ok=True)
        runtime = deepcopy(base)
        runtime["experiment"]["seed"] = int(trace_seed)
        if args.vary_placement_seed:
            runtime["experiment"]["placement_seed"] = int(trace_seed)
        _redirect_outputs(runtime, trace_root)
        runtime_path = trace_root / "runtime_config.yaml"
        runtime_path.write_text(
            yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8"
        )

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_full_pipeline.py"),
            "--config",
            str(runtime_path),
            "--skip-phase1",
            "--phase1-artifact",
            str(args.phase1.resolve()),
            "--operating-points-artifact",
            str(args.operating_points.resolve()),
        ]
        if args.skip_phase4:
            command.append("--skip-phase4")
        if args.skip_phase5:
            command.append("--skip-phase5")
        if args.skip_adaptive_quality:
            command.append("--skip-adaptive-quality")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        manifests.append(
            {
                "trace_seed": int(trace_seed),
                "runtime_config": str(runtime_path),
                "output_root": str(trace_root),
                "phase4_metadata": str(trace_root / "phase4/phase4_metadata.json"),
                "phase5_manifest": str(trace_root / "phase5/phase5_manifest.json"),
                "phase5_quality_metadata": str(
                    trace_root / "phase5/phase5_quality_metadata.json"
                ),
            }
        )

    manifest_path = args.output_root / "multiseed_run_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(args.config.resolve()),
                "phase1": str(args.phase1.resolve()),
                "operating_points": str(args.operating_points.resolve()),
                "runs": manifests,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"Multi-seed manifest: {manifest_path}")


if __name__ == "__main__":
    main()
