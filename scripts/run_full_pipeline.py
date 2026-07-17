#!/usr/bin/env python3
"""Run/resume the complete hybrid digital–analog research pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



def required(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} artifact is required when its producing phase is skipped: {path}")
    return path




def run_digital_selection(config: Path, phase1: Path) -> Path:
    """Generate automatic digital operating points from an existing Phase-1 profile."""
    from experiments.phase1_5_digital_selection.select_digital_operating_points import (
        main as run_selection,
    )
    from experiments.phase1_5_digital_selection.select_greedy_marginal import (
        main as run_measured_greedy_selection,
    )
    from src.common.config import load_yaml, resolve_path

    operating_points = run_selection(config, phase1)
    pipeline_config = load_yaml(config)
    greedy_cfg = pipeline_config.get("digital_selection", {}).get("greedy_marginal", {})
    if bool(greedy_cfg.get("enabled", True)):
        greedy_output = (
            resolve_path(pipeline_config["digital_selection"]["output_root"])
            / "greedy_marginal_points.json"
        )
        run_measured_greedy_selection(
            config,
            phase1,
            greedy_output,
            operating_points,
        )
    return operating_points

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1-artifact", type=Path)
    parser.add_argument("--operating-points-artifact", type=Path)
    parser.add_argument("--trace-artifact", type=Path)
    parser.add_argument("--phase3-manifest", type=Path)
    for phase in (1, 2, 3, 4):
        parser.add_argument(f"--skip-phase{phase}", action="store_true")
    parser.add_argument(
        "--reselect-digital",
        action="store_true",
        help=(
            "Reuse --phase1-artifact but regenerate automatic Phase-1.5 digital "
            "operating points instead of requiring --operating-points-artifact."
        ),
    )
    args = parser.parse_args()

    config = args.config.resolve()
    print(f"Repository: {REPO_ROOT}")
    print(f"Configuration: {config}")

    if args.skip_phase1:
        phase1 = required(args.phase1_artifact, "Phase 1")
        if args.reselect_digital:
            operating_points = run_digital_selection(config, phase1)
        else:
            operating_points = required(args.operating_points_artifact, "Phase 1.5")
    else:
        from experiments.phase1_sensitivity.run_aihwkit_profiling import (
            main as run_phase1,
        )
        from experiments.phase1_sensitivity.analyze_results import (
            main as analyze_phase1,
        )

        phase1 = run_phase1(config)
        analyze_phase1(phase1)
        operating_points = run_digital_selection(config, phase1)

    if args.skip_phase2:
        trace = required(args.trace_artifact, "Phase 2")
    else:
        from experiments.phase2_fidelity.run_fidelity_model import main as run_phase2

        trace = run_phase2(config)

    if args.skip_phase3:
        phase3_manifest = required(args.phase3_manifest, "Phase 3")
    else:
        from experiments.phase3_baselines.run_baseline_mappings import (
            main as run_phase3,
        )

        phase3_manifest = run_phase3(config, phase1, operating_points, trace)

    from scripts.validate_pipeline_contracts import validate_pipeline

    validate_pipeline(config, phase1, operating_points, trace, phase3_manifest)

    phase4_metadata: Path | None = None
    if not args.skip_phase4:
        from experiments.phase4_quality.run_hybrid_quality import main as run_phase4

        phase4_metadata = run_phase4(
            config,
            phase1,
            operating_points,
            trace,
            phase3_manifest,
        )

    print("Hybrid pipeline complete.")
    print(f"Phase 1: {phase1}")
    print(f"Digital operating points: {operating_points}")
    print(f"Phase 2: {trace}")
    print(f"Phase 3: {phase3_manifest}")
    if phase4_metadata is not None:
        print(f"Phase 4: {phase4_metadata}")


if __name__ == "__main__":
    main()
