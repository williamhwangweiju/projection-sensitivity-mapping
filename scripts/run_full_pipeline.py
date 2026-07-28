#!/usr/bin/env python3
"""Run/resume the all-analog deployment pipeline: Phase 0 through Phase 4."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def required(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"{label} artifact is required when its producing phase is skipped: {path}"
        )
    return path


def run_phase0_hwa(config: Path, skip: bool) -> None:
    """Run Phase 0 HWA fine-tuning when enabled; validate the checkpoint contract.

    Downstream phases load weights from model.checkpoint, so a Phase 0 run
    must produce exactly that directory, and a skipped Phase 0 requires it to
    exist already.
    """
    from src.common.config import load_yaml, resolve_path

    pipeline_config = load_yaml(config)
    checkpoint_cfg = pipeline_config["model"].get("checkpoint")

    if skip or not bool(pipeline_config.get("hwa_training", {}).get("enabled", False)):
        if checkpoint_cfg is not None and not resolve_path(checkpoint_cfg).is_dir():
            raise FileNotFoundError(
                f"model.checkpoint does not exist: {resolve_path(checkpoint_cfg)}. "
                "Run Phase 0 first or set model.checkpoint to null."
            )
        return

    if checkpoint_cfg is None:
        raise ValueError(
            "hwa_training.enabled is true but model.checkpoint is null. Point "
            "model.checkpoint at <hwa_training.output_root>/checkpoint_final."
        )

    from experiments.phase0_hwa_training.run_hwa_training import main as run_phase0

    produced = run_phase0(config)
    expected = resolve_path(checkpoint_cfg)
    if produced.resolve() != expected.resolve():
        raise RuntimeError(
            f"Phase 0 produced {produced} but model.checkpoint points at "
            f"{expected}; align hwa_training.output_root and model.checkpoint."
        )


def run_proxy_sensitivity(config: Path, phase1: Path | None) -> Path | None:
    """Compute cheap proxy sensitivity scores when profiling.proxy.enabled."""
    from src.common.config import load_yaml

    pipeline_config = load_yaml(config)
    if not bool(pipeline_config.get("profiling", {}).get("proxy", {}).get("enabled", False)):
        return None

    from experiments.phase1_sensitivity.run_proxy_sensitivity import main as run_proxy

    return run_proxy(config, phase1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1-artifact", type=Path)
    parser.add_argument(
        "--proxy-artifact",
        type=Path,
        help="Existing proxy sensitivity sidecar; used when Phase 1 is skipped.",
    )
    parser.add_argument("--trace-artifact", type=Path)
    parser.add_argument("--phase3-manifest", type=Path)
    for phase in (0, 1, 2, 3, 4):
        parser.add_argument(f"--skip-phase{phase}", action="store_true")
    args = parser.parse_args()

    config = args.config.resolve()
    print(f"Repository: {REPO_ROOT}")
    print(f"Configuration: {config}")

    run_phase0_hwa(config, args.skip_phase0)

    if args.skip_phase1:
        phase1 = required(args.phase1_artifact, "Phase 1")
        proxy = args.proxy_artifact
        if proxy is None:
            proxy = run_proxy_sensitivity(config, phase1)
    else:
        from experiments.phase1_sensitivity.run_aihwkit_profiling import (
            main as run_phase1,
        )
        from experiments.phase1_sensitivity.analyze_results import (
            main as analyze_phase1,
        )

        phase1 = run_phase1(config)
        analyze_phase1(phase1)
        proxy = run_proxy_sensitivity(config, phase1)

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

        phase3_manifest = run_phase3(config, phase1, trace, proxy)

    from scripts.validate_pipeline_contracts import validate_pipeline

    validate_pipeline(config, phase1, trace, phase3_manifest)

    phase4_metadata: Path | None = None
    if not args.skip_phase4:
        from experiments.phase4_quality.run_hybrid_quality import main as run_phase4

        phase4_metadata = run_phase4(config, phase1, trace, phase3_manifest)

    print("Pipeline complete.")
    print(f"Phase 1: {phase1}")
    print(f"Proxy sidecar: {proxy}")
    print(f"Phase 2: {trace}")
    print(f"Phase 3: {phase3_manifest}")
    if phase4_metadata is not None:
        print(f"Phase 4: {phase4_metadata}")


if __name__ == "__main__":
    main()
