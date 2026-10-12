#!/usr/bin/env python3
"""Fisher-proxy diagnostic driver: scores, static_fisher placement, and Phase-4 evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.config import load_json, load_yaml, resolve_path

FISHER_POLICY = "static_fisher"


def derive_config(
    base: dict[str, Any],
    *,
    output_root: Path,
    phase3_policies: list[str],
    phase4_policies: list[str],
    trace_seed: int | None = None,
) -> dict[str, Any]:
    """Copy of the base configuration with Phases 3/4 redirected to the Fisher run.

    Phase 4 pairs every non-Fisher policy (method) against ``static_fisher``
    (baseline), so a positive ``nll_improvement`` means that policy beats the
    proxy placement, the paper's sign convention. ``trace_seed`` mirrors
    ``run_multiseed_pipeline.py --vary-placement-seed``.
    """
    config = deepcopy(base)
    if trace_seed is not None:
        config["experiment"]["seed"] = int(trace_seed)
        config["experiment"]["placement_seed"] = int(trace_seed)
    config["phase3"]["output_root"] = str(output_root / "phase3")
    config["phase3"]["policies"] = list(phase3_policies)
    config["phase4"]["output_root"] = str(output_root / "phase4")
    config["phase4"]["policies"] = list(phase4_policies)
    config["phase4"]["method_policies"] = [p for p in phase4_policies if p != FISHER_POLICY]
    config["phase4"]["baseline_policies"] = [FISHER_POLICY]
    return config


def write_config(config: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def record_run(manifest_path: Path, run: dict[str, Any]) -> None:
    """Merge one trace-seed run into a manifest aggregate_multiseed.py can read."""
    runs: list[dict[str, Any]] = []
    if manifest_path.is_file():
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        runs = list(payload.get("runs", []))
    by_seed = {int(item["trace_seed"]): item for item in runs}
    by_seed[int(run["trace_seed"])] = run
    manifest_path.write_text(
        yaml.safe_dump({"runs": [by_seed[seed] for seed in sorted(by_seed)]}, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fisher-proxy diagnostic: scores, static_fisher placement, Phase-4 evaluation.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/full_pipeline/gpt2_hybrid_3dcim.yaml")
    parser.add_argument("--phase1", type=Path, required=True, help="Measured Phase-1 profile JSON.")
    parser.add_argument("--trace", type=Path, required=True, help="Phase-2 trace.npz for this trace seed.")
    parser.add_argument("--proxy-artifact", type=Path, help="Reuse an existing proxy sidecar instead of recomputing it.")
    parser.add_argument("--max-batches", type=int, default=None, help="Forwarded to run_proxy_sensitivity.py (default 32, the paper setting; 0 = all).")
    parser.add_argument("--phase3-manifest", type=Path, help="Existing Phase-3 manifest (e.g. the frozen campaign's) to extend with static_fisher; otherwise the base policies are placed here first.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data/results/fisher_proxy")
    parser.add_argument("--trace-seed", type=int, help="Sets experiment.seed and placement_seed and records this run for cross-trace aggregation.")
    parser.add_argument("--phase4-policies", nargs="+", default=["static_sensitivity", FISHER_POLICY])
    parser.add_argument("--skip-phase4", action="store_true", help="Stop after the validated static_fisher placement.")
    args = parser.parse_args()

    if FISHER_POLICY not in args.phase4_policies:
        parser.error(f"--phase4-policies must include {FISHER_POLICY}.")
    base_policies = [p for p in args.phase4_policies if p != FISHER_POLICY]
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = load_yaml(args.config)
    config_path = output_root / "fisher_runtime_config.yaml"

    # 1. Proxy scores (+ Spearman against the measured profile).
    if args.proxy_artifact is not None:
        proxy_path = Path(args.proxy_artifact)
    else:
        from experiments.phase1_sensitivity.run_proxy_sensitivity import main as run_proxy

        proxy_path = run_proxy(args.config, args.phase1, args.max_batches)

    # 2. Base placements: the unchanged Phase-3 runner on the non-Fisher policies.
    if args.phase3_manifest is not None:
        base_manifest = Path(args.phase3_manifest)
        present = {str(item["policy"]) for item in load_json(base_manifest)["placements"]}
        missing = sorted(set(base_policies) - present)
        if missing:
            parser.error(f"{base_manifest} lacks placements for {missing}.")
    else:
        write_config(
            derive_config(base, output_root=output_root, phase3_policies=base_policies,
                          phase4_policies=args.phase4_policies, trace_seed=args.trace_seed),
            config_path,
        )
        from experiments.phase3_baselines.run_baseline_mappings import main as run_phase3

        base_manifest = run_phase3(config_path, args.phase1, args.trace)

    # 3. static_fisher placement appended to the base manifest.
    from experiments.phase3_baselines.run_fisher_placement import main as run_fisher

    manifest = run_fisher(args.config, args.phase1, args.trace, proxy_path, base_manifest, output_root / "phase3")
    manifest_policies = [str(item["policy"]) for item in load_json(manifest)["placements"]]
    write_config(
        derive_config(base, output_root=output_root, phase3_policies=manifest_policies,
                      phase4_policies=args.phase4_policies, trace_seed=args.trace_seed),
        config_path,
    )

    # 4. Contracts, then the unchanged Phase-4 runner.
    from scripts.validate_pipeline_contracts import validate_pipeline

    validate_pipeline(config_path, args.phase1, args.trace, manifest)
    print(f"Proxy sidecar: {proxy_path}")
    print(f"Derived configuration: {config_path}")
    if args.skip_phase4:
        print("Phase 4 skipped.")
        return

    from experiments.phase4_quality.run_hybrid_quality import main as run_phase4

    metadata = run_phase4(config_path, args.phase1, args.trace, manifest)
    print(f"Phase 4 metadata: {metadata}")
    print(f"Paired summary (method minus static_fisher, positive = method better): {output_root / 'phase4' / 'paired_policy_summary.csv'}")
    if args.trace_seed is not None:
        runs_manifest = output_root.parent / "fisher_multiseed_manifest.yaml"
        record_run(runs_manifest, {
            "trace_seed": int(args.trace_seed),
            "runtime_config": str(config_path),
            "output_root": str(output_root),
            "phase4_metadata": str(metadata),
        })
        print(f"Cross-trace manifest: {runs_manifest}")


if __name__ == "__main__":
    main()
