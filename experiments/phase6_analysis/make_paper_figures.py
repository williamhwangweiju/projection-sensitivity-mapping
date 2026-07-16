#!/usr/bin/env python3
"""Generate the core paper figures from Phase-4/5 CSV artifacts."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def digital_frontier(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[str, list[tuple[float, float, str]]] = {}
    for row in rows:
        grouped.setdefault(row["selection_method"], []).append(
            (
                float(row["digital_mac_fraction"]),
                float(row["delta_nll_nominal_vs_digital"]),
                row["digital_set_id"],
            )
        )
    plt.figure(figsize=(7.0, 4.5))
    for method, values in sorted(grouped.items()):
        values.sort()
        plt.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            marker="o",
            label=method,
        )
    plt.xlabel("Digital MAC fraction")
    plt.ylabel("Nominal hybrid ΔNLL vs digital")
    plt.title("Digital-protection quality–cost frontier")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    save_current(output)


def static_policy_summary(rows: list[dict[str, str]], output: Path) -> None:
    point_ids = sorted({row["digital_set_id"] for row in rows})
    policies = sorted({row["policy"] for row in rows})
    width = 0.8 / max(len(policies), 1)
    x = list(range(len(point_ids)))
    plt.figure(figsize=(max(7.0, 1.4 * len(point_ids)), 4.8))
    for policy_index, policy in enumerate(policies):
        by_point = {
            row["digital_set_id"]: float(row["mean_delta_nll_tile"])
            for row in rows
            if row["policy"] == policy
        }
        positions = [value - 0.4 + width / 2 + policy_index * width for value in x]
        plt.bar(
            positions,
            [by_point.get(point_id, float("nan")) for point_id in point_ids],
            width=width,
            label=policy,
        )
    plt.xticks(x, point_ids, rotation=30, ha="right")
    plt.ylabel("Mean tile-induced ΔNLL")
    plt.title("Static placement quality within fixed digital sets")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    save_current(output)


def proxy_scatter(rows: list[dict[str, str]], output: Path) -> None:
    plt.figure(figsize=(6.2, 4.6))
    for point_id in sorted({row["digital_set_id"] for row in rows}):
        subset = [row for row in rows if row["digital_set_id"] == point_id]
        plt.scatter(
            [float(row["proxy_variance"]) for row in subset],
            [float(row["delta_nll_tile"]) for row in subset],
            s=18,
            alpha=0.7,
            label=point_id,
        )
    plt.xlabel("Sensitivity-weighted variance proxy")
    plt.ylabel("Measured tile-induced ΔNLL")
    plt.title("Proxy validation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    save_current(output)


def adaptive_events(rows: list[dict[str, str]], output: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for point_id in sorted({row["digital_set_id"] for row in rows}):
        subset = sorted(
            (row for row in rows if row["digital_set_id"] == point_id),
            key=lambda row: int(row["timestep"]),
        )
        cumulative = 0.0
        x: list[int] = []
        y: list[float] = []
        for row in subset:
            if row.get("accepted", "False").lower() == "true":
                cumulative += float(row.get("moved_bytes_fp32", 0.0) or 0.0)
            x.append(int(row["timestep"]))
            y.append(cumulative / (1024**2))
        plt.step(x, y, where="post", label=point_id)
    plt.xlabel("Hardware timestep")
    plt.ylabel("Cumulative remapped data (MiB, FP32 equivalent)")
    plt.title("Adaptive remapping overhead")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7)
    save_current(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-frontier", type=Path, required=True)
    parser.add_argument("--static-summary", type=Path, required=True)
    parser.add_argument("--static-quality", type=Path, required=True)
    parser.add_argument("--adaptive-events", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    digital_frontier(
        read_csv(args.nominal_frontier),
        args.output_root / "figure_digital_protection_frontier.png",
    )
    static_policy_summary(
        read_csv(args.static_summary),
        args.output_root / "figure_static_policy_quality.png",
    )
    proxy_scatter(
        read_csv(args.static_quality),
        args.output_root / "figure_proxy_validation.png",
    )
    if args.adaptive_events is not None and args.adaptive_events.is_file():
        adaptive_events(
            read_csv(args.adaptive_events),
            args.output_root / "figure_adaptive_remapping_overhead.png",
        )
    print(f"Figures saved to: {args.output_root}")


if __name__ == "__main__":
    main()
