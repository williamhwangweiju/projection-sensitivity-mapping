#!/usr/bin/env python3
"""Export Phase 1 ranking and cost-normalized sensitivity tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.common.config import load_json


def main(results_file: Path, output_dir: Path | None = None) -> Path:
    payload = load_json(results_file)
    rows = sorted(payload["projections"], key=lambda row: -float(row["sensitivity_score_for_mapping"]))
    output_dir = output_dir or results_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{results_file.stem}_ranking.csv"
    fields = [
        "rank", "projection_id", "role", "sensitivity_score_for_mapping",
        "delta_nll_analog_reference", "parameter_count", "macs_per_token",
        "sensitivity_per_parameter", "sensitivity_per_mac", "clipped_fraction",
        "tied_to_embedding",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            writer.writerow({"rank": rank, **{field: row.get(field) for field in fields if field != "rank"}})
    print(f"Ranking saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    main(args.results_file, args.output_dir)
