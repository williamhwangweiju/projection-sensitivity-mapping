#!/usr/bin/env python3
"""Export a compact Phase-1 sensitivity ranking CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(results_path: Path) -> Path:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    records = payload["results"]["projections"]
    columns = [
        "projection_id",
        "projection_label",
        "sensitivity_score_for_mapping",
        "sensitivity_score_unit",
        "delta_nll_noise_mean",
        "delta_nll_noise_std",
        "delta_nll_noise_ci95_low",
        "delta_nll_noise_ci95_high",
        "delta_ppl_noise_mean",
        "delta_ppl_total_mean",
        "delta_nll_analog_reference",
        "delta_ppl_analog_reference",
    ]
    frame = pd.DataFrame(records)
    available = [column for column in columns if column in frame.columns]
    frame = frame[available].sort_values(
        "sensitivity_score_for_mapping", ascending=False
    )
    output = results_path.with_name(results_path.stem + "_ranking.csv")
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"Ranking saved to: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    main(args.results)
