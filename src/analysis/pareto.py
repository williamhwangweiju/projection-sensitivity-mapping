"""Small dependency-free helpers for quality/cost Pareto analysis."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def pareto_frontier(
    rows: Iterable[Mapping[str, Any]], *, cost_field: str, quality_field: str
) -> list[dict[str, Any]]:
    """Return nondominated rows when both cost and degradation are minimized."""
    candidates = [dict(row) for row in rows]
    frontier: list[dict[str, Any]] = []
    for index, row in enumerate(candidates):
        cost = float(row[cost_field])
        quality = float(row[quality_field])
        dominated = False
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            other_cost = float(other[cost_field])
            other_quality = float(other[quality_field])
            if (
                other_cost <= cost
                and other_quality <= quality
                and (other_cost < cost or other_quality < quality)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (float(row[cost_field]), float(row[quality_field])))


def _rank(values: list[float]) -> np.ndarray:
    """Average ranks for ties, equivalent to standard Spearman ranking."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rank(x)
    ry = _rank(y)
    if float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
