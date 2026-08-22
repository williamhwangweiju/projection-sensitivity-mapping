#!/usr/bin/env python3
"""Draw Fig. 1 (pipeline / where-the-contribution-sits schematic) as vector PDF.

Usage:  python paper/scripts/make_workflow_figure.py [--out paper/figures/workflow.pdf]

Design notes (drawn at the printed width, 3.45 in = IEEE CAL \\columnwidth, so
the font sizes below are the printed sizes):
  * Row A  - the standard offline LLM-on-CIM deployment flow (compressed).
  * Row B  - the contribution (sensitivity profile + tile-quality map -> mapper),
             the largest row; an explicit "replaces" arrow ties the mapper to
             the toolchain's mapping slot.
  * Row C  - the runtime accelerator: every one of the 96 tiles is drawn
             (12 x 8 grid), shaded by noise class, so proportions read correctly.
  * Two labelled vertical arrows carry "write-verify statistics" (array ->
    tile-quality map) and "programmed weights" (program tiles -> array).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, not Type 3 (IEEE PDF eXpress)
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

GREY_FILL, GREY_EDGE = "#eef1f5", "#8a94a6"
BLUE_EDGE, BLUE_DARK = "#2f6fb3", "#1d4f86"
ORANGE_BAND, ORANGE_EDGE = "#fff4e3", "#d98b2b"
TILE_SHADES = {"quiet": "#fbe3b3", "typical": "#f2b85c", "noisy": "#c4782a"}
RED = "#c0392b"

FS = 6.8          # printed font size (pt) for box titles / row titles
FS_SMALL = 6.2    # printed font size for sub-labels, legend, arrow labels

# Drawing canvas: x in [0, 101.5], y in [0, 76]; figure is 3.45 x 2.32 in.
FIG_W, FIG_H = 3.45, 2.32


def box(ax, x0, y0, w, h, text, *, sub=None, fill=GREY_FILL, edge=GREY_EDGE,
        ls="-", lw=0.7, bold=False, fs=FS, text_color="#1b1f27", sub_color="#4a5263",
        pad=0.25, sub_dy=None, sub_fs=FS_SMALL, title_dy=0.62):
    patch = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad={pad}",
                           linewidth=lw, edgecolor=edge, facecolor=fill, linestyle=ls)
    ax.add_patch(patch)
    cy = y0 + h / 2 if sub is None else y0 + h * title_dy
    ax.text(x0 + w / 2, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=text_color)
    if sub is not None:
        ax.text(x0 + w / 2, y0 + h * (0.27 if sub_dy is None else sub_dy), sub,
                ha="center", va="center", fontsize=sub_fs, color=sub_color, linespacing=0.95)
    return patch


def arrow(ax, p0, p1, *, color="#4a5263", lw=0.7, style="-|>", ms=4, ls="-",
          connection="arc3,rad=0.0"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color,
                        linewidth=lw, linestyle=ls, connectionstyle=connection,
                        shrinkA=0.5, shrinkB=0.5)
    ax.add_patch(a)
    return a


def main(out: Path) -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 101.5)
    ax.set_ylim(0, 76)
    ax.axis("off")

    # ------------------------------------------------------------ Row A (compressed)
    ax.text(2, 73.4, "Offline deployment flow", fontsize=FS, fontweight="bold",
            color="#1b1f27", va="center")
    ya, ha = 62.2, 8.2
    boxes_a = [
        (2.0, 15.5, "Pretrained\nLLM", None, GREY_FILL, GREY_EDGE, "-"),
        (20.0, 17.5, "HWA fine-tune\n(Phase 0)", None, GREY_FILL, GREY_EDGE, "-"),
        (40.0, 14.5, "Shard\n512×512", None, GREY_FILL, GREY_EDGE, "-"),
        (57.0, 22.0, "Mapping slot", "default: sequential\nor random", "#ffffff", BLUE_EDGE, (0, (2.2, 1.4))),
        (81.5, 13.5, "Program\ntiles", None, GREY_FILL, GREY_EDGE, "-"),
    ]
    for x0, w, t, s, f, e, ls in boxes_a:
        box(ax, x0, ya, w, ha, t, sub=s, fill=f, edge=e, ls=ls,
            bold=(t == "Mapping slot"), text_color=(BLUE_DARK if t == "Mapping slot" else "#1b1f27"),
            sub_dy=0.25, sub_fs=FS_SMALL - 0.3, title_dy=0.72)
    for (x0, w, *_), (x1, *_) in zip(boxes_a[:-1], boxes_a[1:]):
        arrow(ax, (x0 + w + 0.4, ya + ha / 2), (x1 - 0.4, ya + ha / 2))
    ax.text(80.0, ya + ha + 1.4, "placement table π", fontsize=FS_SMALL, color=BLUE_DARK,
            ha="center", va="center")

    # ------------------------------------------------------------ Row B (the contribution, largest)
    yb0, yb1 = 31.5, 59.4
    ax.add_patch(FancyBboxPatch((1.5, yb0), 97, yb1 - yb0, boxstyle="round,pad=0.3",
                                linewidth=0.8, edgecolor=BLUE_EDGE, facecolor="#f3f8fd"))
    ax.text(3.0, yb1 - 1.9, "This work: a drop-in for the mapping slot", fontsize=FS,
            fontweight="bold", color=BLUE_DARK, va="center")
    box(ax, 3.0, 35.0, 27.5, 16.5, "Sensitivity profile s(p)",
        sub="Phase 1 · once per model\n49 one-at-a-time probes", fill="#ffffff",
        edge=BLUE_EDGE, fs=FS, sub_dy=0.26)
    box(ax, 34.5, 34.0, 32.5, 18.5, "Sensitivity mapper",
        sub="Phase 3 · one sort, 678 shards\nimportant shards → quiet tiles",
        fill=BLUE_EDGE, edge=BLUE_DARK, bold=True, fs=FS + 0.8, text_color="#ffffff",
        sub_color="#e8f0fa", sub_dy=0.27)
    box(ax, 71.0, 35.0, 26.5, 16.5, "Tile-quality map σᵢ",
        sub="Phase 2 in simulation ·\nper chip, 96 values", fill="#ffffff",
        edge=BLUE_EDGE, fs=FS, sub_dy=0.26)
    arrow(ax, (30.9, 43.2), (34.1, 43.2), color=BLUE_DARK, lw=0.9, ms=5)
    arrow(ax, (70.6, 43.2), (67.4, 43.2), color=BLUE_DARK, lw=0.9, ms=5)
    # "replaces" arrow mapper -> mapping slot
    arrow(ax, (57.5, 52.5), (62.0, 61.8), color=BLUE_DARK, lw=1.2, ms=6,
          connection="arc3,rad=-0.15")
    ax.text(63.6, 57.4, "replaces", fontsize=FS, fontweight="bold", color=BLUE_DARK,
            ha="left", va="center")

    # ------------------------------------------------------------ Row C (runtime)
    yc0, yc1 = 1.0, 29.3
    ax.add_patch(FancyBboxPatch((1.5, yc0), 97, yc1 - yc0, boxstyle="round,pad=0.3",
                                linewidth=0.8, edgecolor=ORANGE_EDGE, facecolor=ORANGE_BAND))
    ax.text(3.0, yc1 - 1.8, "CIM accelerator: runtime inference", fontsize=FS,
            fontweight="bold", color="#8a4b0a", va="center")
    box(ax, 3.0, 5.6, 23.0, 19.2, "Digital periphery", fill="#ffffff", edge=ORANGE_EDGE,
        fs=FS, sub="embedding lookup\nLayerNorm, softmax\nGELU, residual adds\ntoken loop control",
        sub_dy=0.37, title_dy=0.85)
    ax0, ay0, aw, ah = 31.5, 5.3, 62.0, 20.4
    ax.add_patch(FancyBboxPatch((ax0, ay0), aw, ah, boxstyle="round,pad=0.25",
                                linewidth=0.8, edgecolor=ORANGE_EDGE, facecolor="#ffffff"))
    ax.text(ax0 + aw / 2, ay0 + ah - 1.6,
            "96 tiles (12 × 8), 8 stacked 512×512 tiers each",
            fontsize=FS_SMALL, ha="center", va="center", color="#1b1f27")
    rng = np.random.default_rng(7)
    ncol, nrow = 12, 8
    gx0, gy0 = ax0 + 3.2, ay0 + 1.6
    cw, ch, gap = 4.55, 1.62, 0.26
    classes = rng.choice(["quiet", "typical", "noisy"], size=(nrow, ncol), p=[0.25, 0.5, 0.25])
    faults = set()
    while len(faults) < 8:
        faults.add((int(rng.integers(nrow)), int(rng.integers(ncol))))
    quiet_cells = [(r, c) for r in range(nrow) for c in range(ncol)
                   if classes[r, c] == "quiet" and (r, c) not in faults]
    crit = quiet_cells[len(quiet_cells) // 2]
    for r in range(nrow):
        for c in range(ncol):
            x = gx0 + c * (cw + gap)
            y = gy0 + (nrow - 1 - r) * (ch + gap)
            ax.add_patch(Rectangle((x, y), cw, ch, facecolor=TILE_SHADES[classes[r, c]],
                                   edgecolor="#ffffff", linewidth=0.3))
            if (r, c) in faults:
                ax.plot([x + 0.6, x + cw - 0.6], [y + 0.3, y + ch - 0.3], color=RED, lw=0.8)
                ax.plot([x + 0.6, x + cw - 0.6], [y + ch - 0.3, y + 0.3], color=RED, lw=0.8)
    r, c = crit
    ax.add_patch(Rectangle((gx0 + c * (cw + gap) - 0.25, gy0 + (nrow - 1 - r) * (ch + gap) - 0.25),
                           cw + 0.5, ch + 0.5, facecolor="none", edgecolor=BLUE_EDGE, linewidth=1.1))
    # DAC / ADC arrows between periphery and array
    arrow(ax, (26.4, 17.6), (31.2, 17.6), color="#4a5263", lw=0.7, ms=4)
    arrow(ax, (31.2, 11.0), (26.4, 11.0), color="#4a5263", lw=0.7, ms=4)
    ax.text(28.8, 19.9, "DAC", fontsize=FS_SMALL, ha="center", va="center", color="#4a5263")
    ax.text(28.8, 8.3, "ADC", fontsize=FS_SMALL, ha="center", va="center", color="#4a5263")
    # legend (own line, starts at the left so it stays inside the band)
    lx, ly = 3.5, 3.0
    items = [("quiet", TILE_SHADES["quiet"]), ("typical", TILE_SHADES["typical"]),
             ("noisy", TILE_SHADES["noisy"])]
    per_char = 1.05
    for name, col in items:
        ax.add_patch(Rectangle((lx, ly - 0.75), 2.2, 1.5, facecolor=col, edgecolor="#999", lw=0.3))
        ax.text(lx + 2.8, ly, name, fontsize=FS_SMALL, va="center")
        lx += 2.8 + per_char * len(name) + 3.0
    ax.plot([lx, lx + 1.8], [ly - 0.65, ly + 0.65], color=RED, lw=0.8)
    ax.plot([lx, lx + 1.8], [ly + 0.65, ly - 0.65], color=RED, lw=0.8)
    ax.text(lx + 2.5, ly, "fault", fontsize=FS_SMALL, va="center")
    lx += 2.5 + per_char * 5 + 3.0
    ax.add_patch(Rectangle((lx, ly - 0.75), 2.2, 1.5, facecolor="#fbe3b3", edgecolor=BLUE_EDGE, lw=0.9))
    ax.text(lx + 2.8, ly, "critical shard on a quiet tile", fontsize=FS_SMALL, va="center")

    # ------------------------------------------------------------ vertical arrows
    # write-verify statistics: array -> tile-quality map (upward)
    arrow(ax, (84.0, ay0 + ah + 0.3), (84.0, 34.6), color=ORANGE_EDGE, lw=0.9, ms=5)
    # label sits inside the runtime band, beside the arrow, clear of both band borders
    ax.text(82.8, ay0 + ah + 1.9, "write-verify statistics", fontsize=FS_SMALL, color="#8a4b0a",
            ha="right", va="center")
    # programmed weights: program tiles -> array (down the right margin)
    ax.plot([94.9, 97.8], [ya + ha / 2, ya + ha / 2], color="#4a5263", lw=0.8)
    ax.plot([97.8, 97.8], [ya + ha / 2, 14.0], color="#4a5263", lw=0.8)
    arrow(ax, (97.8, 14.0), (ax0 + aw + 0.4, 14.0), color="#4a5263", lw=0.8, ms=4)
    ax.text(99.9, 40.0, "programmed weights", fontsize=FS_SMALL, color="#4a5263",
            rotation=90, ha="center", va="center")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip("."), pad_inches=0.0)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures/workflow.pdf")
    args = parser.parse_args()
    main(args.out)
