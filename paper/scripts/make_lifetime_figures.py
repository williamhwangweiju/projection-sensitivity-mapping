#!/usr/bin/env python3
"""Regenerate Fig. 3 (perplexity under the aging trace) and Fig. 4 (paired improvement)
from the archived multiseed Phase-4 rows (seed, policy, t, r, nll).

Usage:
  python paper/scripts/make_lifetime_figures.py --data paper/data/phase4_nll_multiseed.csv --seed 42

Trace 42 is the weakest of the five traces (smallest lifetime paired improvement
over hardware_only), which is why it is the one plotted; the caption says so.
Bands are the min-max range over the three paired realizations (n = 3), not a
confidence interval.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIAG_PPL = 42.906832451949356        # HWA model, unclipped digital forward (diagnostic)
NOMINAL_PPL = 31.500854659976824     # HWA model, clip + quantization, zero tile noise
PRETRAINED_PPL = 29.009609818103243  # pretrained GPT-2, same tokenization / windows

STYLE = {
    "random": dict(color="#9a9a9a", marker="D", label="random"),
    "sequential": dict(color="#e8a317", marker="^", label="sequential"),
    "hardware_only": dict(color="#1b9e77", marker="s", label="hardware_only"),
    "static_sensitivity": dict(color="#1f5fa8", marker="o", label="static_sensitivity (ours)"),
}


def fig_ppl(df: pd.DataFrame, seed: int, out: Path) -> None:
    d = df[df.seed == seed]
    ts = sorted(d.t.unique())
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=250)
    for pol in ["random", "sequential", "hardware_only", "static_sensitivity"]:
        g = d[d.policy == pol].groupby("t").nll
        mean = np.exp(g.mean().reindex(ts).values)
        lo = np.exp(g.min().reindex(ts).values)
        hi = np.exp(g.max().reindex(ts).values)
        st = STYLE[pol]
        lw = 2.4 if pol == "static_sensitivity" else 1.5
        ax.fill_between(ts, lo, hi, color=st["color"], alpha=0.18, lw=0)
        ax.plot(ts, mean, color=st["color"], marker=st["marker"], lw=lw, ms=6, label=st["label"], zorder=3)
    ax.axhline(DIAG_PPL, color="#555555", ls="-.", lw=1.2)
    ax.text(ts[0] + 0.5, DIAG_PPL - 0.35, f"HWA unclipped digital {DIAG_PPL:.2f} (diagnostic)",
            fontsize=9, color="#555555", va="top")
    ax.axhline(NOMINAL_PPL, color="#1f5fa8", ls=":", lw=1.3)
    ax.text(ts[-1], NOMINAL_PPL + 0.35, f"nominal all-analog {NOMINAL_PPL:.2f}", fontsize=9,
            color="#1f5fa8", ha="right", va="bottom")
    ax.axhline(PRETRAINED_PPL, color="#111111", ls="--", lw=1.3)
    ax.text(ts[-1], PRETRAINED_PPL + 0.35, "pretrained digital 29.0 (deployment alternative)",
            fontsize=9, color="#111111", ha="right", va="bottom")
    band_patch = plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(band_patch)
    labels.append("min-max range (n = 3)")
    ax.legend(handles, labels, loc="upper left", fontsize=9,
              ncol=2, frameon=True, framealpha=0.95, columnspacing=1.0)
    ax.set_xticks(ts)
    ax.set_xlabel("trace timestep (hardware degradation ->)", fontsize=10)
    ax.set_ylabel("held-out test perplexity", fontsize=10)
    ax.set_ylim(28.0, 47.8)
    ax.grid(color="#e5e5e5", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    print("wrote", out)


def fig_paired(df: pd.DataFrame, seed: int, out: Path) -> None:
    d = df[df.seed == seed]
    ts = sorted(d.t.unique())
    m = d[d.policy == "static_sensitivity"].set_index(["t", "r"]).nll
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=250)
    for pol, emph in [("sequential", False), ("random", False), ("hardware_only", True)]:
        b = d[d.policy == pol].set_index(["t", "r"]).nll
        delta = (b - m).reset_index()
        g = delta.groupby("t").nll
        mean, lo, hi = g.mean().reindex(ts).values, g.min().reindex(ts).values, g.max().reindex(ts).values
        st = STYLE[pol]
        ax.fill_between(ts, lo, hi, color=st["color"], alpha=0.18 if emph else 0.13, lw=0)
        ax.plot(ts, mean, color=st["color"], marker=st["marker"], lw=2.6 if emph else 1.3,
                ms=6.5 if emph else 5.5, alpha=1.0 if emph else 0.85,
                label=f"vs {pol}", zorder=4 if emph else 3)
        ax.text(ts[-1] + 2.0, mean[-1], f"vs {pol}", fontsize=9.5 if emph else 8.5,
                fontweight="bold" if emph else "normal", color=st["color"], va="center")
        if not np.all(np.diff(mean) > 0):
            print(f"WARNING: {pol} curve not strictly increasing: {mean}")
    ax.axhline(0.0, color="#333333", lw=0.9)
    band_patch = plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("vs hardware_only"), labels.index("vs random"), labels.index("vs sequential")]
    handles = [handles[i] for i in order] + [band_patch]
    labels = [labels[i] for i in order] + ["min-max range (n = 3)"]
    ax.legend(handles, labels, loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
    ax.set_xticks(ts)
    ax.set_xlim(ts[0] - 3, ts[-1] + 36)
    ax.set_xlabel("trace timestep (hardware degradation ->)", fontsize=10)
    ax.set_ylabel("paired ΔNLL improvement (nats)", fontsize=10)
    ax.grid(color="#e5e5e5", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    print("wrote", out)


def fig_combined(df: pd.DataFrame, seed: int, out: Path) -> None:
    """Single-column two-panel version (shared x): (a) PPL, (b) paired improvement.

    Drawn at the printed width (3.45 in = IEEE CAL columnwidth), so the font sizes
    are the printed sizes. No text is drawn inside the data area: the reference
    lines are named in a second legend, and the curve names live in the legends.
    """
    d = df[df.seed == seed]
    ts = sorted(d.t.unique())
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.45, 2.60), dpi=400, sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.2, 1.0], hspace=0.09))
    FS, FT, FL = 7.5, 7.0, 6.3   # axis labels, ticks, legend
    band_patch = plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=0.5)
    # ---------------------------------------------------------------- (a) perplexity
    pol_handles, pol_labels = [], []
    for pol in ["random", "sequential", "hardware_only", "static_sensitivity"]:
        g = d[d.policy == pol].groupby("t").nll
        mean = np.exp(g.mean().reindex(ts).values)
        lo = np.exp(g.min().reindex(ts).values)
        hi = np.exp(g.max().reindex(ts).values)
        st = STYLE[pol]
        lw = 1.6 if pol == "static_sensitivity" else 1.0
        ax.fill_between(ts, lo, hi, color=st["color"], alpha=0.18, lw=0)
        (h,) = ax.plot(ts, mean, color=st["color"], marker=st["marker"], lw=lw, ms=3.6, zorder=3)
        pol_handles.append(h)
        pol_labels.append(st["label"])
    h_diag = ax.axhline(DIAG_PPL, color="#555555", ls="-.", lw=0.8)
    h_nom = ax.axhline(NOMINAL_PPL, color="#1f5fa8", ls=":", lw=0.9)
    h_pre = ax.axhline(PRETRAINED_PPL, color="#111111", ls="--", lw=0.9)
    leg1 = ax.legend(pol_handles + [band_patch], pol_labels + ["min-max range (n = 3)"],
                     loc="upper left", fontsize=FL, ncol=2, frameon=True, framealpha=0.95,
                     columnspacing=0.8, handlelength=1.6, handletextpad=0.5, borderpad=0.35,
                     labelspacing=0.25)
    ax.add_artist(leg1)
    ax.legend([h_pre, h_nom, h_diag],
              ["pretrained digital 29.0 (deployment alternative)",
               f"nominal all-analog {NOMINAL_PPL:.2f}",
               f"HWA unclipped digital {DIAG_PPL:.2f} (diagnostic)"],
              loc="lower right", fontsize=FL, frameon=True, framealpha=0.95,
              handlelength=1.8, handletextpad=0.5, borderpad=0.35, labelspacing=0.25)
    ax.set_ylabel("test perplexity", fontsize=FS, labelpad=2)
    ax.set_ylim(28.0, 50.5)
    ax.grid(color="#e5e5e5", lw=0.4)
    ax.text(0.985, 0.96, "(a)", transform=ax.transAxes, fontsize=7, fontweight="bold",
            va="top", ha="right")
    # ---------------------------------------------------------------- (b) paired improvement
    m = d[d.policy == "static_sensitivity"].set_index(["t", "r"]).nll
    handles, labels = [], []
    for pol, emph in [("hardware_only", True), ("random", False), ("sequential", False)]:
        b = d[d.policy == pol].set_index(["t", "r"]).nll
        delta = (b - m).reset_index()
        g = delta.groupby("t").nll
        mean, lo, hi = g.mean().reindex(ts).values, g.min().reindex(ts).values, g.max().reindex(ts).values
        st = STYLE[pol]
        bx.fill_between(ts, lo, hi, color=st["color"], alpha=0.18 if emph else 0.13, lw=0)
        (h,) = bx.plot(ts, mean, color=st["color"], marker=st["marker"], lw=1.7 if emph else 0.9,
                       ms=4.0 if emph else 3.3, alpha=1.0 if emph else 0.85,
                       zorder=4 if emph else 3)
        handles.append(h)
        labels.append(f"vs {pol}")
        if not np.all(np.diff(mean) > 0):
            print(f"WARNING: {pol} curve not strictly increasing: {mean}")
    bx.axhline(0.0, color="#333333", lw=0.7)
    leg = bx.legend(handles + [band_patch], labels + ["min-max range (n = 3)"], loc="upper left",
                    fontsize=FL, frameon=True, framealpha=0.95, handlelength=1.6,
                    handletextpad=0.5, borderpad=0.35, labelspacing=0.25)
    leg.get_texts()[0].set_fontweight("bold")   # the headline comparison
    bx.set_xticks(ts)
    bx.set_xlim(ts[0] - 3, ts[-1] + 5)
    bx.set_xlabel("trace timestep (hardware degradation ->)", fontsize=FS, labelpad=2)
    bx.set_ylabel("paired ΔNLL (nats)", fontsize=FS, labelpad=2)
    bx.set_ylim(-0.008, 0.165)
    bx.grid(color="#e5e5e5", lw=0.4)
    bx.text(0.985, 0.05, "(b)", transform=bx.transAxes, fontsize=7, fontweight="bold",
            va="bottom", ha="right")
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_linewidth(0.5)
        a.tick_params(labelsize=FT, length=2, pad=1.5, width=0.5)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.975, bottom=0.13, hspace=0.09)
    fig.savefig(out, dpi=400)
    print("wrote", out)


def main(data: Path, seed: int, figdir: Path) -> None:
    df = pd.read_csv(data)
    # report which trace is weakest (lifetime paired mean vs hardware_only)
    m = df[df.policy == "static_sensitivity"].set_index(["seed", "t", "r"]).nll
    b = df[df.policy == "hardware_only"].set_index(["seed", "t", "r"]).nll
    per = (b - m).reset_index().groupby("seed").nll.mean()
    print("lifetime paired improvement vs hardware_only per trace:", per.round(4).to_dict(),
          "-> weakest trace:", int(per.idxmin()))
    fig_ppl(df, seed, figdir / "p4_ppl.png")
    fig_paired(df, seed, figdir / "p4_paired.png")
    fig_combined(df, seed, figdir / "p4_lifetime.png")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "data/phase4_nll_multiseed.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figdir", type=Path, default=root / "figures")
    args = parser.parse_args()
    main(args.data, args.seed, args.figdir)
