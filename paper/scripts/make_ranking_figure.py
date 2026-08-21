#!/usr/bin/env python3
"""Regenerate Fig. 2 (measured Phase-1 sensitivity profile) from the Phase-1 artifact.

Usage:
  python paper/scripts/make_ranking_figure.py \
      --phase1 paper/data/phase1_sensitivity_seed42.json \
      --out paper/figures/p1_ranking.png

Drawn at the printed width (3.45 in = IEEE CAL \\columnwidth) so the font sizes
below are the printed sizes. Bars are rank-ordered, colored by projection type,
carry +/-1 std error bars over the five antithetic noise pairs, and the
dominant bars are labelled with block and type so the "LM head and early-block
attention" structure can be read directly from the figure. The rank-1 to
rank-49 spread is drawn as an anchored bracket.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

ROLE_COLORS = {
    "lm_head": "#d95f02",
    "attn.c_proj": "#1f5fa8",
    "attn.c_attn": "#5fa8e0",
    "mlp.c_fc": "#1b9e77",
    "mlp.c_proj": "#c6669f",
}
ROLE_ORDER = ["lm_head", "attn.c_proj", "attn.c_attn", "mlp.c_fc", "mlp.c_proj"]


def short_label(projection_id: str) -> str:
    if projection_id == "lm_head":
        return "lm_head"
    block, role = projection_id.split("/")
    return f"{block.replace('block_', 'b')}/{role}"


def main(phase1: Path, out: Path, n_labels: int) -> None:
    payload = json.loads(phase1.read_text(encoding="utf-8"))
    rows = payload["projections"]
    ids = [str(r["projection_id"]) for r in rows]
    roles = [str(r["role"]) for r in rows]
    mean = np.array([float(r["sensitivity_score_for_mapping"]) for r in rows])
    std = np.array([
        np.std([float(x["delta_nll_noise"]) for x in r["realizations"]], ddof=1)
        for r in rows
    ])
    order = np.argsort(-mean)
    ranks = np.arange(1, len(order) + 1)
    m, s = mean[order], std[order]
    colors = [ROLE_COLORS[roles[i]] for i in order]
    labels = [short_label(ids[i]) for i in order]

    fig, ax = plt.subplots(figsize=(3.45, 1.42), dpi=400)
    ax.bar(ranks, m, color=colors, width=0.82, edgecolor="none", zorder=2)
    lo = np.maximum(m - s, m * 0.08)
    ax.errorbar(ranks, m, yerr=[m - lo, s], fmt="none", ecolor="#333333",
                elinewidth=0.5, capsize=1.0, capthick=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(m.min() * 0.25, m.max() * 2.0)
    ax.set_xlim(0.2, len(order) + 0.9)
    ax.set_xticks([1, 10, 20, 30, 40, 49])
    ax.set_xlabel("projection rank (1–49)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel(r"$\Delta\mathrm{NLL}_{\mathrm{noise}}$ (nats, log)", fontsize=6.5, labelpad=1.5)
    ax.tick_params(labelsize=6, length=2, pad=1.5, width=0.5)
    ax.grid(axis="y", which="major", color="#dddddd", lw=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)

    # identities of the dominant bars as a compact one-line header (ranks 1..n_labels);
    # consecutive same-type projections are grouped ("attn.c_attn of blocks 3, 0, 2").
    top_ids = [ids[i] for i in order[:min(n_labels, len(order))]]
    groups: list[tuple[str, list[str]]] = []
    for pid in top_ids:
        if pid == "lm_head":
            groups.append(("lm_head", []))
            continue
        block, role = pid.split("/")
        b = block.replace("block_", "")
        if groups and groups[-1][0] == role:
            groups[-1][1].append(b)
        else:
            groups.append((role, [b]))
    parts = []
    for role, blocks in groups:
        if role == "lm_head":
            parts.append("lm_head")
        elif len(blocks) == 1:
            parts.append(f"b{blocks[0]}/{role}")
        else:
            parts.append(f"{role} of blocks {', '.join(blocks)}")
    ax.set_title(f"ranks 1–{len(top_ids)}: " + "; ".join(parts), fontsize=4.9, loc="left",
                 pad=2.0, color="#222222")

    # anchored ~540x bracket between rank-1 and rank-49 levels
    xb = len(order) + 0.6
    top, bot = m[0], m[-1]
    ax.annotate("", xy=(xb, bot), xytext=(xb, top),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.6, shrinkA=0, shrinkB=0))
    ax.plot([ranks[0] + 0.45, xb], [top, top], color="#555555", lw=0.45, ls=":", zorder=6)
    ax.text(xb - 0.4, np.sqrt(top * bot), f"≈{round(top / bot, -1):.0f}×", rotation=90,
            ha="right", va="center", fontsize=6.0, color="#444444")

    handles = [Patch(facecolor=ROLE_COLORS[r], label=r) for r in ROLE_ORDER]
    handles.append(Line2D([0], [0], color="#333333", lw=0.6, label="±1 std (5 antithetic pairs)"))
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=5.3, frameon=True,
              framealpha=0.85, borderpad=0.4, handlelength=1.3, handletextpad=0.5,
              columnspacing=0.9, labelspacing=0.3, bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(pad=0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    print(f"wrote {out}")
    print("top-8:", [(labels[k], round(float(m[k]), 4)) for k in range(8)])
    print("spread rank1/rank49:", round(float(top / bot), 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--phase1", type=Path, default=root / "data/phase1_sensitivity_seed42.json")
    parser.add_argument("--out", type=Path, default=root / "figures/p1_ranking.png")
    parser.add_argument("--n-labels", type=int, default=8)
    args = parser.parse_args()
    main(args.phase1, args.out, args.n_labels)
