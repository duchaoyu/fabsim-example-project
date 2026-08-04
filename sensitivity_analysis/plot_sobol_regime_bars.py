"""
Sobol regime figure as stacked bars — the reader-facing alternative to the
colour-blend regime map (plot_material_r_regime.py).

One panel per output; one horizontal bar per parameter:

    |========== S1 ==========|///// ST - S1 /////|
     direct effect             interaction share

Bar *length* is the total effect S_T, so the eye reads influence as length; the
segment boundary reads as the split between direct and interactive influence.
Parameters are sorted by S_T descending within each panel, so the ranking is the
vertical order.  Nothing has to be decoded from a colour blend, and there is no
second axis.

All panels share a fixed 0..1 x axis, which is what makes the panels comparable
to each other — an important difference from the heatmap, where every cell was
normalised by the same colormap but the eye still had to compare hues.

Indices come from the same cached tables as every other Sobol figure
(data/sobol_material_r_{group}_{CSV_PREFIX}{output}.csv, log tables preferred for
the cable tensions), so the numbers match fig3_sobol_material_r_combined.

Usage:
    python3 plot_sobol_regime_bars.py [--full-box] [--raw-tensions] [--cols 5]
Products:
    figures/fig3_sobol_material_r_regime_bars{,_fullbox}.{pdf,png}
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization import FIG_DIR
from plot_material_r_sobol_combined import (
    GROUPS, BASE_OUTPUTS, CABLE_OUTPUTS, PARAM_LABELS, OUTPUT_LABELS,
    load_group, _is_masked, FIG_SUFFIX, VALID,
)

# State the box on the figure: the default is the model-validity box, and a
# reader must not have to infer which one from a filename suffix.
_BOX_LABEL = ("model-validity box, $s \\geq 0.95$" if VALID else "full box")

# Same blue/orange language as the regime map: blue = direct, orange =
# interaction.  Here the interaction segment is also hatched and lightened, so
# the split survives greyscale printing and does not out-shout the S1 segment.
C_S1        = "#2E73BA"
C_INT_FACE  = "#FBD3AE"
C_INT_HATCH = "#E07B0A"
C_CONF      = "#555555"

# Bars need short labels; the heatmap row labels carry units, which a
# dimensionless index does not need.  E1 keeps its "(wale)" tag — that one is
# a modelling fact, not a unit.
SHORT_LABELS = {
    "sf_wale":            r"$s_\mathrm{wale}$",
    "sf_course":          r"$s_\mathrm{course}$",
    "knit_dir":           r"$\theta_\mathrm{knit}$",
    "pressure":           r"$p$",
    "cable_wale_lrest":   r"$L^\mathrm{wale}_\mathrm{rest}$",
    "cable_course_lrest": r"$L^\mathrm{course}_\mathrm{rest}$",
    "E1":                 r"$E_1$ (wale)",
    "r":                  r"$r = E_2/E_1$",
    "nu":                 r"$\nu_{12}$",
}


def _short(p):
    return SHORT_LABELS.get(
        p, PARAM_LABELS.get(p, p).split(" (")[0].replace("\n", " "))


# Geometry, inches.  Everything is laid out in absolute units (as in the other
# figures in this chapter) so the bar pitch is identical in both blocks even
# though the cable group has two more parameters.
AX_W     = 2.60    # width of the bar area itself
LAB_W    = 0.92    # room for the y tick labels to the left of each axes
BAR_PITCH = 0.245  # per parameter
GAP_X    = 0.40    # between panels
TITLE_H  = 0.34    # panel title
XLAB_H   = 0.40    # x ticks + axis label, bottom row of a block only
ROW_GAP  = 0.18    # between stacked rows inside a block
BLOCK_TITLE_H = 0.36
BLOCK_GAP = 0.34
PAD_L, PAD_R = 0.16, 0.22
PAD_T, PAD_B = 0.58, 1.02   # suptitle / legend + caption

XMAX = 1.20        # 1.0 of index plus room for the value annotation


def _rows(df, params, out_key):
    """[(param, S1, ST, ST_conf)] for one output, sorted by ST descending."""
    rows = []
    for p in params:
        if _is_masked(p, out_key):
            continue          # structurally inapplicable, not merely small
        s1 = max(0.0, float(df.loc[p, "S1"]))
        st = max(s1, max(0.0, float(df.loc[p, "ST"])))
        rows.append((p, s1, st, float(df.loc[p, "ST_conf"])))
    rows.sort(key=lambda t: -t[2])
    return rows


def _draw_panel(ax, rows, title, show_xlabels):
    y = np.arange(len(rows))[::-1]          # highest ST at the top
    s1   = np.array([r[1] for r in rows])
    intr = np.array([r[2] - r[1] for r in rows])
    st   = np.array([r[2] for r in rows])
    conf = np.array([r[3] for r in rows])

    ax.barh(y, s1, height=0.70, color=C_S1, linewidth=0, zorder=3)
    ax.barh(y, intr, left=s1, height=0.70, color=C_INT_FACE, hatch="////",
            edgecolor=C_INT_HATCH, linewidth=0.5, zorder=3)
    ax.errorbar(st, y, xerr=np.clip(conf, 0, None), fmt="none", ecolor=C_CONF,
                elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)

    for yy, s, c in zip(y, st, conf):
        ax.text(min(s + c + 0.03, XMAX - 0.01), yy, f"{s:.2f}",
                ha="left", va="center", fontsize=8.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([_short(r[0]) for r in rows], fontsize=10)
    ax.set_ylim(-0.62, len(rows) - 0.38)
    ax.set_xlim(0, XMAX)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9, length=2.5)
    if show_xlabels:
        ax.set_xlabel("Sobol index", fontsize=10, labelpad=1)
    else:
        ax.set_xticklabels([])
    ax.set_title(title, fontsize=11.5, pad=4)


def plot_regime_bars(n_col=3, groups=None, tag="", save=True):
    # At 3 panels per row the combined figure runs to ~18 in tall, so --split
    # renders one page-sized figure per group; block letters stay (a)/(b) either
    # way, so a split pair still reads as one pair of panels in the text.
    letter_of = {g: "ab"[i] for i, g in enumerate(GROUPS)}
    groups = list(groups or GROUPS)
    tables, params_of, outs_of = {}, {}, {}
    for g in groups:
        print(f"[{g}]")
        tables[g] = load_group(g)
        if not tables[g]:
            raise SystemExit(f"no Sobol tables found for {g}")
        params_of[g] = list(GROUPS[g][1].keys())
        outs_of[g] = [c for c in (BASE_OUTPUTS +
                                  (CABLE_OUTPUTS if g.endswith("_cable") else []))
                      if c in tables[g]]
        print(f"  {len(outs_of[g])} outputs x {len(params_of[g])} parameters")

    # Row plan per block: panels wrap at n_col, so the bar pitch never changes
    pitch_x = LAB_W + AX_W + GAP_X
    n_rows_of = {g: int(np.ceil(len(outs_of[g]) / n_col)) for g in groups}
    ax_h_of   = {g: BAR_PITCH * len(params_of[g]) for g in groups}

    fig_w = PAD_L + n_col * pitch_x - GAP_X + PAD_R
    block_h = {}
    for g in groups:
        nr = n_rows_of[g]
        block_h[g] = (BLOCK_TITLE_H + nr * (TITLE_H + ax_h_of[g])
                      + (nr - 1) * ROW_GAP + XLAB_H)
    fig_h = PAD_T + sum(block_h.values()) + BLOCK_GAP * (len(groups) - 1) + PAD_B

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "hatch.linewidth": 0.6})
    fig = plt.figure(figsize=(fig_w, fig_h))

    y_cursor = fig_h - PAD_T                      # inches from the bottom
    for bi, g in enumerate(groups):
        label = GROUPS[g][0]
        fig.text(PAD_L / fig_w, (y_cursor - 0.22) / fig_h,
                 f"({letter_of[g]}) {label}", fontsize=12.5,
                 fontweight="bold", ha="left", va="baseline")
        y_cursor -= BLOCK_TITLE_H

        nr = n_rows_of[g]
        for k, o in enumerate(outs_of[g]):
            row, col = divmod(k, n_col)
            ax_top = y_cursor - row * (TITLE_H + ax_h_of[g] + ROW_GAP) - TITLE_H
            x0 = PAD_L + col * pitch_x + LAB_W
            ax = fig.add_axes([x0 / fig_w, (ax_top - ax_h_of[g]) / fig_h,
                               AX_W / fig_w, ax_h_of[g] / fig_h])
            _draw_panel(ax, _rows(tables[g][o], params_of[g], o),
                        OUTPUT_LABELS.get(o, o), show_xlabels=(row == nr - 1))

        y_cursor -= (nr * (TITLE_H + ax_h_of[g]) + (nr - 1) * ROW_GAP
                     + XLAB_H + BLOCK_GAP)

    handles = [
        Patch(facecolor=C_S1, label=r"direct effect  $S_1$"),
        Patch(facecolor=C_INT_FACE, hatch="////", edgecolor=C_INT_HATCH,
              linewidth=0.5, label=r"interaction  $S_T - S_1$"),
        Line2D([0], [0], color=C_CONF, linewidth=0.9, marker="|",
               markersize=5, label=r"95% bootstrap CI on $S_T$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.52 / fig_h))

    fig.text(0.5, 0.20 / fig_h,
             "Bar length is the total effect $S_T$; the segment boundary is how "
             "much of it acts directly.  Parameters are sorted by $S_T$ within "
             "each panel, so vertical order is the ranking.\n"
             "A cable's rest length cannot act on the other cable's tension, so "
             "it is omitted from those two panels rather than drawn as zero.",
             ha="center", va="bottom", fontsize=8.5, linespacing=1.45,
             color="#333333")

    fig.suptitle("Sobol sensitivity regime: direct effect and interaction share, "
                 f"by parameter ({_BOX_LABEL})",
                 fontsize=13, y=1 - 0.20 / fig_h)

    if save:
        base = os.path.join(FIG_DIR,
                            f"fig3_sobol_material_r_regime_bars{tag}{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=3,
                    help="panels per row within a block")
    ap.add_argument("--split", action="store_true",
                    help="one figure per group instead of one tall combined "
                         "figure (recommended at --cols 3)")
    ap.add_argument("--full-box", action="store_true",
                    help="plot the original full box instead of the default "
                         "model-validity box (handled at import; listed so "
                         "argparse accepts the flag)")
    ap.add_argument("--raw-tensions", action="store_true",
                    help="use raw rather than log cable-tension indices "
                         "(handled at import)")
    args = ap.parse_args()
    if args.split:
        for g, (label, _) in GROUPS.items():
            plot_regime_bars(n_col=args.cols, groups=[g],
                             tag="_" + label.replace(" ", ""))
    else:
        plot_regime_bars(n_col=args.cols)
