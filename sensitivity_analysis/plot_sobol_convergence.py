"""
Convergence of the Sobol estimates with the Saltelli base sample size N.

The study reports indices at N = 1024 (config.SOBOL_N_BASE).  That choice needs
evidence rather than assertion, so this re-estimates every index over
N = 32 ... 4096 on the same surrogates and plots the total-order indices against
N.  Every parameter is drawn identically and identified by colour and marker in
a single shared legend; the bootstrap intervals are omitted here for legibility
and are reported per cell in the heatmap figures.

Only surrogate predictions are involved — no FEA — so the whole sweep costs
seconds per output.

Usage:
    python3 plot_sobol_convergence.py [--full-box]
Products:
    figures/figW_sobol_convergence{,_fullbox}.{pdf,png}
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, SOBOL_N_BASE
from surrogate import ScalarSurrogate
from visualization import FIG_DIR
from plot_material_r_sobol_combined import (
    GROUPS, PARAM_LABELS, OUTPUT_LABELS, CABLE_OUTPUTS,
    SUR_PREFIX, FIG_SUFFIX, LOG_TENSIONS, tension_scale,
)

SHORT = {p: lab.split(" (")[0].replace("\n", "") for p, lab in PARAM_LABELS.items()}

N_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096]
SHOWN    = ["crown_height", "H_anisotropy", "mean_stress", "cable_course_tension"]

# Colour-blind-safe qualitative set, keyed by parameter *name* so a parameter
# keeps its colour in both group rows (the cable group has two extra rows before
# E1, so indexing by column would recolour it).
_MARKERS = dict(zip(
    ["sf_wale", "sf_course", "knit_dir", "pressure", "cable_wale_lrest",
     "cable_course_lrest", "E1", "r", "nu"],
    ["o", "s", "^", "v", "P", "X", "D", "*", "d"]))

_PARAM_COLOR = dict(zip(
    ["sf_wale", "sf_course", "knit_dir", "pressure", "cable_wale_lrest",
     "cable_course_lrest", "E1", "r", "nu"],
    ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA4499",
     "#EE3377", "#33BBEE", "#555555", "#BBBB44"]))


def sweep(group, bounds):
    """{output: {param: [(N, ST, conf), ...]}}"""
    sur  = ScalarSurrogate.load(os.path.join(
        DATA_DIR, f"{group}_{SUR_PREFIX}surrogate.pkl"))
    keys = list(bounds)
    problem = {"num_vars": len(keys), "names": keys,
               "bounds": [list(v) for v in bounds.values()]}

    out = {c: {p: [] for p in keys} for c in sur.gps}
    for N in N_VALUES:
        X = saltelli.sample(problem, N, calc_second_order=False)
        preds = sur.predict(X)
        for col, Y in preds.items():
            if np.std(Y) < 1e-10:
                continue
            # Report the scale the study reports: the tensions are zero-inflated
            # and analysed as log1p(T / T_REF) (see run_sobol_robust.py), so
            # converge them on that scale rather than on log(clip(T, 1e-9)),
            # whose spike at the slack plateau dominated the variance.
            if LOG_TENSIONS and col in CABLE_OUTPUTS:
                Y = tension_scale(Y)
            si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                       print_to_console=False)
            for i, p in enumerate(keys):
                out[col][p].append((N, max(0.0, si["ST"][i]), si["ST_conf"][i]))
        print(f"  {group}: N={N:5d} ({len(X)} evaluations)")
    return out


def plot_convergence(save=True):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    n_col = len(SHOWN)
    fig, axes = plt.subplots(len(GROUPS), n_col, figsize=(2.5 * n_col, 5.5),
                             squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.165,
                        wspace=0.28, hspace=0.30)

    for r, (group, (label, bounds)) in enumerate(GROUPS.items()):
        print(f"[{group}]")
        data = sweep(group, bounds)
        keys = list(bounds)
        for c, col in enumerate(SHOWN):
            ax = axes[r][c]
            if col not in data or not data[col][keys[0]]:
                ax.set_visible(False)
                continue
            for p in keys:
                N, st, cf = map(np.array, zip(*data[col][p]))
                ax.plot(N, st, "-", marker=_MARKERS.get(p, "o"),
                        color=_PARAM_COLOR.get(p, "#888888"),
                        linewidth=1.1, markersize=3.0, alpha=0.9, zorder=2)
            ax.axvline(SOBOL_N_BASE, color="#333333", linestyle="--",
                       linewidth=0.7, zorder=0)
            ax.set_xscale("log", base=2)
            # Label every other N; the full set collides at the right-hand end.
            shown = N_VALUES[::2]
            ax.set_xticks(shown)
            ax.set_xticklabels([str(n) for n in shown], fontsize=7)
            ax.set_xticks(N_VALUES, minor=True)
            # A log axis labels its minor ticks by default, which reintroduces
            # the 2^6 / 2^8 labels between the ones we chose.
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_ylim(-0.03, 1.02)
            ax.tick_params(labelsize=7)
            ax.set_title(OUTPUT_LABELS.get(col, col), fontsize=9, pad=3)
            if c == 0:
                ax.set_ylabel(f"{label}\n$S_T$", fontsize=8.5)

    # One shared legend: every parameter appears once, with the colour and marker
    # it carries in every panel of both rows.
    from matplotlib.lines import Line2D
    seen = []
    for group in GROUPS:
        for p in GROUPS[group][1]:
            if p not in seen:
                seen.append(p)
    handles = [Line2D([], [], color=_PARAM_COLOR.get(p, "#888888"),
                      marker=_MARKERS.get(p, "o"), markersize=3.6, linewidth=1.1,
                      label=SHORT.get(p, p)) for p in seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(seen), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005), handlelength=1.8,
               columnspacing=1.4)

    fig.supxlabel("Saltelli base sample size $N$   "
                  "(dashed line: value used, $N = 1024$)", fontsize=9, y=0.075)
    fig.suptitle("Convergence of the total-order Sobol indices with sample size",
                 fontsize=11, y=0.95)

    if save:
        base = os.path.join(FIG_DIR, f"figW_sobol_convergence{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    plot_convergence()
