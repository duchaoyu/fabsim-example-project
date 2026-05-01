"""
Publication-quality figures for sensitivity analysis paper.

Fig A: Sobol ST heatmap — parameters × outputs, faceted by group
Fig B: Spearman correlation heatmap — inputs vs outputs from raw FEA data
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MOTIFS, HAS_CABLE, SCALAR_OUTPUTS
from surrogate import ScalarSurrogate
from sobol_analysis import run_sobol_for_group

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi":       150,
})

PARAM_LABELS = {
    "sf_wale":            r"$s_{wale}$",
    "sf_course":          r"$s_{course}$",
    "knit_dir":           r"$\theta_{knit}$",
    "pressure":           r"$p$",
    "cable_wale_lrest":   r"$l_{wale}$",
    "cable_course_lrest": r"$l_{course}$",
}

OUTPUT_LABELS = {
    "crown_height":           r"$h_{crown}$",
    "H_mean_x0":              r"$\bar{H}_{x0}$",
    "H_mean_y0":              r"$\bar{H}_{y0}$",
    "max_stress":             r"$\sigma_{max}$",
    "mean_stress":            r"$\bar{\sigma}$",
    "cable_wale_tension":     r"$T_{wale}$",
    "cable_course_tension":   r"$T_{course}$",
    "boundary_reaction_mean": r"$\bar{R}_{bdry}$",
}

_CABLE_OUTPUTS = {"cable_wale_tension", "cable_course_tension"}

GROUP_LABELS = {
    "motif1_nocable": "Motif 1\n(no cable)",
    "motif1_cable":   "Motif 1\n(cable)",
    "motif2_nocable": "Motif 2\n(no cable)",
    "motif2_cable":   "Motif 2\n(cable)",
}


# ── Fig A: Sobol ST heatmap ───────────────────────────────────────────────────

def plot_sobol_heatmap(save=True):
    """
    4-panel heatmap: each panel = one (motif, cable) group.
    Rows = input parameters, columns = scalar outputs.
    Cell value = total-order Sobol index ST.
    """
    groups_ordered = [
        "motif1_nocable", "motif1_cable",
        "motif2_nocable", "motif2_cable",
    ]

    # Load surrogates and compute Sobol
    all_ST = {}
    all_ST_conf = {}
    for group in groups_ordered:
        path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        if not os.path.exists(path):
            continue
        surrogate = ScalarSurrogate.load(path)
        has_cable = "cable" in group and "nocable" not in group
        motif = 1 if "motif1" in group else 2
        results = run_sobol_for_group(surrogate, motif, has_cable)

        # Build matrix: rows=params, cols=outputs
        param_names = list(results[list(results.keys())[0]].index)
        out_names   = [o for o in SCALAR_OUTPUTS if o in results
                       and not (o in _CABLE_OUTPUTS and not has_cable)]

        ST_mat   = np.zeros((len(param_names), len(out_names)))
        conf_mat = np.zeros_like(ST_mat)
        for j, col in enumerate(out_names):
            if col in results:
                for i, p in enumerate(param_names):
                    ST_mat[i, j]   = max(0, results[col].loc[p, "ST"])
                    conf_mat[i, j] = results[col].loc[p, "ST_conf"]

        all_ST[group]      = (param_names, out_names, ST_mat)
        all_ST_conf[group] = conf_mat

    if not all_ST:
        print("No surrogate data found.")
        return

    n_groups = len(all_ST)
    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(2.6 * n_groups, 3.2),
        constrained_layout=True,
    )
    if n_groups == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.YlOrRd

    im_ref = None
    for ax, group in zip(axes, groups_ordered):
        if group not in all_ST:
            ax.set_visible(False)
            continue
        param_names, out_names, ST_mat = all_ST[group]
        conf_mat = all_ST_conf[group]

        im = ax.imshow(ST_mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        im_ref = im

        # Annotate cells
        for i in range(len(param_names)):
            for j in range(len(out_names)):
                v = ST_mat[i, j]
                color = "white" if v > 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

        ax.set_xticks(range(len(out_names)))
        ax.set_xticklabels(
            [OUTPUT_LABELS.get(o, o) for o in out_names],
            rotation=30, ha="right",
        )
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(
            [PARAM_LABELS.get(p, p) for p in param_names],
        )
        ax.set_title(GROUP_LABELS.get(group, group), pad=6)
        ax.tick_params(length=0)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, len(out_names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(param_names)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Shared colorbar
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, shrink=0.7, pad=0.02,
                             label=r"Total-order Sobol index $S_T$")
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.suptitle(
        "Variance-based sensitivity analysis: total-order Sobol indices",
        fontsize=10, y=1.02,
    )

    if save:
        path = os.path.join(FIG_DIR, "figA_sobol_heatmap.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig B: Spearman correlation heatmap ──────────────────────────────────────

def plot_correlation_heatmap(save=True):
    """
    2×2 grid of Spearman correlation heatmaps (one per group).
    Rows = input parameters, columns = scalar outputs.
    Only outputs with non-trivial variance are shown.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_curvature.csv"))

    groups_ordered = [
        "motif1_nocable", "motif1_cable",
        "motif2_nocable", "motif2_cable",
    ]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(8.5, 6.0),
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    cmap = plt.cm.RdBu_r
    vmin, vmax = -1.0, 1.0
    im_ref = None

    for ax, group in zip(axes_flat, groups_ordered):
        has_cable = "cable" in group and "nocable" not in group
        sub = df[df["group"] == group].copy()

        input_cols = (["sf_wale", "sf_course", "knit_dir", "pressure",
                       "cable_wale_lrest", "cable_course_lrest"]
                      if has_cable
                      else ["sf_wale", "sf_course", "knit_dir", "pressure"])

        # Only outputs with meaningful variance; exclude cable tensions for nocable
        output_cols = [o for o in SCALAR_OUTPUTS
                       if o in sub.columns and sub[o].std() > 1e-6
                       and not (o in _CABLE_OUTPUTS and not has_cable)]

        mat = np.zeros((len(input_cols), len(output_cols)))
        pmat = np.ones_like(mat)
        for i, inp in enumerate(input_cols):
            for j, out in enumerate(output_cols):
                valid = sub[[inp, out]].dropna()
                if len(valid) < 5:
                    continue
                r, p = stats.spearmanr(valid[inp], valid[out])
                mat[i, j]  = r
                pmat[i, j] = p

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        im_ref = im

        # Annotate with r value; mark significant (p<0.05) with *
        for i in range(len(input_cols)):
            for j in range(len(output_cols)):
                r, p = mat[i, j], pmat[i, j]
                sig  = "*" if p < 0.05 else ""
                color = "white" if abs(r) > 0.65 else "black"
                ax.text(j, i, f"{r:.2f}{sig}",
                        ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_xticks(range(len(output_cols)))
        ax.set_xticklabels(
            [OUTPUT_LABELS.get(o, o) for o in output_cols],
            rotation=30, ha="right",
        )
        ax.set_yticks(range(len(input_cols)))
        ax.set_yticklabels(
            [PARAM_LABELS.get(p, p) for p in input_cols],
        )
        ax.set_title(GROUP_LABELS.get(group, group), pad=4)
        ax.tick_params(length=0)

        # Grid
        ax.set_xticks(np.arange(-0.5, len(output_cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(input_cols)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Shared colorbar
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes_flat, shrink=0.6, pad=0.02,
                             label=r"Spearman correlation $\rho$")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    fig.suptitle(
        r"Spearman rank correlations between design parameters and outputs"
        "\n(* $p < 0.05$; grouped by knit motif and cable configuration)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figB_correlation_heatmap.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    plot_sobol_heatmap()
    plot_correlation_heatmap()
    print("Done.")
