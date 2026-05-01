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


# ── Fig A / A2: Sobol heatmap (ST or S1) ─────────────────────────────────────

def plot_sobol_heatmap(index="ST", save=True):
    """
    4-panel heatmap in figA3 layout: rows = no cable/cable, cols = motif1/motif2.
    Rows = input parameters, columns = scalar outputs.
    Sizing, fonts, and spacing identical to plot_sobol_regime (figA3).
    """
    assert index in ("ST", "S1"), "index must be 'ST' or 'S1'"
    conf_col = "ST_conf" if index == "ST" else "S1_conf"

    MOTIF_TITLES = {1: r"Motif 1 ($E_2/E_1=2.5$)", 2: r"Motif 2 ($E_1/E_2=2.5$)"}
    ROW_LABELS   = {False: "No cable", True: "Cable"}

    layout = [
        [(1, False), (2, False)],
        [(1, True),  (2, True)],
    ]

    def _is_masked(p, o):
        return ((p == "cable_wale_lrest"  and o == "cable_course_tension") or
                (p == "cable_course_lrest" and o == "cable_wale_tension"))

    # Load surrogates and compute Sobol
    group_data = {}
    for motif in [1, 2]:
        for has_cable in [False, True]:
            group = "motif{}_{}".format(motif, "cable" if has_cable else "nocable")
            pkl = os.path.join(DATA_DIR, "{}_scalar_surrogate.pkl".format(group))
            if not os.path.exists(pkl):
                continue
            sur = ScalarSurrogate.load(pkl)
            results = run_sobol_for_group(sur, motif, has_cable)
            param_names = list(results[list(results.keys())[0]].index)
            out_names   = [o for o in SCALAR_OUTPUTS if o in results
                           and not (o in _CABLE_OUTPUTS and not has_cable)]
            mat      = np.zeros((len(param_names), len(out_names)))
            conf_mat = np.zeros_like(mat)
            for j, col in enumerate(out_names):
                for i, p in enumerate(param_names):
                    mat[i, j]      = max(0, results[col].loc[p, index])
                    conf_mat[i, j] = results[col].loc[p, conf_col]
            mask = np.zeros_like(mat, dtype=bool)
            for i, p in enumerate(param_names):
                for j, col in enumerate(out_names):
                    if _is_masked(p, col):
                        mask[i, j] = True
            mat = np.where(mask, np.nan, mat)
            group_data[(motif, has_cable)] = (param_names, out_names, mat, conf_mat, mask)

    if not group_data:
        print("No surrogate data found.")
        return

    # ── Match figA3 sizing exactly ────────────────────────────────────────────
    max_params  = max(len(v[0]) for v in group_data.values())
    max_outputs = max(len(v[1]) for v in group_data.values())
    cell_w, cell_h = 0.72, 0.65
    pad_top, pad_bot, pad_left, pad_right = 1.4, 1.2, 1.6, 0.5
    panel_w = max_outputs * cell_w + 0.4
    panel_h = max_params  * cell_h + 0.6
    fig_w = 2 * panel_w + pad_left + pad_right + 0.6
    fig_h = 2 * panel_h + pad_top  + pad_bot   + 0.4

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=False)
    fig.subplots_adjust(left=pad_left/fig_w, right=1 - pad_right/fig_w,
                        bottom=pad_bot/fig_h, top=1 - pad_top/fig_h,
                        wspace=0.25, hspace=0.45)

    vmin, vmax = 0.0, 1.0
    cmap = plt.cm.YlOrRd
    im_ref = None

    for row_idx, row in enumerate(layout):
        for col_idx, (motif, has_cable) in enumerate(row):
            ax  = axes[row_idx, col_idx]
            key = (motif, has_cable)
            if key not in group_data:
                ax.set_visible(False)
                continue
            param_names, out_names, mat, conf_mat, mask = group_data[key]
            n_p, n_o = len(param_names), len(out_names)

            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            im_ref = im

            # Grey overlay for masked cells
            grey = np.zeros((*mat.shape, 4))
            grey[mask] = [0.85, 0.85, 0.85, 1.0]
            ax.imshow(grey, aspect="auto", interpolation="nearest",
                      extent=(-0.5, n_o-0.5, n_p-0.5, -0.5))

            for i in range(n_p):
                for j in range(n_o):
                    if mask[i, j]:
                        continue
                    v = mat[i, j]
                    c = conf_mat[i, j]
                    color = "white" if v > 0.6 else "black"
                    ax.text(j, i - 0.20, "{:.2f}".format(v),
                            ha="center", va="center",
                            fontsize=7, color=color, fontweight="bold")
                    ax.text(j, i + 0.18, "±{:.2f}".format(c),
                            ha="center", va="center",
                            fontsize=6, color=color)

            ax.set_xticks(range(n_o))
            ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                               rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(n_p))
            ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names],
                               fontsize=9)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, n_o), minor=True)
            ax.set_yticks(np.arange(-0.5, n_p), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)

            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[has_cable], fontsize=10, labelpad=6)
            if row_idx == 0:
                ax.set_title(MOTIF_TITLES[motif], fontsize=10, pad=5)

    # Horizontal colorbar at the bottom
    if im_ref is not None:
        label = (r"Total-order Sobol index $S_T$" if index == "ST"
                 else r"First-order Sobol index $S_1$")
        plot_left  = pad_left / fig_w
        plot_right = 1 - pad_right / fig_w
        cbar_bot   = 0.18 / fig_h
        cbar_h     = 0.18 / fig_h
        cbar_w     = (plot_right - plot_left) * 0.55
        cbar_left  = plot_left + (plot_right - plot_left - cbar_w) / 2
        cbar_ax = fig.add_axes([cbar_left, cbar_bot, cbar_w, cbar_h])
        cbar = fig.colorbar(im_ref, cax=cbar_ax, orientation="horizontal", label=label)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        cbar.set_label(label, fontsize=10)

    title = ("Variance-based sensitivity: total-order Sobol indices $S_T$"
             if index == "ST"
             else "Variance-based sensitivity: first-order Sobol indices $S_1$")
    fig.suptitle(title, fontsize=12, y=1.01)

    if save:
        fname = "figA_sobol_heatmap_ST" if index == "ST" else "figA2_sobol_s1_heatmap"
        path = os.path.join(FIG_DIR, "{}.pdf".format(fname))
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print("Saved: {}".format(path))
    return fig


# ── Fig A3: Sobol regime map ──────────────────────────────────────────────────

def plot_sobol_regime(save=True):
    """
    2×2 grid (rows: nocable/cable, cols: motif1/motif2).
    Each cell coloured by blend: S1 blue + (ST-S1) orange + (1-ST) grey.
    Shows ST= and S1= values; dominant S2 interaction partner in italic.
    """
    from SALib.sample import saltelli as _saltelli
    from SALib.analyze import sobol as _sobol
    from sobol_analysis import _salib_problem

    BLUE   = np.array([0.18, 0.45, 0.73])
    ORANGE = np.array([0.96, 0.50, 0.06])
    GREY   = np.array([0.88, 0.88, 0.88])

    MOTIF_TITLES = {
        1: r"Motif 1 ($E_2/E_1=2.5$)",
        2: r"Motif 2 ($E_1/E_2=2.5$)",
    }
    ROW_LABELS = {False: "No cable", True: "Cable"}

    layout = [
        [(1, False), (2, False)],
        [(1, True),  (2, True)],
    ]

    def _is_masked(p, o):
        return ((p == "cable_wale_lrest"  and o == "cable_course_tension") or
                (p == "cable_course_lrest" and o == "cable_wale_tension"))

    # Pre-compute all Sobol (with S2) for each group
    group_data = {}
    for motif in [1, 2]:
        for has_cable in [False, True]:
            group = "motif{}_{}".format(motif, "cable" if has_cable else "nocable")
            pkl = os.path.join(DATA_DIR, "{}_scalar_surrogate.pkl".format(group))
            if not os.path.exists(pkl):
                continue
            sur = ScalarSurrogate.load(pkl)
            problem = _salib_problem(has_cable)
            X = _saltelli.sample(problem, 512, calc_second_order=True)
            preds = sur.predict(X)

            out_names = [o for o in SCALAR_OUTPUTS if o in preds
                         and not (o in _CABLE_OUTPUTS and not has_cable)]
            param_names = problem["names"]

            results = {}
            for col in out_names:
                Y = preds[col]
                si = _sobol.analyze(problem, Y, calc_second_order=True,
                                    print_to_console=False)
                results[col] = si
            group_data[(motif, has_cable)] = (param_names, out_names, results)

    # Build figure: 2 rows × 2 cols of sub-heatmaps
    # Determine cell sizes from max param/output counts
    max_params  = max(len(v[0]) for v in group_data.values())
    max_outputs = max(len(v[1]) for v in group_data.values())
    cell_w, cell_h = 0.72, 0.65
    pad_top, pad_bot, pad_left, pad_right = 1.4, 1.2, 1.6, 0.5
    panel_w = max_outputs * cell_w + 0.4
    panel_h = max_params  * cell_h + 0.6
    fig_w = 2 * panel_w + pad_left + pad_right + 0.6
    fig_h = 2 * panel_h + pad_top  + pad_bot   + 0.4

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h),
                             constrained_layout=False)
    fig.subplots_adjust(left=pad_left/fig_w, right=1-pad_right/fig_w,
                        bottom=pad_bot/fig_h, top=1-pad_top/fig_h,
                        wspace=0.25, hspace=0.45)

    for row_idx, row in enumerate(layout):
        for col_idx, (motif, has_cable) in enumerate(row):
            ax = axes[row_idx, col_idx]
            key = (motif, has_cable)
            if key not in group_data:
                ax.set_visible(False)
                continue

            param_names, out_names, results = group_data[key]
            n_p, n_o = len(param_names), len(out_names)

            # Build colour array and annotation data
            rgb_arr = np.zeros((n_p, n_o, 3))
            for i, p in enumerate(param_names):
                for j, o in enumerate(out_names):
                    if _is_masked(p, o):
                        rgb_arr[i, j] = GREY * 0.7
                        continue
                    si = results[o]
                    s1 = float(np.clip(si["S1"][param_names.index(p)], 0, 1))
                    st = float(np.clip(si["ST"][param_names.index(p)], 0, 1))
                    st = max(st, s1)
                    rgb_arr[i, j] = np.clip(s1 * BLUE + (st - s1) * ORANGE
                                            + (1 - st) * GREY, 0, 1)

            ax.imshow(rgb_arr, aspect="auto", interpolation="nearest",
                      extent=(-0.5, n_o-0.5, n_p-0.5, -0.5))

            for i, p in enumerate(param_names):
                for j, o in enumerate(out_names):
                    if _is_masked(p, o):
                        continue
                    si   = results[o]
                    pi   = param_names.index(p)
                    s1   = float(np.clip(si["S1"][pi], 0, 1))
                    st   = float(np.clip(si["ST"][pi], 0, 1))
                    st   = max(st, s1)
                    lum  = 0.299*rgb_arr[i,j,0] + 0.587*rgb_arr[i,j,1] + 0.114*rgb_arr[i,j,2]
                    fc   = "white" if lum < 0.55 else "black"

                    # Dominant S2 partner
                    s2_row = si.get("S2", None)
                    partner = ""
                    if s2_row is not None and st - s1 > 0.05:
                        s2_vec = s2_row[pi].copy() if hasattr(s2_row[pi], 'copy') else np.array(s2_row[pi])
                        s2_vec[pi] = -1
                        best = int(np.argmax(s2_vec))
                        if s2_vec[best] > 0.02:
                            partner = "w. " + PARAM_LABELS.get(param_names[best],
                                                                param_names[best])

                    ax.text(j, i - 0.20, r"$S_T$={:.2f}".format(st),
                            ha="center", va="center", fontsize=7,
                            color=fc, fontweight="bold")
                    ax.text(j, i + 0.15, r"$S_1$={:.2f}".format(s1),
                            ha="center", va="center", fontsize=6, color=fc)
                    if partner:
                        ax.text(j, i + 0.38, partner,
                                ha="center", va="center", fontsize=5.5,
                                color=fc, style="italic")

            ax.set_xticks(range(n_o))
            ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                               rotation=30, ha="right", fontsize=9)
            ax.set_yticks(range(n_p))
            ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names],
                               fontsize=9)
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, n_o), minor=True)
            ax.set_yticks(np.arange(-0.5, n_p), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)

            if col_idx == 0:
                ax.set_ylabel(ROW_LABELS[has_cable], fontsize=10, labelpad=6)
            if row_idx == 0:
                ax.set_title(MOTIF_TITLES[motif], fontsize=10, pad=5)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BLUE,   label=r"Direct ($S_1/S_T$ large)"),
        Patch(facecolor=ORANGE, label=r"Interaction ($S_1/S_T$ small)"),
        Patch(facecolor=GREY,   label=r"Negligible ($S_T \to 0$)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        r"Sobol sensitivity regime map""\n"
        r"{\small Color = $S_1$ (blue) + $(S_T-S_1)$ (orange) + $(1-S_T)$ (grey)"
        r" $|$ italic = dominant $S_2$ partner}",
        fontsize=12, y=1.01,
    )

    if save:
        path = os.path.join(FIG_DIR, "figA3_sobol_regime.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print("Saved: {}".format(path))
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
    plot_sobol_heatmap(index="ST")
    plot_sobol_heatmap(index="S1")
    plot_sobol_regime()
    plot_correlation_heatmap()
    print("Done.")
