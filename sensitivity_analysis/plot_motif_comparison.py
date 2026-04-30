"""
Cross-motif sensitivity comparison.

Fig M1: Grouped bar chart — ST(sf_wale) vs ST(sf_course) for motif 1 and motif 2,
         across crown_height, H_mean_x0, H_mean_y0, mean_stress (no-cable groups).

Fig M2: Sensitivity dominance ratio (ST_course / ST_wale) vs E2/E1 ratio,
         using the two existing motifs.  Placeholder for future motifs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth":  0.8,
    "figure.dpi":      150,
})

# Motifs: label, E2/E1, group key
MOTIFS = [
    ("Motif 1\n$E_2/E_1{=}2.5$",  2.50, "motif1_nocable"),
    ("Motif 2\n$E_2/E_1{=}1.6$",  1.60, "motif2_nocable"),
    ("Motif 3\n$E_2/E_1{=}1.0$",  1.00, "motif3_nocable"),
    ("Motif 4\n$E_1/E_2{=}1.6$",  0.625,"motif4_nocable"),
    ("Motif 5\n$E_1/E_2{=}2.5$",  0.40, "motif5_nocable"),
]

OUTPUTS = [
    ("crown_height", "Crown height"),
    ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)"),
    ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)"),
    ("mean_stress",  "Mean stress"),
]

PARAMS = ["sf_wale", "sf_course", "pressure", "knit_dir"]
PARAM_LABELS = {
    "sf_wale":   r"$s_\mathrm{wale}$",
    "sf_course": r"$s_\mathrm{course}$",
    "pressure":  r"$p$",
    "knit_dir":  r"$\theta_\mathrm{knit}$",
}
PARAM_COLORS = {
    "sf_wale":   "#4878CF",
    "sf_course": "#D65F5F",
    "pressure":  "#6ACC65",
    "knit_dir":  "#B47CC7",
}


def _load_sobol(group, output):
    path = os.path.join(DATA_DIR, f"sobol_{group}_{output}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


# ── Figure M1: grouped ST bars, motif × output ────────────────────────────────

def plot_motif_comparison(save=True):
    n_out = len(OUTPUTS)
    n_motif = len(MOTIFS)

    fig, axes = plt.subplots(
        1, n_out,
        figsize=(3.2 * n_out, 3.8),
        constrained_layout=True,
    )

    x = np.arange(n_motif)
    width = 0.18
    offsets = np.linspace(-(len(PARAMS)-1)/2, (len(PARAMS)-1)/2, len(PARAMS)) * width

    for col_idx, (out_key, out_label) in enumerate(OUTPUTS):
        ax = axes[col_idx]

        for p_idx, param in enumerate(PARAMS):
            st_vals  = []
            st_confs = []
            for _, _, group in MOTIFS:
                df = _load_sobol(group, out_key)
                if df is None or param not in df.index:
                    st_vals.append(0.0)
                    st_confs.append(0.0)
                else:
                    st_vals.append(max(0.0, df.loc[param, "ST"]))
                    st_confs.append(df.loc[param, "ST_conf"])

            ax.bar(
                x + offsets[p_idx],
                st_vals,
                width,
                yerr=st_confs,
                color=PARAM_COLORS[param],
                label=PARAM_LABELS[param],
                error_kw={"elinewidth": 0.8, "capsize": 2},
                alpha=0.88,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in MOTIFS], fontsize=7.5)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Total-order Sobol index $S_T$" if col_idx == 0 else "")
        ax.set_title(out_label, pad=5)
        ax.axhline(1.0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_idx == n_out - 1:
            ax.legend(fontsize=7, loc="upper right",
                      framealpha=0.7, handlelength=1.2)

    fig.suptitle(
        "Sobol total-order sensitivity: Motif 1 vs Motif 2 (no cable)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figM1_motif_sobol_comparison.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Figure M2: dominance ratio vs E2/E1 ───────────────────────────────────────

def plot_dominance_ratio(save=True):
    # mean_stress ratio for motif1 is ~17 — plotted separately as annotated value
    outputs_ratio = [
        ("crown_height", "Crown height",          "#4878CF"),
        ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "#D65F5F"),
        ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "#E47833"),
        ("mean_stress",  "Mean stress",            "#6ACC65"),
    ]

    Y_MAX = 4.5

    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)

    e2_e1_vals = [m[1] for m in MOTIFS]

    for out_key, out_label, color in outputs_ratio:
        ratios = []
        for _, e2e1, group in MOTIFS:
            df = _load_sobol(group, out_key)
            if df is None:
                ratios.append(np.nan)
                continue
            st_c = max(1e-6, df.loc["sf_course", "ST"])
            st_w = max(1e-6, df.loc["sf_wale",   "ST"])
            ratios.append(st_c / st_w)

        # Clip to plot range and annotate out-of-scale values
        plotted = [min(r, Y_MAX * 0.97) for r in ratios]
        ax.plot(e2_e1_vals, plotted, "o-", color=color,
                label=out_label, lw=1.5, ms=6)

        for i, (xv, rv, pv) in enumerate(zip(e2_e1_vals, ratios, plotted)):
            if rv > Y_MAX:
                ax.annotate(f"{rv:.0f}×", xy=(xv, pv),
                            xytext=(xv + 0.07, pv - 0.25),
                            fontsize=7, color=color,
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", alpha=0.6, label="Equal sensitivity")
    ax.axvline(1.0, color="gray", lw=0.6, ls=":", alpha=0.4)

    # Shade regions
    ax.fill_between([0.5, 3.0], [1.0, 1.0], [Y_MAX, Y_MAX],
                    color="#D65F5F", alpha=0.05)
    ax.fill_between([0.5, 3.0], [0.0, 0.0], [1.0, 1.0],
                    color="#4878CF", alpha=0.05)

    ax.text(2.7, Y_MAX * 0.92, "course-dominant", fontsize=7.5,
            color="#AA3333", ha="right", va="top")
    ax.text(2.7, 0.08, "wale-dominant", fontsize=7.5,
            color="#224488", ha="right", va="bottom")

    ax.set_xlabel(r"Stiffness anisotropy $E_2/E_1$  (course / wale)", labelpad=4)
    ax.set_ylabel(r"$S_T^\mathrm{course}\;/\;S_T^\mathrm{wale}$", labelpad=4)
    ax.set_title(
        r"Sensitivity dominance ratio vs material anisotropy",
        pad=6, fontsize=9,
    )
    ax.legend(fontsize=8, framealpha=0.7, loc="upper left")
    ax.set_xlim(0.2, 2.8)
    ax.set_ylim(0.0, Y_MAX)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([0.40, 0.625, 1.0, 1.6, 2.5])
    ax.set_xticklabels(["0.40\n(M5)", "0.625\n(M4)", "1.0\n(M3)", "1.6\n(M2)", "2.5\n(M1)"])

    if save:
        path = os.path.join(FIG_DIR, "figM2_dominance_ratio.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


def plot_knitdir_sensitivity(save=True):
    """
    figM3: ST(theta_knit) vs E2/E1 for each output.
    Shows whether knit direction matters more for near-isotropic motifs.
    """
    outputs = [
        ("crown_height", "Crown height",          "#4878CF"),
        ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "#D65F5F"),
        ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "#E47833"),
        ("mean_stress",  "Mean stress",            "#6ACC65"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    e2_e1_vals = [m[1] for m in MOTIFS]

    for out_key, out_label, color in outputs:
        st_vals = []
        st_conf = []
        for _, _, group in MOTIFS:
            df = _load_sobol(group, out_key)
            if df is None or "knit_dir" not in df.index:
                st_vals.append(np.nan)
                st_conf.append(0.0)
            else:
                st_vals.append(max(0.0, df.loc["knit_dir", "ST"]))
                st_conf.append(df.loc["knit_dir", "ST_conf"])

        ax.errorbar(e2_e1_vals, st_vals, yerr=st_conf,
                    fmt="o-", color=color, label=out_label,
                    lw=1.5, ms=5, capsize=3,
                    elinewidth=0.8)

    ax.axhline(0.0, color="gray", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel(r"Stiffness anisotropy $E_2/E_1$  (course / wale)", labelpad=4)
    ax.set_ylabel(r"$S_T^{\theta_\mathrm{knit}}$", labelpad=4)
    ax.set_title(r"Knitting direction sensitivity vs material anisotropy", pad=6, fontsize=9)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_xlim(0.2, 2.8)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks([0.40, 0.625, 1.0, 1.6, 2.5])
    ax.set_xticklabels(["0.40\n(M5)", "0.625\n(M4)", "1.0\n(M3)", "1.6\n(M2)", "2.5\n(M1)"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save:
        path = os.path.join(FIG_DIR, "figM3_knitdir_sensitivity.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting motif comparison figures...")
    plot_motif_comparison()
    plot_dominance_ratio()
    plot_knitdir_sensitivity()
    print("Done.")
