"""
Visualization for sensitivity analysis results.

Figure 1: sf_wale × sf_course → crown_height response surface (contour)
Figure 2: pressure → crown_height curve (motif1 vs motif2)
Figure 3: Sobol S1 + ST bar charts (one per group)
Figure 4: cable vs no-cable curvature field comparison
Figure 5: motif1 vs motif2 stress field comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import (
    MOTIFS, HAS_CABLE, SCALAR_OUTPUTS,
    PARAMS_NO_CABLE, PARAMS_CABLE, DATA_DIR,
)
from surrogate import ScalarSurrogate, FieldSurrogate

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Shared publication style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

PARAM_LABELS = {
    "sf_wale":     r"$s_\mathrm{wale}$",
    "sf_course":   r"$s_\mathrm{course}$",
    "knit_dir":    r"$\theta_\mathrm{knit}$ (°)",
    "pressure":    r"$p$ (Pa)",
    "cable_angle": r"$\phi_\mathrm{cable}$ (°)",
}
OUTPUT_LABELS = {
    "crown_height":           r"$h_\mathrm{crown}$ (m)",
    "H_mean_x0":              r"$\bar{H}_{x=0}$ (m$^{-1}$)",
    "H_mean_y0":              r"$\bar{H}_{y=0}$ (m$^{-1}$)",
    "max_stress":             r"$\sigma_\mathrm{max}$ (Pa)",
    "mean_stress":            r"$\bar{\sigma}$ (Pa)",
    "cable_tension":          r"$T_\mathrm{cable}$ (N)",
    "boundary_reaction_mean": r"$\bar{R}_\mathrm{bdry}$ (N)",
}
GROUP_TITLES = {
    "motif1_nocable": "Motif 1 — no cable",
    "motif1_cable":   "Motif 1 — cable",
    "motif2_nocable": "Motif 2 — no cable",
    "motif2_cable":   "Motif 2 — cable",
}

# Colorblind-friendly pair (blue / vermillion)
_C1, _C2 = "#0077BB", "#EE7733"


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf",           bbox_inches="tight")
    print(f"Saved: {path_no_ext}.png / .pdf")
    plt.close(fig)


def _default_params(has_cable=False):
    bounds = PARAMS_CABLE if has_cable else PARAMS_NO_CABLE
    return {k: (lo + hi) / 2.0 for k, (lo, hi) in bounds.items()}


# ── Figure 1: sf_wale × sf_course response surface ────────────────────────────
def plot_response_surface(surrogate_m1: ScalarSurrogate,
                           output: str = "crown_height",
                           n_grid: int = 60,
                           save: bool = True):
    """sf_wale × sf_course → output, with all other params at midpoint."""
    sf_wale   = np.linspace(0.8, 1.4, n_grid)
    sf_course = np.linspace(0.8, 1.4, n_grid)
    W, C = np.meshgrid(sf_wale, sf_course)

    defaults = _default_params(has_cable=False)
    keys = list(PARAMS_NO_CABLE.keys())
    X = np.column_stack([
        W.ravel() if k == "sf_wale" else
        C.ravel() if k == "sf_course" else
        np.full(n_grid * n_grid, defaults[k])
        for k in keys
    ])
    Z = surrogate_m1.predict(X)[output].reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    cs = ax.contourf(W, C, Z, levels=20, cmap="plasma")
    ax.contour(W, C, Z, levels=10, colors="white", linewidths=0.4, alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = plt.colorbar(cs, cax=cax)
    cb.set_label(OUTPUT_LABELS.get(output, output), labelpad=6)

    ax.set_xlabel(PARAM_LABELS.get("sf_wale",   "sf_wale"))
    ax.set_ylabel(PARAM_LABELS.get("sf_course", "sf_course"))
    ax.set_title(
        f"Response surface — {OUTPUT_LABELS.get(output, output)}, Motif 1\n"
        r"(at median $\theta_\mathrm{knit}$, $p$)",
        pad=8,
    )

    if save:
        _save(fig, os.path.join(FIG_DIR, f"fig1_response_surface_{output}"))
    return fig


# ── Figure 2: pressure → crown_height (motif1 vs motif2) ──────────────────────
def plot_pressure_curve(surrogate_m1: ScalarSurrogate,
                         surrogate_m2: ScalarSurrogate,
                         output: str = "crown_height",
                         save: bool = True):
    pressures = np.linspace(200, 1200, 80)
    defaults  = _default_params(has_cable=False)
    keys      = list(PARAMS_NO_CABLE.keys())

    def _predict(surrogate):
        X = np.column_stack([
            pressures if k == "pressure" else
            np.full(len(pressures), defaults[k])
            for k in keys
        ])
        return surrogate.predict(X)[output]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(pressures, _predict(surrogate_m1), lw=2.0,
            color=_C1, label="Motif 1 (wale-stiff)")
    ax.plot(pressures, _predict(surrogate_m2), lw=2.0, ls="--",
            color=_C2, label="Motif 2 (course-stiff)")
    ax.set_xlabel(r"Inflation pressure $p$ (Pa)")
    ax.set_ylabel(OUTPUT_LABELS.get(output, output))
    ax.set_title(
        f"{OUTPUT_LABELS.get(output, output).split('(')[0].strip()} "
        "vs. inflation pressure",
        pad=8,
    )
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)

    if save:
        _save(fig, os.path.join(FIG_DIR, f"fig2_pressure_curve_{output}"))
    return fig


# ── Figure 3: Sobol S1 + ST bar charts ────────────────────────────────────────
def plot_sobol(sobol_results: dict, save: bool = True):
    """sobol_results: {group: {output: DataFrame(index=params, cols=S1,ST,...)}}"""
    for group, group_results in sobol_results.items():
        has_cable = "cable" in group and "nocable" not in group

        # Drop outputs that are trivially zero (e.g. cable_tension without cable)
        active = {
            col: df for col, df in group_results.items()
            if df["ST"].clip(0).max() > 0.02
        }
        if not active:
            continue

        n_out = len(active)
        ncols = min(n_out, 3)
        nrows = (n_out + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.6 * ncols, 3.0 * nrows),
            squeeze=False,
        )

        legend_added = False
        for idx, (col, df) in enumerate(active.items()):
            ax = axes[idx // ncols][idx % ncols]
            x  = np.arange(len(df))
            w  = 0.35

            kw = dict(capsize=3, error_kw={"linewidth": 0.8, "capthick": 0.8})
            b1 = ax.bar(x - w/2, df["S1"].clip(0), w,
                        color=_C1, alpha=0.85, label=r"$S_1$",
                        yerr=df["S1_conf"], **kw)
            b2 = ax.bar(x + w/2, df["ST"].clip(0), w,
                        color=_C2, alpha=0.85, label=r"$S_T$",
                        yerr=df["ST_conf"], **kw)

            ax.set_xticks(x)
            ax.set_xticklabels(
                [PARAM_LABELS.get(p, p) for p in df.index],
                rotation=30, ha="right",
            )
            ax.set_title(OUTPUT_LABELS.get(col, col), pad=5)
            ax.set_ylim(0, 1.15)
            ax.set_ylabel("Sobol index")
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
            ax.grid(axis="y", linewidth=0.4, alpha=0.4)
            ax.spines["left"].set_visible(True)

            if not legend_added:
                ax.legend(frameon=False, loc="upper right")
                legend_added = True

        # Hide any unused axes
        for idx in range(len(active), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(GROUP_TITLES.get(group, group), fontsize=11, y=1.01)
        fig.tight_layout()

        if save:
            _save(fig, os.path.join(FIG_DIR, f"fig3_sobol_{group}"))


# ── Figure 4: cable vs no-cable curvature field ────────────────────────────────
def plot_cable_vs_nocable(
    V: np.ndarray, F: np.ndarray,
    curvature_no_cable: np.ndarray,
    curvature_cable: np.ndarray,
    save: bool = True,
):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, curv, title in zip(
        axes,
        [curvature_no_cable, curvature_cable],
        ["No cable", "With cable"],
    ):
        sc = ax.tripcolor(V[:, 0], V[:, 1], F,
                          facecolors=curv, cmap="RdBu_r")
        plt.colorbar(sc, ax=ax, label=r"Mean curvature (m$^{-1}$)")
        ax.set_aspect("equal")
        ax.set_title(title, pad=6)
        ax.axis("off")
    fig.suptitle("Mean curvature: cable vs. no-cable (matched parameters)", fontsize=11)
    fig.tight_layout()
    if save:
        _save(fig, os.path.join(FIG_DIR, "fig4_cable_vs_nocable_curvature"))
    return fig


# ── Figure 5: motif1 vs motif2 stress field ────────────────────────────────────
def plot_motif_comparison(
    V: np.ndarray, F: np.ndarray,
    stress_m1: np.ndarray,
    stress_m2: np.ndarray,
    save: bool = True,
):
    vmin = min(stress_m1.min(), stress_m2.min())
    vmax = max(stress_m1.max(), stress_m2.max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, stress, title in zip(
        axes,
        [stress_m1, stress_m2],
        ["Motif 1 (wale-stiff)", "Motif 2 (course-stiff)"],
    ):
        sc = ax.tripcolor(V[:, 0], V[:, 1], F,
                          facecolors=stress, cmap="plasma",
                          vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label=r"Von Mises stress $\sigma$ (Pa)")
        ax.set_aspect("equal")
        ax.set_title(title, pad=6)
        ax.axis("off")
    fig.suptitle("Von Mises stress: Motif 1 vs. Motif 2 (matched parameters)", fontsize=11)
    fig.tight_layout()
    if save:
        _save(fig, os.path.join(FIG_DIR, "fig5_motif_stress_comparison"))
    return fig
