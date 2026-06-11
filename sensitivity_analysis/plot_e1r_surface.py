"""
Fig R — E1 × (E2/E1) response surfaces from the deterministic grid.

X-axis: E1 (wale stiffness), 1–20 kN/m
Y-axis: E2/E1 ratio (= 1/r), 0.2–1.0  (wale-stiffer at bottom, isotropic at top)

Colour panels: Crown height | Mean stress | H_mean x=0 | H_mean y=0

Fixed: sf_wale=sf_course=1.0, knit_dir=0°, pressure=1000 Pa, nu=0.195.

Requires: run_e1r_grid.py (generates data/e1r_grid_results.csv + e1r_grid_sections.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR

FIG_DIR      = os.path.join(os.path.dirname(__file__), "figures")
RESULTS_CSV  = os.path.join(DATA_DIR, "e1r_grid_results.csv")
SECTIONS_CSV = os.path.join(DATA_DIR, "e1r_grid_sections.csv")

os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth":  0.8,
    "figure.dpi":      150,
})

# (output_key, panel_title, cmap, scale, unit, source)
PANELS = [
    ("crown_height", "Crown height",           "viridis", 1000.0, "mm",          "scalar"),
    ("mean_stress",  "Mean stress",            "plasma",  1.0,    "N/m",         "scalar"),
    ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "cividis", 1.0,    r"m$^{-1}$",   "section"),
    ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "cividis", 1.0,    r"m$^{-1}$",   "section"),
]

N_INTERP = 80   # interpolation grid resolution


def _load_data():
    df_sc  = pd.read_csv(RESULTS_CSV)
    df_sec = pd.read_csv(SECTIONS_CSV)
    df = df_sc.merge(df_sec[["sample_id", "H_mean_x0", "H_mean_y0", "vm_x0", "vm_y0"]],
                     on="sample_id", how="left")
    df["E2_over_E1"] = df["E2"] / df["E1"]
    # Keep only valid runs
    df = df[df["crown_height"] > 0.01].copy()
    return df


def _make_interp_grid():
    e1   = np.linspace(1000, 20000, N_INTERP)
    ratio = np.linspace(0.2, 1.0, N_INTERP)
    E1g, Rg = np.meshgrid(e1, ratio)
    return e1, ratio, E1g, Rg


def plot_surface(save=True):
    df = _load_data()
    e1_pts = df["E1"].values
    r_pts  = df["E2_over_E1"].values
    pts    = np.column_stack([e1_pts, r_pts])

    e1_axis, ratio_axis, E1g, Rg = _make_interp_grid()
    query = np.column_stack([E1g.ravel(), Rg.ravel()])

    n_panels = len(PANELS)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (key, title, cmap, scale, unit, _) in enumerate(PANELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        vals = df[key].values * scale
        mask = np.isfinite(vals)
        if mask.sum() < 4:
            ax.set_visible(False)
            continue

        Z = griddata(pts[mask], vals[mask], query, method="cubic")
        # Fill NaN edges with nearest-neighbour
        nan_mask = ~np.isfinite(Z)
        if nan_mask.any():
            Z_nn = griddata(pts[mask], vals[mask], query, method="nearest")
            Z[nan_mask] = Z_nn[nan_mask]
        Z = Z.reshape(N_INTERP, N_INTERP)

        vmin = np.nanpercentile(Z, 2)
        vmax = np.nanpercentile(Z, 98)

        pcm = ax.pcolormesh(e1_axis / 1000, ratio_axis, Z,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="gouraud")
        ax.contour(e1_axis / 1000, ratio_axis, Z, levels=8,
                   colors="white", linewidths=0.5, alpha=0.5)

        # scatter raw data points
        r_pts_plot = df["E2_over_E1"].values
        ax.scatter(e1_pts / 1000, r_pts_plot, s=8, c="white",
                   alpha=0.5, linewidths=0, zorder=3)

        ax.set_xlabel(r"$E_1$ (kN/m)", labelpad=3)
        ax.set_ylabel(r"$E_2/E_1$  (= $1/r$)", labelpad=3)
        ax.set_title(f"{title}  ({unit})", pad=5)
        ax.set_xlim(1, 20)
        ax.set_ylim(0.2, 1.0)

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(unit, fontsize=7)

    fig.suptitle(
        r"$E_1$ (wale) × $E_2/E_1$ response surfaces"
        "\n"
        r"$s_f{=}1.0$,  $p{=}1000\,\mathrm{Pa}$,  "
        r"$\theta_{knit}{=}0°$,  $\nu_{12}{=}0.195$",
        fontsize=10, y=1.02,
    )

    if save:
        path = os.path.join(FIG_DIR, "figR_e1r_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    if not os.path.exists(RESULTS_CSV) or not os.path.exists(SECTIONS_CSV):
        print("ERROR: Run the FEA grid first:")
        print("  python sensitivity_analysis/run_e1r_grid.py --jobs 8")
        sys.exit(1)
    print("Plotting E1 × (E2/E1) response surfaces...")
    plot_surface()
    print("Done.")
