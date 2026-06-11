"""
Fig R — E1 × (E2/E1) response surfaces from the deterministic grid.

X-axis: E1 (wale stiffness), 1–20 kN/m
Y-axis: E2/E1 ratio (= 1/r), 0.2–1.0  (wale-stiffer at bottom, isotropic at top)

Colour panels: Crown height | Mean stress | H_mean x=0 | H_mean y=0

Fixed: sf_wale=sf_course=1.0, knit_dir=0°, pressure=1000 Pa, nu=0.195.

The grid is exactly 10 E1 × 8 r values so results are pivoted directly into
a 2D matrix — no interpolation.  Failed runs (crown_height ≤ 0.01) and
section-metric outliers (> 3× IQR above Q3) are shown as white (NaN) cells.

Requires: run_e1r_grid.py (generates data/e1r_grid_results.csv + e1r_grid_sections.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from run_e1r_grid import E1_VALUES, R_VALUES

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

# (output_key, panel_title, cmap, scale, unit)
PANELS = [
    ("crown_height", "Crown height",           "viridis", 1000.0, "mm"),
    ("mean_stress",  "Mean stress",            "plasma",  1.0,    "N/m"),
    ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
]

# E2/E1 values corresponding to R_VALUES = E1/E2
E2_OVER_E1 = 1.0 / R_VALUES          # [1.0, 0.80, 0.60, 0.50, 0.40, 0.333, 0.25, 0.20]


def _remove_outliers(series: pd.Series) -> pd.Series:
    """Replace values more than 3×IQR above Q3 with NaN."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3.0 * iqr
    return series.where(series <= upper)


def _load_and_pivot():
    df_sc  = pd.read_csv(RESULTS_CSV)
    df_sec = pd.read_csv(SECTIONS_CSV)
    df = df_sc.merge(
        df_sec[["sample_id", "H_mean_x0", "H_mean_y0", "vm_x0", "vm_y0"]],
        on="sample_id", how="left",
    )
    df["E2_over_E1"] = (df["E2"] / df["E1"]).round(6)

    # --- validity filter ---
    # 1. Failed FEA (solver didn't converge / dome collapsed)
    df.loc[df["crown_height"] <= 0.01, ["crown_height", "mean_stress",
                                         "H_mean_x0", "H_mean_y0"]] = np.nan
    # 2. Section-metric outliers (bad section extraction on edge-case geometries)
    for col in ["H_mean_x0", "H_mean_y0"]:
        df[col] = _remove_outliers(df[col])

    # Round grid keys so pivot_table matching is exact
    df["E1_key"]  = df["E1"].round(0)
    df["r2_key"]  = df["E2_over_E1"].round(6)

    e1_keys  = np.round(E1_VALUES, 0)
    r2_keys  = np.round(E2_OVER_E1, 6)

    grids = {}
    for key, _, scale, _ in [(p[0], p[1], p[3], p[4]) for p in PANELS]:
        mat = np.full((len(r2_keys), len(e1_keys)), np.nan)
        for i, r2 in enumerate(r2_keys):
            for j, e1 in enumerate(e1_keys):
                rows = df[(df["E1_key"] == e1) & (df["r2_key"] == r2)]
                if len(rows) == 1 and not pd.isna(rows[key].values[0]):
                    mat[i, j] = rows[key].values[0] * scale
        grids[key] = mat

    return grids, e1_keys, r2_keys


def plot_surface(save=True):
    grids, e1_keys, r2_keys = _load_and_pivot()

    n_panels = len(PANELS)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # x/y bin edges for pcolormesh (cell centres → edges)
    def _edges(centres):
        c = np.asarray(centres, dtype=float)
        half = np.diff(c) / 2
        return np.concatenate([[c[0] - half[0]], c[:-1] + half, [c[-1] + half[-1]]])

    x_edges = _edges(e1_keys / 1000)          # kN/m
    y_edges = _edges(r2_keys)

    for idx, (key, title, cmap, scale, unit) in enumerate(PANELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        Z = grids[key]
        valid = Z[np.isfinite(Z)]
        if valid.size == 0:
            ax.set_visible(False)
            continue

        vmin, vmax = np.nanpercentile(valid, 2), np.nanpercentile(valid, 98)

        # Mask NaN cells (show as white)
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("white")

        pcm = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(Z),
                            cmap=cmap_obj, vmin=vmin, vmax=vmax)

        # Mark invalid cells with a cross
        for i, r2 in enumerate(r2_keys):
            for j, e1 in enumerate(e1_keys):
                if not np.isfinite(Z[i, j]):
                    ax.text(e1 / 1000, r2, "×", ha="center", va="center",
                            fontsize=8, color="0.5")

        ax.set_xlabel(r"$E_1$ (kN/m)", labelpad=3)
        ax.set_ylabel(r"$E_2/E_1$", labelpad=3)
        ax.set_title(f"{title}  ({unit})", pad=5)
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_ylim(y_edges[0], y_edges[-1])
        ax.set_xticks(e1_keys / 1000)
        ax.set_xticklabels([f"{v:.0f}" if v < 10 else f"{v:.0f}" for v in e1_keys / 1000],
                           fontsize=7, rotation=45)
        ax.set_yticks(r2_keys)
        ax.set_yticklabels([f"{v:.2f}" for v in r2_keys], fontsize=7)

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(unit, fontsize=7)

    n_valid = sum(np.isfinite(grids["crown_height"]).sum()
                  for _ in [1])   # just reuse grid shape
    n_total = grids["crown_height"].size
    n_invalid = np.sum(~np.isfinite(grids["crown_height"]))
    fig.suptitle(
        r"$E_1$ (wale) × $E_2/E_1$ response surfaces  "
        f"({n_total - n_invalid}/{n_total} valid runs, × = failed/outlier)"
        "\n"
        r"$s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  "
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
    print("Plotting E1 × (E2/E1) response surfaces (no interpolation)...")
    plot_surface()
    print("Done.")
