"""
Fig S — E1 × nu12 response surfaces at 3 fixed E2/E1 slices (0.25, 0.50, 1.00).
Fig T — E2/E1 × nu12 response surface at fixed E1 = 7000 N/m.

Both use bicubic spline interpolation onto a 200×200 display grid.
Requires: run_nu_grid.py to have been run first.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from run_nu_grid import (
    E1_VALUES, E2_OVER_E1_SLICES, NU12_VALUES,
    E1_FIXED, R_VALUES_B,
    RESULTS_A_CSV, SECTIONS_A_CSV,
    RESULTS_B_CSV, SECTIONS_B_CSV,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_FINE = 200

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

# (key, title, cmap, scale, unit)
PANELS = [
    ("crown_height", "Crown height",           "viridis", 1000.0, "mm"),
    ("mean_stress",  "Mean stress",            "plasma",  1.0,    "N/m"),
    ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
]


def _remove_outliers(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return s.where(s <= q3 + 3.0 * (q3 - q1))


def _smooth(Z, x_keys, y_keys):
    """Bicubic spline on grid (x_keys ascending, y_keys ascending)."""
    Z_clean = Z.copy().astype(float)
    for i in range(Z_clean.shape[0]):
        v = Z_clean[i, np.isfinite(Z_clean[i])]
        if v.size > 0:
            Z_clean[i, ~np.isfinite(Z_clean[i])] = v.mean()
    sp     = RectBivariateSpline(y_keys, x_keys, Z_clean, kx=3, ky=3)
    x_fine = np.linspace(x_keys[0], x_keys[-1], N_FINE)
    y_fine = np.linspace(y_keys[0], y_keys[-1], N_FINE)
    return x_fine, y_fine, sp(y_fine, x_fine)


def _load_grid(results_csv, sections_csv, x_col, y_col, x_vals, y_vals):
    """Load, merge, filter, and pivot into {key: 2D array}."""
    df = pd.read_csv(results_csv).merge(
        pd.read_csv(sections_csv)[["sample_id", "H_mean_x0", "H_mean_y0", "vm_x0", "vm_y0"]],
        on="sample_id", how="left",
    )
    df.loc[df["crown_height"] <= 0.01, ["crown_height", "mean_stress", "H_mean_x0", "H_mean_y0"]] = np.nan
    for col in ["H_mean_x0", "H_mean_y0"]:
        df[col] = _remove_outliers(df[col])

    x_r = np.round(x_vals, 6)
    y_r = np.round(y_vals, 6)
    df["_x"] = df[x_col].round(6)
    df["_y"] = df[y_col].round(6)

    scale_map = {p[0]: p[3] for p in PANELS}
    grids = {}
    for key, _, _, scale, _ in PANELS:
        mat = np.full((len(y_r), len(x_r)), np.nan)
        for i, yv in enumerate(y_r):
            for j, xv in enumerate(x_r):
                rows = df[(df["_x"] == xv) & (df["_y"] == yv)]
                if len(rows) == 1 and not pd.isna(rows[key].values[0]):
                    mat[i, j] = rows[key].values[0] * scale
        grids[key] = mat
    return grids


# ── Fig S: E1 × nu12, columns = 3 E2/E1 slices ──────────────────────────────

def plot_figS(save=True):
    n_rows = len(PANELS)
    n_cols = len(E2_OVER_E1_SLICES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.0 * n_rows),
                             constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.93))

    # Pre-load merged dataframe for vlims
    _dfA_r = pd.read_csv(RESULTS_A_CSV)
    _dfA_s = pd.read_csv(SECTIONS_A_CSV)
    _dfA = _dfA_r.merge(_dfA_s[["sample_id", "H_mean_x0", "H_mean_y0"]], on="sample_id", how="left")
    _dfA.loc[_dfA["crown_height"] <= 0.01, ["crown_height", "mean_stress", "H_mean_x0", "H_mean_y0"]] = np.nan

    # Per-row shared colour limits (across all columns)
    row_vlims = {}
    for key, _, _, scale, _ in PANELS:
        v = _dfA[key].dropna().values * scale
        row_vlims[key] = (np.nanpercentile(v, 2), np.nanpercentile(v, 98)) if v.size > 0 else (0, 1)

    for col_idx, e2r in enumerate(E2_OVER_E1_SLICES):
        sub = _dfA[_dfA["e2_over_e1"].round(6) == round(e2r, 6)].copy()
        for col in ["H_mean_x0", "H_mean_y0"]:
            sub[col] = _remove_outliers(sub[col])

        e1_r  = np.round(E1_VALUES,   6)
        nu_r  = np.round(NU12_VALUES, 6)
        sub["_e1"] = sub["E1"].round(6)
        sub["_nu"] = sub["nu"].round(6)

        for row_idx, (key, title, cmap, scale, unit) in enumerate(PANELS):
            ax = axes[row_idx, col_idx]

            mat = np.full((len(nu_r), len(e1_r)), np.nan)
            for i, nv in enumerate(nu_r):
                for j, e1v in enumerate(e1_r):
                    rows = sub[(sub["_e1"] == e1v) & (sub["_nu"] == nv)]
                    if len(rows) == 1 and not pd.isna(rows[key].values[0]):
                        mat[i, j] = rows[key].values[0] * scale

            valid = mat[np.isfinite(mat)]
            if valid.size == 0:
                ax.set_visible(False)
                continue

            x_fine, y_fine, Z_fine = _smooth(mat, e1_r, nu_r)
            vmin, vmax = row_vlims[key]

            ax.pcolormesh(x_fine / 1000, y_fine, Z_fine,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="gouraud", rasterized=True)
            ax.contour(x_fine / 1000, y_fine, Z_fine, levels=6,
                       colors="white", linewidths=0.5, alpha=0.45)

            if row_idx == 0:
                ax.set_title(rf"$E_2/E_1 = {e2r:.2f}$", pad=6, fontsize=9)

            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$E_1$ (kN/m)", labelpad=3)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(r"$\nu_{12}$", labelpad=3)
                ax.text(-0.30, 0.5, f"{title}\n({unit})",
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.5)
            else:
                ax.set_yticklabels([])

            ax.set_xlim(e1_r[0] / 1000, e1_r[-1] / 1000)
            ax.set_ylim(nu_r[0], nu_r[-1])
            ax.set_xticks(e1_r / 1000)
            ax.set_xticklabels([f"{v:.0f}" for v in e1_r / 1000], fontsize=7, rotation=45)
            ax.set_yticks(nu_r)
            ax.set_yticklabels([f"{v:.2f}" for v in nu_r], fontsize=7)

            # shared colorbar on rightmost column
            if col_idx == n_cols - 1:
                pcm = ax.collections[0]
                cb = fig.colorbar(pcm, ax=ax, pad=0.03, aspect=22)
                cb.set_label(unit, fontsize=7)
                cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"Fig S — $E_1$ × $\nu_{12}$ response surfaces  (3 anisotropy slices)"
        "\n"
        r"$s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  $\theta_\mathrm{knit}{=}0°$",
        fontsize=10, y=0.97,
    )
    if save:
        path = os.path.join(FIG_DIR, "figS_nu_E1_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig T: E2/E1 × nu12 at fixed E1=7000 ────────────────────────────────────

def plot_figT(save=True):
    df_r = pd.read_csv(RESULTS_B_CSV)
    df_s = pd.read_csv(SECTIONS_B_CSV)
    df = df_r.merge(df_s[["sample_id", "H_mean_x0", "H_mean_y0"]], on="sample_id", how="left")
    df.loc[df["crown_height"] <= 0.01, ["crown_height", "mean_stress", "H_mean_x0", "H_mean_y0"]] = np.nan
    for col in ["H_mean_x0", "H_mean_y0"]:
        df[col] = _remove_outliers(df[col])

    e2r_vals = np.round(1.0 / R_VALUES_B, 6)[::-1]   # ascending: 0.20 → 1.00
    nu_r     = np.round(NU12_VALUES, 6)
    df["_e2r"] = df["e2_over_e1"].round(6)
    df["_nu"]  = df["nu"].round(6)

    n_panels = len(PANELS)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (key, title, cmap, scale, unit) in enumerate(PANELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        mat = np.full((len(e2r_vals), len(nu_r)), np.nan)
        for i, e2r in enumerate(e2r_vals):
            for j, nv in enumerate(nu_r):
                rows = df[(df["_e2r"] == e2r) & (df["_nu"] == nv)]
                if len(rows) == 1 and not pd.isna(rows[key].values[0]):
                    mat[i, j] = rows[key].values[0] * scale

        valid = mat[np.isfinite(mat)]
        if valid.size == 0:
            ax.set_visible(False)
            continue

        x_fine, y_fine, Z_fine = _smooth(mat, nu_r, e2r_vals)
        vmin = np.nanpercentile(valid, 2)
        vmax = np.nanpercentile(valid, 98)

        pcm = ax.pcolormesh(x_fine, y_fine, Z_fine,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="gouraud", rasterized=True)
        ax.contour(x_fine, y_fine, Z_fine, levels=6,
                   colors="white", linewidths=0.5, alpha=0.45)

        ax.set_xlabel(r"$\nu_{12}$", labelpad=3)
        ax.set_ylabel(r"$E_2/E_1$", labelpad=3)
        ax.set_title(f"{title}  ({unit})", pad=5)
        ax.set_xlim(nu_r[0], nu_r[-1])
        ax.set_ylim(e2r_vals[0], e2r_vals[-1])
        ax.set_xticks(nu_r)
        ax.set_xticklabels([f"{v:.2f}" for v in nu_r], fontsize=7, rotation=45)
        ax.set_yticks(e2r_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in e2r_vals], fontsize=7)

        cb = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=20)
        cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=7)

    n_total   = len(df)
    n_invalid = int((df["crown_height"].isna() | (df["crown_height"] <= 0.01)).sum())
    fig.suptitle(
        rf"Fig T — $E_2/E_1$ × $\nu_{{12}}$ response surfaces  ($E_1 = {E1_FIXED:.0f}$ N/m)"
        "\n"
        r"$s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  $\theta_\mathrm{knit}{=}0°$"
        f"   ({n_total - n_invalid}/{n_total} valid)",
        fontsize=10, y=1.02,
    )
    if save:
        path = os.path.join(FIG_DIR, "figT_nu_e2r_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    missing = []
    for p in [RESULTS_A_CSV, SECTIONS_A_CSV, RESULTS_B_CSV, SECTIONS_B_CSV]:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print("Missing data files — run first:")
        print("  python sensitivity_analysis/run_nu_grid.py --jobs 8")
        for p in missing:
            print(f"  {p}")
        import sys; sys.exit(1)

    print("Plotting Fig S (E1 × nu12, 3 anisotropy slices)...")
    plot_figS()
    print("Plotting Fig T (E2/E1 × nu12, E1=7000)...")
    plot_figT()
    print("Done.")
