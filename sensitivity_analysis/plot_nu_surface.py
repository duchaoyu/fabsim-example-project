"""
Fig S — E1 × nu12 response surfaces at 3 fixed E2/E1 slices (1.00, 1.60, 2.50).
Fig T — E2/E1 × nu12 response surface at fixed E1 = 7000 N/m, E2/E1 in [1, 5].

E1 is the wale (less stiff) modulus and E2 the course one, so E2/E1 >= 1 on both
figures; the slices are isotropic | motif 2 | motif 1.  Both grids used to sit at
E2/E1 <= 1, the wale-stiff half, which no motif occupies.

The two H_mean panels are replaced by a single dH panel from the pointwise
curvature tensor at the crown, the same estimator and sign convention as
figL/figM/figR: dH = (kappa_y - kappa_x)/(|kappa_y| + |kappa_x|).  The section
estimator disagrees with it in sign on every cell of these grids; see
plot_sf_surface.py for why it cannot carry a directional panel.

figS shares figR's colour limits (plot_e1r_surface.shared_vlims) so a colour
means the same number in both.  figT keeps its own — it is a single E1 slice with
much narrower ranges, and the shared scale would flatten it.

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
from config import DATA_DIR, QUALITY_CROWN_MIN
from plot_e1r_surface import (_dH, _half_diverging, _vlims, shared_vlims,
                              PANEL_IN, W_MARGIN_IN, H_MARGIN_IN)
from run_nu_grid import (
    E1_VALUES, E2_OVER_E1_SLICES, NU12_VALUES,
    E1_FIXED, E2R_VALUES_B,
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

# (key, title, cmap, scale, unit, diverging?)
PANELS = [
    ("crown_height", "Crown height",                 "viridis", 1000.0, "mm",  False),
    ("mean_stress",  "Mean stress",                  "plasma",  1.0,    "N/m", False),
    ("dH_apex",      r"$\Delta H$  (crown tensor)",  "RdBu_r",  1.0,    "",    True),
]

SEC_COLS = ["sample_id", "H_mean_x0", "H_mean_y0", "apex_k_x", "apex_k_y"]


def _prep(df):
    """Derive dH_apex and blank every output on failed runs."""
    # x=0 measures kappa_y, so apex_k_y takes the place H_mean_x0 held
    df["dH_apex"] = _dH(df["apex_k_y"].values, df["apex_k_x"].values)
    df.loc[df["crown_height"] <= QUALITY_CROWN_MIN,
           ["crown_height", "mean_stress", "H_mean_x0", "H_mean_y0",
            "dH_apex"]] = np.nan
    return df


# _vlims and shared_vlims live in plot_e1r_surface so figR and figS cannot
# drift apart.


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


# ── Fig S: E1 × nu12, columns = 3 E2/E1 slices ──────────────────────────────

def plot_figS(save=True):
    n_rows = len(PANELS)
    n_cols = len(E2_OVER_E1_SLICES)
    # Canvas sized from the panel box (see plot_e1r_surface): each row gets a
    # PANEL_IN square per column plus its own inches for tick labels and the
    # spanning colorbar, so neither eats into the panels.
    # Measured, not guessed: at PANEL_IN the panels come out ~3.8 in and the
    # x tick labels + colorbar + its label occupy ~1.0 in, so this is the room
    # each row needs beyond its panel.  Budgeting more just opens dead gaps
    # between rows, because set_box_aspect stops the panels absorbing it.
    ROW_EXTRA = 0.50   # x tick labels + horizontal colorbar + its label
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * PANEL_IN + W_MARGIN_IN,
                                      n_rows * (PANEL_IN + ROW_EXTRA) + H_MARGIN_IN),
                             constrained_layout=True)
    # Reserve a band at the top for the suptitle.  figS has per-column headers
    # directly under it, and without the reservation the two collide; enlarging
    # the figure does not help because constrained_layout gives surplus height to
    # the inter-row gaps instead of the top margin.
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.972))

    # Pre-load merged dataframe for vlims
    _dfA_r = pd.read_csv(RESULTS_A_CSV)
    _dfA_s = pd.read_csv(SECTIONS_A_CSV)
    _dfA = _prep(_dfA_r.merge(_dfA_s[SEC_COLS], on="sample_id", how="left"))

    # Colour limits pooled with figR, so a colour means the same number in both
    # (shared_vlims).  These are shared down each row across all three columns
    # too, which is what makes the slices comparable with one another.
    row_vlims = shared_vlims()

    for col_idx, e2r in enumerate(E2_OVER_E1_SLICES):
        sub = _dfA[_dfA["e2_over_e1"].round(6) == round(e2r, 6)].copy()
        for col in ["H_mean_x0", "H_mean_y0"]:
            sub[col] = _remove_outliers(sub[col])

        e1_r  = np.round(E1_VALUES,   6)
        nu_r  = np.round(NU12_VALUES, 6)
        sub["_e1"] = sub["E1"].round(6)
        sub["_nu"] = sub["nu"].round(6)

        for row_idx, (key, title, cmap, scale, unit, diverging) in enumerate(PANELS):
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
            vmin, vmax, half = row_vlims[key]
            panel_cmap = _half_diverging(cmap, half) if half else cmap

            ax.pcolormesh(x_fine / 1000, y_fine, Z_fine,
                          cmap=panel_cmap, vmin=vmin, vmax=vmax,
                          shading="gouraud", rasterized=True)
            ax.contour(x_fine / 1000, y_fine, Z_fine, levels=6,
                       colors="white", linewidths=0.5, alpha=0.45)
            if diverging and Z_fine.min() < 0 < Z_fine.max():
                ax.contour(x_fine / 1000, y_fine, Z_fine, levels=[0.0],
                           colors="black", linewidths=1.0)

            if row_idx == 0:
                ax.set_title(rf"$E_2/E_1 = {e2r:.2f}$", pad=6, fontsize=9)

            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$E_1$ (kN/m)", labelpad=3)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(r"$\nu_{12}$", labelpad=3)
                ax.text(-0.30, 0.5, f"{title}\n({unit})" if unit else title,
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=8.5)
            else:
                ax.set_yticklabels([])

            ax.set_box_aspect(1)          # keep the data panel square
            ax.set_xlim(e1_r[0] / 1000, e1_r[-1] / 1000)
            ax.set_ylim(nu_r[0], nu_r[-1])
            ax.set_xticks(e1_r / 1000)
            ax.set_xticklabels([f"{v:.0f}" for v in e1_r / 1000], fontsize=7, rotation=45)
            ax.set_yticks(nu_r)
            ax.set_yticklabels([f"{v:.2f}" for v in nu_r], fontsize=7)

            # One shared colorbar per row, horizontal, spanning all columns —
            # the row already shares its limits, so one bar describes all three.
            if col_idx == n_cols - 1:
                pcm = ax.collections[0]
                cb = fig.colorbar(pcm, ax=list(axes[row_idx, :]),
                                  orientation="horizontal", location="bottom",
                                  aspect=55, fraction=0.045, pad=0.02)
                if unit:
                    cb.set_label(unit, fontsize=7)
                cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"Fig S — $E_1$ × $\nu_{12}$ response surfaces,  $s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  $\theta_\mathrm{knit}{=}0°$",
        fontsize=10, y=0.988,
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
    df = _prep(df_r.merge(df_s[SEC_COLS], on="sample_id", how="left"))
    for col in ["H_mean_x0", "H_mean_y0"]:
        df[col] = _remove_outliers(df[col])

    e2r_vals = np.round(E2R_VALUES_B, 6)             # ascending: 1.00 → 5.00
    nu_r     = np.round(NU12_VALUES, 6)
    df["_e2r"] = df["e2_over_e1"].round(6)
    df["_nu"]  = df["nu"].round(6)

    n_panels = len(PANELS)
    ncols = 3 if n_panels <= 3 else 2
    nrows = int(np.ceil(n_panels / ncols))
    ROW_EXTRA = 1.25   # x tick labels + xlabel + colorbar + its label
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * PANEL_IN + W_MARGIN_IN,
                                      nrows * (PANEL_IN + ROW_EXTRA) + H_MARGIN_IN),
                             constrained_layout=True, squeeze=False)

    for idx, (key, title, cmap, scale, unit, diverging) in enumerate(PANELS):
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
        vmin, vmax, half = _vlims(valid, diverging)
        panel_cmap = _half_diverging(cmap, half) if half else cmap

        pcm = ax.pcolormesh(x_fine, y_fine, Z_fine,
                            cmap=panel_cmap, vmin=vmin, vmax=vmax,
                            shading="gouraud", rasterized=True)
        ax.contour(x_fine, y_fine, Z_fine, levels=6,
                   colors="white", linewidths=0.5, alpha=0.45)
        if diverging and Z_fine.min() < 0 < Z_fine.max():
            ax.contour(x_fine, y_fine, Z_fine, levels=[0.0],
                       colors="black", linewidths=1.0)

        ax.set_box_aspect(1)          # keep the data panel square
        ax.set_xlabel(r"$\nu_{12}$", labelpad=3)
        ax.set_ylabel(r"$E_2/E_1$", labelpad=3)
        ax.set_title(f"{title}  ({unit})" if unit else title, pad=5)
        ax.set_xlim(nu_r[0], nu_r[-1])
        ax.set_ylim(e2r_vals[0], e2r_vals[-1])
        ax.set_xticks(nu_r)
        ax.set_xticklabels([f"{v:.2f}" for v in nu_r], fontsize=7, rotation=45)
        ax.set_yticks(e2r_vals)
        ax.set_yticklabels([f"{v:.2f}" for v in e2r_vals], fontsize=7)

        cb = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                          location="bottom", pad=0.06, aspect=30,
                          fraction=0.05)
        if unit:
            cb.set_label(unit, fontsize=7)
        cb.ax.tick_params(labelsize=7)

    for idx in range(n_panels, nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)

    n_total   = len(df)
    n_invalid = int((df["crown_height"].isna() | (df["crown_height"] <= QUALITY_CROWN_MIN)).sum())
    fig.suptitle(
        rf"Fig T — $E_2/E_1$ × $\nu_{{12}}$ response surfaces  ($E_1 = {E1_FIXED:.0f}$ N/m)"
        "\n"
        r"$s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  $\theta_\mathrm{knit}{=}0°$"
        f"   ({n_total - n_invalid}/{n_total} valid)",
        fontsize=10,
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
