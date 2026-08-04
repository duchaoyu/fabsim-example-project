"""
Fig R — E1 × (E2/E1) response surfaces from the deterministic grid.

X-axis: E1 (wale stiffness), 1–20 kN/m
Y-axis: E2/E1, 1–5  (isotropic at the bottom, strongly course-stiff at the top)

Colour panels: Crown height | Mean stress | dH (crown tensor)

E1 is the wale (less stiff) modulus and E2 the course one, so E2/E1 >= 1 over the
whole grid — the regime the motifs occupy (motif 1: 2.50, motif 2: 1.60).  The
grid used to be swept over E2/E1 in [0.2, 1], the wale-stiff half, which contains
no motif.

The two H_mean panels are replaced by a single dH panel taken from the pointwise
curvature tensor at the crown, the same estimator and sign convention as
figL/figM: dH = (kappa_y - kappa_x)/(|kappa_y| + |kappa_x|).  The section
estimator disagrees with it in sign on every cell of this grid (section +0.04 to
+0.16, apex -0.13 to -0.00): it averages |kappa| over a whole diameter and both
cut planes share the apex and the clamped rim, so it cancels the directional
signal.  See plot_sf_surface.py for the fuller argument.

Colour limits are pooled with figS (shared_vlims), so a colour means the same
number in both figures.

Fixed: sf_wale=sf_course=1.1, knit_dir=0°, pressure=1000 Pa, nu=0.195.

The 10×8 FEA grid is bicubic-interpolated onto a 200×200 display grid so
colours blend smoothly.  Failed/outlier cells are excluded before fitting.

Requires: run_e1r_grid.py (generates data/e1r_grid_results.csv + e1r_grid_sections.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, QUALITY_CROWN_MIN
from run_e1r_grid import E1_VALUES, E2R_VALUES
# figR's colour scale is pooled with figS's grid, so this module needs figS's
# data paths.  run_nu_grid imports nothing from here, so there is no cycle.
from run_nu_grid import (RESULTS_A_CSV as NU_A_RESULTS_CSV,
                         SECTIONS_A_CSV as NU_A_SECTIONS_CSV)

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
# (output_key, panel_title, cmap, scale, unit, diverging?)
PANELS = [
    ("crown_height", "Crown height",                    "viridis", 1000.0, "mm",  False),
    ("mean_stress",  "Mean stress",                     "plasma",  1.0,    "N/m", False),
    ("dH_apex",      r"$\Delta H$  (crown tensor)",     "RdBu_r",  1.0,    "",    True),
]

# The sweep variable is E2/E1 itself now, ascending from isotropic.
E2_OVER_E1 = E2R_VALUES              # [1.0, 1.25, 1.67, 2.0, 2.5, 3.0, 4.0, 5.0]


def _dH(a, b):
    """Normalised difference, identical to plot_sf_surface._dH."""
    d = np.abs(a) + np.abs(b)
    return np.where(d > 1e-12, (a - b) / d, np.nan)


def _vlims(valid, diverging):
    """Colour limits for one panel, plus which half of a diverging map to use.

    Returns (vmin, vmax, half) where half is None, "low" or "high".  On a
    diverging panel zero stays at the map's white midpoint so the colour reads as
    a sign, as in figL.  When the data is one-signed a symmetric [-lim, +lim]
    scale would leave half the ramp unused and render the panel almost blank, so
    the map is sliced to the occupied half instead — white still means zero.
    """
    if not diverging:
        return np.nanpercentile(valid, 2), np.nanpercentile(valid, 98), None
    lim = np.abs(valid).max()
    if valid.min() < 0 < valid.max():
        return -lim, lim, None
    if valid.max() <= 0:
        return -lim, 0.0, "low"
    return 0.0, lim, "high"


def _panel_values(results_csv, sections_csv):
    """One grid's panel values, prepared the way both figures prepare them."""
    df = pd.read_csv(results_csv).merge(
        pd.read_csv(sections_csv)[["sample_id", "H_mean_x0", "H_mean_y0",
                                   "apex_k_x", "apex_k_y"]],
        on="sample_id", how="left",
    )
    df["dH_apex"] = _dH(df["apex_k_y"].values, df["apex_k_x"].values)
    df.loc[df["crown_height"] <= QUALITY_CROWN_MIN,
           ["crown_height", "mean_stress", "H_mean_x0", "H_mean_y0",
            "dH_apex"]] = np.nan
    return df


def shared_vlims():
    """Per-panel colour limits pooled over the figR and figS grids.

    figR (E1 × E2/E1 at nu=0.195) and figS (E1 × nu12 at three E2/E1 slices)
    show the same three quantities over the same E1 range and are read side by
    side, so a colour has to mean the same number in both.  Deriving the limits
    from each figure's own data would break that: on their own grids crown height
    spans 50-450 mm vs 100-470 mm and mean stress 1.0-9.2 vs 1.0-5.6 kN/m, so
    equal colours would stand for different numbers.  Pooling both grids and
    applying one rule to the union is what makes the two comparable.

    figT is deliberately excluded — it is a single E1 slice whose ranges are much
    narrower, and forcing it onto this scale would flatten it to near-uniform.
    """
    pool = pd.concat(
        [_panel_values(RESULTS_CSV, SECTIONS_CSV),
         _panel_values(NU_A_RESULTS_CSV, NU_A_SECTIONS_CSV)],
        ignore_index=True,
    )
    out = {}
    for key, _, _, scale, _, diverging in PANELS:
        v = pool[key].dropna().values * scale
        out[key] = _vlims(v, diverging) if v.size else (0.0, 1.0, None)
    return out


def _half_diverging(name, side):
    """The half of a diverging colormap on one side of its midpoint.

    Keeps the midpoint colour (white) attached to zero when the data is
    one-signed, so a colour still means the same value as in a full symmetric
    panel — it just does not waste half the ramp on a sign that never occurs.
    """
    base = plt.get_cmap(name)
    frac = (np.linspace(0.0, 0.5, 256) if side == "low"
            else np.linspace(0.5, 1.0, 256))
    return LinearSegmentedColormap.from_list(f"{name}_{side}", base(frac))


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
        df_sec[["sample_id", "H_mean_x0", "H_mean_y0", "vm_x0", "vm_y0",
                "apex_k_x", "apex_k_y"]],
        on="sample_id", how="left",
    )
    df["E2_over_E1"] = (df["E2"] / df["E1"]).round(6)
    # x=0 measures kappa_y, so apex_k_y takes the place H_mean_x0 held
    df["dH_apex"] = _dH(df["apex_k_y"].values, df["apex_k_x"].values)

    # --- validity filter ---
    # 1. Failed FEA (solver didn't converge / dome collapsed)
    df.loc[df["crown_height"] <= QUALITY_CROWN_MIN, ["crown_height", "mean_stress",
                                         "H_mean_x0", "H_mean_y0",
                                         "dH_apex"]] = np.nan
    # 2. Section-metric outliers (bad section extraction on edge-case geometries)
    for col in ["H_mean_x0", "H_mean_y0"]:
        df[col] = _remove_outliers(df[col])

    # Round grid keys so pivot_table matching is exact
    df["E1_key"]  = df["E1"].round(0)
    df["r2_key"]  = df["E2_over_E1"].round(6)

    e1_keys  = np.round(E1_VALUES, 0)
    r2_keys  = np.round(E2_OVER_E1, 6)

    grids = {}
    for key, scale in [(p[0], p[3]) for p in PANELS]:
        mat = np.full((len(r2_keys), len(e1_keys)), np.nan)
        for i, r2 in enumerate(r2_keys):
            for j, e1 in enumerate(e1_keys):
                rows = df[(df["E1_key"] == e1) & (df["r2_key"] == r2)]
                if len(rows) == 1 and not pd.isna(rows[key].values[0]):
                    mat[i, j] = rows[key].values[0] * scale
        grids[key] = mat

    return grids, e1_keys, r2_keys


N_FINE = 200   # interpolation resolution for smooth gradient display

# Figure sizing, in inches.  PANEL_IN is the square data box; the rest is the
# room reserved for tick labels, axis labels, the suptitle and the colorbar, so
# that none of them eat into the panel itself.
PANEL_IN     = 4.2
ROW_EXTRA_IN = 1.25   # x tick labels + xlabel + horizontal colorbar + its label
W_MARGIN_IN  = 1.3    # y tick labels + ylabel
H_MARGIN_IN  = 0.75   # suptitle


def _smooth_grid(Z, e1_keys, r2_keys):
    """Bicubic-interpolate the 10×8 grid onto an N_FINE×N_FINE display grid."""
    # r2_keys ascends (1.0 → 5.0), which is what RectBivariateSpline wants
    r2_asc = r2_keys
    Z_asc  = Z

    # Fill any NaN cells with the row mean so the spline stays well-conditioned
    Z_clean = Z_asc.copy()
    for i in range(Z_clean.shape[0]):
        row_valid = Z_clean[i, np.isfinite(Z_clean[i])]
        if row_valid.size > 0:
            Z_clean[i, ~np.isfinite(Z_clean[i])] = row_valid.mean()

    spline  = RectBivariateSpline(r2_asc, e1_keys, Z_clean, kx=3, ky=3)
    r2_fine = np.linspace(r2_asc[0],  r2_asc[-1],  N_FINE)
    e1_fine = np.linspace(e1_keys[0], e1_keys[-1], N_FINE)
    return e1_fine, r2_fine, spline(r2_fine, e1_fine)   # (N_FINE, N_FINE)


def plot_surface(save=True):
    grids, e1_keys, r2_keys = _load_and_pivot()
    vlims = shared_vlims()

    n_panels = len(PANELS)
    ncols = 3 if n_panels <= 3 else 2
    nrows = int(np.ceil(n_panels / ncols))
    # Size the canvas from the panel box, not the other way round: each data
    # panel gets a PANEL_IN square and the tick labels, axis labels and the
    # horizontal colorbar are given their own inches on top of that.  Sizing the
    # figure first and letting the colorbar steal from the axes is what left the
    # panels short and wide.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * PANEL_IN + W_MARGIN_IN,
                                      nrows * (PANEL_IN + ROW_EXTRA_IN) + H_MARGIN_IN),
                             constrained_layout=True, squeeze=False)

    for idx, (key, title, cmap, scale, unit, diverging) in enumerate(PANELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        Z = grids[key]
        valid = Z[np.isfinite(Z)]
        if valid.size == 0:
            ax.set_visible(False)
            continue

        e1_fine, r2_fine, Z_fine = _smooth_grid(Z, e1_keys, r2_keys)

        # Colour limits are pooled with figS so a colour means the same number
        # in both figures (see shared_vlims).
        vmin, vmax, half = vlims[key]
        panel_cmap = _half_diverging(cmap, half) if half else cmap

        pcm = ax.pcolormesh(e1_fine / 1000, r2_fine, Z_fine,
                            cmap=panel_cmap, vmin=vmin, vmax=vmax,
                            shading="gouraud", rasterized=True)
        ax.contour(e1_fine / 1000, r2_fine, Z_fine, levels=6,
                   colors="white", linewidths=0.5, alpha=0.45)
        if diverging and Z_fine.min() < 0 < Z_fine.max():
            ax.contour(e1_fine / 1000, r2_fine, Z_fine, levels=[0.0],
                       colors="black", linewidths=1.0)

        # Mark any invalid original grid cells with a cross
        for i, r2 in enumerate(r2_keys):
            for j, e1 in enumerate(e1_keys):
                if not np.isfinite(Z[i, j]):
                    ax.text(e1 / 1000, r2, "×", ha="center", va="center",
                            fontsize=9, color="white", fontweight="bold")

        ax.set_box_aspect(1)          # keep the data panel square
        ax.set_xlabel(r"$E_1$ (kN/m)", labelpad=3)
        ax.set_ylabel(r"$E_2/E_1$", labelpad=3)
        ax.set_title(f"{title}  ({unit})" if unit else title, pad=5)
        ax.set_xlim(e1_fine[0] / 1000, e1_fine[-1] / 1000)
        ax.set_ylim(r2_fine[0], r2_fine[-1])
        ax.set_xticks(e1_keys / 1000)
        ax.set_xticklabels([f"{v:.0f}" for v in e1_keys / 1000],
                           fontsize=7, rotation=45)
        ax.set_yticks(r2_keys)
        ax.set_yticklabels([f"{v:.2f}" for v in r2_keys], fontsize=7)

        cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                            location="bottom", pad=0.06, aspect=30,
                            fraction=0.05)
        cbar.ax.tick_params(labelsize=7)
        if unit:
            cbar.set_label(unit, fontsize=7)

    for idx in range(n_panels, nrows * ncols):
        axes[divmod(idx, ncols)].set_visible(False)

    n_total   = grids["crown_height"].size
    n_invalid = int(np.sum(~np.isfinite(grids["crown_height"])))
    fig.suptitle(
        r"$E_1$ (wale) × $E_2/E_1$ response surfaces  "
        f"({n_total - n_invalid}/{n_total} valid runs)"
        "\n"
        r"$s_f{=}1.1$,  $p{=}1000\,\mathrm{Pa}$,  "
        r"$\theta_{knit}{=}0°$,  $\nu_{12}{=}0.195$",
        fontsize=10,
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
    print("Plotting E1 × (E2/E1) response surfaces (bicubic interpolation)...")
    plot_surface()
    print("Done.")
