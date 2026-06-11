"""
Fig Q — sf_wale × sf_course response surfaces across E1/E2 anisotropy combinations.

Layout: 4 rows × 5 columns
  Rows:    Crown height  |  H x=0  |  H y=0  |  Mean stress
  Columns: motif 5→1, i.e. E2/E1 = 0.40, 0.625, 1.0, 1.6, 2.50
           (wale-stiff → isotropic → course-stiff)

Uses per-motif scalar surrogates (trained on sf_wale, sf_course, knit_dir, pressure).
Fixed: knit_dir=0°, pressure=1000 Pa.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from surrogate import ScalarSurrogate

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

SF_RANGE = (0.8, 1.4)
N_GRID   = 60
KNIT_DIR = 0.0
PRESSURE = 1000.0

# Columns: ordered from wale-stiff to course-stiff
COLS = [
    ("motif5_nocable", r"$E_2/E_1{=}0.40$",  r"$E_1{=}12507$, $E_2{=}5000$"),
    ("motif4_nocable", r"$E_2/E_1{=}0.625$", r"$E_1{=}8000$, $E_2{=}5000$"),
    ("motif3_nocable", r"$E_2/E_1{=}1.0$",   r"$E_1{=}E_2{=}5000$"),
    ("motif2_nocable", r"$E_2/E_1{=}1.6$",   r"$E_1{=}5000$, $E_2{=}8000$"),
    ("motif1_nocable", r"$E_2/E_1{=}2.5$",   r"$E_1{=}5000$, $E_2{=}12507$"),
]

# Rows: (output_key, row_label, cmap, scale, unit)
ROWS = [
    ("crown_height", "Crown height",           "viridis", 1000.0, "mm"),
    ("H_mean_x0",    r"$\bar{H}$  ($x{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("H_mean_y0",    r"$\bar{H}$  ($y{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("mean_stress",  "Mean stress",            "plasma",  1.0,    "N/m"),
]


def _make_grid():
    sf_w = np.linspace(*SF_RANGE, N_GRID)
    sf_c = np.linspace(*SF_RANGE, N_GRID)
    W, C = np.meshgrid(sf_w, sf_c)
    n = N_GRID * N_GRID
    X = np.column_stack([
        W.ravel(),
        C.ravel(),
        np.full(n, KNIT_DIR),
        np.full(n, PRESSURE),
    ])
    return sf_w, sf_c, W, C, X


def plot_e1e2_sf_surface(save=True):
    sf_w, sf_c, W, C, X = _make_grid()

    # Pre-load surrogates and compute predictions
    all_Z = [[None] * len(COLS) for _ in range(len(ROWS))]
    for col_idx, (group, ratio_label, val_label) in enumerate(COLS):
        path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        sur = ScalarSurrogate.load(path)
        preds = sur.predict(X)
        for row_idx, (key, _, _, scale, _) in enumerate(ROWS):
            vals = preds.get(key)
            if vals is not None:
                all_Z[row_idx][col_idx] = vals.reshape(N_GRID, N_GRID) * scale

    # Per-row shared colour limits
    row_vlims = []
    for row_idx in range(len(ROWS)):
        all_vals = np.concatenate([z.ravel() for z in all_Z[row_idx] if z is not None])
        row_vlims.append((np.percentile(all_vals, 2), np.percentile(all_vals, 98)))

    n_rows = len(ROWS)
    n_cols = len(COLS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )

    for row_idx, (key, row_label, cmap, scale, unit) in enumerate(ROWS):
        vmin, vmax = row_vlims[row_idx]
        for col_idx, (group, ratio_label, val_label) in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            Z  = all_Z[row_idx][col_idx]
            if Z is None:
                ax.set_visible(False)
                continue

            cs = ax.contourf(W, C, Z, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(W, C, Z, levels=10, colors="white",
                       linewidths=0.4, alpha=0.5)
            # dashed diagonal: sf_wale = sf_course
            ax.plot([SF_RANGE[0], SF_RANGE[1]],
                    [SF_RANGE[0], SF_RANGE[1]],
                    "w--", lw=1.0, alpha=0.7)

            ax.set_xlim(*SF_RANGE)
            ax.set_ylim(*SF_RANGE)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)

            if row_idx == n_rows - 1:
                ax.set_xlabel(r"$s_\mathrm{wale}$", labelpad=2)
            else:
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(r"$s_\mathrm{course}$", labelpad=2)
            else:
                ax.set_yticklabels([])

            if row_idx == 0:
                ax.set_title(f"{ratio_label}\n{val_label}", pad=5, fontsize=8.5)

            # row label on left edge
            if col_idx == 0:
                ax.text(-0.30, 0.5, row_label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9)

            # shared colorbar on right edge of each row
            if col_idx == n_cols - 1:
                cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
                cb.set_label(unit, fontsize=8)
                cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"Response surfaces: $s_\mathrm{wale}$ × $s_\mathrm{course}$"
        r" across $E_1$/$E_2$ anisotropy"
        "\n"
        r"($\theta_\mathrm{knit}{=}0°$,  $p{=}1000\,\mathrm{Pa}$;"
        r"  dashed = uniform $s_f$)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figQ_e1e2_sf_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting sf × sf surfaces across E1/E2 combinations...")
    plot_e1e2_sf_surface()
    print("Done.")
