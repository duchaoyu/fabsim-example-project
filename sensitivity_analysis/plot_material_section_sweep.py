"""
Fig P — E1 / r / nu12 parameter sweeps for section outputs (no cable).

3 columns × 3 rows:
  cols: E1 sweep | r=E1/E2 sweep | nu12 sweep
  rows: section curvature H_x0 & H_y0
        section stress vm_x0 & vm_y0
        curvature anisotropy index (H_x0-H_y0)/(H_x0+H_y0)

All other parameters fixed at midpoints:
  sf_wale = sf_course = 1.0,  knit_dir = 45°,  pressure = 700 Pa,
  and the non-swept material param held at its midpoint.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_MATERIAL_NO_CABLE
from surrogate import ScalarSurrogate

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SURROGATE_PATH = os.path.join(DATA_DIR, "material_nocable_section_surrogate.pkl")

# ── fixed background values ──────────────────────────────────────────────────
FIXED = {
    "sf_wale":   1.0,
    "sf_course": 1.0,
    "knit_dir":  45.0,
    "pressure":  700.0,
    "E1":        4500.0,   # midpoint of (1000, 8000)
    "r":         4.0,      # midpoint of (3, 5)
    "nu":        0.675,    # midpoint of (0.45, 0.9)
}

PARAM_KEYS = list(PARAMS_MATERIAL_NO_CABLE.keys())
N_GRID = 120

SWEEP_PARAMS = [
    ("E1",  r"$E_1$  (N/m)",    PARAMS_MATERIAL_NO_CABLE["E1"]),
    ("r",   r"$r = E_1/E_2$",   PARAMS_MATERIAL_NO_CABLE["r"]),
    ("nu",  r"$\nu_{12}$",      PARAMS_MATERIAL_NO_CABLE["nu"]),
]

COLOR_X0 = "#2E8B57"
COLOR_Y0 = "#e07b39"

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


def _sweep_X(param_name: str, grid: np.ndarray) -> np.ndarray:
    """Build (N_GRID, 7) input array with one param swept, rest fixed."""
    rows = []
    for v in grid:
        row = [FIXED[k] if k != param_name else v for k in PARAM_KEYS]
        rows.append(row)
    return np.array(rows)


def plot_sweep(save=True):
    sur = ScalarSurrogate.load(SURROGATE_PATH)

    n_cols = len(SWEEP_PARAMS)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 9.5),
                              gridspec_kw={"hspace": 0.52, "wspace": 0.35})

    for col_idx, (pname, xlabel, (plo, phi)) in enumerate(SWEEP_PARAMS):
        grid = np.linspace(plo, phi, N_GRID)
        X    = _sweep_X(pname, grid)
        pred = sur.predict(X)

        hx = pred.get("H_mean_x0", np.full(N_GRID, np.nan))
        hy = pred.get("H_mean_y0", np.full(N_GRID, np.nan))
        vx = pred.get("vm_x0",     np.full(N_GRID, np.nan))
        vy = pred.get("vm_y0",     np.full(N_GRID, np.nan))

        denom = np.abs(hx) + np.abs(hy)
        dH    = np.where(denom > 1e-9, (hx - hy) / denom, 0.0)

        ax_c = axes[0, col_idx]
        ax_s = axes[1, col_idx]
        ax_a = axes[2, col_idx]

        # --- curvature ---
        ax_c.plot(grid, hx, color=COLOR_X0, lw=2, ls="-")
        ax_c.plot(grid, hy, color=COLOR_Y0, lw=2, ls="--")
        ax_c.set_xlabel(xlabel)

        # --- stress ---
        ax_s.plot(grid, vx, color=COLOR_X0, lw=2, ls="-")
        ax_s.plot(grid, vy, color=COLOR_Y0, lw=2, ls="--")
        ax_s.set_xlabel(xlabel)

        # --- anisotropy ---
        ax_a.plot(grid, dH, color="#1a6fa8", lw=2)
        ax_a.axhline(0, color="0.75", lw=0.8, ls=":")
        ax_a.set_xlabel(xlabel)

        # column titles on top row only
        if col_idx == 0:
            ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
            ax_s.set_ylabel(r"Mean von Mises stress  (N/m)")
            ax_a.set_ylabel(r"$(H_{x=0}{-}H_{y=0})\,/\,(H_{x=0}{+}H_{y=0})$")

        for ax in (ax_c, ax_s, ax_a):
            ax.set_xlim(plo, phi)

    # ── row titles ───────────────────────────────────────────────────────────
    axes[0, 1].set_title(r"Section curvature  $\bar{H}$",       pad=6)
    axes[1, 1].set_title("Section stress",                       pad=6)
    axes[2, 1].set_title("Curvature anisotropy index",           pad=6)

    # ── shared legend ─────────────────────────────────────────────────────────
    plane_handles = [
        Line2D([0], [0], color=COLOR_X0, lw=2, ls="-",  label=r"$x{=}0$ section"),
        Line2D([0], [0], color=COLOR_Y0, lw=2, ls="--", label=r"$y{=}0$ section"),
    ]
    axes[0, 0].legend(handles=plane_handles, fontsize=8, loc="best")
    axes[1, 0].legend(handles=plane_handles, fontsize=8, loc="best")

    fixed_str = (r"$s_{wale}{=}s_{course}{=}1.0$,  "
                 r"$p{=}700\,\mathrm{Pa}$,  $\theta_{knit}{=}45°$"
                 "\n"
                 r"non-swept: $E_1{=}4500\,\mathrm{N/m}$,  "
                 r"$r{=}4.0$,  $\nu_{12}{=}0.675$")
    fig.suptitle(
        r"Effect of material parameters on section outputs  (no cable)"
        "\n" + fixed_str,
        fontsize=10, y=1.02,
    )

    if save:
        path = os.path.join(FIG_DIR, "figP_material_section_sweep.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Fig P: material parameter sweeps on section outputs...")
    plot_sweep()
    print("Done.")
