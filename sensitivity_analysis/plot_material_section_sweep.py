"""
Fig P — E1 / r / nu12 parameter sweeps for section outputs (no cable).

3 columns × 3 rows:
  cols: E1 sweep | r=E1/E2 sweep | nu12 sweep
  rows: crown height
        section curvature H_x0 & H_y0
        section stress vm_x0 & vm_y0

Fixed: sf_wale = sf_course = 1.0,  knit_dir = 0°,  pressure = 1000 Pa,
       non-swept material param held at its midpoint.
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

SCALAR_SURROGATE_PATH  = os.path.join(DATA_DIR, "material_nocable_scalar_surrogate.pkl")
SECTION_SURROGATE_PATH = os.path.join(DATA_DIR, "material_nocable_section_surrogate.pkl")

# ── fixed background values ──────────────────────────────────────────────────
FIXED = {
    "sf_wale":   1.0,
    "sf_course": 1.0,
    "knit_dir":  0.0,
    "pressure":  1000.0,
    "E1":        10500.0,  # midpoint of (1000, 20000)
    "r":         3.0,      # midpoint of (1, 5)
    "nu":        0.195,    # midpoint of (0.09, 0.3)
}

PARAM_KEYS = list(PARAMS_MATERIAL_NO_CABLE.keys())
N_GRID = 120

SWEEP_PARAMS = [
    ("E1",  r"$E_1$  (N/m)",   (1000.0, 20000.0)),
    ("r",   r"$r = E_1/E_2$",  (1.0,    5.0)),
    ("nu",  r"$\nu_{12}$",     (0.09,   0.3)),
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
    rows = [[FIXED[k] if k != param_name else v for k in PARAM_KEYS] for v in grid]
    return np.array(rows)


def plot_sweep(save=True):
    sur_scalar  = ScalarSurrogate.load(SCALAR_SURROGATE_PATH)
    sur_section = ScalarSurrogate.load(SECTION_SURROGATE_PATH)

    n_cols = len(SWEEP_PARAMS)
    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 9.5),
                              gridspec_kw={"hspace": 0.52, "wspace": 0.35})

    for col_idx, (pname, xlabel, (plo, phi)) in enumerate(SWEEP_PARAMS):
        grid = np.linspace(plo, phi, N_GRID)
        X    = _sweep_X(pname, grid)

        pred_sc  = sur_scalar.predict(X)
        pred_sec = sur_section.predict(X)

        crown = pred_sc.get("crown_height",  np.full(N_GRID, np.nan))
        hx    = pred_sec.get("H_mean_x0",   np.full(N_GRID, np.nan))
        hy    = pred_sec.get("H_mean_y0",   np.full(N_GRID, np.nan))
        vx    = pred_sec.get("vm_x0",       np.full(N_GRID, np.nan))
        vy    = pred_sec.get("vm_y0",       np.full(N_GRID, np.nan))

        ax_h = axes[0, col_idx]
        ax_c = axes[1, col_idx]
        ax_s = axes[2, col_idx]

        # --- crown height ---
        ax_h.plot(grid, crown, color="#1a6fa8", lw=2)
        ax_h.set_xlabel(xlabel)

        # --- section curvature ---
        ax_c.plot(grid, hx, color=COLOR_X0, lw=2, ls="-")
        ax_c.plot(grid, hy, color=COLOR_Y0, lw=2, ls="--")
        ax_c.set_xlabel(xlabel)

        # --- section stress ---
        ax_s.plot(grid, vx, color=COLOR_X0, lw=2, ls="-")
        ax_s.plot(grid, vy, color=COLOR_Y0, lw=2, ls="--")
        ax_s.set_xlabel(xlabel)

        if col_idx == 0:
            ax_h.set_ylabel(r"Crown height  (m)")
            ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
            ax_s.set_ylabel(r"Mean von Mises stress  (N/m)")

        # Training-data bounds for each swept param
        train_bounds = {"E1": (1000.0, 8000.0), "r": (3.0, 5.0), "nu": (0.45, 0.9)}
        tlo, thi = train_bounds[pname]
        for ax in (ax_h, ax_c, ax_s):
            ax.set_xlim(plo, phi)
            # shade extrapolation regions
            if plo < tlo:
                ax.axvspan(plo, tlo, color="0.88", zorder=0)
            if phi > thi:
                ax.axvspan(thi, phi, color="0.88", zorder=0)
            # training boundary lines
            for xb in [tlo, thi]:
                if plo < xb < phi:
                    ax.axvline(xb, color="0.55", lw=0.8, ls=":", zorder=1)

    # ── row titles (centre column) ────────────────────────────────────────────
    axes[0, 1].set_title("Crown height  $h_{crown}$",   pad=6)
    axes[1, 1].set_title(r"Section curvature  $\bar{H}$", pad=6)
    axes[2, 1].set_title("Section stress",                pad=6)

    # ── shared legend ─────────────────────────────────────────────────────────
    plane_handles = [
        Line2D([0], [0], color=COLOR_X0, lw=2, ls="-",  label=r"$x{=}0$ section"),
        Line2D([0], [0], color=COLOR_Y0, lw=2, ls="--", label=r"$y{=}0$ section"),
    ]
    axes[1, 0].legend(handles=plane_handles, fontsize=8, loc="best")
    axes[2, 0].legend(handles=plane_handles, fontsize=8, loc="best")

    fixed_str = (r"$s_{wale}{=}s_{course}{=}1.0$,  "
                 r"$p{=}1000\,\mathrm{Pa}$,  $\theta_{knit}{=}0°$"
                 "\n"
                 r"non-swept: $E_1{=}10500\,\mathrm{N/m}$,  "
                 r"$r{=}3.0$,  $\nu_{12}{=}0.195$"
                 r"  (dashed region = GP extrapolation)")
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
