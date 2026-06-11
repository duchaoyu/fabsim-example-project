"""
Fig P — E1 × E2 response surfaces (no cable).

Layout: 3 rows × 2 columns
  Row 1:  Crown height           |  Curvature anisotropy  (H_x0 - H_y0) / (H_x0 + H_y0)
  Row 2:  Section curvature H_x0 |  Section curvature H_y0
  Row 3:  Section stress vm_x0   |  Section stress vm_y0

Fixed: sf_wale = sf_course = 1.0,  knit_dir = 0°,  pressure = 1000 Pa,
       nu12 = 0.195 (midpoint of training range 0.09–0.3).

Requires material_ext_nocable_scalar_surrogate.pkl and
         material_ext_nocable_section_surrogate.pkl  (produced by run_material_ext.py).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_MATERIAL_EXT_NO_CABLE
from surrogate import ScalarSurrogate

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EXT_SCALAR_SUR_PATH  = os.path.join(DATA_DIR, "material_ext_nocable_scalar_surrogate.pkl")
EXT_SECTION_SUR_PATH = os.path.join(DATA_DIR, "material_ext_nocable_section_surrogate.pkl")

E_RANGE  = (1000.0, 20000.0)
N_GRID   = 60
NU_FIXED = 0.195      # midpoint of (0.09, 0.3)
SF_FIXED = 1.0
P_FIXED  = 1000.0
KNIT_FIXED = 0.0

PARAM_KEYS = list(PARAMS_MATERIAL_EXT_NO_CABLE.keys())
# keys: sf_wale, sf_course, knit_dir, pressure, E1, E2, nu

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

# (surrogate_type, output_key, row_label, cmap, scale, unit)
PANELS = [
    ("scalar",  "crown_height",  "Crown height",           "viridis", 1000.0, "mm"),
    ("section", "H_anisotropy",  r"Curvature anisotropy $\Delta H/\bar{H}$",
                                                            "RdBu_r",  1.0,    ""),
    ("section", "H_mean_x0",     r"$\bar{H}$  ($x{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("section", "H_mean_y0",     r"$\bar{H}$  ($y{=}0$)", "cividis", 1.0,    r"m$^{-1}$"),
    ("section", "vm_x0",         r"Stress  ($x{=}0$)",    "plasma",  1.0,    "N/m"),
    ("section", "vm_y0",         r"Stress  ($y{=}0$)",    "plasma",  1.0,    "N/m"),
]


def _build_grid():
    e1 = np.linspace(*E_RANGE, N_GRID)
    e2 = np.linspace(*E_RANGE, N_GRID)
    E1g, E2g = np.meshgrid(e1, e2)
    n = N_GRID * N_GRID
    X = np.column_stack([
        np.full(n, SF_FIXED),    # sf_wale
        np.full(n, SF_FIXED),    # sf_course
        np.full(n, KNIT_FIXED),  # knit_dir
        np.full(n, P_FIXED),     # pressure
        E1g.ravel(),             # E1
        E2g.ravel(),             # E2
        np.full(n, NU_FIXED),    # nu
    ])
    return e1, e2, E1g, E2g, X


def plot_surface(save=True):
    if not os.path.exists(EXT_SCALAR_SUR_PATH):
        raise FileNotFoundError(
            f"Scalar surrogate not found: {EXT_SCALAR_SUR_PATH}\n"
            "Run: python run_material_ext.py --steps generate,sections,train"
        )
    if not os.path.exists(EXT_SECTION_SUR_PATH):
        raise FileNotFoundError(
            f"Section surrogate not found: {EXT_SECTION_SUR_PATH}\n"
            "Run: python run_material_ext.py --steps generate,sections,train"
        )

    sur_sc  = ScalarSurrogate.load(EXT_SCALAR_SUR_PATH)
    sur_sec = ScalarSurrogate.load(EXT_SECTION_SUR_PATH)

    e1, e2, E1g, E2g, X = _build_grid()
    pred_sc  = sur_sc.predict(X)
    pred_sec = sur_sec.predict(X)
    all_preds = {**pred_sc, **pred_sec}

    fig, axes = plt.subplots(3, 2, figsize=(9, 12),
                              gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    for idx, (sur_type, key, label, cmap, scale, unit) in enumerate(PANELS):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        vals = all_preds.get(key)
        if vals is None:
            ax.set_visible(False)
            continue

        Z = (vals * scale).reshape(N_GRID, N_GRID)

        # symmetric colormap for anisotropy
        if "anisotropy" in key.lower() or cmap == "RdBu_r":
            lim = np.nanpercentile(np.abs(Z), 98)
            vmin, vmax = -lim, lim
        else:
            vmin = np.nanpercentile(Z, 2)
            vmax = np.nanpercentile(Z, 98)

        pcm = ax.pcolormesh(e1 / 1000, e2 / 1000, Z,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            shading="gouraud")
        ax.contour(e1 / 1000, e2 / 1000, Z, levels=8,
                   colors="white", linewidths=0.5, alpha=0.5)

        # diagonal: E1 = E2 (isotropic line)
        ax.plot([E_RANGE[0]/1000, E_RANGE[1]/1000],
                [E_RANGE[0]/1000, E_RANGE[1]/1000],
                "w--", lw=1.0, alpha=0.7, label="$E_1 = E_2$")

        ax.set_xlabel(r"$E_1$  (kN/m)", labelpad=3)
        ax.set_ylabel(r"$E_2$  (kN/m)", labelpad=3)
        title = f"{label}"
        if unit:
            title += f"  ({unit})"
        ax.set_title(title, pad=5)

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, aspect=20)
        cbar.ax.tick_params(labelsize=7)
        if unit:
            cbar.set_label(unit, fontsize=7)

        if idx == 0:
            ax.legend(fontsize=7, loc="lower right",
                      framealpha=0.6, handlelength=1.5)

    fig.suptitle(
        r"Response surfaces: $E_1$ (wale) $\times$ $E_2$ (course)  |  no cable"
        "\n"
        r"$s_{wale}{=}s_{course}{=}1.0$,  $p{=}1000\,\mathrm{Pa}$,  "
        r"$\theta_{knit}{=}0°$,  $\nu_{12}{=}0.195$"
        "\n"
        r"(dashed diagonal: $E_1{=}E_2$, isotropic)",
        fontsize=10, y=1.01,
    )

    if save:
        path = os.path.join(FIG_DIR, "figP_material_e1e2_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot_surface()
