"""
Fig L — response surfaces over sf_wale x sf_course, from a direct FEA grid.

Rows:  crown height,  section stress x=0,  section stress y=0,
       curvature anisotropy index dH
Cols:  motif 1 | motif 2
Fixed: knit_dir = 0 deg, pressure = 1000 Pa.  sf in [0.9, 1.4] on both axes.

Colour scales are shared across BOTH motifs and, where the quantities are
comparable, across rows: the two section-stress rows are drawn on one scale with
one colourbar, so a colour means the same stress in the x=0 and y=0 panels as
well as between the motifs.  The previous version gave every panel its own
scale, which made equal colours mean different numbers everywhere.  Axis limits
are identical on all panels.

The H_{x=0} and H_{y=0} rows are gone; the directional information they carried is
now in the two dH rows, which is what the figure is for.

dH = (kappa_y - kappa_x) / (|kappa_y| + |kappa_x|), the same form as in figM, but
taken from the POINTWISE curvature tensor at the crown (apex_curvature.py,
apex_k_x / apex_k_y) rather than from the profile-averaged section estimator.

The section estimator is not usable for this panel.  It averages |kappa| over a
whole diameter, and because both cut planes share the apex and the clamped rim
they must turn through nearly the same total angle, so the average cancels the
directional signal: at (s_wale, s_course) = (0.92, 1.12) the crown curvatures are
0.71 vs 0.92 m^-1 (dH = -0.13) while the 80%-span averages are 0.914 vs 0.912
(dH = +0.0007), opposite in sign.  Over this grid the two agree on the sign of
the anisotropy at only 17% / 21% of points; the section field is non-monotone in
s_wale (19% vs 93%), 4-5x rougher, and changes sign along the s_wale = s_course
diagonal where the material anisotropy is the only source and cannot change sign.
dH_section is still computed in _derive() if it is wanted for a methods figure.

Data: run_sf_grid.py -> data/sf_grid.csv (direct FEA, one run per grid point).

This replaces GP surrogates fitted to the *_nocable groups of
results_with_sections.csv.  Per validate_fem_runs.py those runs used the wrong
mesh, and motif2_nocable additionally used motif 5's material (wale-stiff, the
opposite anisotropy to motif 2), so the motif comparison this figure exists to
make was confounded by both geometry and material.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR

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

GRID_CSV   = os.path.join(DATA_DIR, "sf_grid.csv")
SF_RANGE   = (0.9, 1.4)
MOTIFS     = (1, 2)
COL_TITLES = {1: "Motif 1  ($E_2/E_1=2.50$)", 2: "Motif 2  ($E_2/E_1=1.60$)"}
N_LEVELS   = 20


def _dH(a, b):
    d = np.abs(a) + np.abs(b)
    return np.where(d > 1e-12, (a - b) / d, np.nan)


# (key, row label, colourbar unit, cmap, diverging?, scale group)
# Rows sharing a scale group are drawn on ONE set of contour levels and share a
# single colourbar, so a colour means the same number across those rows as well
# as across the motif columns.  The two section-stress rows are only comparable
# to each other if they are on the same scale.
ROWS = [
    ("crown_height", "Crown height",              "mm", "viridis", False, "crown"),
    ("von_mises_x0", r"Section stress  $x{=}0$",  "Pa", "plasma",  False, "stress"),
    ("von_mises_y0", r"Section stress  $y{=}0$",  "Pa", "plasma",  False, "stress"),
    ("dH_apex",      r"$\Delta H$",               "",   "RdBu_r",  True,  "dH"),
]


def _derive(sub):
    """Add the two dH columns and scale crown height to mm."""
    sub = sub.copy()
    sub["crown_height"] = sub["crown_height"] * 1000.0
    sub["dH_section"] = _dH(sub["H_fit_x0"].values, sub["H_fit_y0"].values)
    if "apex_k_x" in sub and "apex_k_y" in sub:
        # x=0 section measures kappa_y, so pair apex_k_y with H_fit_x0
        sub["dH_apex"] = _dH(sub["apex_k_y"].values, sub["apex_k_x"].values)
    else:
        sub["dH_apex"] = np.nan
    return sub


def _load():
    if not os.path.exists(GRID_CSV):
        raise FileNotFoundError(
            f"{GRID_CSV} not found — run:  python3 run_sf_grid.py")
    df = pd.read_csv(GRID_CSV)
    df = df[~df["sim_failed"].astype(bool)]
    df = df[df["sf_wale"].between(*SF_RANGE) &
            df["sf_course"].between(*SF_RANGE)]
    return {m: _derive(s) for m, s in df.groupby("motif")}


def plot_sf_surface(save=True):
    data = _load()
    motifs = [m for m in MOTIFS if m in data]

    fig, axes = plt.subplots(len(ROWS), len(motifs),
                             figsize=(4.6 * len(motifs), 3.9 * len(ROWS)),
                             constrained_layout=True, squeeze=False)

    # rows are grouped by scale: every row in a group gets the same levels and
    # they all hang off one colourbar
    groups = []
    for r, row in enumerate(ROWS):
        if groups and groups[-1][0] == row[5]:
            groups[-1][1].append(r)
        else:
            groups.append((row[5], [r]))

    for _, row_idx in groups:
        vals = np.concatenate([data[m][ROWS[r][0]].values
                               for r in row_idx for m in motifs])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            for r in row_idx:
                for c in range(len(motifs)):
                    axes[r, c].set_visible(False)
            continue
        diverging = ROWS[row_idx[0]][4]
        if diverging:
            lim = np.abs(vals).max()
            levels = np.linspace(-lim, lim, N_LEVELS + 1)
        else:
            levels = np.linspace(vals.min(), vals.max(), N_LEVELS + 1)

        cs = None
        for r in row_idx:
            key, label, unit, cmap, diverging, _ = ROWS[r]
            for c, m in enumerate(motifs):
                ax = axes[r, c]
                sub = data[m]
                w, cc, z = (sub["sf_wale"].values, sub["sf_course"].values,
                            sub[key].values)
                ok = np.isfinite(z)
                tri = Triangulation(w[ok], cc[ok])
                cs = ax.tricontourf(tri, z[ok], levels=levels, cmap=cmap,
                                    extend="both")
                ax.tricontour(tri, z[ok], levels=levels[::2], colors="white",
                              linewidths=0.4, alpha=0.5)
                if diverging:
                    ax.tricontour(tri, z[ok], levels=[0.0], colors="black",
                                  linewidths=1.0, linestyles="-")
                ax.plot(SF_RANGE, SF_RANGE, color="white", lw=1.0, ls="--",
                        alpha=0.8)
                ax.set_xlim(*SF_RANGE)
                ax.set_ylim(*SF_RANGE)
                ax.set_aspect("equal")
                ax.set_xlabel(r"$s_{wale}$", labelpad=2)
                ax.set_ylabel(r"$s_{course}$", labelpad=2)
                ax.tick_params(labelsize=7)
                ax.set_title(f"{COL_TITLES[m]}  —  {label}", pad=6,
                             fontsize=8.5)

        # one colourbar for the whole group
        cb_axes = [axes[r, c] for r in row_idx for c in range(len(motifs))]
        cb = fig.colorbar(cs, ax=cb_axes, fraction=0.030, pad=0.02)
        cb.set_label(ROWS[row_idx[0]][2], fontsize=8)
        cb.ax.tick_params(labelsize=7)

    n_pts = sum(len(data[m]) for m in motifs)
    fig.suptitle(
        r"Response surfaces over $s_{wale}\times s_{course}$   "
        r"($\theta_{knit}=0°$,  $p=1000$ Pa)"
        "\n"
        rf"direct FEA grid, {n_pts} runs;  each row shares one colour scale "
        r"across motifs;  dashed = uniform $s_f$,  black = $\Delta H = 0$",
        fontsize=9.5,
    )

    if save:
        path = os.path.join(FIG_DIR, "figL_sf_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


def report():
    data = _load()
    for m, sub in data.items():
        print(f"  motif{m}  n={len(sub)}")
        for key, label, *_ in ROWS:
            v = sub[key].values
            v = v[np.isfinite(v)]
            print(f"    {key:14s} {v.min():+9.3f} .. {v.max():+9.3f}")


if __name__ == "__main__":
    print("Plotting sf_wale x sf_course response surfaces (direct FEA grid)...")
    plot_sf_surface()
    report()
    print("Done.")
