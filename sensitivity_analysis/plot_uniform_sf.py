"""
Effect of uniform stretch factor (sf_wale = sf_course = sf) on:
  - crown height
  - mean stress
  - mean curvature along the x=0 and y=0 sections

All three panels come from a direct FEA sweep along the diagonal
(data/uniform_sf_sweep.csv, produced by run_uniform_sf_sweep.py) — one
simulation per plotted sf, at knit_dir = 0 deg and p = 1000 Pa.  Curves only
interpolate between simulated points; the per-run markers are not drawn.

This replaces the previous GP-surrogate slice.  The surrogate was fitted to the
Sobol samples, which barely cover this line — knit_dir = 0 sits on the edge of
the sampled range and the corners of the (sf_wale, sf_course) square are sparse
— so it contributed most of the structure in the curvature panel and got the
sign of the trend wrong below sf = 1.  A run costs ~0.25 s, so there is no
reason to interpolate a surrogate here.

Crown height and mean stress are numerically smooth along the sweep (rms second
difference < 0.1% of range) and are drawn as-is.  Section curvature uses the
polynomial-fit estimator (section_curvature.py, columns H_fit_*), which is stable
along a sweep and matches the spherical-cap reference to 1%; the binned estimator
behind H_mean_* steps by up to 14% between adjacent runs and reads 24-28% high.
That panel is drawn as a Savitzky-Golay trend through the per-run values.

Plotted from sf = 0.9: below that the Newton solve stops converging (it returns
the undeformed state, which the sweep flags as failed).

No-cable groups only; motif 1 vs motif 2 overlaid.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SWEEP_CSV = os.path.join(DATA_DIR, "uniform_sf_sweep.csv")

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.8,
    "figure.dpi":       150,
})

# ── seagreen palette ──────────────────────────────────────────────────────────
COLORS = {1: "#2E8B57", 2: "#20B2AA"}   # seagreen / lightseagreen
LABELS = {1: "Motif 1", 2: "Motif 2"}

SF_RANGE   = (0.9, 1.4)   # below 0.9 the Newton solve stops converging
SMOOTH_WIN = 9       # Savitzky-Golay window (points) for the curvature trend
SMOOTH_ORD = 2


def _smooth(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay trend; falls back to the raw values if too few points."""
    win = min(SMOOTH_WIN if SMOOTH_WIN % 2 else SMOOTH_WIN + 1, len(y))
    if win <= SMOOTH_ORD + 1:
        return y
    if win % 2 == 0:
        win -= 1
    return savgol_filter(y, win, SMOOTH_ORD)


def plot_uniform_sf(save=True):
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"{SWEEP_CSV} not found — run:  python3 run_uniform_sf_sweep.py")

    df = pd.read_csv(SWEEP_CSV)
    n_all = len(df)
    df = df[~df["sim_failed"].astype(bool)].sort_values(["motif", "sf"])
    knit_dir = df["knit_dir"].iloc[0]
    pressure = df["pressure"].iloc[0]
    # Plot only the range shown, so the trend lines are fitted to the same points
    # the reader sees.
    df = df[df["sf"].between(*SF_RANGE)]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 8.5), constrained_layout=True,
                             sharex=True)
    ax_h, ax_s, ax_c = axes

    for motif, sub in df.groupby("motif"):
        color = COLORS[motif]
        label = LABELS[motif]
        sf    = sub["sf"].values

        # row 1: crown height — smooth enough to plot directly
        ax_h.plot(sf, sub["crown_height"].values * 1000, color=color, lw=1.6,
                  label=label)

        # row 2: mean stress
        ax_s.plot(sf, sub["mean_stress"].values, color=color, lw=1.6,
                  label=label)

        # row 3: section curvature — Savitzky-Golay trend through the per-run
        # values (one FEA run per sf; the markers are no longer drawn).
        # H_fit_*: polynomial-fit estimator (section_curvature.py).  The binned
        # estimator behind H_mean_* takes |z''| from finite differences on binned
        # data, which rectifies noise into a positive bias: it sits 24-28% above
        # the spherical-cap reference kappa = 2h/(a^2+h^2) and steps by up to 14%
        # between adjacent runs.  The fit matches that reference to 1%.
        for col, ls, marker, sec in [("H_fit_x0", "-",  "o", "x=0 section"),
                                     ("H_fit_y0", "--", "s", "y=0 section")]:
            y = sub[col].values
            ax_c.plot(sf, _smooth(y), color=color, ls=ls, lw=1.8,
                      label=f"{label} ({sec})")

    # ── formatting ────────────────────────────────────────────────────────────
    ax_h.set_ylabel("Crown height  (mm)")
    ax_h.set_title(r"Crown height  vs uniform $s_f$")
    ax_h.legend(fontsize=8, loc="upper right")
    ax_h.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    ax_s.set_ylabel(r"Mean stress  (Pa)")
    ax_s.set_title(r"Mean stress  vs uniform $s_f$")
    ax_s.legend(fontsize=8, loc="upper left")

    ax_c.set_ylabel(r"Mean curvature  $\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  vs uniform $s_f$"
                   "\n(Savitzky-Golay trend through one FEA run per $s_f$)")
    ax_c.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax_c.set_xlabel(r"Uniform stretch factor  $s_f$  ($s_{wale} = s_{course}$)")
    ax_c.set_ylim(0, None)

    for ax in axes:
        ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlim(*SF_RANGE)

    fig.suptitle(
        r"Effect of uniform stretch factor on dome geometry and stress"
        "\n(direct FEA sweep, one run per point;  "
        rf"$\theta_{{knit}}={knit_dir:.0f}°$,  $p={pressure:.0f}$ Pa)",
        fontsize=9,
    )

    print(f"  {len(df)} runs plotted (of {n_all} in the sweep) over "
          f"sf {SF_RANGE[0]}-{SF_RANGE[1]}")

    if save:
        path = os.path.join(FIG_DIR, "figK_uniform_sf.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting uniform stretch factor influence...")
    plot_uniform_sf()
    print("Done.")
