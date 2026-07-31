"""
Effect of uniform stretch factor (sf_wale = sf_course = sf) on:
  - crown height
  - mean stress
  - mean curvature along the x=0 and y=0 sections

All three panels come from a direct FEA sweep along the diagonal
(data/uniform_sf_sweep.csv, produced by run_uniform_sf_sweep.py) — one
simulation per plotted sf, at knit_dir = 0 deg and p = 1000 Pa.  Every run is
drawn as a marker; lines only interpolate between simulated points.

This replaces the previous GP-surrogate slice.  The surrogate was fitted to the
Sobol samples, which barely cover this line — knit_dir = 0 sits on the edge of
the sampled range and the corners of the (sf_wale, sf_course) square are sparse
— so it contributed most of the structure in the curvature panel and got the
sign of the trend wrong below sf = 1.  A run costs ~0.25 s, so there is no
reason to interpolate a surrogate here.

Crown height and mean stress are numerically smooth along the sweep (rms second
difference < 0.1% of range) and are drawn as-is.  The section curvature carries
8-12% discretisation noise from _profile_curvature (5 mm binning of the slice
crossings, then two numerical derivatives), so the curvature panel adds a
Savitzky-Golay trend line on top of the per-run markers.

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

SF_RANGE   = (0.8, 1.4)
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

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 8.5), constrained_layout=True,
                             sharex=True)
    ax_h, ax_s, ax_c = axes

    # Left edge of the region where at least one motif converged; below it no
    # run produced a dome at all.  Each curve simply starts where its own
    # solves began to converge (motif 1 at a higher sf than motif 2).
    sf_converged = df["sf"].min()

    for motif, sub in df.groupby("motif"):
        color = COLORS[motif]
        label = LABELS[motif]
        sf    = sub["sf"].values

        # row 1: crown height — smooth enough to plot directly
        ax_h.plot(sf, sub["crown_height"].values * 1000, color=color, lw=1.6,
                  marker="o", ms=2.4, mew=0, label=f"{label}  ({len(sf)} runs)")

        # row 2: mean stress
        ax_s.plot(sf, sub["mean_stress"].values, color=color, lw=1.6,
                  marker="o", ms=2.4, mew=0, label=f"{label}  ({len(sf)} runs)")

        # row 3: section curvature — per-run markers + Savitzky-Golay trend
        for col, ls, marker, sec in [("H_mean_x0", "-",  "o", "x=0 section"),
                                     ("H_mean_y0", "--", "s", "y=0 section")]:
            y = sub[col].values
            ax_c.plot(sf, y, ls="none", marker=marker, ms=2.6,
                      mfc="none", mec=color, mew=0.7, alpha=0.55)
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
                   "\n(markers = individual runs,  lines = Savitzky-Golay trend)")
    ax_c.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax_c.set_xlabel(r"Uniform stretch factor  $s_f$  ($s_{wale} = s_{course}$)")
    ax_c.set_ylim(0, None)

    for ax in axes:
        ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlim(*SF_RANGE)
        if sf_converged > SF_RANGE[0]:
            ax.axvspan(SF_RANGE[0], sf_converged, color="0.85", alpha=0.55,
                       lw=0, zorder=0)
    if sf_converged > SF_RANGE[0]:
        ax_h.text((SF_RANGE[0] + sf_converged) / 2, 0.5,
                  "Newton solve does not converge",
                  transform=ax_h.get_xaxis_transform(), ha="center",
                  va="center", fontsize=6.5, color="0.35", rotation=90)

    fig.suptitle(
        r"Effect of uniform stretch factor on dome geometry and stress"
        "\n(direct FEA sweep, one run per point;  "
        rf"$\theta_{{knit}}={knit_dir:.0f}°$,  $p={pressure:.0f}$ Pa)",
        fontsize=9,
    )

    print(f"  {len(df)}/{n_all} runs converged; "
          f"plotted range sf >= {sf_converged:.2f}")

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
