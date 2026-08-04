"""
Fig P — knitting direction on a NON-axisymmetric boundary (ellipse, 2:1 in x).

Companion to figM (circular dome).  Same four quantities, same estimators, same
fixed conditions (s_wale = s_course = 1.0, p = 1000 Pa), one FEA run per degree:

  Row 1  crown height              — a rotational invariant on the circle, and
                                     dead flat there (0.02%).  Not an invariant
                                     here, so any variation is the whole point.
  Row 2  section curvature H_fit    — x=0 cuts the SHORT axis, y=0 the LONG axis,
                                     so the two curves are far apart for purely
                                     geometric reasons; the theta-modulation on
                                     top of that offset is the material effect.
  Row 3  section von Mises stress
  Row 4  curvature anisotropy index dH, defined exactly as in figM: from the
         pointwise crown curvature tensor, (kappa_y-kappa_x)/(|kappa_y|+|kappa_x|),
         not from the section estimator of row 2.
  Row 5  apex principal curvatures and the direction of maximum curvature
         (apex_curvature.py).  This row carries the mechanism.  On the circle
         the principal direction tracks the material frame 1:1 (180 deg -> 90 deg
         as theta goes 0 -> 90); on the ellipse it is pinned to the geometry,
         moving only 2.3 deg over the same sweep.  So when the boundary is
         symmetric, material anisotropy can rotate the response and nothing
         invariant changes; when it is not, the rotation is blocked and the
         anisotropy has to change the magnitudes instead.

Deliberately NOT symmetrised.  figM averages each curve against its mirror
partner using X_{x=0}(theta) = X_{y=0}(90-theta), which holds only because the
circle has an x<->y mirror symmetry.  The ellipse has none, the identity is
false, and imposing it would erase the asymmetry about 45 degrees that is the
result of this study.  Markers are the individual runs; the line is a light
Savitzky-Golay trend through them.

Data: run_knit_dir_sweep_ellipse.py -> data/knit_dir_sweep_ellipse.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

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

SWEEP_CSV  = os.path.join(DATA_DIR, "knit_dir_sweep_ellipse.csv")
CIRC_CSV   = os.path.join(DATA_DIR, "knit_dir_sweep.csv")
COLORS     = {1: "#2E8B57", 2: "#20B2AA"}
LABELS     = {1: "Motif 1", 2: "Motif 2"}
SMOOTH_WIN = 9
SMOOTH_ORD = 2


def _smooth(y):
    y = np.asarray(y, float)
    win = min(SMOOTH_WIN, len(y))
    if win % 2 == 0:
        win -= 1
    if win <= SMOOTH_ORD + 1:
        return y
    return savgol_filter(y, win, SMOOTH_ORD)


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run:  python3 run_knit_dir_sweep_ellipse.py")
    df = pd.read_csv(path)
    return df[~df["sim_failed"].astype(bool)].sort_values(["motif", "knit_dir"])


def _swing(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (v.max() - v.min()) / v.mean() * 100 if len(v) else np.nan


def plot(save=True):
    df = _load(SWEEP_CSV)
    circ = _load(CIRC_CSV) if os.path.exists(CIRC_CSV) else None

    fig, axes = plt.subplots(5, 1, figsize=(6, 15),
                             gridspec_kw={"hspace": 0.55}, sharex=True)
    ax_h, ax_c, ax_s, ax_a, ax_k = axes
    ax_d = ax_k.twinx()          # principal direction, right-hand scale

    notes = []
    for motif, sub in df.groupby("motif"):
        color, label = COLORS[motif], LABELS[motif]
        th = sub["knit_dir"].values

        # ── row 1: crown height ───────────────────────────────────────────────
        h = sub["crown_height"].values * 1000
        ax_h.plot(th, h, ls="none", marker="o", ms=2.4, mfc="none", mec=color,
                  mew=0.7, alpha=0.5)
        ax_h.plot(th, _smooth(h), color=color, lw=2, label=label)

        # ── rows 2-3: section curvature and stress, both cut planes ───────────
        for ax, cx, cy in ((ax_c, "H_fit_x0", "H_fit_y0"),
                           (ax_s, "von_mises_x0", "von_mises_y0")):
            for col, ls, marker in ((cx, "-", "o"), (cy, "--", "s")):
                v = sub[col].values
                ax.plot(th, v, ls="none", marker=marker, ms=2.4, mfc="none",
                        mec=color, mew=0.7, alpha=0.45)
                ax.plot(th, _smooth(v), color=color, lw=2, ls=ls)

        # ── row 4: anisotropy index, same definition as figM ─────────────────
        # From the pointwise crown curvature tensor (apex_k_x / apex_k_y), like
        # figM and figL/figR/figS.  The x=0 cut measures kappa_y, so apex_k_y
        # takes the place H_fit_x0 held and the sign convention is unchanged.
        kx, ky = _smooth(sub["apex_k_x"].values), _smooth(sub["apex_k_y"].values)
        denom = np.abs(ky) + np.abs(kx)
        dH = np.where(denom > 1e-9, (ky - kx) / denom, 0.0)
        ax_a.plot(th, dH, color=color, lw=2, label=label)
        kxr, kyr = sub["apex_k_x"].values, sub["apex_k_y"].values
        ax_a.plot(th, (kyr - kxr) / (np.abs(kyr) + np.abs(kxr)), ls="none",
                  marker="o", ms=2.4, mfc="none", mec=color, mew=0.7, alpha=0.45)

        # ── row 5: apex principal curvatures + principal direction ───────────
        if "apex_k_min" in sub:
            for col, ls in (("apex_k_max", "-"), ("apex_k_min", "--")):
                ax_k.plot(th, _smooth(sub[col].values), color=color, lw=2, ls=ls)
            ax_d.plot(th, sub["apex_k_max_dir_deg"].values, color=color, lw=1.2,
                      ls=":", alpha=0.9)

        # ── note line: ellipse swing vs circle swing ─────────────────────────
        bits = []
        for col, lab in (("crown_height", r"$h_{crown}$"),
                         ("H_fit_x0", r"$\bar{H}_{x=0}$"),
                         ("von_mises_x0", r"$\sigma_{x=0}$")):
            e = _swing(sub[col].values)
            c = (_swing(circ[circ.motif == motif][col].values)
                 if circ is not None else np.nan)
            bits.append(rf"{lab} {e:.0f}% (circle {c:.2f}%)")
        notes.append(f"{label}:  " + ";  ".join(bits))

    # ── formatting ────────────────────────────────────────────────────────────
    ax_h.set_ylabel(r"$h_{crown}$  (mm)")
    ax_h.set_title("Crown height  —  a rotational invariant on the circle, "
                   "not here", fontsize=8.5)
    ax_h.legend(fontsize=8, loc="best")

    ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  $\bar{H}$", fontsize=8.5)

    ax_s.set_ylabel("Mean stress  (Pa)")
    ax_s.set_title("Section stress", fontsize=8.5)

    ax_a.set_ylabel(r"$(\kappa_y - \kappa_x)\,/\,(|\kappa_y| + |\kappa_x|)$")
    ax_a.set_title(r"Curvature anisotropy index  $\Delta H$  (crown tensor)",
                   fontsize=8.5)
    ax_a.legend(fontsize=8, loc="best")

    # ── row 5 formatting, incl. the circle reference for the direction ────────
    ax_k.set_ylabel(r"$\kappa$ at crown  (m$^{-1}$)")
    ax_d.set_ylabel(r"direction of $\kappa_{max}$  (°)", fontsize=8)
    ax_k.set_title("Apex principal curvatures, and the direction of "
                   r"$\kappa_{max}$ (dotted, right axis)", fontsize=8.5)
    ax_k.set_xlabel(r"Knitting direction  $\theta_{knit}$  (°)")
    if circ is not None and "apex_k_max_dir_deg" in circ:
        ref = circ[circ.motif == 1].sort_values("knit_dir")
        ax_d.plot(ref["knit_dir"].values, ref["apex_k_max_dir_deg"].values,
                  color="0.55", lw=1.2, ls=":",
                  label="circle: tracks material frame 1:1")
    # keep the 90 deg line clear of the kappa_min curves on the left scale
    ax_d.set_ylim(55, 195)
    ax_d.axhline(90, color="0.85", lw=0.8, zorder=0)
    ax_k.legend(handles=[
        Line2D([0], [0], color="0.4", lw=1.5, ls="-",  label=r"$\kappa_{max}$"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label=r"$\kappa_{min}$"),
        Line2D([0], [0], color="0.4", lw=1.2, ls=":",
               label=r"ellipse: $\kappa_{max}$ direction pinned to geometry"),
        Line2D([0], [0], color="0.55", lw=1.2, ls=":",
               label="circle: direction follows the material, $180°\\!\\to\\!90°$"),
    ], fontsize=6.5, loc="center left")

    for ax in axes:
        ax.set_xlim(0, 90)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax.axvline(45, color="0.85", lw=0.8, ls=":", zorder=0)

    plane_handles = [
        Line2D([0], [0], color="0.4", lw=1.5, ls="-",
               label="$x=0$ section  (short axis, 1196 mm)"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--",
               label="$y=0$ section  (long axis, 2389 mm)"),
    ]
    ax_c.legend(handles=plane_handles, fontsize=7, loc="best")
    ax_s.legend(handles=plane_handles, fontsize=7, loc="best")

    n_runs = len(df)
    fig.suptitle(
        r"Effect of knitting direction $\theta_{knit}$ on a "
        "non-axisymmetric boundary\n"
        rf"(ellipse 2:1 in $x$, semi-axes 1194 $\times$ 598 mm; direct FEA "
        rf"sweep, {n_runs} runs;  $s_{{wale}}=s_{{course}}=1.0$,  $p=1000$ Pa)",
        fontsize=10, y=0.995,
    )
    fig.text(0.5, 0.062,
             "markers = individual runs;  line = Savitzky-Golay trend.  "
             "No symmetrisation: the identity "
             r"$X_{x=0}(\theta)=X_{y=0}(90^\circ\!-\theta)$"
             "\nused in figM follows from the circle's mirror symmetry and is "
             "false here, so the curves are asymmetric about 45°.\n"
             "Peak-to-peak variation over $\\theta$, ellipse vs circle:\n"
             + "\n".join(notes),
             ha="center", va="top", fontsize=6.5, color="0.35")

    if save:
        path = os.path.join(FIG_DIR, "figP_knit_dir_ellipse.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    plot()
