"""
Sensitivity analysis of knitting direction (theta_knit).

Fig M — Line sweep + anisotropy index, from a direct FEA sweep
  Row 1: Section curvature H_x0 and H_y0 vs theta
  Row 2: Section stress sigma_x0 and sigma_y0 vs theta
  Row 3: Anisotropy index  dH = (kappa_y-kappa_x)/(|kappa_y|+|kappa_x|), from the
         pointwise curvature tensor at the crown (apex_curvature.py) — the same
         estimator and sign convention as figL/figR/figS.  Rows 1-2 stay on the
         section estimator because they are section quantities; row 3 does not,
         because the section average cancels most of the directional signal
         (0.8-0.9% peak-to-peak against 10-18% for the crown tensor).
  Fixed: sf_wale = sf_course = 1.0, pressure = 1000 Pa.  Data:
  run_knit_dir_sweep.py -> data/knit_dir_sweep.csv (one FEA run per angle).
  All curves are symmetrised with the mirror identity below, so the plotted
  x=0 / y=0 pairs are exact reflections and dH is exactly antisymmetric about
  45 degrees; the residual that symmetrising removes is quoted in the caption.

  What the sweep shows, once the section-curvature estimator is stable
  (H_fit_*, section_curvature.py):
    - crown height is independent of theta to 0.01-0.02%, as rotational
      invariance requires (theta rotates the material and stretch frames
      together, so on a circular domain the problem is merely rotated);
    - the section curvature is likewise nearly independent of theta, varying
      under 1%: the inflated shape is close to axisymmetric;
    - the section stress is not — it varies 13-22% sinusoidally, with the two
      cut planes exchanging roles at 45 degrees, because a fixed cut plane
      samples a rotating orthotropic tension field.
  The earlier version of this figure was built from GP surrogates fitted to the
  Sobol samples (wrong mesh; motif 5 material for motif 2) and read with the
  binned curvature estimator, which steps by up to 26% along this sweep.  It
  therefore showed curvature structure that is not there and a flat section
  stress that should vary.

Fig N — Section profile gallery at theta = 0, 30, 45, 60, 90 degrees
  Shows z(s) and H(s) along x=0 and y=0.  Reads the same direct FEA sweep as
  figM, so the runs are exactly on the (sf=1.0, p=1000 Pa) slice and exactly at
  the plotted angles.  It previously read results_with_sections.csv — the Sobol
  samples, on the wrong mesh and with motif 5 material for motif 2 — and picked
  the nearest sample within 15 degrees of each target, so the columns were
  labelled with one angle and drawn from another.  Not symmetrised: this row
  shows what the mesh actually produced.
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
from config import DATA_DIR, MESH_PATH, PARAMS_NO_CABLE
from curvature import read_off, compute_curvatures
from plot_section_profiles import _slice_plane

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

_REST_VERTS, _FACES = read_off(MESH_PATH)

GROUPS = ["motif1_nocable", "motif2_nocable"]
COLORS = {"motif1_nocable": "#2E8B57", "motif2_nocable": "#20B2AA"}
LABELS = {"motif1_nocable": "Motif 1", "motif2_nocable": "Motif 2"}

THETA_RANGE = (0.0, 90.0)

GALLERY_TARGETS = [0, 30, 45, 60, 90]
GALLERY_TOL     = 15.0   # deg; slack for a truncated sweep, the grid is 1 deg


# ── direct FEA sweep ─────────────────────────────────────────────────────────
#
# figM used to read this slice off symmetry-augmented GPs fitted to the Sobol
# samples.  Those samples were run on the wrong mesh and, for motif 2, with
# motif 5 material (see validate_fem_runs.py), and the GP added its own
# artifacts on top.  The slice is now simulated directly by
# run_knit_dir_sweep.py — one FEA run per plotted angle.
#
# The augmentation the GPs used to need is now exact by construction: reflecting
# about y = x maps wale(theta) -> wale(90-theta) and swaps the cut planes, so
#   H_x0(theta) = H_y0(90 - theta),
# up to the mesh itself not being exactly mirror-symmetric.  run_knit_dir_sweep
# --check-only reports the residual.

SWEEP_CSV  = os.path.join(DATA_DIR, "knit_dir_sweep.csv")
SMOOTH_WIN = 9      # Savitzky-Golay window for the section-metric trend lines
SMOOTH_ORD = 2


def _load_sweep():
    if not os.path.exists(SWEEP_CSV):
        raise FileNotFoundError(
            f"{SWEEP_CSV} not found — run:  python3 run_knit_dir_sweep.py")
    df = pd.read_csv(SWEEP_CSV)
    return df[~df["sim_failed"].astype(bool)].sort_values(["motif", "knit_dir"])


def _smooth(y):
    """Trend line through the per-run values; the section-curvature and
    section-stress estimators carry a few percent of discretisation noise."""
    win = min(SMOOTH_WIN, len(y))
    if win % 2 == 0:
        win -= 1
    if win <= SMOOTH_ORD + 1:
        return np.asarray(y)
    return savgol_filter(np.asarray(y), win, SMOOTH_ORD)


def _load_verts(path):
    """Deformed vertices of one sweep run, from the verts_path stored in the CSV."""
    if not isinstance(path, str) or not os.path.exists(path):
        return None
    return pd.read_csv(path).sort_values("vid")[["x", "y", "z"]].values


def _nearest_angle(sub, target_deg):
    """The sweep run closest to target_deg.  The grid is 1 deg, so this is exact
    at every gallery target; the tolerance only guards a truncated sweep."""
    if sub.empty:
        return None
    dist = (sub["knit_dir"] - target_deg).abs()
    if dist.min() > GALLERY_TOL:
        return None
    return sub.loc[dist.idxmin()]


# ── Fig M ─────────────────────────────────────────────────────────────────────

def plot_sweep(save=True):
    df = _load_sweep()

    fig, axes = plt.subplots(3, 1, figsize=(6, 9.5),
                             gridspec_kw={"hspace": 0.52}, sharex=True)
    ax_c, ax_s, ax_a = axes

    notes = []
    for motif, sub in df.groupby("motif"):
        group = f"motif{motif}_nocable"
        color = COLORS[group]
        label = LABELS[group]
        theta = sub["knit_dir"].values
        # H_fit_*: polynomial-fit estimator (section_curvature.py).  The binned
        # estimator behind H_mean_* steps by up to 26% along this sweep and sits
        # 24-28% above the spherical-cap reference; the fit is stable to 0.03%
        # and matches that reference to 1%.
        hx, hy = sub["H_fit_x0"].values, sub["H_fit_y0"].values
        sx, sy = sub["von_mises_x0"].values, sub["von_mises_y0"].values

        def mirror(v):
            """v evaluated at 90-theta — the partner an exactly mirror-symmetric
            mesh would return for the other cut plane."""
            return np.interp(90.0 - theta, theta, v)

        def mirror_of(a, b, th):
            return np.interp(90.0 - th, th, b)

        # Curvature and stress sections: each curve is the mean of the two
        # estimates the exact identity forces to agree, X(theta) and its mirror
        # partner X_other(90-theta).  Symmetrising this way makes the plotted
        # x=0 and y=0 curves exact mirror images of one another, which is what
        # the continuum problem requires; the discrepancy that symmetrising
        # removes is the discretisation error of the section extraction on this
        # mesh (no exact mirror symmetry: all 399 vertices unmatched under
        # x<->y).  Its size is reported in the caption as the mirror residual.
        for ax, vx, vy in ((ax_c, hx, hy), (ax_s, sx, sy)):
            for v, v_partner, ls in ((vx, vy, "-"), (vy, vx, "--")):
                a, b = _smooth(v), mirror(_smooth(v_partner))
                ax.plot(theta, 0.5 * (a + b), color=color, lw=2, ls=ls)

        # Anisotropy index, from the POINTWISE curvature tensor at the crown
        # (apex_curvature.py: apex_k_x / apex_k_y), not from the section
        # estimator that rows 1-2 use.  Same definition and sign convention as
        # figL/figR/figS: the x=0 cut measures kappa_y, so apex_k_y takes the
        # place H_fit_x0 held, and dH > 0 still means the x=0 direction is the
        # more curved one.
        #
        # The section estimator cannot carry this panel.  It averages |kappa|
        # over a whole diameter, and both cut planes share the apex and the
        # clamped rim, so they must turn through nearly the same total angle and
        # the average cancels most of the directional signal: over this sweep it
        # spans only 0.8-0.9% against 10-18% for the crown tensor, i.e. it
        # understates the anisotropy by 12-19x.
        kx, ky = sub["apex_k_x"].values, sub["apex_k_y"].values
        kxs, kys = _smooth(kx), _smooth(ky)
        denom = np.abs(kys) + np.abs(kxs)
        dH = np.where(denom > 1e-9, (kys - kxs) / denom, 0.0)
        # Reflection about y = x maps theta -> 90-theta and swaps the axes, so
        # kappa_x(theta) = kappa_y(90-theta) and hence dH(theta) = -dH(90-theta),
        # exactly as for the section index.  Averaging against -dH(90-theta)
        # imposes that antisymmetry.
        dH_anti = -mirror(dH)
        dH_sym = 0.5 * (dH + dH_anti)
        ax_a.plot(theta, dH_sym, color=color, lw=2, label=label)

        # exact identities and effect sizes, reported on the figure
        crown = sub["crown_height"].values * 1000
        inv = (crown.max() - crown.min()) / crown.mean() * 100
        mirror_resid = np.abs(hx - mirror_of(hx, hy, theta))
        h_var = (hx.max() - hx.min()) / hx.mean() * 100
        s_var = (sx.max() - sx.min()) / sx.mean() * 100
        # section-estimator index, quoted in the caption only as the contrast
        hxs, hys = _smooth(hx), _smooth(hy)
        d_sec = np.abs(hxs) + np.abs(hys)
        dH_sec = np.where(d_sec > 1e-9, (hxs - hys) / d_sec, 0.0)
        notes.append(
            f"{label}: crown height varies {inv:.2f}% over $\\theta$ (exact 0%);  "
            r"mirror residual $|H_{x=0}(\theta)-H_{y=0}(90^\circ\!-\theta)|$ "
            f"{np.median(mirror_resid)/hx.mean()*100:.2f}%;  "
            rf"$\bar{{H}}$ varies {h_var:.1f}%,  section stress {s_var:.0f}%;  "
            rf"$\Delta H$ spans {dH_sym.min():+.3f}..{dH_sym.max():+.3f} "
            rf"({dH_sym.max()-dH_sym.min():.3f} p-p, vs "
            rf"{dH_sec.max()-dH_sec.min():.3f} for the section estimator)")

    # ── formatting ────────────────────────────────────────────────────────────
    ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  $\bar{H}$")

    ax_s.set_ylabel("Mean stress  (Pa)")
    ax_s.set_title("Section stress")

    ax_a.set_ylabel(r"$(\kappa_y - \kappa_x)\,/\,(|\kappa_y| + |\kappa_x|)$")
    ax_a.set_title(r"Curvature anisotropy index  $\Delta H$  (crown tensor)")
    ax_a.axhline(0, color="0.75", lw=0.8, ls=":")
    ax_a.legend(fontsize=8)
    ax_a.set_xlabel(r"Knitting direction  $\theta_{knit}$  (°)")

    for ax in axes:
        ax.set_xlim(*THETA_RANGE)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

    # combined legend: plane styles + motif colours
    plane_handles = [
        Line2D([0], [0], color="0.4", lw=1.5, ls="-",  label="$x=0$ section"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="$y=0$ section"),
    ]
    motif_handles = [
        Line2D([0], [0], color=COLORS[g], lw=2, label=LABELS[g])
        for g in GROUPS
    ]
    ax_c.legend(handles=motif_handles + plane_handles, fontsize=7.5, loc="best")
    ax_s.legend(handles=plane_handles, fontsize=7.5, loc="best")

    n_runs = len(df)
    fig.suptitle(
        r"Effect of knitting direction $\theta_{knit}$"
        "\n"
        rf"(direct FEA sweep, {n_runs} runs;  "
        r"$s_{wale}=s_{course}=1.0$,  $p=1000$ Pa)",
        fontsize=10, y=1.02,
    )
    fig.text(0.5, -0.005,
             "curves are symmetrised: each is the mean of the two estimates the "
             r"identity $X_{x=0}(\theta)=X_{y=0}(90^\circ\!-\theta)$ forces to "
             "agree,\nso the $x=0$ and $y=0$ curves are exact mirror images and "
             r"the anisotropy index is exactly antisymmetric about $45^\circ$"
             "\n" + "\n".join(notes),
             ha="center", va="top", fontsize=6.5, color="0.35")

    if save:
        path = os.path.join(FIG_DIR, "figM_knit_dir_sweep.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig N ─────────────────────────────────────────────────────────────────────
# Profiles are normalised to remove the sf/p effect:
#   shape row:     z_norm = z / z_max          (0 → 1)
#   curvature row: H_norm = H(s) / H_mean_interior  (relative distribution)
# This makes profiles comparable across samples with different (sf, p),
# focusing purely on the directional effect of theta.

def _normalise_profile(pos, vals, trim=0.10):
    """Return interior mask and normalised values (divided by interior mean)."""
    span = pos.max() - pos.min()
    mask = (pos > pos.min() + trim * span) & (pos < pos.max() - trim * span)
    interior = vals[mask]
    interior = interior[np.isfinite(interior)]
    mean_val = np.mean(interior) if len(interior) else 1.0
    if abs(mean_val) < 1e-9:
        mean_val = 1.0
    return vals / mean_val, mask


def plot_gallery(save=True):
    # Same source as figM: the direct FEA sweep, one run per degree at
    # s_wale = s_course = 1.0, p = 1000 Pa.  This used to read
    # results_with_sections.csv, i.e. the Sobol samples, which were run on the
    # wrong mesh and — for motif 2 — with motif 5 material (validate_fem_runs.py).
    # Because every run in the sweep is already at the fixed (sf, p) slice, the
    # nearest-sample search the old version needed is now an exact lookup.
    df = _load_sweep()

    n_cols = len(GALLERY_TARGETS)
    n_rows = 2 * len(GROUPS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        gridspec_kw={"hspace": 0.60, "wspace": 0.35},
    )

    plane_colors = {"x=0": "#2E8B57", "y=0": "#e07b39"}

    for g_idx, group in enumerate(GROUPS):
        motif = int(group.replace("motif", "").replace("_nocable", ""))
        sub  = df[df["motif"] == motif]
        ax_z_row = 2 * g_idx
        ax_H_row = 2 * g_idx + 1

        for c_idx, target in enumerate(GALLERY_TARGETS):
            ax_z = axes[ax_z_row, c_idx]
            ax_H = axes[ax_H_row, c_idx]

            row = _nearest_angle(sub, target)
            if row is None:
                ax_z.set_visible(False); ax_H.set_visible(False); continue

            verts = _load_verts(row["verts_path"])
            if verts is None:
                ax_z.set_visible(False); ax_H.set_visible(False); continue

            curv = compute_curvatures(verts, _FACES)
            H    = curv["H"]

            for plane, fixed_axis in [("x=0", 0), ("y=0", 1)]:
                pos, z, Hv = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
                if len(pos) < 3:
                    continue
                ls = "-" if fixed_axis == 0 else "--"
                c  = plane_colors[plane]

                # normalised shape: z / z_max  (0 → 1)
                z_norm = z / z.max() if z.max() > 1e-3 else z
                s_norm = pos / (pos.max() - pos.min()) * 2   # -1 … +1

                # normalised curvature: H(s) / H_mean_interior
                Hv_norm, _ = _normalise_profile(pos, Hv)
                # suppress boundary spikes in plot
                span = pos.max() - pos.min()
                interior = (pos > pos.min() + 0.1*span) & (pos < pos.max() - 0.1*span)

                ax_z.plot(s_norm, z_norm, color=c, ls=ls, lw=1.8, label=plane)
                ax_H.plot(s_norm[interior], Hv_norm[interior],
                          color=c, ls=ls, lw=1.8)

            ax_z.set_ylim(-0.05, 1.15)
            ax_z.axhline(0, color="0.8", lw=0.5, ls=":")
            ax_H.axhline(1, color="0.8", lw=0.5, ls=":")   # H/H_mean = 1 reference
            ax_z.set_xlim(-1.05, 1.05)
            ax_H.set_xlim(-1.05, 1.05)
            ax_z.tick_params(labelsize=7)
            ax_H.tick_params(labelsize=7)

            # column title: the sweep hits every target exactly, and (sf, p) are
            # fixed across the whole sweep, so they go in the suptitle instead
            if g_idx == 0:
                ax_z.set_title(rf"$\theta_{{knit}} = {row['knit_dir']:.0f}\degree$",
                               fontsize=8, pad=4)

            if c_idx == 0:
                ax_z.set_ylabel(f"{LABELS[group]}\n$z/z_{{max}}$", fontsize=8)
                ax_H.set_ylabel(r"$H\,/\,\bar{H}_{interior}$", fontsize=8)

            if g_idx == len(GROUPS) - 1:
                ax_H.set_xlabel(r"$s\,/\,2R$", fontsize=7)

            if c_idx == 0 and g_idx == 0:
                ax_z.legend(fontsize=7, loc="upper center",
                            handlelength=1.2, ncol=2, framealpha=0.7)

    fig.suptitle(
        "Normalised cross-section profiles at key knitting directions\n"
        r"(direct FEA sweep, $s_{wale}=s_{course}=1.0$, $p=1000$ Pa;  "
        r"shape: $z/z_{max}$;  curvature: $H/\bar{H}$;  "
        r"solid=$x{=}0$,  dashed=$y{=}0$)",
        fontsize=9, y=1.01,
    )

    if save:
        path = os.path.join(FIG_DIR, "figN_knit_dir_gallery.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fig M: knit direction sweep...")
    plot_sweep()
    print("Fig N: section profile gallery...")
    plot_gallery()
    print("Done.")
