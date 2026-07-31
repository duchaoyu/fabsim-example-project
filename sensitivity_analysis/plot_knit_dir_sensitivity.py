"""
Sensitivity analysis of knitting direction (theta_knit).

Fig M — Line sweep + anisotropy index, from a direct FEA sweep
  Row 1: Section curvature H_x0 and H_y0 vs theta
  Row 2: Section stress sigma_x0 and sigma_y0 vs theta
  Row 3: Anisotropy index  dH = (H_x0-H_y0)/(H_x0+H_y0)
  Fixed: sf_wale = sf_course = 1.0, pressure = 1000 Pa.  Data:
  run_knit_dir_sweep.py -> data/knit_dir_sweep.csv (one FEA run per angle).

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

Fig N — Section profile gallery at theta ≈ 0, 30, 45, 60, 90 degrees
  Shows z(s) and H(s) along x=0 and y=0 for representative samples.
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

FIXED_SF   = 1.0     # figN only: how close a stored sample must be to the slice
FIXED_P    = 1000.0
THETA_RANGE = (0.0, 90.0)

GALLERY_TARGETS = [0, 30, 45, 60, 90]
GALLERY_TOL     = 15.0   # deg


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


def _load_verts(sid):
    p = os.path.join(DATA_DIR, f"{sid:05d}_verts.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).sort_values("vid")[["x", "y", "z"]].values


def _best_sample(sub, target_deg):
    sub = sub.copy()
    sub["angle_dist"] = (sub["knit_dir"] - target_deg).abs()
    cands = sub[sub["angle_dist"] < GALLERY_TOL].copy()
    if cands.empty:
        return None
    cands["score"] = (
        ((cands["sf_wale"]   - FIXED_SF) / 0.3) ** 2 +
        ((cands["sf_course"] - FIXED_SF) / 0.3) ** 2 +
        ((cands["pressure"]  - FIXED_P)  / 500)  ** 2
    )
    return cands.nsmallest(1, "score").iloc[0]


# ── Fig M ─────────────────────────────────────────────────────────────────────

def _add_section_insets(ax_a, theta_targets=(0, 45, 90)):
    """
    Add small cross-section profile insets on the anisotropy panel
    at representative theta values using actual FEA samples.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]

    # Use motif1 for illustration; pick samples closest to sf~1, p~1000
    sub = df[df["group"] == "motif1_nocable"].copy()
    sub["sf_dist"] = ((sub["sf_wale"] - 1.0)**2 + (sub["sf_course"] - 1.0)**2 +
                      ((sub["pressure"] - 1000.0) / 500)**2)

    ax_xlim = ax_a.get_xlim()
    ax_ylim = ax_a.get_ylim()

    inset_w = 0.12   # axes fraction
    inset_h = 0.28
    # positions: (theta_target, x_anchor_in_axes_frac, va)
    positions = {
        0:  (0.03,  0.68),   # left
        45: (0.41,  0.12),   # centre-bottom
        90: (0.78,  0.68),   # right
    }

    for target in theta_targets:
        cands = sub[np.abs(sub["knit_dir"] - target) < 12].copy()
        if cands.empty:
            continue
        row = cands.nsmallest(1, "sf_dist").iloc[0]
        sid = int(row["sample_id"])
        verts = _load_verts(sid)
        if verts is None:
            continue

        curv  = compute_curvatures(verts, _FACES)
        H     = curv["H"]

        xf, yf = positions[target]
        ax_in = ax_a.inset_axes([xf, yf, inset_w, inset_h])
        ax_in.set_facecolor("white")

        for fixed_axis, ls, color in [(0, "-", "#2E8B57"), (1, "--", "#888")]:
            pos, z, _ = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(pos) < 3:
                continue
            s = (pos - pos.min()) / (pos.max() - pos.min())
            z_n = z / z.max() if z.max() > 1e-3 else z
            ax_in.plot(s, z_n, color=color, ls=ls, lw=1.2)

        ax_in.set_xlim(0, 1)
        ax_in.set_ylim(-0.05, 1.15)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(f"θ={target}°", fontsize=6.5, pad=2)
        for sp in ax_in.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("0.5")

        # arrow from inset to the corresponding point on the dH=0 line at target theta
        if ax_xlim[1] > ax_xlim[0]:
            tx = (target - ax_xlim[0]) / (ax_xlim[1] - ax_xlim[0])
            ty = (0.0    - ax_ylim[0]) / (ax_ylim[1] - ax_ylim[0])
            # center-bottom of inset → data point
            ix = xf + inset_w / 2
            iy = yf if yf < 0.5 else yf
            ax_a.annotate("", xy=(tx, ty), xytext=(ix, iy if yf < 0.5 else yf),
                          xycoords="axes fraction", textcoords="axes fraction",
                          arrowprops=dict(arrowstyle="-", color="0.5",
                                          lw=0.7, connectionstyle="arc3,rad=0"))


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
        hx_old, hy_old = sub["H_mean_x0"].values, sub["H_mean_y0"].values
        sx, sy = sub["von_mises_x0"].values, sub["von_mises_y0"].values

        def mirror(v):
            """v evaluated at 90-theta — the partner an exactly mirror-symmetric
            mesh would return for the other cut plane."""
            return np.interp(90.0 - theta, theta, v)

        def mirror_of(a, b, th):
            return np.interp(90.0 - th, th, b)

        # Curvature and stress sections: markers are individual runs, the line is
        # the mean of the two estimates the exact identity says must agree
        # (X(theta) and its mirror partner), and the band spans them.  The band is
        # therefore the discretisation error of the section extraction, measured
        # rather than assumed — this mesh has no mirror symmetry at all (all 399
        # vertices unmatched under x<->y), so the identity only holds in the
        # continuum.
        # the previous estimator, for reference: its scatter is the reason the
        # earlier version of this figure showed structure that is not there
        ax_c.plot(theta, hx_old, ls="none", marker=".", ms=2.2, color="0.62",
                  alpha=0.55, zorder=1,
                  label="binned estimator (previous)" if motif == 1 else None)
        ax_c.plot(theta, hy_old, ls="none", marker=".", ms=2.2, color="0.62",
                  alpha=0.55, zorder=1)

        for ax, vx, vy in ((ax_c, hx, hy), (ax_s, sx, sy)):
            for v, v_partner, ls, marker in ((vx, vy, "-", "o"),
                                             (vy, vx, "--", "s")):
                a, b = _smooth(v), mirror(_smooth(v_partner))
                ax.plot(theta, v, ls="none", marker=marker, ms=2.6, mfc="none",
                        mec=color, mew=0.7, alpha=0.5)
                ax.plot(theta, 0.5 * (a + b), color=color, lw=2, ls=ls)
                ax.fill_between(theta, np.minimum(a, b), np.maximum(a, b),
                                color=color, alpha=0.13, lw=0, zorder=0)

        # Anisotropy index.  The same identity forces dH(theta) = -dH(90-theta),
        # so the band between the two is again the discretisation error.
        hxs, hys = _smooth(hx), _smooth(hy)
        denom = np.abs(hxs) + np.abs(hys)
        dH = np.where(denom > 1e-9, (hxs - hys) / denom, 0.0)
        dH_anti = -mirror(dH)
        ax_a.plot(theta, 0.5 * (dH + dH_anti), color=color, lw=2, label=label)
        ax_a.fill_between(theta, np.minimum(dH, dH_anti),
                          np.maximum(dH, dH_anti), color=color, alpha=0.13,
                          lw=0, zorder=0)
        ax_a.plot(theta, np.where(np.abs(hx) + np.abs(hy) > 1e-9,
                                  (hx - hy) / (np.abs(hx) + np.abs(hy)), 0.0),
                  ls="none", marker="o", ms=2.4, mfc="none", mec=color,
                  mew=0.7, alpha=0.45)

        # exact identities and effect sizes, reported on the figure
        crown = sub["crown_height"].values * 1000
        inv = (crown.max() - crown.min()) / crown.mean() * 100
        mirror = np.abs(hx - mirror_of(hx, hy, theta))
        h_var = (hx.max() - hx.min()) / hx.mean() * 100
        s_var = (sx.max() - sx.min()) / sx.mean() * 100
        notes.append(
            f"{label}: crown height varies {inv:.2f}% over $\\theta$ (exact 0%);  "
            r"mirror residual $|H_{x=0}(\theta)-H_{y=0}(90^\circ\!-\theta)|$ "
            f"{np.median(mirror)/hx.mean()*100:.2f}%;  "
            rf"$\bar{{H}}$ varies {h_var:.1f}%,  section stress {s_var:.0f}%")

    # ── formatting ────────────────────────────────────────────────────────────
    ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  $\bar{H}$")

    ax_s.set_ylabel("Mean stress  (Pa)")
    ax_s.set_title("Section stress")

    ax_a.set_ylabel(r"$(H_{x=0} - H_{y=0})\,/\,(H_{x=0} + H_{y=0})$")
    ax_a.set_title("Curvature anisotropy index")
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
    # markers are individual simulations; lines are the Savitzky-Golay trend
    fig.text(0.5, -0.005,
             "markers = individual runs;  line = mean of the two estimates the "
             r"identity $X_{x=0}(\theta)=X_{y=0}(90^\circ\!-\theta)$ forces to "
             "agree;\nband = the gap between them, i.e. the discretisation error "
             "of the section extraction on this (non-symmetric) mesh\n"
             + "\n".join(notes),
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
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]

    n_cols = len(GALLERY_TARGETS)
    n_rows = 2 * len(GROUPS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        gridspec_kw={"hspace": 0.60, "wspace": 0.35},
    )

    plane_colors = {"x=0": "#2E8B57", "y=0": "#e07b39"}

    for g_idx, group in enumerate(GROUPS):
        sub  = df[df["group"] == group]
        ax_z_row = 2 * g_idx
        ax_H_row = 2 * g_idx + 1

        for c_idx, target in enumerate(GALLERY_TARGETS):
            ax_z = axes[ax_z_row, c_idx]
            ax_H = axes[ax_H_row, c_idx]

            row = _best_sample(sub, target)
            if row is None:
                ax_z.set_visible(False); ax_H.set_visible(False); continue

            sid   = int(row["sample_id"])
            verts = _load_verts(sid)
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

            # column title: target angle + actual sample parameters
            col_title = (f"θ ≈ {target}°  (actual {row['knit_dir']:.0f}°)\n"
                         f"$s_w$={row['sf_wale']:.2f}  "
                         f"$s_c$={row['sf_course']:.2f}  "
                         f"$p$={row['pressure']:.0f} Pa")
            if g_idx == 0:
                ax_z.set_title(col_title, fontsize=7, pad=4)

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
        r"(shape: $z/z_{max}$;  curvature: $H/\bar{H}$;  "
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
