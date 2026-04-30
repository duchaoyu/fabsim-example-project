"""
figQ: Cable rest-length sweep — section metrics comparison.

For each L_rest (no-cable + 1.20, 1.30, 1.40 m and extended 0.90, 1.00, 1.10 m),
plot four panels:
  A. Crown height + cable tension vs L_rest
  B. Shape profiles z(s) along x=0 and y=0
  C. Mean-curvature profiles H(s) along x=0 and y=0
  D. Von Mises stress maps (heatmap on mesh) for selected cases

Motif 1 and Motif 2 shown in separate columns.
"""

import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MESH_PATH
from curvature import read_off
from plot_section_profiles import _slice_plane

FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
SWEEP_DIR = os.path.join(os.path.dirname(__file__), "data", "lrest_sweep")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

_VERTS_REST, _FACES = read_off(MESH_PATH)

# ── Gaussian curvature helpers ────────────────────────────────────────────────

def _find_boundary_verts(faces):
    cnt = defaultdict(int)
    for f in faces:
        for i in range(3):
            cnt[tuple(sorted([int(f[i]), int(f[(i+1)%3])]))] += 1
    bv = set()
    for e, c in cnt.items():
        if c == 1:
            bv.update(e)
    return bv

_BDR_VERTS = _find_boundary_verts(_FACES)

def _gaussian_curvature_vertices(verts):
    """Discrete angle-defect Gaussian curvature per vertex (NaN at boundary)."""
    n = len(verts)
    angle_sum = np.zeros(n)
    area_sum  = np.zeros(n)
    for f in _FACES:
        pts = [verts[f[i]] for i in range(3)]
        cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        face_area = 0.5 * np.linalg.norm(cross)
        if face_area < 1e-15:
            continue
        for i in range(3):
            a = pts[(i+1)%3] - pts[i]
            b = pts[(i+2)%3] - pts[i]
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if la < 1e-15 or lb < 1e-15:
                continue
            angle_sum[f[i]] += np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1))
            area_sum[f[i]]  += face_area / 3.0
    K = np.full(n, np.nan)
    for i in range(n):
        if i not in _BDR_VERTS and area_sum[i] > 1e-15:
            K[i] = (2*np.pi - angle_sum[i]) / area_sum[i]
    return K


def _gauss_curv_section(verts, fixed_axis, band=0.08, n_bins=40, trim=0.0, n_eval=200):
    """Return (pos_mm, K_m-2) along the section, spline fit without symmetry."""
    K_vert = _gaussian_curvature_vertices(verts)
    R = 0.6
    near = (np.abs(verts[:, fixed_axis]) < band * R) & np.isfinite(K_vert)
    if near.sum() < 6:
        return np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    pos = verts[near, pos_axis] * 1000
    K   = K_vert[near]

    order = np.argsort(pos)
    pos, K = pos[order], K[order]
    span = pos.max() - pos.min()
    mask = (pos > pos.min() + trim*span) & (pos < pos.max() - trim*span)
    pos, K = pos[mask], K[mask]
    if len(pos) < 6:
        return np.array([]), np.array([])

    # Symmetrize before binning
    pos_s2 = np.concatenate([pos, -pos])
    K_s2   = np.concatenate([K,    K])
    ord2   = np.argsort(pos_s2)
    pos, K = pos_s2[ord2], K_s2[ord2]

    p_lo, p_hi = pos.min(), pos.max()
    edges   = np.linspace(p_lo, p_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means_K = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = (pos >= edges[k]) & (pos < edges[k+1])
        if sel.sum() >= 1:
            means_K[k] = np.mean(K[sel])
    ok = np.isfinite(means_K)
    if ok.sum() < 4:
        return np.array([]), np.array([])

    c_ok, K_ok = centres[ok], means_K[ok]
    poly_K = np.poly1d(np.polyfit(c_ok, K_ok, min(4, len(c_ok)-1)))
    pos_s  = np.linspace(p_lo, p_hi, n_eval)
    return pos_s, poly_K(pos_s)


# ── colour map: orange (no-cable) → green→blue (cables by tension) ───────────
NOCABLE_COLOR = "#CC6600"   # warm orange — distinct from the cool cable palette
CABLE_CMAP    = "GnBu"      # light green (slack) → dark blue (tight)

def _cable_color(tension, t_max=2000.0):
    return cm.GnBu(0.25 + 0.70 * min(tension / t_max, 1.0))


def _load_verts(prefix):
    p = prefix + "_verts.csv"
    if not os.path.exists(p): return None
    return pd.read_csv(p).sort_values("vid")[["x","y","z"]].values


def _load_stress(prefix):
    p = prefix + "_stress.csv"
    if not os.path.exists(p): return None
    return pd.read_csv(p).sort_values("face")


def _smooth_section(verts, fixed_axis, n_eval=200, band=0.08, n_bins=40, trim=0.0,
                    z_crown_mm=None, piecewise=False):
    """
    Bin vertices near the section plane then fit a cubic spline — no symmetry
    enforced so creases or kinks at the centre are preserved.
    κ = z'' / (1+z'^2)^(3/2),  returned in m⁻¹.
    Returns (pos_mm, z_mm, kappa_per_m).
    """
    R = 0.6
    near = np.abs(verts[:, fixed_axis]) < band * R
    if near.sum() < 6:
        return np.array([]), np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    pos = verts[near, pos_axis] * 1000
    z   = verts[near, 2]        * 1000

    order = np.argsort(pos)
    pos, z = pos[order], z[order]
    span = pos.max() - pos.min()
    mask = (pos > pos.min() + trim*span) & (pos < pos.max() - trim*span)
    pos, z = pos[mask], z[mask]
    if len(pos) < 6:
        return np.array([]), np.array([]), np.array([])

    # For the piecewise (crease) case, mirror data about pos=0 before binning
    # so both halves are fit from symmetrized averages, giving a symmetric kink.
    if piecewise:
        pos_all = np.concatenate([pos, -pos])
        z_all   = np.concatenate([z,    z])
        order2  = np.argsort(pos_all)
        pos, z  = pos_all[order2], z_all[order2]

    # Save full data extent — used for evaluation range (bin centres fall short)
    p_lo, p_hi = pos.min(), pos.max()

    # Bin into means — removes per-vertex scatter while keeping real features
    edges   = np.linspace(p_lo, p_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means_z = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = (pos >= edges[k]) & (pos < edges[k+1])
        if sel.sum() >= 1:
            means_z[k] = np.mean(z[sel])
    ok = np.isfinite(means_z)
    if ok.sum() < 4:
        return np.array([]), np.array([]), np.array([])

    c_ok, z_ok = centres[ok], means_z[ok]

    if not piecewise:
        # Single polynomial — used for sections along the cable (no crease expected)
        deg = min(4, len(c_ok) - 1)
        poly_z   = np.poly1d(np.polyfit(c_ok, z_ok, deg))
        dpoly_z  = poly_z.deriv(1)
        ddpoly_z = poly_z.deriv(2)
        pos_s = np.linspace(p_lo, p_hi, n_eval)
        z_s   = poly_z(pos_s)
        if z_crown_mm is not None:
            z_s = z_s + (z_crown_mm - float(poly_z(0.0)))
        dz    = dpoly_z(pos_s)
        ddz   = ddpoly_z(pos_s)
        kappa = ddz / (1.0 + dz**2)**1.5 * 1000.0
        return pos_s, z_s, kappa

    # Fit two separate polynomials: pos<0 and pos>=0, so the crease at pos=0
    # is preserved as a discontinuity in slope rather than being smoothed away.
    def _half_fit(mask):
        if mask.sum() < 2:
            return None, None, None
        deg = min(4, mask.sum() - 1)
        c = np.polyfit(c_ok[mask], z_ok[mask], deg)
        p = np.poly1d(c)
        return p, p.deriv(1), p.deriv(2)

    neg_mask = c_ok < 0
    pos_mask = c_ok >= 0

    p_neg, dp_neg, ddp_neg = _half_fit(neg_mask)
    p_pos, dp_pos, ddp_pos = _half_fit(pos_mask)

    # Build piecewise evaluation arrays (NaN gap keeps the two halves separate)
    half_neg = n_eval // 2
    half_pos = n_eval - half_neg
    ps_neg = np.linspace(p_lo,   -1e-6, half_neg) if p_neg else np.array([])
    ps_pos = np.linspace(1e-6,    p_hi, half_pos) if p_pos else np.array([])

    pos_s = np.concatenate([ps_neg, [np.nan], ps_pos])
    z_s   = np.concatenate([
        p_neg(ps_neg) if p_neg is not None else np.array([]),
        [np.nan],
        p_pos(ps_pos) if p_pos is not None else np.array([])
    ])

    # Anchor: shift both halves so they meet at the shared crown z at pos=0
    if z_crown_mm is not None:
        if p_neg is not None:
            z_s[:half_neg]        += z_crown_mm - float(p_neg(0.0))
        if p_pos is not None:
            z_s[half_neg+1:]      += z_crown_mm - float(p_pos(0.0))

    # Curvature: computed per half (NaN at the gap)
    def _kappa(ps, dp, ddp):
        if dp is None or len(ps) == 0:
            return np.array([])
        dz  = dp(ps)
        ddz = ddp(ps)
        return ddz / (1.0 + dz**2)**1.5 * 1000.0

    kappa = np.concatenate([
        _kappa(ps_neg, dp_neg, ddp_neg),
        [np.nan],
        _kappa(ps_pos, dp_pos, ddp_pos)
    ])
    return pos_s, z_s, kappa


def _section_profiles(verts):
    """Return dict with x=0 and y=0 smooth profiles (pos, z, kappa)."""
    # Shared crown: vertex closest to (0,0) in xy — anchors both polynomial fits
    r2 = verts[:, 0]**2 + verts[:, 1]**2
    z_crown_mm = float(verts[np.argmin(r2), 2]) * 1000.0
    out = {}
    # x=0 section: along the cable — single smooth fit
    out["x0"] = _smooth_section(verts, 0, z_crown_mm=z_crown_mm, piecewise=False)
    # y=0 section: perpendicular to cable — piecewise fit reveals crease at x=0
    out["y0"] = _smooth_section(verts, 1, z_crown_mm=z_crown_mm, piecewise=True)
    return out


def _stress_section(verts, sdf, fixed_axis, n_bins=40):
    """Return (pos_mm, mean_von_mises) along the section plane (y=0), symmetrized."""
    SECTION_TOL = 0.04
    face_ids  = sdf["face"].values.astype(int)
    centroids = verts[_FACES[face_ids]].mean(axis=1)
    vm        = sdf["von_mises"].values

    mask = np.abs(centroids[:, fixed_axis]) < SECTION_TOL
    if not mask.any():
        return np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    pts  = centroids[mask, pos_axis] * 1000
    vm_m = vm[mask]

    # Symmetrize before binning
    pts_sym = np.concatenate([pts,  -pts])
    vm_sym  = np.concatenate([vm_m,  vm_m])
    ord2    = np.argsort(pts_sym)
    pts, vm_m = pts_sym[ord2], vm_sym[ord2]

    p_lo, p_hi = pts.min(), pts.max()
    edges   = np.linspace(p_lo, p_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means   = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = (pts >= edges[k]) & (pts < edges[k+1])
        if sel.sum() >= 1:
            means[k] = np.mean(vm_m[sel])

    ok = np.isfinite(means)
    if ok.sum() < 4:
        return centres[ok], means[ok]

    c_ok, m_ok = centres[ok], means[ok]
    poly_vm = np.poly1d(np.polyfit(c_ok, m_ok, min(4, len(c_ok)-1)))
    pos_s   = np.linspace(p_lo, p_hi, 200)
    return pos_s, np.maximum(0, poly_vm(pos_s))


# ─────────────────────────────────────────────────────────────────────────────

def plot_lrest_sweep(save=True):
    df = pd.read_csv(os.path.join(SWEEP_DIR, "sweep_results.csv"))

    # Build ordered label list (no_cable first, then increasing L_rest)
    L_VALS = sorted([v for v in df["L_rest_m"].dropna().unique()])
    LABELS  = ["no cable"] + [f"{v:.2f} m" for v in L_VALS]

    SECTION_TOL = 0.03  # faces within 30mm of plane

    motifs = [1, 2]
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5*n_cols, 3.2*n_rows),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    for col_idx, motif in enumerate(motifs):
        sub = df[df["motif"] == motif].copy()
        nocable_row  = sub[sub["L_rest_m"].isna()].iloc[0]
        cable_rows   = sub[sub["L_rest_m"].isin([1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5])].sort_values("L_rest_m")

        # ── Row 0: crown height + tension summary ──────────────────────────────
        ax0 = axes[0, col_idx]
        ax0b = ax0.twinx()

        l_vals = cable_rows["L_rest_m"].values
        h_vals = cable_rows["crown_height"].values * 1000
        t_vals = cable_rows["cable_tension"].values

        ax0.axhline(nocable_row["crown_height"]*1000, color=NOCABLE_COLOR,
                    lw=1.8, ls="--", label="no cable")
        ax0.plot(l_vals, h_vals, "o-", color="#2E8B57", lw=2, ms=6,
                 label="crown height")
        ax0b.bar(l_vals, t_vals, width=0.06, color="#E07B39", alpha=0.55,
                 label="cable tension")
        ax0b.axhline(0, color="0.7", lw=0.5)

        ax0.set_ylabel("Crown height  (mm)", color="#2E8B57")
        ax0b.set_ylabel("Cable tension  (N)", color="#E07B39")
        ax0.set_xlabel(r"$L_{rest}$  (m)")
        ax0.set_title(f"Motif {motif} — crown height & cable tension")
        # Combined legend
        h1, l1 = ax0.get_legend_handles_labels()
        h2, l2 = ax0b.get_legend_handles_labels()
        ax0.legend(h1+h2, l1+l2, fontsize=7.5, loc="upper left")
        ax0.set_xlim(l_vals.min()-0.05, l_vals.max()+0.05)

        # ── Load section profiles for all cases ────────────────────────────────
        profiles = {}
        # no cable
        verts_nc = _load_verts(nocable_row["prefix"])
        if verts_nc is not None:
            profiles["no_cable"] = _section_profiles(verts_nc)

        for _, crow in cable_rows.iterrows():
            key = f"{crow['L_rest_m']:.2f}"
            verts_c = _load_verts(crow["prefix"])
            if verts_c is not None:
                profiles[key] = _section_profiles(verts_c)

        # ── Row 1: shape profiles z(s) ─────────────────────────────────────────
        ax1 = axes[1, col_idx]
        t_max = cable_rows["cable_tension"].max()

        for key, prof in profiles.items():
            if key == "no_cable":
                color, lw, alpha = NOCABLE_COLOR, 2.0, 1.0
            else:
                T = cable_rows[cable_rows["L_rest_m"].round(2) == float(key)
                               ]["cable_tension"].values
                T_val = T[0] if len(T) else 0
                color = _cable_color(T_val, t_max)
                lw, alpha = 1.5, 0.85

            for plane, ls in [("x0", "-"), ("y0", "--")]:
                pos, z, _ = prof[plane]   # already smooth from poly fit
                if len(pos) < 3: continue
                ax1.plot(pos, z, color=color, ls=ls, lw=lw, alpha=alpha)

        ax1.set_xlabel("position along section  (mm)")
        ax1.set_ylabel("z  (mm)")
        ax1.set_title(f"Motif {motif} — shape profiles  (solid=x=0, dashed=y=0)")
        ax1.autoscale(axis="y", tight=False)
        y0, y1 = ax1.get_ylim()
        pad = 0.08 * (y1 - y0)
        ax1.set_ylim(y0 - pad, y1 + pad)
        # Legend
        legend_handles = [
            Line2D([0],[0], color=NOCABLE_COLOR, lw=2, label="no cable"),
            Line2D([0],[0], color=_cable_color(t_max*0.3,t_max), lw=1.5,
                   label="low tension"),
            Line2D([0],[0], color=_cable_color(t_max*0.9,t_max), lw=1.5,
                   label="high tension"),
            Line2D([0],[0], color="0.5", ls="-",  lw=1.2, label="x=0 plane"),
            Line2D([0],[0], color="0.5", ls="--", lw=1.2, label="y=0 plane"),
        ]
        ax1.legend(handles=legend_handles, fontsize=7, loc="upper center",
                   ncol=3, framealpha=0.8)

        # ── Row 2: curvature profiles H(s) — from spline fit ──────────────────
        ax2 = axes[2, col_idx]
        h_all = []

        for key, prof in profiles.items():
            if key == "no_cable":
                color, lw, alpha = NOCABLE_COLOR, 2.0, 1.0
            else:
                T = cable_rows[cable_rows["L_rest_m"].round(2) == float(key)
                               ]["cable_tension"].values
                T_val = T[0] if len(T) else 0
                color = _cable_color(T_val, t_max)
                lw, alpha = 1.5, 0.85

            for plane, ls in [("x0", "-"), ("y0", "--")]:
                pos, z, kappa = prof[plane]
                if len(pos) < 3: continue
                # kappa already in m⁻¹ from _smooth_section
                h_all.extend(kappa[np.isfinite(kappa)])
                ax2.plot(pos, kappa, color=color, ls=ls, lw=lw, alpha=alpha)

        if h_all:
            h_hi = np.percentile(h_all, 97)
            h_lo = np.percentile(h_all, 3)
            pad = 0.15 * (h_hi - h_lo + 1e-9)
            ax2.set_ylim(h_lo - pad, h_hi + pad)

        ax2.axhline(0, color="0.8", lw=0.5, ls=":")
        ax2.set_xlabel("position along section  (mm)")
        ax2.set_ylabel(r"Curvature  $\kappa$  (m$^{-1}$)")
        ax2.set_title(f"Motif {motif} — section curvature  (solid=x=0, dashed=y=0)")

        # ── Row 3: von Mises stress — binned along sections ────────────────────
        ax3 = axes[3, col_idx]

        for case_key, row_data in ([("no_cable", nocable_row)]
                                    + [(f"{r['L_rest_m']:.2f}", r)
                                       for _, r in cable_rows.iterrows()]):
            sdf    = _load_stress(row_data["prefix"])
            verts_d = _load_verts(row_data["prefix"])
            if sdf is None or verts_d is None: continue

            if case_key == "no_cable":
                color, lw, alpha = NOCABLE_COLOR, 2.0, 1.0
            else:
                T_val = cable_rows[
                    cable_rows["L_rest_m"].round(2) == float(case_key)
                ]["cable_tension"].values
                T_val = T_val[0] if len(T_val) else 0
                color = _cable_color(T_val, t_max)
                lw, alpha = 1.5, 0.85

            pts, vm_mean = _stress_section(verts_d, sdf, fixed_axis=1)
            if len(pts) < 2: continue
            ax3.plot(pts, vm_mean, color=color, lw=lw, alpha=alpha)

        ax3.set_xlabel("position along section  (mm)")
        ax3.set_ylabel("Von Mises stress  (Pa)")
        ax3.set_title(f"Motif {motif} — section stress  (y=0 plane)")
        ax3.set_ylim(bottom=0)


    # Shared colorbar for cable tension
    sm = cm.ScalarMappable(cmap=CABLE_CMAP,
                           norm=mcolors.Normalize(vmin=0, vmax=t_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, :], shrink=0.4, pad=0.02,
                        location="right", label="Cable tension  (N)")
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        r"Frictionless sliding steel cable — rest-length sweep  ($s_f$=1.0, $\theta$=0°, $p$=1000 Pa)"
        "\n"
        r"Colour = cable tension (orange=no cable; green$\to$blue = increasing tension)",
        fontsize=10, y=1.005)

    if save:
        path = os.path.join(FIG_DIR, "figQ_lrest_sweep.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting L_rest sweep metrics...")
    plot_lrest_sweep()
    print("Done.")
