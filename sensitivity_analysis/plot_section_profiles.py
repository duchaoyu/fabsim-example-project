"""
Cross-section curvature profiles along x=0 and y=0 planes.

For each selected sample, computes:
  - Shape profile z(s) along the two cutting planes
  - Mean-curvature profile H(s) along each plane

where s is arc length from the left boundary to the right boundary.

Comparing the two planes reveals anisotropy: if sf_wale ≠ sf_course the
dome is stiffer in one direction, and the curvature concentration differs
between the x=0 (one material axis) and y=0 (other material axis) sections.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off, compute_curvatures

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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

# ── mesh (loaded once) ────────────────────────────────────────────────────────
_REST_VERTS, _FACES = read_off(MESH_PATH)


def _load_verts(sample_id):
    path = os.path.join(DATA_DIR, f"{sample_id:05d}_verts.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("vid")
    return df[["x", "y", "z"]].values


def _slice_plane(verts, faces, H_per_vertex, fixed_axis, fixed_val=0.0):
    """
    Intersect mesh with the plane coord[fixed_axis] = fixed_val.
    Returns (s, z, H) where s is arc length in mm, z in mm, H in m⁻¹.
    fixed_axis: 0 → x=0 plane (YZ section), 1 → y=0 plane (XZ section)
    """
    other_axis = 1 if fixed_axis == 0 else 0   # the "horizontal" coordinate in the cut

    pts_other, pts_z, pts_H = [], [], []

    for f in faces:
        xs = verts[f, fixed_axis] - fixed_val
        # Skip face if it doesn't straddle the plane
        if np.all(xs >= 0) or np.all(xs <= 0):
            continue
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            xa, xb = verts[a, fixed_axis] - fixed_val, verts[b, fixed_axis] - fixed_val
            if xa * xb < 0:
                t  = xa / (xa - xb)
                pt = (1 - t) * verts[a] + t * verts[b]
                h  = (1 - t) * H_per_vertex[a] + t * H_per_vertex[b]
                pts_other.append(pt[other_axis])
                pts_z.append(pt[2])
                pts_H.append(h)

    if len(pts_other) < 2:
        return np.array([]), np.array([]), np.array([])

    pts_other = np.array(pts_other)
    pts_z     = np.array(pts_z)
    pts_H     = np.array(pts_H)

    order     = np.argsort(pts_other)
    pts_other = pts_other[order]
    pts_z     = pts_z[order]
    pts_H     = pts_H[order]

    return pts_other * 1000, pts_z * 1000, pts_H   # pos, z in mm; H in m⁻¹


def pick_samples(df, group, n=4):
    """
    Select samples that span sf_wale vs sf_course asymmetry.
    Returns a list of (label, row) tuples.
    """
    sub = df[df["group"] == group].dropna(subset=["crown_height"])
    sub = sub[sub["crown_height"] > 0.02].copy()   # skip collapsed cases
    if len(sub) == 0:
        return []

    # quartiles of sf_wale and sf_course
    qw = sub["sf_wale"].quantile([0.25, 0.75])
    qc = sub["sf_course"].quantile([0.25, 0.75])

    # Four corners: (low sw, low sc), (high sw, low sc), (low sw, high sc), (high sw, high sc)
    buckets = {
        r"$s_w{\downarrow}s_c{\downarrow}$": sub[(sub["sf_wale"] < qw[0.25]) & (sub["sf_course"] < qc[0.25])],
        r"$s_w{\uparrow}s_c{\downarrow}$":   sub[(sub["sf_wale"] > qw[0.75]) & (sub["sf_course"] < qc[0.25])],
        r"$s_w{\downarrow}s_c{\uparrow}$":   sub[(sub["sf_wale"] < qw[0.25]) & (sub["sf_course"] > qc[0.75])],
        r"$s_w{\uparrow}s_c{\uparrow}$":     sub[(sub["sf_wale"] > qw[0.75]) & (sub["sf_course"] > qc[0.75])],
    }

    chosen = []
    for lbl, bucket in buckets.items():
        if len(bucket) == 0:
            continue
        row = bucket.iloc[len(bucket) // 2]
        chosen.append((lbl, row))

    return chosen[:n]


# ── Main figure ───────────────────────────────────────────────────────────────

def plot_profiles(group="motif1_nocable", save=True):
    """
    2 × N_samples grid:
      row 0: shape profiles z(s) — x=0 (solid) vs y=0 (dashed)
      row 1: curvature profiles H(s) — x=0 (solid) vs y=0 (dashed)
    One column per sample, chosen to span sf_wale/sf_course combinations.
    """
    df      = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    samples = pick_samples(df, group)
    if not samples:
        print(f"  No valid samples for {group}")
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 5.5), constrained_layout=True)
    if n == 1:
        axes = axes[:, None]

    colors = {"x=0": "#1f77b4", "y=0": "#d62728"}

    for col, (lbl, row) in enumerate(samples):
        sid   = int(row["sample_id"])
        verts = _load_verts(sid)
        if verts is None:
            continue

        curv  = compute_curvatures(verts, _FACES)
        H     = curv["H"]

        ax_shape = axes[0, col]
        ax_curv  = axes[1, col]

        for plane_label, fixed_axis in [("x=0", 0), ("y=0", 1)]:
            s, z, Hv = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(s) < 3:
                continue

            ls = "-" if fixed_axis == 0 else "--"
            c  = colors[plane_label]

            ax_shape.plot(s, z, color=c, ls=ls, lw=1.8, label=plane_label)
            ax_curv.plot(s, Hv, color=c, ls=ls, lw=1.8, label=plane_label)

        # Formatting
        ax_shape.axhline(0, color="0.8", lw=0.5, ls=":")
        ax_curv.axhline(0, color="0.8", lw=0.5, ls=":")

        # Clip H axis to interior range (exclude first/last 10% of span)
        # to suppress boundary spikes
        all_H = []
        for fixed_axis in [0, 1]:
            pos_, _, Hv_ = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(pos_) < 3:
                continue
            span = pos_.max() - pos_.min()
            lo, hi = pos_.min() + 0.1 * span, pos_.min() + 0.9 * span
            mask = (pos_ > lo) & (pos_ < hi)
            if mask.any():
                all_H.extend(Hv_[mask])
        if all_H:
            h_hi = np.percentile(all_H, 97)
            h_lo = min(0, np.percentile(all_H, 3))
            ax_curv.set_ylim(h_lo - 0.1 * abs(h_hi), h_hi * 1.2)

        title = (f"{lbl}\n"
                 f"$s_w$={row['sf_wale']:.2f} $s_c$={row['sf_course']:.2f}\n"
                 f"$h$={row['crown_height']*1000:.0f} mm  $p$={row['pressure']:.0f} Pa")
        ax_shape.set_title(title, fontsize=7.5, pad=4)

        if col == 0:
            ax_shape.set_ylabel("z  (mm)")
            ax_curv.set_ylabel(r"$H$  (m$^{-1}$)")
        ax_curv.set_xlabel("position  (mm)")

        ax_shape.legend(fontsize=7, loc="upper center",
                        handlelength=1.5, ncol=2, framealpha=0.7)

    fig.suptitle(
        f"Cross-section profiles along x=0 and y=0 planes — {group.replace('_', ' ')}\n"
        r"(solid = $x{=}0$,  dashed = $y{=}0$;  anisotropy visible when profiles differ)",
        fontsize=9,
    )

    if save:
        path = os.path.join(FIG_DIR, f"figH_profiles_{group}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


def plot_profiles_overlay(group="motif1_nocable", save=True):
    """
    Overlay all selected samples on a single 1×2 axes:
      left:  all z(s) profiles, x=0 in warm colours, y=0 in cool colours
      right: all H(s) profiles, same colouring
    Colour intensity = crown height.
    """
    df      = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    samples = pick_samples(df, group, n=4)
    if not samples:
        print(f"  No valid samples for {group}")
        return

    h_vals = [row["crown_height"] for _, row in samples]
    norm   = mcolors.Normalize(vmin=min(h_vals), vmax=max(h_vals))
    cmap_x = cm.Oranges
    cmap_y = cm.Blues

    fig, (ax_z, ax_H) = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    for _, row in samples:
        sid   = int(row["sample_id"])
        verts = _load_verts(sid)
        if verts is None:
            continue
        curv  = compute_curvatures(verts, _FACES)
        H     = curv["H"]
        t     = norm(row["crown_height"])
        lbl   = (f"$s_w$={row['sf_wale']:.2f}, $s_c$={row['sf_course']:.2f}, "
                 f"$h$={row['crown_height']*1000:.0f} mm")

        for fixed_axis, cmap, ls, pname in [(0, cmap_x, "-", "x=0"), (1, cmap_y, "--", "y=0")]:
            s, z, Hv = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(s) < 3:
                continue
            color = cmap(0.4 + 0.5 * t)
            ax_z.plot(s, z,  color=color, ls=ls, lw=1.8,
                      label=f"{pname}  {lbl}" if fixed_axis == 0 else None)
            ax_H.plot(s, Hv, color=color, ls=ls, lw=1.8)

    # Clip H axis to interior range across all shown profiles
    all_H_interior = []
    for _, row in samples:
        sid = int(row["sample_id"])
        verts = _load_verts(sid)
        if verts is None:
            continue
        curv = compute_curvatures(verts, _FACES)
        H    = curv["H"]
        for fixed_axis in [0, 1]:
            pos_, _, Hv_ = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(pos_) < 3:
                continue
            span = pos_.max() - pos_.min()
            lo, hi = pos_.min() + 0.1 * span, pos_.min() + 0.9 * span
            mask = (pos_ > lo) & (pos_ < hi)
            if mask.any():
                all_H_interior.extend(Hv_[mask])
    if all_H_interior:
        h_hi = np.percentile(all_H_interior, 97)
        ax_H.set_ylim(-0.1 * h_hi, h_hi * 1.3)

    for ax in (ax_z, ax_H):
        ax.axhline(0, color="0.8", lw=0.5, ls=":")
        ax.set_xlabel("position  (mm)")

    ax_z.set_ylabel("z  (mm)")
    ax_H.set_ylabel(r"$H$  (m$^{-1}$)")
    ax_z.set_title("Shape profiles  z(s)", pad=5)
    ax_H.set_title(r"Mean-curvature profiles  $H(s)$", pad=5)
    ax_z.legend(fontsize=6.5, loc="upper center", handlelength=1.5,
                ncol=1, framealpha=0.7)

    # Legend patches for plane encoding
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="0.4", ls="-",  lw=1.5, label="x=0 plane"),
               Line2D([0], [0], color="0.4", ls="--", lw=1.5, label="y=0 plane")]
    ax_H.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.7)

    fig.suptitle(
        f"Shape and curvature profiles — {group.replace('_', ' ')}\n"
        r"Warm = $x{=}0$, cool = $y{=}0$; deeper colour = taller dome",
        fontsize=9,
    )

    if save:
        path = os.path.join(FIG_DIR, f"figI_profiles_overlay_{group}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting cross-section curvature profiles...")
    for g in ["motif1_nocable", "motif1_cable", "motif2_nocable", "motif2_cable"]:
        plot_profiles(g)
        plot_profiles_overlay(g)
    print("Done.")
