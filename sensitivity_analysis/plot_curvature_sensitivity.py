"""
Curvature sensitivity figures.

Fig E: Spearman correlation heatmap for curvature outputs (H, K, k1, k2 stats)
Fig F: Scatter plots — sf_wale/sf_course vs mean curvature, coloured by pressure
Fig G: Cross-section gallery coloured by local mean curvature H
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH, MOTIFS, HAS_CABLE
from curvature import enrich_results, read_off, compute_curvatures, boundary_vertices


def _slice_at_x0(verts, faces, H_per_vertex, x_tol=0.0):
    """
    Compute a smooth YZ cross-section by intersecting each mesh face with the
    x=0 plane (or x=x_tol plane).  Returns (y, z, H) arrays sorted by y,
    where H is linearly interpolated from per-vertex values.
    """
    pts_y, pts_z, pts_H = [], [], []
    for f in faces:
        xs = verts[f, 0]
        # Check if the face straddles x = x_tol
        xs_shifted = xs - x_tol
        signs = np.sign(xs_shifted)
        if np.all(signs >= 0) or np.all(signs <= 0):
            continue
        # Find the two edges that cross x = x_tol
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            xa, xb = verts[a, 0] - x_tol, verts[b, 0] - x_tol
            if xa * xb < 0:  # edge crosses
                t = xa / (xa - xb)
                pt = (1 - t) * verts[a] + t * verts[b]
                h  = (1 - t) * H_per_vertex[a] + t * H_per_vertex[b]
                pts_y.append(pt[1])
                pts_z.append(pt[2])
                pts_H.append(h)

    if len(pts_y) < 2:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(pts_y)
    return np.array(pts_y)[order], np.array(pts_z)[order], np.array(pts_H)[order]

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

PARAM_LABELS = {
    "sf_wale":    r"$s_{wale}$",
    "sf_course":  r"$s_{course}$",
    "knit_dir":   r"$\theta_{knit}$",
    "pressure":   r"$p$",
    "cable_angle": r"$\phi_{cable}$",
}

CURV_LABELS = {
    "H_mean":  r"$\bar{H}$",
    "H_max":   r"$H_{max}$",
    "K_mean":  r"$\bar{K}$",
    "K_max":   r"$|K|_{max}$",
    "k1_mean": r"$\bar{\kappa}_1$",
    "k2_mean": r"$\bar{\kappa}_2$",
}

CURV_OUTPUTS = list(CURV_LABELS.keys())

GROUP_LABELS = {
    "motif1_nocable": "Motif 1\n(no cable)",
    "motif1_cable":   "Motif 1\n(cable)",
    "motif2_nocable": "Motif 2\n(no cable)",
    "motif2_cable":   "Motif 2\n(cable)",
}


def load_enriched():
    """Load or compute the curvature-enriched results DataFrame."""
    path = os.path.join(DATA_DIR, "results_with_curvature.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print("Computing curvature statistics for all samples...")
    return enrich_results(
        results_csv=os.path.join(DATA_DIR, "results.csv"),
        mesh_off=MESH_PATH,
        data_dir=DATA_DIR,
        out_csv=path,
    )


# ── Fig E: Spearman correlation heatmap (curvature outputs) ───────────────────

def plot_curvature_correlation(save=True):
    df = load_enriched()

    groups_ordered = [
        "motif1_nocable", "motif1_cable",
        "motif2_nocable", "motif2_cable",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
    axes_flat = axes.flatten()

    cmap = plt.cm.RdBu_r
    im_ref = None

    for ax, group in zip(axes_flat, groups_ordered):
        has_cable = "cable" in group and "nocable" not in group
        sub = df[df["group"] == group].copy()

        input_cols = (["sf_wale", "sf_course", "knit_dir", "pressure", "cable_angle"]
                      if has_cable
                      else ["sf_wale", "sf_course", "knit_dir", "pressure"])

        out_cols = [o for o in CURV_OUTPUTS
                    if o in sub.columns and sub[o].std(skipna=True) > 1e-10]

        if not out_cols:
            ax.set_visible(False)
            continue

        mat  = np.zeros((len(input_cols), len(out_cols)))
        pmat = np.ones_like(mat)
        for i, inp in enumerate(input_cols):
            for j, out in enumerate(out_cols):
                valid = sub[[inp, out]].dropna()
                if len(valid) < 5:
                    continue
                r, p = stats.spearmanr(valid[inp], valid[out])
                mat[i, j]  = r
                pmat[i, j] = p

        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        im_ref = im

        for i in range(len(input_cols)):
            for j in range(len(out_cols)):
                r, p = mat[i, j], pmat[i, j]
                sig   = "*" if p < 0.05 else ""
                color = "white" if abs(r) > 0.65 else "black"
                ax.text(j, i, f"{r:.2f}{sig}",
                        ha="center", va="center", fontsize=7, color=color)

        ax.set_xticks(range(len(out_cols)))
        ax.set_xticklabels([CURV_LABELS.get(o, o) for o in out_cols],
                           rotation=30, ha="right")
        ax.set_yticks(range(len(input_cols)))
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in input_cols])
        ax.set_title(GROUP_LABELS.get(group, group), pad=4)
        ax.tick_params(length=0)

        ax.set_xticks(np.arange(-0.5, len(out_cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(input_cols)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes_flat, shrink=0.6, pad=0.02,
                            label=r"Spearman $\rho$")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle(
        r"Spearman correlations: design parameters vs surface curvature metrics"
        "\n(* $p < 0.05$;  "
        r"$\bar{H}$=mean curvature,  $\bar{K}$=Gaussian,  "
        r"$\bar{\kappa}_1/\bar{\kappa}_2$=principal)",
        fontsize=9,
    )

    if save:
        path = os.path.join(FIG_DIR, "figE_curvature_correlation.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig F: Scatter — (sf_wale, sf_course) → mean curvature ───────────────────

def plot_curvature_landscape(save=True):
    df = load_enriched()

    groups = ["motif1_nocable", "motif2_nocable"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), constrained_layout=True)

    metrics = [("H_mean", r"Mean curvature $\bar{H}$ (m$^{-1}$)"),
               ("K_mean", r"Gaussian curvature $\bar{K}$ (m$^{-2}$)")]

    for col_idx, (metric, clabel) in enumerate(metrics):
        for row_idx, group in enumerate(groups):
            ax  = axes[row_idx, col_idx]
            sub = df[df["group"] == group].dropna(subset=[metric])
            if sub.empty:
                ax.set_visible(False)
                continue

            sc = ax.scatter(sub["sf_wale"], sub["sf_course"],
                            c=sub[metric], cmap="viridis",
                            s=22, alpha=0.8, linewidths=0)
            fig.colorbar(sc, ax=ax, label=clabel, shrink=0.85, pad=0.02)
            ax.set_xlabel(r"$s_{wale}$")
            ax.set_ylabel(r"$s_{course}$")
            ax.set_title(f"{GROUP_LABELS[group].replace(chr(10), ' ')} — {metric}",
                         pad=5)

    fig.suptitle(
        r"Curvature landscape in $(s_{wale},\, s_{course})$ space"
        "\n(motif 1 vs motif 2, uniform pressure range)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figF_curvature_landscape.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig G: Cross-section coloured by local mean curvature H ──────────────────

def plot_curvature_sections(group="motif1_nocable", save=True):
    """
    Show 3 cross-sections (low/mid/high crown height) and colour each point
    by its local mean curvature H.
    """
    df      = load_enriched()
    sub     = df[df["group"] == group].dropna(subset=["crown_height", "H_mean"])

    # Pick low / mid / high crown height samples
    q33, q66 = sub["crown_height"].quantile([0.2, 0.8])
    samples  = {
        "low h":  sub[sub["crown_height"] <= q33].nsmallest(1, "crown_height").iloc[0],
        "mid h":  sub.iloc[(sub["crown_height"] - sub["crown_height"].median()).abs().argsort()[:1].iloc[0]],
        "high h": sub[sub["crown_height"] >= q66].nlargest(1, "crown_height").iloc[0],
    }

    # Load mesh faces for curvature re-computation
    rest_verts, faces = read_off(MESH_PATH)
    bdry_set = boundary_vertices(faces, len(rest_verts))
    interior_mask = np.array([i not in bdry_set for i in range(len(rest_verts))])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    for ax, (label, row) in zip(axes, samples.items()):
        sid  = int(row["sample_id"])
        vpath = os.path.join(DATA_DIR, f"{sid:05d}_verts.csv")
        if not os.path.exists(vpath):
            ax.set_visible(False)
            continue

        vdf   = pd.read_csv(vpath).sort_values("vid")
        verts = vdf[["x", "y", "z"]].values
        curv  = compute_curvatures(verts, faces)
        H     = curv["H"]

        # Near-centreline vertices
        y_vals, z_vals, h_vals = _slice_at_x0(verts, faces, H, x_tol=0.0)
        if len(y_vals) < 3:
            ax.set_visible(False)
            continue

        y_vals *= 1000
        z_vals *= 1000

        norm = mcolors.Normalize(vmin=np.nanpercentile(h_vals, 5),
                                 vmax=np.nanpercentile(h_vals, 95))
        cmap = cm.plasma

        for k in range(len(y_vals) - 1):
            ax.plot(y_vals[k:k+2], z_vals[k:k+2],
                    color=cmap(norm(h_vals[k])), lw=2.5, solid_capstyle="round")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=r"$H$ (m$^{-1}$)", shrink=0.85, pad=0.02)

        ax.set_xlabel("y  (mm)")
        ax.set_ylabel("z  (mm)")
        ax.set_title(
            f"{label}  —  $h$={row['crown_height']*1000:.0f} mm\n"
            f"$s_w$={row['sf_wale']:.2f}, $s_c$={row['sf_course']:.2f}, "
            f"$p$={row['pressure']:.0f} Pa",
            pad=5, fontsize=8,
        )
        ax.axhline(0, color="0.8", lw=0.6, ls="--")

    fig.suptitle(
        f"YZ cross-sections coloured by local mean curvature $H$ — {group.replace('_', ' ')}",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, f"figG_curvature_sections_{group}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Generating curvature figures...")
    plot_curvature_correlation()
    plot_curvature_landscape()
    for g in ["motif1_nocable", "motif1_cable", "motif2_nocable", "motif2_cable"]:
        plot_curvature_sections(g)
    print("Done.")
