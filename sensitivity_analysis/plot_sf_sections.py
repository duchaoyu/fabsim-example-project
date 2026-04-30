"""
Cross-section gallery: how spatially-varying sf_wale / sf_course shapes the dome.

Selects samples from the existing FEA runs that span the extremes of the
(sf_wale, sf_course) parameter space and overlays their YZ cross-sections,
coloured by crown height.  Also plots a 2-D scatter of (sf_wale, sf_course)
→ crown_height to contextualise where each section sits.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from curvature import read_off as _read_off


def _load_mesh_faces():
    from config import MESH_PATH
    _, faces = _read_off(MESH_PATH)
    return faces


def _slice_yz(verts, faces):
    """YZ cross-section via edge-plane intersection at x=0. Returns (y, z) in metres."""
    pts_y, pts_z = [], []
    for f in faces:
        xs = verts[f, 0]
        for k in range(3):
            a, b = f[k], f[(k + 1) % 3]
            xa, xb = verts[a, 0], verts[b, 0]
            if xa * xb < 0:
                t  = xa / (xa - xb)
                pt = (1 - t) * verts[a] + t * verts[b]
                pts_y.append(pt[1])
                pts_z.append(pt[2])
    if not pts_y:
        return np.array([]), np.array([])
    order = np.argsort(pts_y)
    return np.array(pts_y)[order], np.array(pts_z)[order]

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
FIG_DIR  = os.path.join(_HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "figure.dpi":     150,
})


_FACES = None  # lazy-loaded mesh faces

def _get_faces():
    global _FACES
    if _FACES is None:
        _FACES = _load_mesh_faces()
    return _FACES


def load_verts(sample_id):
    path = os.path.join(DATA_DIR, f"{sample_id:05d}_verts.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).sort_values("vid")
    return df[["x", "y", "z"]].values


def pick_samples(df, group="motif1_nocable", n=8):
    """
    Choose n samples that best cover the corners + centre of (sf_wale, sf_course).
    Includes the four corners, two mid-range, plus highest/lowest crown_height.
    """
    sub = df[df["group"] == group].copy()
    sub = sub.dropna(subset=["crown_height"])

    # corner quartets in sf space
    q25w, q75w = sub["sf_wale"].quantile(0.25),   sub["sf_wale"].quantile(0.75)
    q25c, q75c = sub["sf_course"].quantile(0.25),  sub["sf_course"].quantile(0.75)

    masks = [
        (sub["sf_wale"] < q25w) & (sub["sf_course"] < q25c),   # low-low
        (sub["sf_wale"] > q75w) & (sub["sf_course"] < q25c),   # high-low
        (sub["sf_wale"] < q25w) & (sub["sf_course"] > q75c),   # low-high
        (sub["sf_wale"] > q75w) & (sub["sf_course"] > q75c),   # high-high
    ]
    chosen = []
    for m in masks:
        bucket = sub[m]
        if len(bucket):
            chosen.append(bucket.iloc[len(bucket) // 2])   # pick median within bucket

    # Also add highest and lowest crown_height overall
    chosen.append(sub.nlargest(1, "crown_height").iloc[0])
    chosen.append(sub.nsmallest(1, "crown_height").iloc[0])

    out = pd.DataFrame(chosen).drop_duplicates(subset="sample_id")
    return out.head(n)


# ── Figure 1: cross-section overlay ──────────────────────────────────────────

def plot_sections(group="motif1_nocable", save=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    picks = pick_samples(df, group)

    h_min = picks["crown_height"].min()
    h_max = picks["crown_height"].max()
    norm  = mcolors.Normalize(vmin=h_min, vmax=h_max)
    cmap  = cm.plasma

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    ax_sec, ax_scat = axes

    faces = _get_faces()

    # — left: cross-sections —
    for _, row in picks.iterrows():
        sid   = int(row["sample_id"])
        verts = load_verts(sid)
        if verts is None:
            continue
        y, z = _slice_yz(verts, faces)
        if len(y) < 3:
            continue
        color = cmap(norm(row["crown_height"]))
        lbl = (f"$s_{{w}}$={row['sf_wale']:.2f}, "
               f"$s_{{c}}$={row['sf_course']:.2f}\n"
               f"$h$={row['crown_height']*1000:.0f} mm")
        ax_sec.plot(y * 1000, z * 1000, color=color, lw=1.6, label=lbl)

    ax_sec.set_xlabel("y  (mm)")
    ax_sec.set_ylabel("z  (mm)")
    ax_sec.set_title(f"YZ cross-sections — {group.replace('_', ' ')}", pad=6)
    ax_sec.axhline(0, color="0.7", lw=0.6, ls="--")
    ax_sec.legend(fontsize=6.5, loc="upper left",
                  handlelength=1.2, labelspacing=0.35)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_sec, label="Crown height  (m)", shrink=0.8, pad=0.02)

    # — right: scatter sf_wale vs sf_course coloured by crown_height —
    sub  = df[df["group"] == group].dropna(subset=["crown_height"])
    sc   = ax_scat.scatter(
        sub["sf_wale"], sub["sf_course"],
        c=sub["crown_height"], cmap="plasma",
        s=18, alpha=0.7, linewidths=0,
    )
    # mark the chosen samples
    ax_scat.scatter(picks["sf_wale"], picks["sf_course"],
                    s=55, facecolors="none", edgecolors="black", lw=1.2,
                    zorder=5, label="shown sections")
    fig.colorbar(sc, ax=ax_scat, label="Crown height  (m)", shrink=0.8, pad=0.02)
    ax_scat.set_xlabel(r"$s_{wale}$")
    ax_scat.set_ylabel(r"$s_{course}$")
    ax_scat.set_title(r"$h_{crown}$ landscape in $(s_w,\ s_c)$ space", pad=6)
    ax_scat.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        r"Effect of stretch factors on dome geometry  —  "
        + group.replace("_", " "),
        fontsize=10,
    )

    if save:
        tag  = group
        path = os.path.join(FIG_DIR, f"figC_sections_{tag}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Figure 2: 2×2 panel — all four groups side-by-side ───────────────────────

def plot_section_grid(save=True):
    """
    Single figure with 4 subplots (2×2), each showing the YZ cross-section
    overlay for one (motif, cable) group.  Useful for direct comparison.
    """
    df     = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    groups = ["motif1_nocable", "motif1_cable",
              "motif2_nocable", "motif2_cable"]
    titles = ["Motif 1  (no cable)", "Motif 1  (cable)",
              "Motif 2  (no cable)", "Motif 2  (cable)"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes_flat = axes.flatten()
    cmap = cm.plasma

    for ax, group, title in zip(axes_flat, groups, titles):
        picks = pick_samples(df, group, n=6)
        if picks.empty:
            ax.set_visible(False)
            continue

        h_min = df[df["group"] == group]["crown_height"].min()
        h_max = df[df["group"] == group]["crown_height"].max()
        norm  = mcolors.Normalize(vmin=h_min, vmax=h_max)

        faces = _get_faces()
        for _, row in picks.iterrows():
            sid   = int(row["sample_id"])
            verts = load_verts(sid)
            if verts is None:
                continue
            y, z = _slice_yz(verts, faces)
            if len(y) < 3:
                continue
            color = cmap(norm(row["crown_height"]))
            lbl = (f"$s_w$={row['sf_wale']:.2f} "
                   f"$s_c$={row['sf_course']:.2f}")
            ax.plot(y * 1000, z * 1000, color=color, lw=1.4, label=lbl)

        ax.set_title(title, pad=5)
        ax.set_xlabel("y  (mm)")
        ax.set_ylabel("z  (mm)")
        ax.axhline(0, color="0.75", lw=0.5, ls="--")
        ax.legend(fontsize=6, handlelength=1, labelspacing=0.3, loc="upper left")

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="h (m)", shrink=0.85, pad=0.02)

    fig.suptitle(
        r"YZ cross-sections for extreme $(s_{wale},\, s_{course})$ samples"
        "\n(colour = crown height)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figD_section_grid.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting cross-section figures...")
    for g in ["motif1_nocable", "motif1_cable", "motif2_nocable", "motif2_cable"]:
        plot_sections(g)
    plot_section_grid()
    print("Done.")
