"""
Visualise D5 optimisation results: 4-region adaptive vs 10-region Laplacian.

Produces a 3-row figure:
  Row 1 — top-down region map (coloured by region)
  Row 2 — vertex displacement error heatmap (|simulated - target|, mm)
  Row 3 — bar chart: sf_wale and sf_course per region
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE   = os.path.dirname(os.path.abspath(__file__))
MESH   = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
OUT    = os.path.join(HERE, "figures")

RUNS = [
    dict(
        label     = "4-region adaptive\n(per-face knit dirs)",
        opt_json  = os.path.join(HERE, "optimisation", "d5_4ra_v2_optimised.json"),
        map_json  = os.path.join(HERE, "optimisation", "d5_4ra_v2_map.json"),
        verts_csv = os.path.join(HERE, "optimisation", "d5_4ra_v2_03244_verts.csv"),
    ),
    dict(
        label     = "10-region Laplacian\n(per-face knit dirs)",
        opt_json  = os.path.join(HERE, "optimisation", "d5_10lap_v2_optimised.json"),
        map_json  = os.path.join(HERE, "optimisation", "d5_10lap_v2_map.json"),
        verts_csv = os.path.join(HERE, "optimisation", "d5_10lap_v2_01241_verts.csv"),
    ),
]


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()]       for i in range(nv)])
    F = np.array([[int(x)   for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


def load_verts_csv(path, nv):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    V = np.zeros((nv, 3))
    for row in data:
        vid = int(row[0])
        V[vid] = row[1:4]
    return V


def face_centroids(V, F):
    return V[F].mean(axis=1)


def per_vertex_error(V_sim, V_tgt):
    """Mean of nodal distances in mm."""
    return np.linalg.norm(V_sim - V_tgt, axis=1) * 1000   # → mm


def make_figure(runs, V_tgt, F):
    n_cols = len(runs)
    fig, axes = plt.subplots(3, n_cols, figsize=(5.5 * n_cols, 14))
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    triang = mtri.Triangulation(V_tgt[:, 0], V_tgt[:, 1], F)

    for col, run in enumerate(runs):
        with open(run["opt_json"])  as f: opt  = json.load(f)
        with open(run["map_json"])  as f: rmap = json.load(f)

        face_regions = np.array(rmap["face_regions"])
        n_reg = int(face_regions.max()) + 1
        regions = opt["regions"]
        rmse_mm = opt.get("rmse_mm") or opt.get("rmse_m", 0) * 1000

        V_sim = load_verts_csv(run["verts_csv"], len(V_tgt))
        err   = per_vertex_error(V_sim, V_tgt)

        # ── Row 0: region map (top-down, coloured by region) ─────────────────
        ax = axes[0, col]
        centroids = face_centroids(V_tgt, F)
        cmap_reg  = plt.get_cmap("tab10" if n_reg <= 10 else "tab20")
        colors_f  = [cmap_reg(r / max(n_reg - 1, 1)) for r in face_regions]

        for fi in range(len(F)):
            tri_xy = V_tgt[F[fi], :2]
            poly = plt.Polygon(tri_xy, color=colors_f[fi], linewidth=0)
            ax.add_patch(poly)

        # Region centroids with labels
        for r in range(n_reg):
            mask = face_regions == r
            cx, cy = centroids[mask, 0].mean(), centroids[mask, 1].mean()
            ax.text(cx, cy, str(r), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="0.25", alpha=0.6, ec="none"))

        ax.set_aspect("equal"); ax.autoscale_view()
        ax.set_title(f"{run['label']}\nRegion map  ({n_reg} regions)", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.tick_params(labelsize=7)

        # ── Row 1: per-vertex displacement error ─────────────────────────────
        ax = axes[1, col]
        vmax = max(r["verts_csv"] and 1 or 1 for r in runs)   # shared scale below
        tcf = ax.tripcolor(triang, err, cmap="hot_r", vmin=0)
        fig.colorbar(tcf, ax=ax, label="error (mm)", shrink=0.85)
        ax.set_aspect("equal")
        ax.set_title(f"Nodal error  RMSE={rmse_mm:.2f} mm", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.tick_params(labelsize=7)

        # ── Row 2: bar chart sf_wale / sf_course per region ──────────────────
        ax = axes[2, col]
        x      = np.arange(n_reg)
        sw     = [regions[r]["sf_wale"]   for r in range(n_reg)]
        sc     = [regions[r]["sf_course"] for r in range(n_reg)]
        width  = 0.35

        bar1 = ax.bar(x - width/2, sw, width, label="sf_wale",  color="#1f77b4", alpha=0.85)
        bar2 = ax.bar(x + width/2, sc, width, label="sf_course", color="#ff7f0e", alpha=0.85)
        ax.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{r}" for r in range(n_reg)], fontsize=7 if n_reg <= 10 else 6)
        ax.set_ylabel("stretch factor")
        ax.set_ylim(0.85, 1.35)
        ax.set_title("Stretch factors per region", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=7)

        # Annotate bars with values (only for ≤6 regions to avoid clutter)
        if n_reg <= 6:
            for bar in [*bar1, *bar2]:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6)

    # Shared colour scale for error maps
    err_all = []
    for run in runs:
        V_s = load_verts_csv(run["verts_csv"], len(V_tgt))
        err_all.append(per_vertex_error(V_s, V_tgt))
    vmax_shared = max(e.max() for e in err_all)
    for col in range(n_cols):
        # re-draw error map with shared scale
        ax = axes[1, col]
        ax.cla()
        with open(runs[col]["opt_json"]) as f: opt2 = json.load(f)
        rmse_mm2 = opt2.get("rmse_mm") or opt2.get("rmse_m", 0) * 1000
        tcf2 = ax.tripcolor(triang, err_all[col], cmap="hot_r",
                             vmin=0, vmax=vmax_shared)
        ax.set_aspect("equal")
        ax.set_title(f"Nodal error  RMSE={rmse_mm2:.2f} mm", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.tick_params(labelsize=7)

    # Single shared colorbar for error row
    sm = ScalarMappable(cmap="hot_r", norm=Normalize(vmin=0, vmax=vmax_shared))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.37, 0.015, 0.24])
    fig.colorbar(sm, cax=cbar_ax, label="nodal error (mm)")

    fig.suptitle("D5 optimisation: 4-region adaptive vs 10-region Laplacian\n"
                 "(per-face knit directions from directional field)",
                 fontsize=11, y=0.98)
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    V_tgt, F = load_off(MESH)

    fig = make_figure(RUNS, V_tgt, F)
    out_path = os.path.join(OUT, "D5_optimisation_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
