"""
Visualise D5 3-region BFS optimisation result.
Saves D5_3region_result.png — does NOT overwrite D5_optim_result.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from collections import deque, defaultdict

HERE   = os.path.dirname(os.path.abspath(__file__))
MESH   = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
OPT_J  = os.path.join(HERE, "optimisation", "D5_3region_optimised.json")
RMAP_J = os.path.join(HERE, "optimisation", "D5_3region_map.json")
VERTS  = os.path.join(HERE, "optimisation", "d5_3r_run2_00408_verts.csv")
OUT    = os.path.join(HERE, "data", "D5", "D5_3region_result.png")
PREV   = os.path.join(HERE, "data", "D5", "D5_optim_result.png")

REGION_COLORS = ["#4C72B0", "#DD8452", "#55A868"]  # blue, orange, green


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


def main():
    V_rest, F = load_off(MESH)
    V_target  = V_rest.copy()   # rest = target for D5

    verts_data = np.loadtxt(VERTS, delimiter=",", skiprows=1)
    V_opt = verts_data[:, 1:]   # drop index column

    with open(OPT_J) as f:  opt  = json.load(f)
    with open(RMAP_J) as f: rmap = json.load(f)
    with open(CABLE_J) as f: cable = json.load(f)

    face_region = np.array(rmap["face_regions"])
    n_regions   = opt["n_regions"]
    rmse_3r     = opt["rmse_m"] * 1000        # mm
    rmse_1r     = 7.61                        # mm (global 8-sector run)

    # Per-vertex region (mode of adjacent faces)
    vert_region = np.zeros(len(V_rest), dtype=int)
    vert_votes  = defaultdict(lambda: defaultdict(int))
    for fi, (a, b, c) in enumerate(F):
        r = face_region[fi]
        for v in (a, b, c):
            vert_votes[v][r] += 1
    for v, votes in vert_votes.items():
        vert_region[v] = max(votes, key=votes.get)

    # Per-vertex deviation magnitude
    dev = np.linalg.norm(V_opt - V_target, axis=1) * 1000  # mm

    # Cable vertices for overlay
    cable_verts = np.array(cable["vertices"])
    cable_path  = cable["vertex_indices"]
    cx = V_target[cable_path, 0]
    cy = V_target[cable_path, 1]

    # Triangulation for matplotlib
    tri = mtri.Triangulation(V_target[:, 0], V_target[:, 1], F)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")

    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, 0])   # BFS regions top-down
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")  # 3D surface
    ax2 = fig.add_subplot(gs[1, 0])   # deviation heatmap
    ax3 = fig.add_subplot(gs[1, 1])   # per-region parameters

    fig.suptitle(
        f"D5  ·  3-region BFS optimisation\n"
        f"RMSE = {rmse_3r:.2f} mm  (↓ {rmse_1r - rmse_3r:.2f} mm vs 1-region global = {rmse_1r:.2f} mm)",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Panel 0: BFS region map ───────────────────────────────────────────────
    face_colors = np.array([REGION_COLORS[r] for r in face_region])
    ax0.set_title("BFS region assignment", fontsize=10)
    for fi, (a, b, c) in enumerate(F):
        xs = [V_target[a, 0], V_target[b, 0], V_target[c, 0], V_target[a, 0]]
        ys = [V_target[a, 1], V_target[b, 1], V_target[c, 1], V_target[a, 1]]
        ax0.fill(xs, ys, color=face_colors[fi], linewidth=0, alpha=0.85)
    ax0.plot(np.append(cx, cx[0]), np.append(cy, cy[0]),
             "k-", lw=1.5, label="inner cable")
    # seed vertices
    for r, rv in enumerate(opt["seed_vertices"]):
        ax0.scatter(V_target[rv, 0], V_target[rv, 1],
                    s=80, color=REGION_COLORS[r], edgecolors="k",
                    linewidths=1.2, zorder=5)
    legend_patches = [Patch(facecolor=REGION_COLORS[r],
                            edgecolor="k", linewidth=0.6,
                            label=f"R{r} ({opt['regions'][r]['n_faces']} faces)")
                      for r in range(n_regions)]
    legend_patches.append(plt.Line2D([0], [0], color="k", lw=1.5, label="inner cable"))
    ax0.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax0.set_aspect("equal"); ax0.axis("off")

    # ── Panel 1: 3D optimised surface ─────────────────────────────────────────
    vert_col = np.array([REGION_COLORS[vert_region[v]] for v in range(len(V_opt))])
    ax1.set_title("Optimised surface (colour = region)", fontsize=10)
    ax1.plot_trisurf(V_opt[:, 0], V_opt[:, 1], V_opt[:, 2], triangles=F,
                     alpha=0.7, color="lightgray", linewidth=0)
    # highlight region boundaries via scatter of seed verts
    for r in range(n_regions):
        mask = vert_region == r
        ax1.scatter(V_opt[mask, 0], V_opt[mask, 1], V_opt[mask, 2],
                    c=REGION_COLORS[r], s=2, alpha=0.4, linewidths=0)
    ax1.set_xlabel("X (m)", fontsize=7); ax1.set_ylabel("Y (m)", fontsize=7)
    ax1.set_zlabel("Z (m)", fontsize=7)
    ax1.tick_params(labelsize=6)
    crown = V_opt[:, 2].max()
    ax1.set_title(f"Optimised surface  (crown={crown:.3f} m)", fontsize=10)

    # ── Panel 2: deviation heatmap ────────────────────────────────────────────
    ax2.set_title("Vertex deviation  |opt − target|  (mm)", fontsize=10)
    tc = ax2.tripcolor(tri, dev, cmap="plasma", shading="gouraud",
                       vmin=0, vmax=dev.max())
    ax2.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), "w-", lw=1.2)
    cb = fig.colorbar(tc, ax=ax2, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("mm", fontsize=8)
    ax2.set_aspect("equal"); ax2.axis("off")
    ax2.text(0.02, 0.02, f"RMSE = {rmse_3r:.2f} mm\nmax = {dev.max():.2f} mm",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 3: per-region parameter table ───────────────────────────────────
    ax3.axis("off")
    ax3.set_title("Per-region optimised parameters", fontsize=10)

    col_labels = ["Region", "Seed v", "Faces", "knit θ (°)", "sf_wale", "sf_course"]
    rows = []
    for r in range(n_regions):
        rr = opt["regions"][r]
        rows.append([f"R{r}", str(rr["seed_vertex"]), str(rr["n_faces"]),
                     f"{rr['knit_dir_deg']:.1f}",
                     f"{rr['sf_wale']:.4f}", f"{rr['sf_course']:.4f}"])

    table = ax3.table(cellText=rows, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.25, 1.0, 0.55])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and col == 0:
            cell.set_facecolor(REGION_COLORS[row - 1])
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#cccccc")

    # comparison bar
    ax_bar = ax3.inset_axes([0.1, 0.02, 0.8, 0.18])
    xs = [0, 1]
    ax_bar.bar(xs, [rmse_1r, rmse_3r],
               color=["#aaaaaa", "#4C72B0"], edgecolor="k", linewidth=0.7)
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(["1-region\n(global)", "3-region\n(BFS)"], fontsize=7)
    ax_bar.set_ylabel("RMSE (mm)", fontsize=8)
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_ylim(0, max(rmse_1r, rmse_3r) * 1.3)
    for i, v in enumerate([rmse_1r, rmse_3r]):
        ax_bar.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax_bar.set_title("RMSE comparison", fontsize=8, pad=3)

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    assert os.path.exists(PREV), f"Previous result missing: {PREV}"
    print(f"Preserved: {PREV}")


if __name__ == "__main__":
    main()
