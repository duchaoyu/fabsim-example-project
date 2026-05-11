"""
Visualise D5 4-region adaptive optimisation result.
Saves D5_4region_adaptive_result.png.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from collections import defaultdict

HERE    = os.path.dirname(os.path.abspath(__file__))
MESH    = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
OPT_J   = os.path.join(HERE, "optimisation", "D5_4region_adaptive_optimised.json")
RMAP_J  = os.path.join(HERE, "optimisation", "D5_4region_adaptive_map.json")
VERTS   = os.path.join(HERE, "optimisation", "d5_4ra_02756_verts.csv")
OUT     = os.path.join(HERE, "data", "D5", "D5_4region_adaptive_result.png")
OUT_OFF = os.path.join(HERE, "data", "D5", "D5_4region_adaptive_optimised.off")

REGION_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]  # blue, orange, green, red
RMSE_1R = 7.61
RMSE_3R = 5.94


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


def main():
    V_rest, F = load_off(MESH)
    V_target  = V_rest.copy()

    verts_data = np.loadtxt(VERTS, delimiter=",", skiprows=1)
    V_opt = verts_data[:, 1:]

    with open(OPT_J) as f:  opt  = json.load(f)
    with open(RMAP_J) as f: rmap = json.load(f)
    with open(CABLE_J) as f: cable = json.load(f)

    face_region = np.array(rmap["face_regions"])
    n_regions   = opt["n_regions"]
    rmse_4r     = opt["rmse_m"] * 1000

    # Per-vertex region (mode of adjacent faces)
    vert_votes = defaultdict(lambda: defaultdict(int))
    for fi, (a, b, c) in enumerate(F):
        r = face_region[fi]
        for v in (a, b, c):
            vert_votes[v][r] += 1
    vert_region = np.zeros(len(V_rest), dtype=int)
    for v, votes in vert_votes.items():
        vert_region[v] = max(votes, key=votes.get)

    dev = np.linalg.norm(V_opt - V_target, axis=1) * 1000  # mm

    cable_path = cable["vertex_indices"]
    cx = V_target[cable_path, 0]
    cy = V_target[cable_path, 1]

    tri = mtri.Triangulation(V_target[:, 0], V_target[:, 1], F)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"D5  ·  4-region adaptive optimisation\n"
        f"RMSE = {rmse_4r:.2f} mm  (vs 3-region = {RMSE_3R:.2f} mm, 1-region = {RMSE_1R:.2f} mm)",
        fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Panel 0: region map ───────────────────────────────────────────────────
    ax0.set_title("Adaptive region assignment (top-down)", fontsize=10)
    face_colors = [REGION_COLORS[r] for r in face_region]
    for fi, (a, b, c) in enumerate(F):
        xs = [V_target[a,0], V_target[b,0], V_target[c,0], V_target[a,0]]
        ys = [V_target[a,1], V_target[b,1], V_target[c,1], V_target[a,1]]
        ax0.fill(xs, ys, color=face_colors[fi], linewidth=0, alpha=0.85)
    ax0.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), "k-", lw=1.5, label="inner cable")
    seed_verts = opt["seed_vertices"]
    for r, sv in enumerate(seed_verts):
        ax0.scatter(V_target[sv, 0], V_target[sv, 1],
                    s=80, color=REGION_COLORS[r], edgecolors="k",
                    linewidths=1.2, zorder=5)
    legend_patches = [Patch(facecolor=REGION_COLORS[r], edgecolor="k", linewidth=0.6,
                            label=f"R{r} ({opt['regions'][r]['n_faces']} faces)")
                      for r in range(n_regions)]
    legend_patches.append(plt.Line2D([0],[0], color="k", lw=1.5, label="inner cable"))
    ax0.legend(handles=legend_patches, fontsize=7, loc="lower right")
    ax0.set_aspect("equal"); ax0.axis("off")

    # ── Panel 1: 3D surface ───────────────────────────────────────────────────
    ax1.plot_trisurf(V_opt[:,0], V_opt[:,1], V_opt[:,2], triangles=F,
                     alpha=0.7, color="lightgray", linewidth=0)
    for r in range(n_regions):
        mask = vert_region == r
        ax1.scatter(V_opt[mask,0], V_opt[mask,1], V_opt[mask,2],
                    c=REGION_COLORS[r], s=2, alpha=0.4, linewidths=0)
    ax1.set_xlabel("X (m)", fontsize=7); ax1.set_ylabel("Y (m)", fontsize=7)
    ax1.set_zlabel("Z (m)", fontsize=7); ax1.tick_params(labelsize=6)
    crown = V_opt[:,2].max()
    ax1.set_title(f"Optimised surface  (crown={crown:.3f} m)", fontsize=10)

    # ── Panel 2: deviation heatmap ────────────────────────────────────────────
    ax2.set_title("Vertex deviation  |opt − target|  (mm)", fontsize=10)
    tc = ax2.tripcolor(tri, dev, cmap="plasma", shading="gouraud", vmin=0, vmax=dev.max())
    ax2.plot(np.append(cx, cx[0]), np.append(cy, cy[0]), "w-", lw=1.2)
    cb = fig.colorbar(tc, ax=ax2, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7); cb.set_label("mm", fontsize=8)
    ax2.set_aspect("equal"); ax2.axis("off")
    ax2.text(0.02, 0.02, f"RMSE = {rmse_4r:.2f} mm\nmax = {dev.max():.2f} mm",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ── Panel 3: parameter table + RMSE comparison ────────────────────────────
    ax3.axis("off")
    ax3.set_title("Per-region optimised parameters", fontsize=10)

    col_labels = ["Region", "Faces", "sf_wale", "sf_course"]
    rows = []
    for r in range(n_regions):
        rr = opt["regions"][r]
        rows.append([f"R{r}", str(rr["n_faces"]),
                     f"{rr['sf_wale']:.4f}", f"{rr['sf_course']:.4f}"])

    table = ax3.table(cellText=rows, colLabels=col_labels,
                      cellLoc="center", loc="center",
                      bbox=[0.0, 0.28, 1.0, 0.52])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif row > 0 and col == 0:
            cell.set_facecolor(REGION_COLORS[row - 1])
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#cccccc")

    ax3.text(0.5, 0.27, "Knit direction fixed from directional field (not optimised)",
             transform=ax3.transAxes, fontsize=7, ha="center", color="#666666", style="italic")

    ax_bar = ax3.inset_axes([0.1, 0.02, 0.8, 0.18])
    rmse_vals = [RMSE_1R, RMSE_3R, rmse_4r]
    bar_colors = ["#aaaaaa", "#4C72B0", "#C44E52"]
    ax_bar.bar([0,1,2], rmse_vals, color=bar_colors, edgecolor="k", linewidth=0.7)
    ax_bar.set_xticks([0,1,2])
    ax_bar.set_xticklabels(["1-region\n(global)", "3-region\n(BFS)", "4-region\n(adaptive)"], fontsize=7)
    ax_bar.set_ylabel("RMSE (mm)", fontsize=8); ax_bar.tick_params(labelsize=7)
    ax_bar.set_ylim(0, max(rmse_vals) * 1.3)
    for i, v in enumerate(rmse_vals):
        ax_bar.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax_bar.set_title("RMSE comparison", fontsize=8, pad=3)

    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")

    with open(OUT_OFF, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V_opt)} {len(F)} 0\n")
        for v in V_opt:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    print(f"Saved: {OUT_OFF}")


if __name__ == "__main__":
    main()
