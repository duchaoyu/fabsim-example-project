"""Extract 8 D8 spokes on the dense C5 FDM result and render cables on it.

The dense .obj has different vertex IDs from the .off, so cable_paths_C5.json
indices don't apply. We rebuild the 8 cables by greedy edge-walks from the apex
along each ridge angle θ = 11.25° + k·45° on the form-found mesh.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh

HERE     = os.path.dirname(os.path.abspath(__file__))
RESULT   = os.path.join(HERE, "data", "C5", "mesh_out_C5_dense_latest.json")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_dense_cables.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "cable_paths_C5_dense.json")

SPOKE_ANGLES = [11.25 + 45.0 * k for k in range(8)]
APEX_RADIUS_TOL = 1e-3      # apex = unique vertex with smallest xy radius


# ── Load form-found dense mesh ────────────────────────────────────────────────
mesh = Mesh.from_json(RESULT)
verts_keys = list(mesh.vertices())
V = np.array([mesh.vertex_coordinates(v) for v in verts_keys])
key_to_idx = {k: i for i, k in enumerate(verts_keys)}

# Adjacency (per-vertex neighbour key list)
adj = {k: list(mesh.vertex_neighbors(k)) for k in verts_keys}

# Boundary set
bdry_keys = set(mesh.vertices_on_boundary())

print(f"Loaded {RESULT}")
print(f"  {len(V)} verts, {mesh.number_of_faces()} faces, {len(bdry_keys)} boundary")
print(f"  x[{V[:,0].min():.4f},{V[:,0].max():.4f}]  z[{V[:,2].min():.4f},{V[:,2].max():.4f}]")


# ── Apex vertex (smallest xy radius, ideally exactly at origin) ───────────────
r_xy = np.linalg.norm(V[:, :2], axis=1)
apex_idx = int(np.argmin(r_xy))
apex_key = verts_keys[apex_idx]
print(f"Apex: vert {apex_key}, xy_r={r_xy[apex_idx]:.5f}, z={V[apex_idx,2]:.4f}")


# ── Greedy spoke walk ─────────────────────────────────────────────────────────
def angle_diff_deg(a, b):
    """Smallest unsigned diff in degrees, mod 360."""
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def walk_spoke(theta_deg, max_steps=200):
    """From apex, walk to boundary along edges minimising angular deviation
    from theta_deg while strictly increasing xy radius."""
    cur = apex_key
    path = [cur]
    visited = {cur}
    for _ in range(max_steps):
        if cur in bdry_keys and cur != apex_key:
            break
        cur_xy = V[key_to_idx[cur], :2]
        cur_r  = np.linalg.norm(cur_xy)

        best, best_score = None, np.inf
        for nb in adj[cur]:
            if nb in visited:
                continue
            nb_xy = V[key_to_idx[nb], :2]
            nb_r  = np.linalg.norm(nb_xy)
            if nb_r <= cur_r + 1e-9:
                continue                          # must move outward
            # Score by angular deviation of the neighbour position from theta_deg
            nb_theta = np.degrees(np.arctan2(nb_xy[1], nb_xy[0]))
            score = angle_diff_deg(nb_theta, theta_deg)
            if score < best_score:
                best, best_score = nb, score
        if best is None:
            break
        path.append(best)
        visited.add(best)
        cur = best
    return path


cables = {}
for k, theta in enumerate(SPOKE_ANGLES):
    path = walk_spoke(theta)
    name = f"S{k}_{theta:.2f}deg"
    cables[name] = [int(v) for v in path]
    end_xy = V[key_to_idx[path[-1]], :2]
    end_th = np.degrees(np.arctan2(end_xy[1], end_xy[0]))
    end_r  = np.linalg.norm(end_xy)
    print(f"  {name}: {len(path)} verts, end θ={end_th:6.2f}° (target {theta:5.2f}°), "
          f"end r={end_r:.4f}, on boundary={path[-1] in bdry_keys}")

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"Saved cable JSON: {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
F = np.array([[key_to_idx[k] for k in mesh.face_vertices(f)[:3]]
              for f in mesh.faces()])
face_verts = [[V[F[i, j]] for j in range(3)] for i in range(len(F))]

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
zmax = max(V[:, 2].max(), 1e-6)
poly = Poly3DCollection(face_verts, alpha=0.55,
                        facecolor=cm.viridis(V[F].mean(axis=1)[:, 2] / zmax),
                        edgecolor="white", linewidth=0.12)
ax1.add_collection3d(poly)

for name, path in cables.items():
    pts = np.array([V[key_to_idx[v]] for v in path])
    ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
             color="red", linewidth=2.0, alpha=0.95)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                color="orange", s=14, edgecolors="red", linewidths=0.4, zorder=5)

ax1.set_title(f"C5 dense FDM result — {len(V)}v / {len(F)}f  "
              f"(red = 8 spoke cables)",
              color="white", fontsize=11)
ax1.set_xlabel("x", color="white", fontsize=8)
ax1.set_ylabel("y", color="white", fontsize=8)
ax1.set_zlabel("z", color="white", fontsize=8)
ax1.tick_params(colors="white", labelsize=7)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
xr = float(np.abs(V[:, :2]).max()) * 1.05
ax1.set_xlim(-xr, xr); ax1.set_ylim(-xr, xr); ax1.set_zlim(0, zmax * 1.2)
ax1.view_init(elev=25, azim=-50)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

for face in F:
    p = V[face]
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        ax2.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                 color="white", linewidth=0.18, alpha=0.28)

bdry_pts = np.array([V[key_to_idx[k]] for k in bdry_keys])
ax2.scatter(bdry_pts[:, 0], bdry_pts[:, 1], c="cyan", s=18, zorder=4,
            label=f"boundary ({len(bdry_keys)})")

for name, path in cables.items():
    pts = np.array([V[key_to_idx[v]] for v in path])
    ax2.plot(pts[:, 0], pts[:, 1], color="red", linewidth=2.4, alpha=0.95)
    ax2.scatter(pts[:, 0], pts[:, 1], color="orange", s=18,
                edgecolors="red", linewidths=0.5, zorder=5)
    end = pts[-1]
    angle = float(name.split("_")[1].rstrip("deg"))
    ax2.scatter(end[0], end[1], c="yellow", s=90, zorder=6, marker="*",
                edgecolors="red", linewidths=1)
    ax2.annotate(f"{angle:.1f}°", xy=(end[0], end[1]),
                 xytext=(end[0]*1.13, end[1]*1.13), color="red", fontsize=8,
                 ha="center", va="center")

ax2.set_title("Top-down — cables walked along ridge angles",
              color="white", fontsize=11)
ax2.set_xlabel("x", color="white", fontsize=8)
ax2.set_ylabel("y", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)

fig.suptitle(f"C5 dense FDM result with stiffener spokes",
             color="white", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved PNG: {OUT_PNG}")
