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
import matplotlib.colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh

HERE     = os.path.dirname(os.path.abspath(__file__))
RESULT   = os.path.join(HERE, "data", "C5", "mesh_out_C5_dense_latest.json")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_dense_cables.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "cable_paths_C5_dense.json")

APEX_RADIUS_TOL = 1e-3      # apex = unique vertex with smallest xy radius

# Spoke angles are detected from the FDM result itself (see code below) — no
# longer hardcoded to 11.25° + k·45° (which was the .off remesher's choice).


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


# ── Force-density-driven cable extraction ────────────────────────────────────
# Cables = paths from apex to boundary that maximise force-density. We use
# Dijkstra from apex with edge weight = 1/q (so high-q edges are short),
# pick 8 distinct boundary endpoints with smallest distance, then take their
# Dijkstra paths back to apex. This recovers exactly the radial high-q
# "spokes" that the FDM optimisation produced.
import heapq

edges_list = list(mesh.edges())
edge_q = {tuple(sorted([u, v])): float(mesh.edge_attribute((u, v), "qpre"))
          for u, v in edges_list}
print(f"FDM q range: [{min(edge_q.values()):.3f}, {max(edge_q.values()):.3f}], "
      f"mean={np.mean(list(edge_q.values())):.3f}")


# ── Dijkstra from apex with weight = 1/q (force-density-driven) ──────────────
def dijkstra_from_apex():
    dist = {k: np.inf for k in verts_keys}
    prev = {k: None for k in verts_keys}
    dist[apex_key] = 0.0
    pq = [(0.0, apex_key)]
    while pq:
        d, uk = heapq.heappop(pq)
        if d > dist[uk]: continue
        for vk in adj[uk]:
            qe = edge_q[tuple(sorted([uk, vk]))]
            w = 1.0 / max(qe, 1e-6)             # high q → short edge → preferred
            nd = d + w
            if nd < dist[vk]:
                dist[vk] = nd
                prev[vk] = uk
                heapq.heappush(pq, (nd, vk))
    return dist, prev

dist, prev = dijkstra_from_apex()
def reconstruct(target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

# Pick 8 boundary endpoints: smallest-distance per 45° angular bucket
def angle_diff_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)

apex_xy = V[apex_idx, :2]
bdry_list = list(bdry_keys)
bdry_th = []
for bk in bdry_list:
    bv = V[key_to_idx[bk], :2] - apex_xy
    bdry_th.append(np.degrees(np.arctan2(bv[1], bv[0])) % 360.0)
bdry_th = np.array(bdry_th)
bdry_dist = np.array([dist[k] for k in bdry_list])

# Angular non-maximum suppression:
#   sort boundary verts by ascending q-distance, greedily take the lowest-distance
#   vertex that is more than MIN_SEP_DEG away from anything already picked.
N_CABLES    = 8
MIN_SEP_DEG = 25.0
order = np.argsort(bdry_dist)
selected, selected_th = [], []
for i in order:
    if len(selected) == N_CABLES:
        break
    th_i = float(bdry_th[i])
    if all(angle_diff_deg(th_i, t) > MIN_SEP_DEG for t in selected_th):
        selected.append(bdry_list[i])
        selected_th.append(th_i)
# Sort by angle for nicer printing
ord_th = np.argsort(selected_th)
selected    = [selected[i] for i in ord_th]
selected_th = [selected_th[i] for i in ord_th]
SPOKE_ANGLES = list(selected_th)
print(f"\n8 cable endpoints (angle, dist) — by min-q-distance from apex:")
for k, (bk, th) in enumerate(zip(selected, selected_th)):
    print(f"  cable {k}: vertex {bk}  θ={th:6.2f}°  dist={dist[bk]:.3f}")

cables = {}
print("\nReconstructing high-q paths:")
for k, (bk, theta) in enumerate(zip(selected, selected_th)):
    path = reconstruct(bk)
    name = f"S{k}_{theta:.2f}deg"
    cables[name] = [int(v) for v in path]
    end_xy = V[key_to_idx[path[-1]], :2]
    end_th = np.degrees(np.arctan2(end_xy[1], end_xy[0]))
    end_r  = np.linalg.norm(end_xy)
    # mean q along path
    qs = [edge_q[tuple(sorted([path[i], path[i+1]]))] for i in range(len(path)-1)]
    print(f"  {name}: {len(path)} verts, end θ={end_th:6.2f}°, r={end_r:.4f}, "
          f"mean q={np.mean(qs):.3f}, min q={min(qs):.3f}")

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"Saved cable JSON: {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
r_xy_all = np.linalg.norm(V[:, :2], axis=1)
F = np.array([[key_to_idx[k] for k in mesh.face_vertices(f)[:3]]
              for f in mesh.faces()])
face_verts = [[V[F[i, j]] for j in range(3)] for i in range(len(F))]
face_z_mean = V[F].mean(axis=1)[:, 2]
zmax = max(V[:, 2].max(), 1e-6)

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
poly = Poly3DCollection(face_verts, alpha=0.7,
                        facecolor=cm.viridis(face_z_mean / zmax),
                        edgecolor="white", linewidth=0.08)
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

# Top-down: edges coloured & sized by q (force-density flow). Cables overlay.
edges_xy = [(V[key_to_idx[u], :2], V[key_to_idx[v], :2]) for u, v in edges_list]
qs = np.array([edge_q[tuple(sorted([u, v]))] for u, v in edges_list])
q_norm = matplotlib.colors.LogNorm(vmin=max(qs.min(), 0.01), vmax=qs.max())
order = np.argsort(qs)
for ei in order:
    p0, p1 = edges_xy[ei]
    qq = qs[ei]
    lw = 0.2 + 4 * (np.log10(qq + 0.01) - np.log10(0.01)) / (
            np.log10(qs.max()) - np.log10(0.01))
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]],
             color=cm.plasma(q_norm(qq)), linewidth=lw, alpha=0.7,
             solid_capstyle='round')

R_max = r_xy_all.max() * 1.05

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

ax2.set_title("Top-down — edge colour/thickness ∝ q (FDM force flow), "
              "red = 8 cables along high-q paths from apex",
              color="white", fontsize=9)
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
