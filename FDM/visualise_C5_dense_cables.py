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


# ── Detect 8 ridge angles directly from this FDM result ──────────────────────
from scipy.interpolate import LinearNDInterpolator
from scipy.signal import find_peaks
interp = LinearNDInterpolator(V[:, :2], V[:, 2])

N_TH = 1440
th_grid = np.linspace(0, 360, N_TH, endpoint=False)
radii_sample = np.linspace(0.20 * r_xy.max(), 0.92 * r_xy.max(), 19)
ridge_avg = np.zeros(N_TH)
for r in radii_sample:
    z_ring = np.array([interp(r * np.cos(np.radians(t)), r * np.sin(np.radians(t)))
                       for t in th_grid])
    if np.any(~np.isfinite(z_ring)):
        z_ring = np.where(np.isfinite(z_ring), z_ring, np.nanmean(z_ring))
    z_centred = z_ring - z_ring.mean()
    if z_centred.std() > 1e-6:
        ridge_avg += z_centred / z_centred.std()
ridge_avg /= len(radii_sample)

# Smooth the ridge profile
rs = ridge_avg.copy()
for _ in range(5):
    rs = (np.roll(rs, -1) + 2*rs + np.roll(rs, 1)) / 4

peaks, props = find_peaks(rs, distance=N_TH // 12, prominence=0.001)
order = np.argsort(props["prominences"])[::-1][:8]
SPOKE_ANGLES = sorted(float(th_grid[p]) for p in peaks[order])
print(f"Detected ridge angles from FDM result:")
for i, a in enumerate(SPOKE_ANGLES):
    print(f"  spoke {i}: θ = {a:6.2f}°")


# ── Dijkstra cable extraction along radial spoke lines ───────────────────────
# For each spoke at angle θ, build a graph whose edge weight = midpoint
# perpendicular distance to the radial line (apex → boundary @ θ), plus a tiny
# Euclidean term to break ties. Dijkstra from apex to the boundary vertex
# nearest θ gives the cleanest mesh-edge path that hugs the spoke.
import heapq

def angle_diff_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)

# All mesh edges (both directions) for adjacency
edges_set = set()
for k1 in verts_keys:
    for k2 in adj[k1]:
        if k1 < k2:
            edges_set.add((k1, k2))
        else:
            edges_set.add((k2, k1))

apex_xy = V[apex_idx, :2]

def dijkstra_spoke(theta_deg):
    # Spoke direction unit vector and its 90° normal
    th = np.radians(theta_deg)
    u  = np.array([np.cos(th), np.sin(th)])
    n  = np.array([-np.sin(th), np.cos(th)])

    # Pick boundary target: vertex with smallest angular deviation from θ that
    # also has a positive projection along u (i.e. on the correct half-plane).
    proj_u = (V[:, :2] - apex_xy) @ u
    proj_n = (V[:, :2] - apex_xy) @ n
    best_t, best_score = None, np.inf
    for k in bdry_keys:
        i = key_to_idx[k]
        if proj_u[i] <= 0:
            continue
        # angular deviation from spoke direction
        bv = V[i, :2] - apex_xy
        bth = np.degrees(np.arctan2(bv[1], bv[0]))
        s = angle_diff_deg(bth, theta_deg)
        if s < best_score:
            best_t, best_score = k, s
    target = best_t

    # Edge weight: midpoint |perpendicular distance from spoke line|
    # plus tiny eucl-length term to break ties for vertices already on the line.
    dist = {k: np.inf for k in verts_keys}
    prev = {k: None for k in verts_keys}
    dist[apex_key] = 0.0
    pq = [(0.0, apex_key)]
    while pq:
        d, u_k = heapq.heappop(pq)
        if d > dist[u_k]: continue
        if u_k == target: break
        u_idx = key_to_idx[u_k]
        for v_k in adj[u_k]:
            v_idx = key_to_idx[v_k]
            mid_xy = 0.5 * (V[u_idx, :2] + V[v_idx, :2])
            perp   = abs((mid_xy - apex_xy) @ n)
            eucl   = np.linalg.norm(V[v_idx, :2] - V[u_idx, :2])
            w = perp + 0.001 * eucl              # heavy bias toward the radial line
            nd = d + w
            if nd < dist[v_k]:
                dist[v_k] = nd
                prev[v_k] = u_k
                heapq.heappush(pq, (nd, v_k))

    # Reconstruct
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))


cables = {}
print("\nExtracting cables along detected ridges:")
for k, theta in enumerate(SPOKE_ANGLES):
    path = dijkstra_spoke(theta)
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

# Top-down: face elevation (viridis). Brighter = higher z.
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
patches = [MplPoly(V[F[i], :2]) for i in range(len(F))]
pc = PatchCollection(patches, alpha=0.85, edgecolor="white", linewidths=0.1)
pc.set_facecolor(cm.viridis(face_z_mean / zmax))
ax2.add_collection(pc)

# Dashed guide lines at the detected ridge angles
R_max = r_xy_all.max() * 1.05
for sa_deg in SPOKE_ANGLES:
    sa = np.radians(sa_deg)
    ax2.plot([0, R_max * np.cos(sa)], [0, R_max * np.sin(sa)],
             color="white", linestyle="--", linewidth=0.6, alpha=0.4)

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

ax2.set_title("Top-down — viridis = z elevation, red = 8 cables, "
              "white-- = ridge angles detected from FDM result",
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
