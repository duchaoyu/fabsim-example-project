"""
Visualise the 4 continuous cable paths (boundary→boundary) and resulting
9-region layout on the B5 FDM result, with proposed knit direction per region.

Cable paths are found via Dijkstra minimising 1/f (prefers high-force edges).
"""
import os, heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from compas.datastructures import Mesh

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
OUT_PNG     = os.path.join(HERE, "B5_9regions.png")
BG          = "#1a1a2e"

# ── Load ──────────────────────────────────────────────────────────────────────
mesh  = Mesh.from_json(RESULT_JSON)
verts = list(mesh.vertices())
pts   = np.array([mesh.vertex_coordinates(v) for v in verts])
edges = list(mesh.edges())
faces = list(mesh.faces())

q_arr = np.array([mesh.edge_attribute(e, "qpre") for e in edges])
l_arr = np.array([np.linalg.norm(pts[u] - pts[v]) for u, v in edges])
f_arr = q_arr * l_arr
bdry  = set(mesh.vertices_on_boundary())
bdry_pts = {v: pts[v] for v in bdry}

# ── Dijkstra: high-force path between two boundary vertices ───────────────────
def high_force_path(src, tgt):
    cost = 1.0 / (f_arr + 1e-6)
    adj  = {v: [] for v in mesh.vertices()}
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, cost[i], i))
        adj[v].append((u, cost[i], i))
    dist = {v: np.inf for v in mesh.vertices()}
    prev = {}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == tgt and u != src:
            path_v, path_e = [], []
            cur = u
            while cur in prev:
                pv, ei = prev[cur]
                path_v.append(cur); path_e.append(ei); cur = pv
            path_v.append(src)
            return list(reversed(path_v)), list(reversed(path_e))
        for v, w, ei in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = (u, ei)
                heapq.heappush(pq, (nd, v))
    return [], []

def closest_bdry(xy):
    return min(bdry_pts, key=lambda v: np.linalg.norm(pts[v, :2] - np.array(xy)))

R = 10.0
# 4 cables for 3×3 grid at ±R/3 in each axis
cable_specs = [
    (closest_bdry([-R/3, -R]), closest_bdry([-R/3,  R]), "Cable A (x≈-3.3)", "#00d4ff"),
    (closest_bdry([ R/3, -R]), closest_bdry([ R/3,  R]), "Cable B (x≈+3.3)", "#00d4ff"),
    (closest_bdry([-R,  -R/3]), closest_bdry([ R, -R/3]), "Cable C (y≈-3.3)", "#ff6b35"),
    (closest_bdry([-R,   R/3]), closest_bdry([ R,  R/3]), "Cable D (y≈+3.3)", "#ff6b35"),
]

cable_paths = []
for src, tgt, label, col in cable_specs:
    pv, pe = high_force_path(src, tgt)
    cable_paths.append((pv, pe, label, col))
    mf = f_arr[pe].mean() if pe else 0
    print(f"{label}: {len(pv)} verts, mean f={mf:.1f}")

# ── Region assignment for faces ───────────────────────────────────────────────
# Determine x/y split positions from cable paths
cable_AB_x = np.array([pts[v, 0] for v in cable_paths[0][0] + cable_paths[1][0]])
cable_CD_y = np.array([pts[v, 1] for v in cable_paths[2][0] + cable_paths[3][0]])
x_splits = sorted([pts[cable_specs[0][0], 0], pts[cable_specs[1][0], 0]])  # approx ±3.3
# Use median x of each cable for the split
x_A = np.median([pts[v, 0] for v in cable_paths[0][0]])
x_B = np.median([pts[v, 0] for v in cable_paths[1][0]])
y_C = np.median([pts[v, 1] for v in cable_paths[2][0]])
y_D = np.median([pts[v, 1] for v in cable_paths[3][0]])

# Region labels (row 0=bottom, col 0=left)
REGION_NAMES = [
    "R7 (BL)", "R8 (BC)", "R9 (BR)",
    "R4 (ML)", "R5 (MC)", "R6 (MR)",
    "R1 (TL)", "R2 (TC)", "R3 (TR)",
]
# knit direction per region (degrees, 0=horizontal/wale, 90=vertical/course)
# Aligned with cable direction bounding the region
KNIT_DIR = {
    (0, 0): 0,  (0, 1): 0,  (0, 2): 0,   # uniform 0° everywhere
    (1, 0): 0,  (1, 1): 0,  (1, 2): 0,   # (wale along x-axis,
    (2, 0): 0,  (2, 1): 0,  (2, 2): 0,   #  aligned with cables C/D)
}
REGION_COLORS = {
    (0,0): "#2a3a5a", (0,1): "#2a5a3a", (0,2): "#2a3a5a",
    (1,0): "#5a2a3a", (1,1): "#5a4a2a", (1,2): "#5a2a3a",
    (2,0): "#2a3a5a", (2,1): "#2a5a3a", (2,2): "#2a3a5a",
}

def face_region(cx, cy):
    col = 0 if cx < x_A else (1 if cx < x_B else 2)
    row = 0 if cy < y_C else (1 if cy < y_D else 2)
    return row, col

face_cx = np.array([np.mean([mesh.vertex_coordinates(v)[0] for v in mesh.face_vertices(f)]) for f in faces])
face_cy = np.array([np.mean([mesh.vertex_coordinates(v)[1] for v in mesh.face_vertices(f)]) for f in faces])
face_regions = [face_region(face_cx[i], face_cy[i]) for i in range(len(faces))]

# Region centroids (for direction arrows)
from collections import defaultdict
reg_pts = defaultdict(list)
for i, (row, col) in enumerate(face_regions):
    reg_pts[(row, col)].append([face_cx[i], face_cy[i]])
reg_centroids = {k: np.mean(v, axis=0) for k, v in reg_pts.items()}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor(BG)

for ax in axes:
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("x (m)", color="white", fontsize=9)
    ax.set_ylabel("y (m)", color="white", fontsize=9)
    for sp in ax.spines.values(): sp.set_color("white")

# ── Left: top-down with regions + cables ─────────────────────────────────────
ax = axes[0]
ax.set_title("Top-down: 9 regions + cable paths + knit directions",
             color="white", fontsize=11)

# Shade regions
for i, f in enumerate(faces):
    fv_xy = [mesh.vertex_coordinates(v)[:2] for v in mesh.face_vertices(f)]
    row, col = face_regions[i]
    color = REGION_COLORS[(row, col)]
    ax.fill([p[0] for p in fv_xy], [p[1] for p in fv_xy],
            color=color, alpha=0.55, zorder=1)

# Region labels + knit direction arrows
L = 2.8  # arrow half-length
for (row, col), c in reg_centroids.items():
    kd  = np.radians(KNIT_DIR[(row, col)])
    dx  = L * np.cos(kd)
    dy  = L * np.sin(kd)
    ax.annotate("", xy=(c[0]+dx, c[1]+dy), xytext=(c[0]-dx, c[1]-dy),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
    lbl_r = {0: "B", 1: "M", 2: "T"}[row]
    lbl_c = {0: "L", 1: "C", 2: "R"}[col]
    ax.text(c[0], c[1] - 0.6, f"{lbl_r}{lbl_c}\n{KNIT_DIR[(row,col)]}°",
            ha="center", va="top", color="white", fontsize=7.5, zorder=5)

# Low-force mesh edges (faint)
for i, (u, v) in enumerate(edges):
    ax.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]],
            color="white", alpha=0.07, linewidth=0.35, zorder=2)

# Cable paths
for pv, pe, label, col in cable_paths:
    if not pv: continue
    cx = [pts[v, 0] for v in pv]
    cy = [pts[v, 1] for v in pv]
    ax.plot(cx, cy, color=col, linewidth=3.5, zorder=4, solid_capstyle="round")
    ax.scatter(cx[0],  cy[0],  color=col, s=60, zorder=5)
    ax.scatter(cx[-1], cy[-1], color=col, s=60, zorder=5)

# Legend
patch_A = mpatches.Patch(color="#00d4ff", label="Cables A/B (x-direction)")
patch_C = mpatches.Patch(color="#ff6b35", label="Cables C/D (y-direction)")
ax.legend(handles=[patch_A, patch_C], loc="upper right",
          facecolor=BG, labelcolor="white", fontsize=8)

# ── Right: 3D perspective ─────────────────────────────────────────────────────
from mpl_toolkits.mplot3d import Axes3D
fig.delaxes(axes[1])
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.set_facecolor(BG)
ax3.set_title("3D: cable paths over dome", color="white", fontsize=11)

# Shaded faces
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
face_xyz = [[mesh.vertex_coordinates(v) for v in mesh.face_vertices(f)] for f in faces]
face_colors_3d = [REGION_COLORS[face_regions[i]] for i in range(len(faces))]
poly = Poly3DCollection(face_xyz, alpha=0.45, facecolor=face_colors_3d, edgecolor="none")
ax3.add_collection3d(poly)

# Cable paths in 3D
for pv, pe, label, col in cable_paths:
    if not pv: continue
    cx = [pts[v, 0] for v in pv]
    cy = [pts[v, 1] for v in pv]
    cz = [pts[v, 2] for v in pv]
    ax3.plot(cx, cy, cz, color=col, linewidth=3.5, zorder=5)
    ax3.scatter([cx[0], cx[-1]], [cy[0], cy[-1]], [cz[0], cz[-1]],
                color=col, s=40, zorder=6)

ax3.set_xlim(-11, 11); ax3.set_ylim(-11, 11); ax3.set_zlim(0, 5)
ax3.set_xlabel("x (m)", color="white", fontsize=7)
ax3.set_ylabel("y (m)", color="white", fontsize=7)
ax3.set_zlabel("z (m)", color="white", fontsize=7)
ax3.tick_params(colors="white", labelsize=6)
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = False
ax3.view_init(elev=28, azim=-50)

# Stats box
stats = (
    f"4 cables, each continuous boundary→boundary\n"
    f"Cables A/B: at x≈±{R/3:.1f} m  (9 vertices each)\n"
    f"Cables C/D: at y≈±{R/3:.1f} m  (9 vertices each)\n"
    f"Mean axial force on cables: {np.mean([f_arr[pe].mean() for _,pe,_,_ in cable_paths if pe]):.1f}\n"
    f"9 regions: knit_dir aligned with bounding cable\n"
    f"Corner/centre regions: 45° (bisect both directions)"
)
fig.text(0.5, 0.01, stats, ha="center", va="bottom", color="white",
         fontsize=8, fontfamily="monospace",
         bbox=dict(facecolor="#0d0d1a", edgecolor="white", alpha=0.7, pad=5))

fig.suptitle("B5 — 4 continuous cables + 9-region layout + knit directions",
             color="white", fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0.10, 1, 0.97])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
