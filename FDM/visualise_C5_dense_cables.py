"""C5 cable visualiser — user-specified cables on OBJ mesh.

Cable endpoints are given as 0-based OBJ vertex indices (C5_remeshed.obj,
1309 vertices). Force-density q values come from the COMPAS form-found JSON
and are mapped to OBJ vertices via nearest-neighbour on (x,y).

Layout:
  - lime  : inner hoop ring (re-detected on OBJ mesh from FDM q)
  - orange: innermost cable section  (apex → r≈0.30)
  - gold  : middle cable section     (r≈0.30 → r≈0.45)
  - red   : outer cable section      (r≈0.45 → boundary)
  8 cables × 3 sections = 24 cable sections + 1 hoop
"""
import os, json
import numpy as np
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh

HERE     = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(HERE, "data", "C5", "C5_remeshed.obj")
RESULT   = os.path.join(HERE, "data", "C5", "mesh_out_C5_dense_latest.json")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_dense_cables.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "cable_paths_C5_dense.json")

# ── User-specified cable waypoints (0-based OBJ indices) ─────────────────────
# Each tuple: (centroid, hoop_vertex, boundary_vertex)
CABLES_DEF = [
    (1091,  995,  949),
    (1091, 1059, 1032),
    (1091, 1138, 1193),
    (1091, 1187, 1218),
    (1091, 1170, 1212),
    (1091, 1105, 1128),
    (1091, 1028,  973),
    (1091,  977,  942),
]
HOOP_VERTS = [c[1] for c in CABLES_DEF]   # the 8 spoke-hoop junction vertices


# ── Load OBJ mesh (0-based vertices) ─────────────────────────────────────────
def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t: continue
            if t[0] == "v":
                V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                F.append([int(s.split("/")[0]) - 1 for s in t[1:4]])
    return np.array(V), np.array(F, dtype=int)

V_obj, F_obj = load_obj(OBJ_PATH)
N_obj = len(V_obj)
print(f"OBJ: {N_obj} verts, {len(F_obj)} faces")

# Build adjacency on OBJ mesh
adj_obj = [set() for _ in range(N_obj)]
for tri in F_obj:
    for i, j in [(0,1),(1,2),(2,0)]:
        adj_obj[tri[i]].add(tri[j])
        adj_obj[tri[j]].add(tri[i])

r_xy_obj = np.linalg.norm(V_obj[:, :2], axis=1)
apex_obj  = int(np.argmin(r_xy_obj))
apex_xy   = V_obj[apex_obj, :2]
print(f"OBJ apex: idx={apex_obj}, r={r_xy_obj[apex_obj]:.5f}, z={V_obj[apex_obj,2]:.4f}")


# ── Load form-found COMPAS mesh → extract q values ───────────────────────────
cmesh      = Mesh.from_json(RESULT)
ck_list    = list(cmesh.vertices())
V_c        = np.array([cmesh.vertex_coordinates(k) for k in ck_list])
ck_to_i    = {k: i for i, k in enumerate(ck_list)}
bdry_c     = set(cmesh.vertices_on_boundary())
edges_c    = list(cmesh.edges())
edge_q_c   = {tuple(sorted([u, v])): float(cmesh.edge_attribute((u,v), "qpre"))
              for u, v in edges_c}
print(f"COMPAS mesh: {len(ck_list)} verts, q∈[{min(edge_q_c.values()):.3f},{max(edge_q_c.values()):.3f}]")

# Map each OBJ vertex to nearest COMPAS vertex by (x,y) distance
obj_to_compas = np.argmin(
    np.sum((V_c[:, :2][None, :, :] - V_obj[:, :2][:, None, :])**2, axis=2),
    axis=1)   # shape (N_obj,)

# Map q to OBJ edges via COMPAS nearest-neighbours
edge_q_obj = {}    # tuple(sorted(u_obj, v_obj)) → q
for u_obj in range(N_obj):
    cu = ck_list[obj_to_compas[u_obj]]
    for v_obj in adj_obj[u_obj]:
        if v_obj < u_obj: continue
        cv = ck_list[obj_to_compas[v_obj]]
        key = tuple(sorted([cu, cv]))
        q = edge_q_c.get(key, 0.1)    # default 0.1 if edge not found
        edge_q_obj[tuple(sorted([u_obj, v_obj]))] = q

qs_all = np.array(list(edge_q_obj.values()))
print(f"OBJ mapped q: [{qs_all.min():.3f},{qs_all.max():.3f}] mean={qs_all.mean():.3f}")


# ── Shortest-path Dijkstra on COMPAS mesh (Euclidean 3D edge length) ─────────
# OBJ mesh is disconnected; COMPAS is fully connected.
# Weight = 3D edge length → true geodesic shortest path.
adj_c_nb = {k: list(cmesh.vertex_neighbors(k)) for k in ck_list}
r_xy_c   = np.linalg.norm(V_c[:, :2], axis=1)

def dijkstra_compas(src_key):
    dist = {k: np.inf for k in ck_list}
    prev = {k: None  for k in ck_list}
    dist[src_key] = 0.0
    pq = [(0.0, src_key)]
    while pq:
        d, uk = heapq.heappop(pq)
        if d > dist[uk]: continue
        iu = ck_to_i[uk]
        for vk in adj_c_nb[uk]:
            iv = ck_to_i[vk]
            w  = float(np.linalg.norm(V_c[iv] - V_c[iu]))   # 3D edge length
            nd = d + w
            if nd < dist[vk]:
                dist[vk] = nd
                prev[vk] = uk
                heapq.heappush(pq, (nd, vk))
    return dist, prev

def reconstruct_compas(target_key, prev):
    path = []
    cur  = target_key
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

def obj_to_compas_key(obj_idx):
    """Nearest COMPAS vertex to a given OBJ vertex (3D distance)."""
    p    = V_obj[obj_idx]
    best = int(np.argmin(np.sum((V_c - p)**2, axis=1)))
    return ck_list[best]

def compas_path_to_obj(path_c):
    """Map a sequence of COMPAS keys to nearest OBJ vertex indices (by xy)."""
    return [int(np.argmin(np.sum((V_obj[:, :2] - V_c[ck_to_i[ck], :2])**2, axis=1)))
            for ck in path_c]


# ── Inner hoop detection on COMPAS mesh (accurate q, then mapped to OBJ) ──────
# Detect the hoop on the original form-found COMPAS mesh where q values are
# exact, then map the resulting vertex sequence to OBJ indices via (x,y) NN.
print("\nDetecting inner hoop on COMPAS mesh...")
R_max_c  = float(np.linalg.norm(V_c[:, :2], axis=1).max())
R_BDRY_C = 0.85 * R_max_c
qs_c_arr = np.array(list(edge_q_c.values()))
q_thr_c  = float(np.percentile(qs_c_arr, 70))

edge_circ_c = {}
for (u, v), qe in edge_q_c.items():
    if u in bdry_c or v in bdry_c: continue
    iu, iv = ck_to_i[u], ck_to_i[v]
    pu, pv = V_c[iu, :2], V_c[iv, :2]
    ru, rv = np.linalg.norm(pu), np.linalg.norm(pv)
    r_mean = 0.5 * (ru + rv)
    if r_mean > R_BDRY_C: continue
    if qe < q_thr_c: continue
    L  = np.linalg.norm(pv - pu) + 1e-9
    dr = abs(rv - ru)
    if (1.0 - dr / L) > 0.6:
        edge_circ_c[(u, v)] = (qe, r_mean)

print(f"  circumferential COMPAS edges (q>{q_thr_c:.3f}): {len(edge_circ_c)}")

bin_w = 0.04
bins  = np.arange(0, R_BDRY_C + bin_w, bin_w)
bin_count = np.zeros(len(bins) - 1)
bin_qsum  = np.zeros(len(bins) - 1)
for (qe, r) in edge_circ_c.values():
    b = int(r // bin_w)
    if 0 <= b < len(bin_count):
        bin_count[b] += 1
        bin_qsum[b]  += qe

candidates = [b for b in range(len(bin_count)) if bin_count[b] >= 6]
hoop_path = []
if candidates:
    peak = max(candidates, key=lambda b: bin_qsum[b])
    r_lo, r_hi = bins[peak], bins[peak + 1]
    if peak + 1 < len(bin_count) and bin_count[peak + 1] >= 4:
        r_hi = bins[peak + 2]
    if peak > 0 and bin_count[peak - 1] >= 4:
        r_lo = bins[peak - 1]
    ring_edges_c = [(u, v) for (u, v), (qe, r) in edge_circ_c.items() if r_lo <= r < r_hi]
    ring_adj_c = {}
    for u, v in ring_edges_c:
        ring_adj_c.setdefault(u, []).append(v)
        ring_adj_c.setdefault(v, []).append(u)
    start_c = min(ring_adj_c, key=lambda k: np.arctan2(V_c[ck_to_i[k], 1], V_c[ck_to_i[k], 0]))
    path_c, prev_v, cur = [start_c], None, start_c
    while True:
        nbs = [n for n in ring_adj_c.get(cur, []) if n != prev_v and n not in path_c]
        if not nbs:
            if start_c in ring_adj_c.get(cur, []) and len(path_c) > 4:
                path_c.append(start_c)
            break
        cur_th = np.arctan2(V_c[ck_to_i[cur], 1], V_c[ck_to_i[cur], 0])
        nxt = min(nbs, key=lambda n: (np.arctan2(V_c[ck_to_i[n], 1], V_c[ck_to_i[n], 0]) - cur_th) % (2*np.pi))
        path_c.append(nxt)
        prev_v, cur = cur, nxt
        if cur == start_c or len(path_c) > 300: break
    # Map COMPAS keys → nearest OBJ vertex by (x,y)
    hoop_path = [int(np.argmin(np.sum((V_obj[:, :2] - V_c[ck_to_i[ck], :2])**2, axis=1)))
                 for ck in path_c]
    print(f"  hoop: {len(hoop_path)} verts at r∈[{r_lo:.3f},{r_hi:.3f}], closed={path_c[0]==path_c[-1]}")
else:
    print("  no hoop found")


# ── Build cables from exact user-specified waypoints ─────────────────────────
# Each cable: centroid --[inner]--> hoop_vertex --[outer]--> boundary_vertex
# Paths computed on COMPAS mesh (OBJ is disconnected), mapped back to OBJ.
cables = {}

# Hoop ring without closing duplicate (for arc extraction)
hoop_closed = len(hoop_path) > 1 and hoop_path[0] == hoop_path[-1]
hoop_ring   = hoop_path[:-1] if hoop_closed else list(hoop_path)
n_ring      = len(hoop_ring)

print(f"\n{'k':>2}  {'θ°':>7}  inner  outer")
spoke_ring_pos = []   # position of each hoop_vertex in hoop_ring

for k, (v_src, v_hoop, v_dst) in enumerate(CABLES_DEF):
    theta = float(np.degrees(np.arctan2(V_obj[v_dst, 1], V_obj[v_dst, 0])) % 360)

    # inner leg: centroid → hoop vertex
    src_key  = obj_to_compas_key(v_src)
    hoop_key = obj_to_compas_key(v_hoop)
    _, prev_inner = dijkstra_compas(src_key)
    path_inner_c  = reconstruct_compas(hoop_key, prev_inner)
    path_inner    = compas_path_to_obj(path_inner_c)
    # ensure it ends exactly at v_hoop
    if path_inner[-1] != v_hoop:
        path_inner.append(v_hoop)

    # outer leg: hoop vertex → boundary
    dst_key  = obj_to_compas_key(v_dst)
    _, prev_outer = dijkstra_compas(hoop_key)
    path_outer_c  = reconstruct_compas(dst_key, prev_outer)
    path_outer    = compas_path_to_obj(path_outer_c)
    # ensure it starts exactly at v_hoop
    if path_outer[0] != v_hoop:
        path_outer.insert(0, v_hoop)

    print(f"{k:>2}  {theta:7.2f}  {len(path_inner):5d}  {len(path_outer):5d}")

    cables[f"Si{k}_{theta:.1f}"] = path_inner   # orange
    cables[f"So{k}_{theta:.1f}"] = path_outer   # red

    # Find v_hoop's position in the hoop ring (for arc extraction)
    if v_hoop in hoop_ring:
        pos = hoop_ring.index(v_hoop)
    else:
        dists = np.sum((V_obj[hoop_ring, :2] - V_obj[v_hoop, :2])**2, axis=1)
        pos   = int(np.argmin(dists))
    spoke_ring_pos.append(pos)

# ── Hoop arcs: split ring at the 8 spoke connection points ───────────────────
order     = np.argsort(spoke_ring_pos)
positions = [spoke_ring_pos[i] for i in order]
thetas    = [float(np.degrees(np.arctan2(V_obj[CABLES_DEF[i][2], 1],
                                          V_obj[CABLES_DEF[i][2], 0])) % 360)
             for i in order]

print(f"\nHoop arcs:")
for i in range(8):
    p0 = positions[i]
    p1 = positions[(i + 1) % 8]
    if p1 > p0:
        arc = [hoop_ring[j] for j in range(p0, p1 + 1)]
    else:
        arc = ([hoop_ring[j] for j in range(p0, n_ring)] +
               [hoop_ring[j] for j in range(0, p1 + 1)])
    print(f"  arc {i}: ring[{p0}..{p1}] → {len(arc)} verts")
    cables[f"Ha{i}_{thetas[i]:.1f}"] = arc   # lime

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"\nSaved {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
face_verts  = [[V_obj[F_obj[i,j]] for j in range(3)] for i in range(len(F_obj))]
face_z_mean = V_obj[F_obj].mean(axis=1)[:, 2]
zmax        = float(V_obj[:, 2].max())
xr          = float(np.abs(V_obj[:, :2]).max()) * 1.08

BG = "#0d0d1a"
fig = plt.figure(figsize=(16, 8))
fig.patch.set_facecolor(BG)

# cable colour/width lookup
def cable_style(name):
    if name.startswith("Ha"): return "#00ff88", 3.0   # lime hoop arcs
    if name.startswith("Si"): return "#ff8c00", 2.8   # orange inner spokes
    return                           "#ff2244", 2.8   # red outer spokes

# ── 3D view ───────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor(BG)
poly = Poly3DCollection(face_verts, alpha=0.55,
                        facecolor=cm.Blues(face_z_mean / max(zmax, 1e-6) * 0.7 + 0.15),
                        edgecolor="#334466", linewidth=0.06)
ax1.add_collection3d(poly)

for name, path in cables.items():
    pts = np.array([V_obj[v] for v in path])
    col, lw = cable_style(name)
    ax1.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=lw,
             alpha=1.0, solid_capstyle="round")

# junction dots: hoop vertices (white) + boundary endpoints (cyan) + centroid (yellow)
centroid_pt = V_obj[CABLES_DEF[0][0]]
ax1.scatter(*centroid_pt, color="yellow", s=60, zorder=10)
for v_src, v_hoop, v_dst in CABLES_DEF:
    ax1.scatter(*V_obj[v_hoop], color="white",  s=30, zorder=10)
    ax1.scatter(*V_obj[v_dst],  color="#00cfff", s=30, zorder=10)

ax1.set_title("C5 — 3D  (orange=inner · red=outer · green=hoop)",
              color="white", fontsize=10, pad=6)
ax1.tick_params(colors="#556677", labelsize=6)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False; pane.set_edgecolor("#223344")
ax1.set_xlim(-xr, xr); ax1.set_ylim(-xr, xr); ax1.set_zlim(0, zmax * 1.15)
ax1.view_init(elev=30, azim=-55)

# ── Top-down — clean (mesh edges faint, cables bold) ─────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor(BG)
ax2.set_aspect("equal")

# faint mesh wireframe
for tri in F_obj:
    for i, j in [(0,1),(1,2),(2,0)]:
        u, v = tri[i], tri[j]
        ax2.plot([V_obj[u,0], V_obj[v,0]], [V_obj[u,1], V_obj[v,1]],
                 color="#1e2a3a", linewidth=0.25, alpha=0.8)

# cables
for name, path in cables.items():
    pts = np.array([V_obj[v] for v in path])
    col, lw = cable_style(name)
    ax2.plot(pts[:,0], pts[:,1], color=col, linewidth=lw,
             alpha=1.0, solid_capstyle="round", zorder=4)

# centroid marker
cx, cy = V_obj[CABLES_DEF[0][0], :2]
ax2.scatter(cx, cy, color="yellow", s=80, zorder=10, marker="*")

# hoop junction dots + boundary dots + angle labels
for v_src, v_hoop, v_dst in CABLES_DEF:
    ax2.scatter(*V_obj[v_hoop, :2], color="white",  s=40, zorder=8)
    ax2.scatter(*V_obj[v_dst,  :2], color="#00cfff", s=40, zorder=8)
    bx, by = V_obj[v_dst, :2]
    th = float(np.degrees(np.arctan2(by, bx)) % 360)
    ax2.annotate(f"{th:.0f}°", xy=(bx, by),
                 xytext=(bx * 1.14, by * 1.14),
                 color="#00cfff", fontsize=8, ha="center", va="center",
                 fontweight="bold")

ax2.set_title("Top-down  (orange=inner · red=outer · green=hoop · ●=junction · ★=centroid)",
              color="white", fontsize=9)
ax2.tick_params(colors="#556677", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#223344")
ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)
ax2.set_xlabel("x", color="#556677", fontsize=8)
ax2.set_ylabel("y", color="#556677", fontsize=8)

fig.suptitle("C5  —  8 inner spokes  +  8 outer spokes  +  8 hoop arcs  =  24 cable sections",
             color="white", fontsize=13, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
