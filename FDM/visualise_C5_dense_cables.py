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

N_CABLES = 8   # D8 symmetry → 8 spokes


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


# ── Dijkstra on COMPAS mesh: weight = (1/q²) · (1 + λ·sin²θ_radial) ─────────
# OBJ mesh is disconnected; COMPAS is fully connected.
# Tilt penalty: sin² of angle between edge and outward radial at current vertex.
# No target angle — radial direction computed locally from apex at each step.
# Multiplicative form: q dominates on true cable edges; tilt only breaks ties.
LAMBDA_TILT = 4.0

adj_c_nb  = {k: list(cmesh.vertex_neighbors(k)) for k in ck_list}
r_xy_c    = np.linalg.norm(V_c[:, :2], axis=1)
apex_c_i  = int(np.argmin(r_xy_c))
apex_c_xy = V_c[apex_c_i, :2]

def dijkstra_compas(src_key):
    dist = {k: np.inf for k in ck_list}
    prev = {k: None  for k in ck_list}
    dist[src_key] = 0.0
    pq = [(0.0, src_key)]
    while pq:
        d, uk = heapq.heappop(pq)
        if d > dist[uk]: continue
        iu   = ck_to_i[uk]
        u_xy = V_c[iu, :2]
        r_u  = float(np.linalg.norm(u_xy - apex_c_xy))
        for vk in adj_c_nb[uk]:
            iv    = ck_to_i[vk]
            qe    = edge_q_c.get(tuple(sorted([uk, vk])), 0.1)
            e_xy  = V_c[iv, :2] - u_xy
            e_len = float(np.linalg.norm(e_xy))
            if r_u > 1e-6 and e_len > 1e-9:
                radial = (u_xy - apex_c_xy) / r_u
                cos_r  = float(np.dot(e_xy / e_len, radial))
                sin2_r = max(0.0, 1.0 - cos_r ** 2)   # sin²(tilt from radial)
            else:
                sin2_r = 0.0
            w  = (1.0 / max(qe, 1e-6) ** 2) * (1.0 + LAMBDA_TILT * sin2_r)
            nd = d + w
            if nd < dist[vk]:
                dist[vk] = nd
                prev[vk] = uk
                heapq.heappush(pq, (nd, vk))
    return dist, prev

# ── Cable-node detection ──────────────────────────────────────────────────────
# A cable node is a vertex where high-q edges arrive from ≥2 distinct directions.
# These are the junctions where cables meet: centroid, hoop joints, boundary anchors.
CABLE_NODE_Q_THR   = float(np.percentile(list(edge_q_c.values()), 65))
CABLE_NODE_ANG_THR = 30.0   # degrees

def is_cable_node(vk):
    angles = []
    for nk in adj_c_nb[vk]:
        qe = edge_q_c.get(tuple(sorted([vk, nk])), 0.0)
        if qe < CABLE_NODE_Q_THR: continue
        iv = ck_to_i[nk]; iu = ck_to_i[vk]
        a  = float(np.degrees(np.arctan2(V_c[iv,1]-V_c[iu,1],
                                          V_c[iv,0]-V_c[iu,0])) % 360)
        angles.append(a)
    for i in range(len(angles)):
        for j in range(i+1, len(angles)):
            diff = abs(((angles[i]-angles[j]+180) % 360) - 180)
            if diff > CABLE_NODE_ANG_THR:
                return True
    return False

cable_nodes = {k for k in ck_list if is_cable_node(k)}
print(f"Cable nodes detected: {len(cable_nodes)}")

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


# ── Auto-detect boundary anchors (cable nodes on boundary, one per sector) ────
# A good boundary anchor is a boundary vertex that is also a cable node (high-q
# edges from ≥2 directions), one picked per 45° sector.
bdry_cable_nodes = [(k, r_xy_c[ck_to_i[k]],
                     float(np.degrees(np.arctan2(V_c[ck_to_i[k],1],
                                                  V_c[ck_to_i[k],0])) % 360))
                    for k in cable_nodes if k in bdry_c]
print(f"Boundary cable nodes: {len(bdry_cable_nodes)}")

# One per 45° sector: pick by highest max-q of incident edges
sector_best = {}   # sector → (key, max_q)
for ck, r, th in bdry_cable_nodes:
    sec = int(th // 45)
    mq  = max(edge_q_c.get(tuple(sorted([ck, nk])), 0.0) for nk in adj_c_nb[ck])
    if sec not in sector_best or mq > sector_best[sec][1]:
        sector_best[sec] = (ck, mq)

boundary_anchors = [sector_best[s][0] for s in sorted(sector_best)]
print(f"Selected {len(boundary_anchors)} boundary anchors (sectors: {sorted(sector_best)})")

# ── Centroid: COMPAS vertex nearest the geometric apex ───────────────────────
src_key = ck_list[apex_c_i]
print(f"Centroid: COMPAS key {src_key}  r={r_xy_c[apex_c_i]:.4f}")

# ── Hoop ring (OBJ) ─────────────────────────────────────────────────────────
hoop_closed = len(hoop_path) > 1 and hoop_path[0] == hoop_path[-1]
hoop_ring   = hoop_path[:-1] if hoop_closed else list(hoop_path)
n_ring      = len(hoop_ring)

# ── Spoke paths: centroid → boundary anchor, split at first hoop cable node ──
cables = {}
spoke_ring_pos = []

print(f"\n{'k':>2}  {'θ°':>7}  inner  outer  hoop_junction")
_, prev_from_centroid = dijkstra_compas(src_key)

for k, anchor_key in enumerate(boundary_anchors):
    theta = float(np.degrees(np.arctan2(V_c[ck_to_i[anchor_key],1],
                                         V_c[ck_to_i[anchor_key],0])) % 360)
    # Full path: centroid → boundary anchor
    full_path_c = reconstruct_compas(anchor_key, prev_from_centroid)

    # Split at first cable node beyond apex that also lies on/near hoop ring
    # (= hoop junction). Walk from centroid end; first cable node with r ≥ r_hoop_mid.
    r_hoop_mid = 0.5 * (r_lo + r_hi)
    split_i = 1
    for i, ck in enumerate(full_path_c[1:-1], start=1):
        ri = r_xy_c[ck_to_i[ck]]
        if ri >= r_hoop_mid * 0.85 and is_cable_node(ck):
            split_i = i
            break

    path_inner_c = full_path_c[:split_i + 1]
    path_outer_c = full_path_c[split_i:]

    path_inner = compas_path_to_obj(path_inner_c)
    path_outer = compas_path_to_obj(path_outer_c)

    hoop_v_obj = path_inner[-1]
    # Position of hoop junction on the OBJ hoop ring
    dists = np.sum((V_obj[hoop_ring,:2] - V_obj[hoop_v_obj,:2])**2, axis=1)
    ring_pos = int(np.argmin(dists))
    spoke_ring_pos.append(ring_pos)

    print(f"{k:>2}  {theta:7.2f}  {len(path_inner):5d}  {len(path_outer):5d}"
          f"  OBJ={hoop_v_obj} ring[{ring_pos}]")

    cables[f"Si{k}_{theta:.1f}"] = path_inner
    cables[f"So{k}_{theta:.1f}"] = path_outer

# ── Hoop arcs: split ring at the 8 junction positions ────────────────────────
order     = np.argsort(spoke_ring_pos)
positions = [spoke_ring_pos[i] for i in order]
thetas_arc = [float(np.degrees(np.arctan2(V_c[ck_to_i[boundary_anchors[i]],1],
                                            V_c[ck_to_i[boundary_anchors[i]],0])) % 360)
              for i in order]

print(f"\nHoop arcs:")
for i in range(N_CABLES):
    p0 = positions[i]
    p1 = positions[(i + 1) % N_CABLES]
    arc = ([hoop_ring[j] for j in range(p0, p1 + 1)] if p1 > p0
           else [hoop_ring[j] for j in range(p0, n_ring)] +
                [hoop_ring[j] for j in range(0, p1 + 1)])
    print(f"  arc {i}: ring[{p0}..{p1}] → {len(arc)} verts")
    cables[f"Ha{i}_{thetas_arc[i]:.1f}"] = arc

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"\nSaved {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
face_verts  = [[V_obj[F_obj[i,j]] for j in range(3)] for i in range(len(F_obj))]
face_z_mean = V_obj[F_obj].mean(axis=1)[:, 2]
zmax        = float(V_obj[:, 2].max())
xr          = float(np.abs(V_obj[:, :2]).max()) * 1.08

# ── Identify 16 mesh regions (connected components of the OBJ) ───────────────
from collections import deque
_visited, _comps = set(), []
for _s in range(N_obj):
    if _s in _visited: continue
    _comp = set([_s]); _q = deque([_s])
    while _q:
        _u = _q.popleft()
        for _v in adj_obj[_u]:
            if _v not in _comp: _comp.add(_v); _q.append(_v)
    _visited |= _comp; _comps.append(sorted(_comp))
_comps.sort(key=lambda c: float(np.linalg.norm(V_obj[c, :2], axis=1).mean()))

# Assign each component: wedge 0-7 (which 45° sector) + inner/outer
region_info = []
for comp in _comps:
    xy    = V_obj[comp, :2]
    r_m   = float(np.linalg.norm(xy, axis=1).mean())
    th_m  = float(np.degrees(np.arctan2(xy[:,1].mean(), xy[:,0].mean())) % 360)
    wedge = int(th_m // 45)
    rtype = "inner" if r_m < 0.318 else "outer"
    region_info.append({"comp": comp, "wedge": wedge, "type": rtype, "r": r_m})

# 8 distinct hues (one per wedge); inner = light, outer = saturated
_hues = [i/8 for i in range(8)]
import colorsys
def wedge_color(wedge, rtype):
    h = _hues[wedge]
    s = 0.55 if rtype == "inner" else 0.95
    v = 0.80 if rtype == "inner" else 0.65
    return colorsys.hsv_to_rgb(h, s, v)

region_colors = [wedge_color(r["wedge"], r["type"]) for r in region_info]

# Face → component mapping for 3D colouring
face_comp = np.full(len(F_obj), -1, dtype=int)
vert_comp  = np.full(N_obj, -1, dtype=int)
for ci, ri in enumerate(region_info):
    for v in ri["comp"]: vert_comp[v] = ci
for fi, tri in enumerate(F_obj):
    c = vert_comp[tri[0]]
    if vert_comp[tri[1]] == c and vert_comp[tri[2]] == c:
        face_comp[fi] = c

BG = "#0d0d1a"
fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor(BG)

def cable_style(name):
    if name.startswith("Ha"): return "white",   3.2   # hoop arcs
    if name.startswith("Si"): return "#ffdd00", 2.8   # inner spokes
    return                           "#ffdd00", 2.8   # outer spokes

# ── Layout: top-down (large) left, two 3D angles right ───────────────────────
gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.08, wspace=0.05)
ax2 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1], projection="3d")
ax3 = fig.add_subplot(gs[1, 1], projection="3d")

# ── Top-down — regions coloured ──────────────────────────────────────────────
ax2.set_facecolor(BG); ax2.set_aspect("equal")

# filled triangles per region
for fi, tri in enumerate(F_obj):
    ci = face_comp[fi]
    if ci < 0: continue
    poly2d = plt.Polygon(V_obj[tri, :2], facecolor=region_colors[ci],
                         edgecolor=region_colors[ci], linewidth=0.1, alpha=0.82)
    ax2.add_patch(poly2d)

# cables
for name, path in cables.items():
    pts = np.array([V_obj[v] for v in path])
    col, lw = cable_style(name)
    ax2.plot(pts[:,0], pts[:,1], color=col, linewidth=lw,
             solid_capstyle="round", zorder=5)

# centroid + junction markers + vertex labels (derived from auto-detected paths)
centroid_obj = compas_path_to_obj([src_key])[0]
ax2.scatter(*V_obj[centroid_obj, :2], color="white", s=100, zorder=10, marker="*")
for name, path in cables.items():
    if name.startswith("Si"):   # mark start (hoop junction) and end
        hx, hy = V_obj[path[-1], :2]
        ax2.scatter(hx, hy, color="white", s=40, zorder=9)
        ax2.annotate(str(path[-1]), (hx, hy), xytext=(hx*0.80, hy*0.80),
                     color="white", fontsize=6.5, ha="center", va="center", zorder=11)
    if name.startswith("So"):
        bx, by = V_obj[path[-1], :2]
        ax2.scatter(bx, by, color="white", s=40, zorder=9)
        ax2.annotate(str(path[-1]), (bx, by), xytext=(bx*1.15, by*1.15),
                     color="white", fontsize=6.5, ha="center", va="center",
                     fontweight="bold", zorder=11)

ax2.set_title("16 regions coloured by wedge  (light=inner · saturated=outer)",
              color="white", fontsize=10, pad=5)
ax2.tick_params(colors="#445566", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#223344")
ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)
ax2.set_xlabel("x", color="#445566", fontsize=8)
ax2.set_ylabel("y", color="#445566", fontsize=8)

# ── 3D oblique + 3D top ───────────────────────────────────────────────────────
# Per-face colour list for 3D
face_colors_3d = [region_colors[face_comp[fi]] if face_comp[fi] >= 0 else (0.1,0.1,0.15)
                  for fi in range(len(F_obj))]

for ax, elev, azim in [(ax1, 28, -55), (ax3, 88, -90)]:
    ax.set_facecolor(BG)
    poly3 = Poly3DCollection(face_verts, alpha=0.88,
                             facecolor=face_colors_3d,
                             edgecolor=(0,0,0,0.15), linewidth=0.05)
    ax.add_collection3d(poly3)
    for name, path in cables.items():
        pts = np.array([V_obj[v] for v in path])
        col, lw = cable_style(name)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=lw, alpha=1.0)
    ax.scatter(*V_obj[centroid_obj], color="white", s=40, zorder=10)
    ax.tick_params(colors="#334455", labelsize=5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#1a2a3a")
    ax.set_xlim(-xr,xr); ax.set_ylim(-xr,xr); ax.set_zlim(0, zmax*1.1)
    ax.view_init(elev=elev, azim=azim)

ax1.set_title("3D oblique", color="white", fontsize=9, pad=3)
ax3.set_title("3D top",     color="white", fontsize=9, pad=3)

fig.suptitle("C5  ·  8 inner spokes + 8 outer spokes + 8 hoop arcs = 24 sections",
             color="white", fontsize=13, y=1.01)
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
