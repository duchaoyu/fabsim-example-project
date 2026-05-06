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

# ── User-specified cable endpoints (0-based OBJ indices) ─────────────────────
SRC_OBJ = 1091
DST_OBJ = [949, 1032, 1193, 1218, 1212, 1128, 973, 942]

# Radii where each cable is split into 3 sections
R_SPLIT1 = 0.30   # inner hoop radius (coincides with lime ring)
R_SPLIT2 = 0.45   # outer split (midway between hoop and boundary)


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


# ── Dijkstra on COMPAS mesh (OBJ is disconnected; COMPAS is fully connected) ──
# Weight = 1/q^4 + λ*(1 - cos θ) where θ is the angle between edge and the
# outward radial direction from the apex.
LAMBDA_RADIAL = 3.0
adj_c_nb = {k: list(cmesh.vertex_neighbors(k)) for k in ck_list}
r_xy_c   = np.linalg.norm(V_c[:, :2], axis=1)
apex_c_i = int(np.argmin(r_xy_c))
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
            qe     = edge_q_c.get(tuple(sorted([uk, vk])), 0.1)
            q_cost = 1.0 / max(qe, 1e-6) ** 4
            iv     = ck_to_i[vk]
            e_vec  = V_c[iv, :2] - u_xy
            e_len  = float(np.linalg.norm(e_vec))
            if r_u > 1e-6 and e_len > 1e-9:
                radial_dir = (u_xy - apex_c_xy) / r_u
                cos_r = float(np.dot(e_vec / e_len, radial_dir))
            else:
                cos_r = 0.0
            w  = q_cost + LAMBDA_RADIAL * (1.0 - cos_r)
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


# ── Find 8 cable paths via COMPAS-mesh Dijkstra, then map to OBJ ─────────────
# The OBJ is disconnected (16 components), so we run Dijkstra on the
# connected COMPAS form-found mesh and map the resulting paths to OBJ vertices.
src_key  = obj_to_compas_key(SRC_OBJ)
dst_keys = [obj_to_compas_key(d) for d in DST_OBJ]
print(f"\nMapped OBJ-{SRC_OBJ} → COMPAS key {src_key}  r={r_xy_c[ck_to_i[src_key]]:.4f}")
print(f"Finding COMPAS paths to {len(dst_keys)} destinations...")
dist_src, prev_src = dijkstra_compas(src_key)

cables = {}
cables["H0_inner_hoop"] = hoop_path

print(f"\n{'cable':>6}  {'θ°':>7}  {'len':>4}  {'qmin':>6}  sec1  sec2  sec3")
for k, (dst_obj, dst_key) in enumerate(zip(DST_OBJ, dst_keys)):
    path_c = reconstruct_compas(dst_key, prev_src)
    path   = compas_path_to_obj(path_c)
    r_path = r_xy_obj[path]
    theta  = float(np.degrees(np.arctan2(V_obj[dst_obj, 1], V_obj[dst_obj, 0])) % 360)

    s1 = int(np.argmin(np.abs(r_path - R_SPLIT1)))
    s2 = s1 + int(np.argmin(np.abs(r_path[s1:] - R_SPLIT2)))
    s1 = max(s1, 1);  s2 = max(s2, s1 + 1);  s2 = min(s2, len(path) - 2)

    sec1 = path[:s1 + 1]
    sec2 = path[s1:s2 + 1]
    sec3 = path[s2:]

    qs = [edge_q_obj.get(tuple(sorted([path[i], path[i+1]])), 0.1) for i in range(len(path)-1)]
    qmin_str = f"{min(qs):.3f}" if qs else "N/A  "
    print(f"  {k:>4d}  {theta:7.2f}  {len(path):4d}  {qmin_str:>6}  "
          f"{len(sec1):4d}  {len(sec2):4d}  {len(sec3):4d}")

    cables[f"Si{k}_{theta:.1f}"] = sec1   # innermost (orange)
    cables[f"Sm{k}_{theta:.1f}"] = sec2   # middle    (gold)
    cables[f"So{k}_{theta:.1f}"] = sec3   # outer     (red)

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"\nSaved {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
face_verts  = [[V_obj[F_obj[i,j]] for j in range(3)] for i in range(len(F_obj))]
face_z_mean = V_obj[F_obj].mean(axis=1)[:, 2]
zmax        = float(V_obj[:, 2].max())

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

# ── 3D view ──────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
poly = Poly3DCollection(face_verts, alpha=0.65,
                        facecolor=cm.viridis(face_z_mean / max(zmax, 1e-6)),
                        edgecolor="white", linewidth=0.08)
ax1.add_collection3d(poly)

def cable_style(name):
    if name.startswith("H"):  return "lime",   "yellow",  2.5, 10
    if name.startswith("Si"): return "orange", "gold",    2.2, 12
    if name.startswith("Sm"): return "gold",   "white",   2.2, 12
    return                           "red",    "tomato",  2.2, 12

for name, path in cables.items():
    pts = np.array([V_obj[v] for v in path])
    col, mcol, lw, ms = cable_style(name)
    ax1.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=lw, alpha=0.95)
    ax1.scatter(pts[:,0], pts[:,1], pts[:,2], color=mcol, s=ms,
                edgecolors=col, linewidths=0.4, zorder=5)

ax1.set_title(f"C5 FDM — {N_obj}v/{len(F_obj)}f\n"
              f"orange=inner·gold=mid·red=outer·lime=hoop (24+1)",
              color="white", fontsize=10)
for lab in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
    lab.label.set_color("white"); lab.label.set_fontsize(8)
ax1.tick_params(colors="white", labelsize=7)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
xr = float(np.abs(V_obj[:,:2]).max()) * 1.05
ax1.set_xlim(-xr,xr); ax1.set_ylim(-xr,xr); ax1.set_zlim(0, zmax*1.2)
ax1.view_init(elev=25, azim=-50)

# ── Top-down with FDM q flow ──────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

qs_edges = np.array([edge_q_obj.get(tuple(sorted([u_obj,v_obj])), 0.1)
                     for u_obj in range(N_obj) for v_obj in adj_obj[u_obj] if v_obj > u_obj])
q_norm = matplotlib.colors.LogNorm(vmin=max(qs_all.min(), 0.01), vmax=qs_all.max())
for u_obj in range(N_obj):
    for v_obj in adj_obj[u_obj]:
        if v_obj <= u_obj: continue
        qe = edge_q_obj.get(tuple(sorted([u_obj,v_obj])), 0.1)
        lw = 0.15 + 3.5*(np.log10(qe+0.01)-np.log10(0.01))/(np.log10(qs_all.max())-np.log10(0.01))
        ax2.plot([V_obj[u_obj,0],V_obj[v_obj,0]], [V_obj[u_obj,1],V_obj[v_obj,1]],
                 color=cm.plasma(q_norm(qe)), linewidth=max(lw,0.1), alpha=0.6,
                 solid_capstyle='round')

for name, path in cables.items():
    pts = np.array([V_obj[v] for v in path])
    col, mcol, lw, ms = cable_style(name)
    ax2.plot(pts[:,0], pts[:,1], color=col, linewidth=lw+0.3, alpha=0.95, zorder=4)
    ax2.scatter(pts[:,0], pts[:,1], color=mcol, s=ms, edgecolors=col,
                linewidths=0.5, zorder=5)
    if name.startswith("So"):
        end = pts[-1]
        th  = float(name.split("_")[1])
        ax2.annotate(f"{th:.0f}°", xy=(end[0], end[1]),
                     xytext=(end[0]*1.12, end[1]*1.12), color="white", fontsize=7,
                     ha="center", va="center")

ax2.set_title("Top-down — q flow (plasma) · orange=inner · gold=mid · red=outer · lime=hoop",
              color="white", fontsize=8)
for lab, txt in [(ax2.xaxis,"x"), (ax2.yaxis,"y")]:
    ax2.set_xlabel(txt, color="white", fontsize=8) if txt=="x" else ax2.set_ylabel(txt, color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.set_xlim(-xr,xr); ax2.set_ylim(-xr,xr)

fig.suptitle("C5 dense FDM — 8 cables × 3 sections = 24 (+ inner hoop)",
             color="white", fontsize=12, y=0.99)
plt.tight_layout(rect=[0,0.01,1,0.96])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
