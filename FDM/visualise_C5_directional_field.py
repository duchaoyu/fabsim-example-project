"""Directional (cross) field on the C5 dome guided by cable stiffeners.

For every mesh face, the two principal directions are:
  d1 : tangent projection of the nearest cable segment onto the face plane
  d2 : face_normal × d1  (perpendicular to cable, in the tangent plane)

The result is saved as a JSON of {face_index: [d1_xyz, d2_xyz]} and
visualised as a cross field overlaid on the coloured-region mesh.
"""
import os, json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque
import colorsys

HERE     = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(HERE, "data", "C5", "C5_remeshed.obj")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_directional_field.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "directional_field_C5.json")

# ── Correct cable waypoints (0-based OBJ indices) ─────────────────────────────
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

# ── Load OBJ ──────────────────────────────────────────────────────────────────
def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t: continue
            if t[0] == "v":  V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f": F.append([int(s.split("/")[0])-1 for s in t[1:4]])
    return np.array(V), np.array(F, dtype=int)

V, F = load_obj(OBJ_PATH)
print(f"OBJ: {len(V)} verts, {len(F)} faces")

# ── Build cable segments from waypoints via geodesic shortest path on COMPAS ──
# (Re-use the COMPAS mesh + Dijkstra from the main pipeline)
from compas.datastructures import Mesh
import heapq

RESULT = os.path.join(HERE, "data", "C5", "mesh_out_C5_dense_latest.json")
cmesh   = Mesh.from_json(RESULT)
ck_list = list(cmesh.vertices())
V_c     = np.array([cmesh.vertex_coordinates(k) for k in ck_list])
ck_to_i = {k: i for i, k in enumerate(ck_list)}
adj_c   = {k: list(cmesh.vertex_neighbors(k)) for k in ck_list}
edges_c = list(cmesh.edges())
edge_q  = {tuple(sorted([u,v])): float(cmesh.edge_attribute((u,v),"qpre"))
           for u,v in edges_c}

r_xy_c    = np.linalg.norm(V_c[:,:2], axis=1)
apex_c_i  = int(np.argmin(r_xy_c))
apex_c_xy = V_c[apex_c_i, :2]
LAMBDA_TILT = 4.0

def dijkstra(src_key):
    dist = {k: np.inf for k in ck_list}
    prev = {k: None   for k in ck_list}
    dist[src_key] = 0.0
    pq = [(0.0, src_key)]
    while pq:
        d, uk = heapq.heappop(pq)
        if d > dist[uk]: continue
        iu   = ck_to_i[uk]
        u_xy = V_c[iu, :2]
        r_u  = float(np.linalg.norm(u_xy - apex_c_xy))
        for vk in adj_c[uk]:
            iv    = ck_to_i[vk]
            qe    = edge_q.get(tuple(sorted([uk,vk])), 0.1)
            e_xy  = V_c[iv,:2] - u_xy
            e_len = float(np.linalg.norm(e_xy))
            if r_u > 1e-6 and e_len > 1e-9:
                radial = (u_xy - apex_c_xy) / r_u
                cos_r  = float(np.dot(e_xy/e_len, radial))
                sin2_r = max(0.0, 1.0 - cos_r**2)
            else:
                sin2_r = 0.0
            w  = (1.0/max(qe,1e-6)**2) * (1.0 + LAMBDA_TILT*sin2_r)
            nd = d + w
            if nd < dist[vk]:
                dist[vk] = nd; prev[vk] = uk
                heapq.heappush(pq, (nd, vk))
    return dist, prev

def reconstruct(tgt, prev):
    path=[]; cur=tgt
    while cur is not None: path.append(cur); cur=prev[cur]
    return list(reversed(path))

def nearest_compas(obj_idx):
    best = int(np.argmin(np.sum((V_c - V[obj_idx])**2, axis=1)))
    return ck_list[best]

def path_to_obj(path_c):
    return [int(np.argmin(np.sum((V[:,:2]-V_c[ck_to_i[ck],:2])**2,axis=1)))
            for ck in path_c]

# Build cable vertex lists
print("Computing cable paths...")
cable_paths = {}   # name → list of OBJ vertex indices
for k, (v_src, v_hoop, v_dst) in enumerate(CABLES_DEF):
    theta = float(np.degrees(np.arctan2(V[v_dst,1], V[v_dst,0])) % 360)

    src_key  = nearest_compas(v_src)
    hoop_key = nearest_compas(v_hoop)
    dst_key  = nearest_compas(v_dst)

    _, pi = dijkstra(src_key)
    inner = path_to_obj(reconstruct(hoop_key, pi))
    if inner[-1] != v_hoop: inner.append(v_hoop)

    _, po = dijkstra(hoop_key)
    outer = path_to_obj(reconstruct(dst_key, po))
    if outer[0] != v_hoop: outer.insert(0, v_hoop)

    cable_paths[f"Si{k}_{theta:.1f}"] = inner
    cable_paths[f"So{k}_{theta:.1f}"] = outer
    print(f"  cable {k}: θ={theta:.1f}°  inner={len(inner)}  outer={len(outer)}")

# Add hoop arcs (reuse hoop detection)
qs_all = np.array(list(edge_q.values()))
bdry_c = set(cmesh.vertices_on_boundary())
R_max  = float(np.linalg.norm(V_c[:,:2],axis=1).max())
R_BDRY = 0.85 * R_max
q_thr  = float(np.percentile(qs_all, 70))

edge_circ = {}
for (u,v), qe in edge_q.items():
    if u in bdry_c or v in bdry_c: continue
    iu,iv = ck_to_i[u], ck_to_i[v]
    pu,pv = V_c[iu,:2], V_c[iv,:2]
    r_m = 0.5*(np.linalg.norm(pu)+np.linalg.norm(pv))
    if r_m > R_BDRY or qe < q_thr: continue
    L = np.linalg.norm(pv-pu)+1e-9; dr = abs(np.linalg.norm(pv)-np.linalg.norm(pu))
    if (1.0-dr/L) > 0.6: edge_circ[(u,v)] = (qe, r_m)

bin_w=0.04; bins=np.arange(0,R_BDRY+bin_w,bin_w)
bc=np.zeros(len(bins)-1); bq=np.zeros(len(bins)-1)
for (qe,r) in edge_circ.values():
    b=int(r//bin_w)
    if 0<=b<len(bc): bc[b]+=1; bq[b]+=qe
cands=[b for b in range(len(bc)) if bc[b]>=6]
hoop_obj=[]
if cands:
    pk=max(cands,key=lambda b:bq[b])
    r_lo,r_hi=bins[pk],bins[pk+1]
    if pk+1<len(bc) and bc[pk+1]>=4: r_hi=bins[pk+2]
    if pk>0 and bc[pk-1]>=4: r_lo=bins[pk-1]
    re=[(u,v) for (u,v),(qe,r) in edge_circ.items() if r_lo<=r<r_hi]
    ra={}
    for u,v in re: ra.setdefault(u,[]).append(v); ra.setdefault(v,[]).append(u)
    st=min(ra,key=lambda k: np.arctan2(V_c[ck_to_i[k],1],V_c[ck_to_i[k],0]))
    hp,pv2,cur=[st],None,st
    while True:
        nbs=[n for n in ra.get(cur,[]) if n!=pv2 and n not in hp]
        if not nbs:
            if st in ra.get(cur,[]) and len(hp)>4: hp.append(st)
            break
        ct=np.arctan2(V_c[ck_to_i[cur],1],V_c[ck_to_i[cur],0])
        nxt=min(nbs,key=lambda n:(np.arctan2(V_c[ck_to_i[n],1],V_c[ck_to_i[n],0])-ct)%(2*np.pi))
        hp.append(nxt); pv2,cur=cur,nxt
        if cur==st or len(hp)>300: break
    hoop_obj=[int(np.argmin(np.sum((V[:,:2]-V_c[ck_to_i[ck],:2])**2,axis=1))) for ck in hp]
    cable_paths["H0_hoop"] = hoop_obj
    print(f"Hoop: {len(hoop_obj)} verts")

# ── Build cable segment list in 3D ────────────────────────────────────────────
seg_p0, seg_p1, seg_tan = [], [], []
for path in cable_paths.values():
    for i in range(len(path)-1):
        p0, p1 = V[path[i]], V[path[i+1]]
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-9: continue
        seg_p0.append(p0); seg_p1.append(p1); seg_tan.append(d/L)

seg_p0  = np.array(seg_p0)   # (N_segs, 3)
seg_p1  = np.array(seg_p1)
seg_tan = np.array(seg_tan)
print(f"Total cable segments: {len(seg_p0)}")

# ── Face geometry ─────────────────────────────────────────────────────────────
centroids = V[F].mean(axis=1)                              # (nf, 3)
e1 = V[F[:,1]] - V[F[:,0]]
e2 = V[F[:,2]] - V[F[:,0]]
normals = np.cross(e1, e2)
norms   = np.linalg.norm(normals, axis=1, keepdims=True)
normals = normals / np.maximum(norms, 1e-10)               # (nf, 3) unit normals

# ── Nearest cable segment for each face (via KD-tree on segment midpoints) ────
midpoints = 0.5 * (seg_p0 + seg_p1)
tree      = cKDTree(midpoints)
_, nn_idx = tree.query(centroids, k=1)   # nearest segment index per face

# ── Directional field: project cable tangent onto face tangent plane ──────────
d1 = np.zeros((len(F), 3))
d2 = np.zeros((len(F), 3))

for fi in range(len(F)):
    n   = normals[fi]
    tan = seg_tan[nn_idx[fi]]
    # remove normal component
    along = tan - np.dot(tan, n) * n
    L = np.linalg.norm(along)
    if L < 1e-9:
        # degenerate: cable parallel to normal — pick arbitrary tangent
        along = e1[fi] - np.dot(e1[fi], n) * n
        L = np.linalg.norm(along)
    d1[fi] = along / max(L, 1e-10)
    d2[fi] = np.cross(n, d1[fi])
    L2 = np.linalg.norm(d2[fi])
    if L2 > 1e-10: d2[fi] /= L2

print("Directional field computed.")

# ── Save JSON ─────────────────────────────────────────────────────────────────
field_out = {str(fi): {"d1": d1[fi].tolist(), "d2": d2[fi].tolist(),
                        "centroid": centroids[fi].tolist()}
             for fi in range(len(F))}
with open(OUT_JSON, "w") as f:
    json.dump(field_out, f, indent=1)
print(f"Saved {OUT_JSON}")

# ── Region colouring (same as main script) ────────────────────────────────────
adj_obj = [set() for _ in range(len(V))]
for tri in F:
    for i,j in [(0,1),(1,2),(2,0)]:
        adj_obj[tri[i]].add(tri[j]); adj_obj[tri[j]].add(tri[i])

visited, comps = set(), []
for s in range(len(V)):
    if s in visited: continue
    comp=set([s]); q=deque([s])
    while q:
        u=q.popleft()
        for v in adj_obj[u]:
            if v not in comp: comp.add(v); q.append(v)
    visited|=comp; comps.append(sorted(comp))
comps.sort(key=lambda c: float(np.linalg.norm(V[c,:2],axis=1).mean()))

r_hoop_mid = 0.5*(r_lo+r_hi)
region_info=[]
for comp in comps:
    xy=V[comp,:2]
    r_m=float(np.linalg.norm(xy,axis=1).mean())
    th_m=float(np.degrees(np.arctan2(xy[:,1].mean(),xy[:,0].mean()))%360)
    wedge=int(th_m//45)
    rtype="inner" if r_m<r_hoop_mid else "outer"
    region_info.append({"comp":comp,"wedge":wedge,"type":rtype})

def wedge_color(wedge, rtype):
    h=wedge/8; s=0.5 if rtype=="inner" else 0.92; v=0.82 if rtype=="inner" else 0.62
    return colorsys.hsv_to_rgb(h,s,v)

region_colors=[wedge_color(r["wedge"],r["type"]) for r in region_info]
vert_comp=np.full(len(V),-1,dtype=int)
for ci,ri in enumerate(region_info):
    for v in ri["comp"]: vert_comp[v]=ci
face_comp=np.full(len(F),-1,dtype=int)
for fi,tri in enumerate(F):
    c=vert_comp[tri[0]]
    if vert_comp[tri[1]]==c and vert_comp[tri[2]]==c: face_comp[fi]=c
face_colors_2d=[region_colors[face_comp[fi]] if face_comp[fi]>=0 else (0.08,0.08,0.12)
                for fi in range(len(F))]

# ── Plot ──────────────────────────────────────────────────────────────────────
BG  = "#0d0d1a"
xr  = float(np.abs(V[:,:2]).max())*1.08
zmax= float(V[:,2].max())
face_verts   = [[V[F[i,j]] for j in range(3)] for i in range(len(F))]
face_z_mean  = V[F].mean(axis=1)[:,2]

fig = plt.figure(figsize=(18,9))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[1.1,1], wspace=0.04)
ax2 = fig.add_subplot(gs[0])                          # top-down
ax1 = fig.add_subplot(gs[1], projection="3d")         # 3D oblique

# ── Top-down ──────────────────────────────────────────────────────────────────
ax2.set_facecolor(BG); ax2.set_aspect("equal")

for fi, tri in enumerate(F):
    fc = face_colors_2d[fi]
    poly = plt.Polygon(V[tri,:2], facecolor=fc, edgecolor=fc, linewidth=0.08, alpha=0.85)
    ax2.add_patch(poly)

# Cross field: draw d1 and d2 as short line segments through face centroid
# Subsample to avoid overdraw (every other face)
ARROW_LEN = 0.018
for fi in range(len(F)):
    if face_comp[fi] < 0: continue
    c2 = centroids[fi,:2]
    a1 = d1[fi,:2]; a2 = d2[fi,:2]
    # d1: along cable (white)
    ax2.plot([c2[0]-ARROW_LEN*a1[0], c2[0]+ARROW_LEN*a1[0]],
             [c2[1]-ARROW_LEN*a1[1], c2[1]+ARROW_LEN*a1[1]],
             color="white", linewidth=0.55, alpha=0.75, solid_capstyle="round")
    # d2: perp cable (cyan)
    ax2.plot([c2[0]-ARROW_LEN*a2[0], c2[0]+ARROW_LEN*a2[0]],
             [c2[1]-ARROW_LEN*a2[1], c2[1]+ARROW_LEN*a2[1]],
             color="#00eeff", linewidth=0.55, alpha=0.75, solid_capstyle="round")

# cable overlay
CABLE_COLS = {"Si":"#ffcc00","So":"#ffcc00","H":"#00ff88"}
for name, path in cable_paths.items():
    pts=V[path]
    col="#ffcc00" if not name.startswith("H") else "#00ff88"
    ax2.plot(pts[:,0],pts[:,1],color=col,linewidth=2.2,solid_capstyle="round",zorder=5)

ax2.set_xlim(-xr,xr); ax2.set_ylim(-xr,xr)
ax2.tick_params(colors="#445566",labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#223344")
ax2.set_xlabel("x",color="#445566",fontsize=8); ax2.set_ylabel("y",color="#445566",fontsize=8)
ax2.set_title("Top-down — cross field  (white = along cable · cyan = perpendicular)",
              color="white",fontsize=10,pad=5)
ax2.legend(handles=[mpatches.Patch(color="#ffcc00",label="spokes"),
                    mpatches.Patch(color="#00ff88",label="hoop"),
                    mpatches.Patch(color="white",  label="d₁ ∥ cable"),
                    mpatches.Patch(color="#00eeff",label="d₂ ⊥ cable")],
           loc="lower right",fontsize=8,facecolor="#111827",labelcolor="white",
           edgecolor="#334455",framealpha=0.85)

# ── 3D oblique ────────────────────────────────────────────────────────────────
ax1.set_facecolor(BG)
face_colors_3d=[region_colors[face_comp[fi]] if face_comp[fi]>=0 else (0.08,0.08,0.12)
                for fi in range(len(F))]
poly3=Poly3DCollection(face_verts,alpha=0.88,facecolor=face_colors_3d,
                       edgecolor=(0,0,0,0.12),linewidth=0.05)
ax1.add_collection3d(poly3)

# Cross field in 3D (subsample every 4th face to keep readable)
ARROW3 = 0.025
for fi in range(0, len(F), 4):
    if face_comp[fi] < 0: continue
    c3 = centroids[fi]
    ax1.plot([c3[0]-ARROW3*d1[fi,0],c3[0]+ARROW3*d1[fi,0]],
             [c3[1]-ARROW3*d1[fi,1],c3[1]+ARROW3*d1[fi,1]],
             [c3[2]-ARROW3*d1[fi,2],c3[2]+ARROW3*d1[fi,2]],
             color="white",linewidth=0.6,alpha=0.8)
    ax1.plot([c3[0]-ARROW3*d2[fi,0],c3[0]+ARROW3*d2[fi,0]],
             [c3[1]-ARROW3*d2[fi,1],c3[1]+ARROW3*d2[fi,1]],
             [c3[2]-ARROW3*d2[fi,2],c3[2]+ARROW3*d2[fi,2]],
             color="#00eeff",linewidth=0.6,alpha=0.8)

for name, path in cable_paths.items():
    pts=V[path]
    col="#ffcc00" if not name.startswith("H") else "#00ff88"
    ax1.plot(pts[:,0],pts[:,1],pts[:,2],color=col,linewidth=2.0,alpha=1.0)

ax1.set_xlim(-xr,xr); ax1.set_ylim(-xr,xr); ax1.set_zlim(0,zmax*1.1)
ax1.view_init(elev=28, azim=-55)
ax1.tick_params(colors="#334455",labelsize=5)
for pane in (ax1.xaxis.pane,ax1.yaxis.pane,ax1.zaxis.pane):
    pane.fill=False; pane.set_edgecolor("#1a2a3a")
ax1.set_title("3D — cross field (white=∥cable · cyan=⊥cable)",color="white",fontsize=9,pad=3)

fig.suptitle("C5  —  directional field aligned with cable stiffeners",
             color="white",fontsize=13,y=1.01)
plt.savefig(OUT_PNG,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
