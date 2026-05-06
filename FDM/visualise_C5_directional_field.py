"""Directional (cross) field on the C5 dome guided by cable stiffeners.

For every mesh face, the two principal directions are:
  d1 : radial (along spokes / perpendicular to hoop and boundary)
  d2 : face_normal × d1  (circumferential)

Cables are hard constraints; interior faces are smoothed via a
Laplacian linear solve (penalty formulation).
"""
import os, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
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
HOOP_VERTS_OBJ = [ 995, 1059, 1138, 1187, 1170, 1105, 1028,  977]

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

# ── Cable segments: straight connections between waypoints ────────────────────
cable_paths = {}
for k, (v_src, v_hoop, v_dst) in enumerate(CABLES_DEF):
    theta = float(np.degrees(np.arctan2(V[v_dst,1], V[v_dst,0])) % 360)
    cable_paths[f"Si{k}_{theta:.1f}"] = [v_src,  v_hoop]   # inner spoke
    cable_paths[f"So{k}_{theta:.1f}"] = [v_hoop, v_dst ]   # outer spoke

# Hoop ring: connect consecutive hoop vertices in CCW order
hoop_ring_cable = HOOP_VERTS_OBJ + [HOOP_VERTS_OBJ[0]]   # closed
cable_paths["H0_hoop"] = hoop_ring_cable
print(f"Cable segments built from waypoints: {sum(len(p)-1 for p in cable_paths.values())} segments")

# ── Hoop mid-radius (for region colouring) ────────────────────────────────────
r_hoop_mid = float(np.linalg.norm(V[HOOP_VERTS_OBJ, :2], axis=1).mean())

# ── Build cable segment list in 3D ────────────────────────────────────────────
# seg_is_hoop=True → d1 should be ⊥ to the segment (radial), not ∥ to it
seg_p0, seg_p1, seg_tan, seg_is_hoop = [], [], [], []
for name, path in cable_paths.items():
    is_hoop = name.startswith("H")
    for i in range(len(path)-1):
        p0, p1 = V[path[i]], V[path[i+1]]
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-9: continue
        seg_p0.append(p0); seg_p1.append(p1)
        seg_tan.append(d/L); seg_is_hoop.append(is_hoop)

seg_p0      = np.array(seg_p0)    # (N_segs, 3)
seg_p1      = np.array(seg_p1)
seg_tan     = np.array(seg_tan)
seg_is_hoop = np.array(seg_is_hoop, dtype=bool)
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

# ── Directional field ─────────────────────────────────────────────────────────
# Spokes:   d1 ∥ cable tangent (radial)
# Hoop:     d1 ⊥ hoop tangent  (also radial — perpendicular to ring)
# In both cases d2 = n × d1 (circumferential)
d1 = np.zeros((len(F), 3))
d2 = np.zeros((len(F), 3))

for fi in range(len(F)):
    n   = normals[fi]
    tan = seg_tan[nn_idx[fi]]
    along = tan - np.dot(tan, n) * n
    L = np.linalg.norm(along)
    if L < 1e-9:
        along = e1[fi] - np.dot(e1[fi], n) * n
        L = np.linalg.norm(along)
    along /= max(L, 1e-10)
    if seg_is_hoop[nn_idx[fi]]:
        # rotate 90° in tangent plane: d1 = n × along  (radial direction)
        along = np.cross(n, along)
        L2 = np.linalg.norm(along)
        if L2 > 1e-10: along /= L2
    d1[fi] = along
    d2[fi] = np.cross(n, d1[fi])
    L2 = np.linalg.norm(d2[fi])
    if L2 > 1e-10: d2[fi] /= L2

print("Directional field computed (initial, cable-aligned).")

# ── Laplacian smoothing — tiered constraints ──────────────────────────────────
# Priority 1 (λ=1000): radial spokes (Si, So)
# Priority 2 (λ=150):  inner hoop ring + outer mesh boundary (d1 ⊥ those curves)

nf = len(F)

# Face-face adjacency (shared edge)
edge_to_faces = {}
for fi, tri in enumerate(F):
    for a, b in [(0,1),(1,2),(2,0)]:
        key = tuple(sorted([tri[a], tri[b]]))
        edge_to_faces.setdefault(key, []).append(fi)
face_adj = [[] for _ in range(nf)]
for fl in edge_to_faces.values():
    if len(fl) == 2:
        face_adj[fl[0]].append(fl[1])
        face_adj[fl[1]].append(fl[0])

# Outer boundary faces (edges with only one adjacent face)
bdry_face_set = {fl[0] for fl in edge_to_faces.values() if len(fl) == 1}

# Distance from each face centroid to cable segments, split by type
def pt_seg_dist_batch(pts, p0, p1):
    d = p1 - p0
    L2 = float(np.dot(d, d))
    if L2 < 1e-12:
        return np.linalg.norm(pts - p0, axis=1)
    t = np.clip(((pts - p0) @ d) / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (p0 + t[:, None] * d), axis=1)

dist_spoke = np.full(nf, np.inf)
dist_hoop  = np.full(nf, np.inf)
for si in range(len(seg_p0)):
    d = pt_seg_dist_batch(centroids, seg_p0[si], seg_p1[si])
    if seg_is_hoop[si]:
        dist_hoop = np.minimum(dist_hoop, d)
    else:
        dist_spoke = np.minimum(dist_spoke, d)

edge_vecs = (V[F[:, 1]] - V[F[:, 0]],
             V[F[:, 2]] - V[F[:, 1]],
             V[F[:, 0]] - V[F[:, 2]])
mean_edge = float(np.mean([np.linalg.norm(ev, axis=1).mean() for ev in edge_vecs]))
CDIST = 0.8 * mean_edge

LAMBDA_SPOKE = 1000.0
LAMBDA_HOOP  =  150.0
LAMBDA_BDRY  =  150.0

# Consistent outward orientation before averaging
apex_pos = V[1091]
for fi in range(nf):
    if np.dot(d1[fi], centroids[fi] - apex_pos) < 0:
        d1[fi] = -d1[fi]

# Build per-face penalty and target; higher priority overwrites lower
face_lambda = np.zeros(nf)
face_d1_tgt = d1.copy()

# Priority 2a: hoop constraints (d1 already set to radial for hoop faces)
hoop_mask = dist_hoop < CDIST
face_lambda[hoop_mask] = LAMBDA_HOOP

# Priority 2b: outer boundary faces — d1 = radial (outward from apex)
for fi in bdry_face_set:
    if face_lambda[fi] < LAMBDA_BDRY:
        n = normals[fi]
        radial = centroids[fi] - apex_pos
        radial -= np.dot(radial, n) * n
        L = np.linalg.norm(radial)
        if L > 1e-10:
            face_lambda[fi] = LAMBDA_BDRY
            face_d1_tgt[fi] = radial / L

# Priority 1: spoke faces — hard Dirichlet
spoke_mask = dist_spoke < CDIST
face_d1_tgt[spoke_mask] = d1[spoke_mask]

# Immediate neighbours of spoke faces — also hard Dirichlet (same radial direction)
spoke_nbr_mask = np.zeros(nf, dtype=bool)
for fi in range(nf):
    if spoke_mask[fi]:
        for fj in face_adj[fi]:
            if not spoke_mask[fj]:
                spoke_nbr_mask[fj] = True

for fi in np.where(spoke_nbr_mask)[0]:
    n = normals[fi]
    radial = centroids[fi] - apex_pos
    radial -= np.dot(radial, n) * n
    L = np.linalg.norm(radial)
    if L > 1e-10:
        face_d1_tgt[fi] = radial / L

# Hard mask = spokes + their neighbours
hard_mask = spoke_mask | spoke_nbr_mask

# Remaining soft constraints (hoop / boundary) — penalty only where not hard
LAMBDA_SOFT = 150.0
soft_mask = ~hard_mask & (
    (dist_hoop < CDIST) |
    np.array([fi in bdry_face_set for fi in range(nf)])
)
for fi in np.where(soft_mask & np.array([fi in bdry_face_set for fi in range(nf)]))[0]:
    n = normals[fi]
    radial = centroids[fi] - apex_pos
    radial -= np.dot(radial, n) * n
    L = np.linalg.norm(radial)
    if L > 1e-10:
        face_d1_tgt[fi] = radial / L

print(f"Hard Dirichlet: {spoke_mask.sum()} spoke  {spoke_nbr_mask.sum()} spoke-neighbours")
print(f"Soft penalty:   {soft_mask.sum()} hoop/boundary  (λ={LAMBDA_SOFT:.0f})")

# ── Reduced Laplacian: unknowns = all non-hard faces ─────────────────────────
unk_idx = np.where(~hard_mask)[0]
unk_map  = {int(fi): i for i, fi in enumerate(unk_idx)}
n_unk    = len(unk_idx)

L_red = lil_matrix((n_unk, n_unk))
rhs   = np.zeros((n_unk, 3))

for ii, fi in enumerate(unk_idx):
    lam = LAMBDA_SOFT if soft_mask[fi] else 0.0
    L_red[ii, ii] = len(face_adj[fi]) + lam
    for fj in face_adj[fi]:
        if hard_mask[fj]:
            rhs[ii] += face_d1_tgt[fj]      # hard neighbour → RHS
        else:
            L_red[ii, unk_map[fj]] = -1.0
    if soft_mask[fi]:
        rhs[ii] += lam * face_d1_tgt[fi]

L_red_csr = L_red.tocsr()
d1_smooth = np.zeros((nf, 3))
d1_smooth[hard_mask] = face_d1_tgt[hard_mask]   # exact
for c in range(3):
    d1_smooth[unk_idx, c] = spsolve(L_red_csr, rhs[:, c])

# Project back onto tangent planes and renormalise
for fi in range(nf):
    n = normals[fi]
    v = d1_smooth[fi] - np.dot(d1_smooth[fi], n) * n
    L = np.linalg.norm(v)
    if L < 1e-10:
        v = d1[fi]
        L = np.linalg.norm(v)
    d1[fi] = v / max(L, 1e-10)
    d2[fi] = np.cross(n, d1[fi])
    L2 = np.linalg.norm(d2[fi])
    if L2 > 1e-10: d2[fi] /= L2

print("Directional field smoothed (hard: spokes+neighbours · soft: hoop/boundary).")

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

r_hoop_mid = float(np.linalg.norm(V[HOOP_VERTS_OBJ, :2], axis=1).mean())
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

fig.suptitle("C5  —  smoothed directional field  (cables = hard constraints)",
             color="white",fontsize=13,y=1.01)
plt.savefig(OUT_PNG,dpi=160,bbox_inches="tight",facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
