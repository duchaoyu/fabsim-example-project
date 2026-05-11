"""
Directional (cross) field on D5 guided by the inner cable boundary.

For every face:
  d1 : tangent to the inner cable (wale direction)
  d2 : face_normal × d1  (course direction)

The inner cable is the hard constraint; all other faces are smoothed
via a face-adjacency Laplacian (complex-number representation of the
in-plane angle).
"""
import os, json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE      = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH  = os.path.join(HERE, "data", "D5", "D5_remeshed.obj")
CABLE_JSON= os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
OUT_PNG   = os.path.join(HERE, "data", "D5", "D5_directional_field.png")
OUT_JSON  = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
BG        = "#0d0d1a"


# ── Load mesh ─────────────────────────────────────────────────────────────────
def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t: continue
            if t[0] == "v":  V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f": F.append([int(s.split("/")[0])-1 for s in t[1:4]])
    return np.array(V, float), np.array(F, int)

V, F = load_obj(OBJ_PATH)
n_v, n_f = len(V), len(F)
print(f"Mesh: {n_v} verts, {n_f} faces")


# ── Load inner cable ──────────────────────────────────────────────────────────
with open(CABLE_JSON) as f:
    cable_data = json.load(f)
cable_idx = cable_data["vertex_indices"]   # ordered 42-vertex loop
n_cv = len(cable_idx)
print(f"Inner cable: {n_cv} vertices")


# ── Face geometry ─────────────────────────────────────────────────────────────
centroids = V[F].mean(axis=1)             # (n_f, 3)
e1 = V[F[:, 1]] - V[F[:, 0]]
e2 = V[F[:, 2]] - V[F[:, 0]]
normals = np.cross(e1, e2)
norms   = np.linalg.norm(normals, axis=1, keepdims=True)
normals /= np.maximum(norms, 1e-10)       # (n_f, 3) unit normals

# Per-face orthonormal tangent frame  (u, v) spanning the tangent plane
# u = normalised e1 projected onto tangent plane
u_frame = e1 - (e1 * normals).sum(axis=1, keepdims=True) * normals
u_frame /= np.maximum(np.linalg.norm(u_frame, axis=1, keepdims=True), 1e-10)
v_frame = np.cross(normals, u_frame)     # (n_f, 3)


# ── Cable edge tangents ───────────────────────────────────────────────────────
# Build ordered edge tangents for the closed loop
cable_segs_p0  = []
cable_segs_p1  = []
cable_segs_tan = []   # unit tangent of each cable edge

for i in range(n_cv):
    a = cable_idx[i]
    b = cable_idx[(i + 1) % n_cv]
    d = V[b] - V[a]
    L = np.linalg.norm(d)
    if L > 1e-10:
        cable_segs_p0.append(V[a])
        cable_segs_p1.append(V[b])
        cable_segs_tan.append(d / L)

cable_segs_p0  = np.array(cable_segs_p0)   # (n_cv, 3)
cable_segs_p1  = np.array(cable_segs_p1)
cable_segs_tan = np.array(cable_segs_tan)
cable_mids     = 0.5 * (cable_segs_p0 + cable_segs_p1)


# ── Project nearest cable tangent onto each face ───────────────────────────────
def project_to_tangent(vec3d, normal):
    """Project vec3d onto tangent plane defined by normal, then normalise."""
    t = vec3d - np.dot(vec3d, normal) * normal
    L = np.linalg.norm(t)
    return t / L if L > 1e-10 else None


# For each face, find nearest cable edge midpoint → get that edge's tangent
dists = np.linalg.norm(
    centroids[:, np.newaxis, :] - cable_mids[np.newaxis, :, :], axis=2)  # (n_f, n_cv)
nearest_seg = np.argmin(dists, axis=1)   # (n_f,) index into cable_segs

d1_init = np.zeros((n_f, 3))
d2_init = np.zeros((n_f, 3))
for fi in range(n_f):
    seg_tan  = cable_segs_tan[nearest_seg[fi]]
    n        = normals[fi]
    t        = project_to_tangent(seg_tan, n)
    if t is None:
        t = u_frame[fi]
    d1_init[fi] = t
    d2_init[fi] = np.cross(n, t)
    d2_init[fi] /= max(np.linalg.norm(d2_init[fi]), 1e-10)


# ── Identify cable-adjacent faces (hard constraints) ─────────────────────────
cable_verts_set = set(cable_idx)
face_on_cable   = np.array([
    len(set(F[fi]) & cable_verts_set) >= 2   # face has ≥2 cable vertices
    for fi in range(n_f)
], dtype=bool)
print(f"Cable-adjacent faces: {face_on_cable.sum()}")


# ── Build face adjacency ──────────────────────────────────────────────────────
edge_to_faces = defaultdict(list)
for fi, tri in enumerate(F):
    for k in range(3):
        e = tuple(sorted([tri[k], tri[(k+1)%3]]))
        edge_to_faces[e].append(fi)

face_adj = defaultdict(list)   # fi → list of neighbour face indices
for e, flist in edge_to_faces.items():
    if len(flist) == 2:
        face_adj[flist[0]].append(flist[1])
        face_adj[flist[1]].append(flist[0])


# ── Laplacian smoothing via complex representation ────────────────────────────
# Represent d1 as a complex number z = e^(i*theta) in the (u_frame, v_frame) basis.
# Average neighbours in complex space (parallel transport = project then re-express).
# Cable faces are soft constraints with high weight.

SMOOTH_ITERS   = 300
CABLE_WEIGHT   = 10.0   # relative weight of cable-face constraint

# Initialise angles from d1_init
def d1_to_angle(fi, d1):
    """Express d1[fi] as angle in face's (u,v) frame."""
    uu = float(np.dot(d1, u_frame[fi]))
    vv = float(np.dot(d1, v_frame[fi]))
    return np.arctan2(vv, uu)

angles = np.array([d1_to_angle(fi, d1_init[fi]) for fi in range(n_f)])

def transport_angle(fi, fj, ang_i):
    """
    Parallel-transport d1 from face fi to face fj:
    express d1_fi in 3D, project onto face fj tangent plane, re-express as angle.
    """
    d = (np.cos(ang_i) * u_frame[fi] +
         np.sin(ang_i) * v_frame[fi])
    d -= np.dot(d, normals[fj]) * normals[fj]
    L = np.linalg.norm(d)
    if L < 1e-10:
        return ang_i
    d /= L
    return np.arctan2(float(np.dot(d, v_frame[fj])),
                      float(np.dot(d, u_frame[fj])))

print(f"Smoothing {SMOOTH_ITERS} iterations …")
for it in range(SMOOTH_ITERS):
    new_angles = angles.copy()
    for fi in range(n_f):
        if face_on_cable[fi]:
            # Soft constraint: keep cable direction with higher weight
            zsum  = CABLE_WEIGHT * np.exp(1j * angles[fi])
            count = CABLE_WEIGHT
        else:
            zsum  = 0j
            count = 0.0

        for fj in face_adj[fi]:
            ang_j_transported = transport_angle(fj, fi, angles[fj])
            # Resolve π-ambiguity (cross field: d and -d are the same direction)
            da = ang_j_transported - angles[fi]
            da = (da + np.pi/2) % np.pi - np.pi/2   # nearest equivalent in (-π/2, π/2)
            zsum  += np.exp(1j * (angles[fi] + da))
            count += 1.0

        if count > 0:
            new_angles[fi] = np.angle(zsum)

    angles = new_angles
    if (it + 1) % 50 == 0:
        print(f"  iter {it+1}")

print("Smoothing done.")

# Reconstruct d1, d2 from smoothed angles
d1 = np.cos(angles)[:, None] * u_frame + np.sin(angles)[:, None] * v_frame
d2 = np.cross(normals, d1)
d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-10)


# ── Save JSON ─────────────────────────────────────────────────────────────────
field_out = {str(fi): {"d1": d1[fi].tolist(), "d2": d2[fi].tolist(),
                        "centroid": centroids[fi].tolist()}
             for fi in range(n_f)}
with open(OUT_JSON, "w") as f:
    json.dump(field_out, f, indent=1)
print(f"Saved {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 9), facecolor=BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.04,
                        left=0.03, right=0.97, top=0.92, bottom=0.05)
ax2 = fig.add_subplot(gs[0])                     # top-down
ax1 = fig.add_subplot(gs[1], projection="3d")    # 3D oblique

xr   = float(np.abs(V[:, :2]).max()) * 1.08
zmax = float(V[:, 2].max())
ARROW = 0.016

# ── Top-down ──────────────────────────────────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_aspect("equal")

# Mesh face fill (dark grey)
for fi, tri in enumerate(F):
    poly = plt.Polygon(V[tri, :2], facecolor=(0.10, 0.10, 0.14),
                       edgecolor=(0.3, 0.3, 0.35), linewidth=0.08, alpha=0.9)
    ax2.add_patch(poly)

# Cross-field ticks on every face
for fi in range(n_f):
    c  = centroids[fi, :2]
    a1 = d1[fi, :2]
    a2 = d2[fi, :2]
    ax2.plot([c[0] - ARROW*a1[0], c[0] + ARROW*a1[0]],
             [c[1] - ARROW*a1[1], c[1] + ARROW*a1[1]],
             color="white", lw=0.55, alpha=0.80, solid_capstyle="round")
    ax2.plot([c[0] - ARROW*a2[0], c[0] + ARROW*a2[0]],
             [c[1] - ARROW*a2[1], c[1] + ARROW*a2[1]],
             color="#00eeff", lw=0.55, alpha=0.80, solid_capstyle="round")

# Inner cable overlay
cable_pts = V[cable_idx + [cable_idx[0]]]
ax2.plot(cable_pts[:, 0], cable_pts[:, 1],
         color="#ffcc00", lw=2.2, solid_capstyle="round", zorder=6, label="inner cable")
ax2.scatter(V[cable_idx, 0], V[cable_idx, 1],
            c="#ffcc00", s=12, zorder=7)

ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)
ax2.tick_params(colors="#445566", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#223344")
ax2.set_xlabel("x", color="#445566", fontsize=8)
ax2.set_ylabel("y", color="#445566", fontsize=8)
ax2.set_title("Top-down — cross field  (white = ∥ cable · cyan = ⊥ cable)",
              color="white", fontsize=10, pad=5)
ax2.legend(handles=[
    mpatches.Patch(color="#ffcc00", label="inner cable"),
    mpatches.Patch(color="white",   label="d₁ ∥ cable"),
    mpatches.Patch(color="#00eeff", label="d₂ ⊥ cable"),
], loc="lower right", fontsize=8, facecolor="#111827",
   labelcolor="white", edgecolor="#334455", framealpha=0.85)


# ── 3D oblique ────────────────────────────────────────────────────────────────
ax1.set_facecolor(BG)

poly3 = Poly3DCollection(
    [V[F[fi]] for fi in range(n_f)],
    alpha=0.75, facecolor=(0.10, 0.10, 0.14, 0.75),
    edgecolor=(0.25, 0.25, 0.3, 0.15), linewidth=0.05)
ax1.add_collection3d(poly3)

ARROW3 = 0.022
for fi in range(0, n_f, 3):
    c = centroids[fi]
    ax1.plot([c[0]-ARROW3*d1[fi,0], c[0]+ARROW3*d1[fi,0]],
             [c[1]-ARROW3*d1[fi,1], c[1]+ARROW3*d1[fi,1]],
             [c[2]-ARROW3*d1[fi,2], c[2]+ARROW3*d1[fi,2]],
             color="white", lw=0.6, alpha=0.8)
    ax1.plot([c[0]-ARROW3*d2[fi,0], c[0]+ARROW3*d2[fi,0]],
             [c[1]-ARROW3*d2[fi,1], c[1]+ARROW3*d2[fi,1]],
             [c[2]-ARROW3*d2[fi,2], c[2]+ARROW3*d2[fi,2]],
             color="#00eeff", lw=0.6, alpha=0.8)

# Inner cable in 3D
ax1.plot(cable_pts[:, 0], cable_pts[:, 1], cable_pts[:, 2],
         color="#ffcc00", lw=2.5, alpha=1.0)

ax1.set_xlim(-xr, xr); ax1.set_ylim(-xr, xr); ax1.set_zlim(0, zmax * 1.1)
ax1.view_init(elev=28, azim=-55)
ax1.tick_params(colors="#334455", labelsize=5)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False; pane.set_edgecolor("#1a2a3a")
ax1.set_xlabel("x", color="#445566", fontsize=7)
ax1.set_ylabel("y", color="#445566", fontsize=7)
ax1.set_zlabel("z", color="#445566", fontsize=7)
ax1.set_title("3D — cross field (white = ∥ cable · cyan = ⊥ cable)",
              color="white", fontsize=9, pad=3)

fig.suptitle("D5 — directional field guided by inner cable boundary",
             color="white", fontsize=13, y=0.98)

plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor=BG)
print(f"Saved {OUT_PNG}")
