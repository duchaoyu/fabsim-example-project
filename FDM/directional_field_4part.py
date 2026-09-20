"""
Directional (cross) field on the 4-part mesh, guided by the extracted cables.

Same construction as FDM/directional_field_D5.py:
  d1 : wale direction, tangent to the nearest cable
  d2 : course direction, face_normal x d1
The cable-adjacent faces are held fixed; every other face is found by minimising
the Dirichlet energy of the field in its doubled (pi-periodic) complex
representation, which is one sparse linear solve over the free faces.

The two diagonal cables are not the only lines the field has to follow.  The
shape is D4, so the x and y axes are mirror lines too, and a mirror line forces
a cross field to be either tangent or normal to it - never oblique.  With the
cables as the only constraint those two axes were left free and drifted 8-18 deg
(max 36) off, while the diagonals held to 3 deg.  The axes are therefore added
as guide bands of their own (AXIS_GUIDES): every face whose centroid lies within
GUIDE_BAND of an axis, and outside GUIDE_R_MIN of the centre, is constrained to
the axis tangent, on the same footing as the cable faces.

This is consistent, not over-constrained, precisely because the field is a CROSS
field: aligning a branch with the 45 deg diagonal puts the other branch on the
135 deg diagonal, and a radial/circumferential cross field is simultaneously
aligned with every line through the origin.  The centre is the index-1
singularity of that field, which is why GUIDE_R_MIN excludes it.

This field is what fixes knit_dir_deg per region in FDM/optimise_4part.py.  The
knit direction is NOT an optimisation variable.

    .venv/bin/python FDM/directional_field_4part.py
"""
import os, json
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA       = os.path.join(HERE, "data", "4part")
OFF_PATH   = os.path.join(DATA, "4part_tri_m.off")
CABLE_JSON = os.path.join(DATA, "cable_paths_4part.json")
OUT_JSON   = os.path.join(DATA, "directional_field_4part.json")
OUT_PNG    = os.path.join(DATA, "4part_directional_field.png")

# The cable and guide faces are hard constraints in the solve, so no weight
# is needed; the iteration count that the old Jacobi smoother required is
# gone with it.

# ── axis guides (the D4 mirror lines the cables do not cover) ────────────────
AXIS_GUIDES  = True
AXIS_DIRS    = [(1.0, 0.0), (0.0, 1.0)]   # the x and y axes, in plan
GUIDE_BAND   = 0.030   # m, half-width of the constrained band around each axis
GUIDE_R_MIN  = 0.060   # m, skip the singular neighbourhood of the centre


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


V, F = load_off(OFF_PATH)
n_v, n_f = len(V), len(F)
print(f"Mesh: {n_v} verts, {n_f} faces")

cables = list(json.load(open(CABLE_JSON)).values())
print(f"Cables: {len(cables)}  ({sum(len(c) - 1 for c in cables)} segments)")

centroids = V[F].mean(axis=1)
e1 = V[F[:, 1]] - V[F[:, 0]]
e2 = V[F[:, 2]] - V[F[:, 0]]
normals = np.cross(e1, e2)
normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)

u_frame = e1 - (e1 * normals).sum(axis=1, keepdims=True) * normals
u_frame /= np.maximum(np.linalg.norm(u_frame, axis=1, keepdims=True), 1e-10)
v_frame = np.cross(normals, u_frame)

# ── cable segment tangents (open polylines, so no wrap-around) ───────────────
p0, p1, tan = [], [], []
for path in cables:
    for a, b in zip(path[:-1], path[1:]):
        d = V[b] - V[a]
        L = np.linalg.norm(d)
        if L > 1e-10:
            p0.append(V[a]); p1.append(V[b]); tan.append(d / L)
p0, p1, tan = np.array(p0), np.array(p1), np.array(tan)
mids = 0.5 * (p0 + p1)

dists = np.linalg.norm(centroids[:, None, :] - mids[None, :, :], axis=2)
nearest = np.argmin(dists, axis=1)

d1_init = np.zeros((n_f, 3))
for fi in range(n_f):
    t = tan[nearest[fi]]
    t = t - np.dot(t, normals[fi]) * normals[fi]
    L = np.linalg.norm(t)
    d1_init[fi] = t / L if L > 1e-10 else u_frame[fi]

cable_verts = {v for c in cables for v in c}
face_on_cable = np.array([len(set(F[fi].tolist()) & cable_verts) >= 2
                          for fi in range(n_f)], dtype=bool)
print(f"Cable-adjacent faces: {face_on_cable.sum()}")

# ── axis guide bands ─────────────────────────────────────────────────────────
# A face is taken by at most one guide, and the cables always win, so the two
# constraint sets can never fight over the same face.
face_on_guide = np.zeros(n_f, dtype=bool)
if AXIS_GUIDES:
    r_plan = np.linalg.norm(centroids[:, :2], axis=1)
    for ax in AXIS_DIRS:
        a = np.array(ax, dtype=float)
        a /= np.linalg.norm(a)
        perp = np.abs(centroids[:, 0] * -a[1] + centroids[:, 1] * a[0])
        band = (perp < GUIDE_BAND) & (r_plan > GUIDE_R_MIN) & ~face_on_cable
        t3 = np.array([a[0], a[1], 0.0])
        for fi in np.flatnonzero(band):
            t = t3 - np.dot(t3, normals[fi]) * normals[fi]
            L = np.linalg.norm(t)
            if L > 1e-10:
                d1_init[fi] = t / L
                face_on_guide[fi] = True
    print(f"Axis-guide faces:     {face_on_guide.sum()}  "
          f"(band {GUIDE_BAND*1e3:.0f} mm, r > {GUIDE_R_MIN*1e3:.0f} mm)")

face_fixed = face_on_cable | face_on_guide

edge_to_faces = defaultdict(list)
for fi, tri in enumerate(F):
    for k in range(3):
        edge_to_faces[tuple(sorted((tri[k], tri[(k + 1) % 3])))].append(fi)
face_adj = defaultdict(list)
for e, fl in edge_to_faces.items():
    if len(fl) == 2:
        face_adj[fl[0]].append(fl[1]); face_adj[fl[1]].append(fl[0])

angles = np.array([np.arctan2(float(d1_init[fi] @ v_frame[fi]),
                              float(d1_init[fi] @ u_frame[fi])) for fi in range(n_f)])


def edge_transport(fi, fj, va, vb):
    """Rotation r_ij carrying face fj's frame into face fi's, about their shared
    edge (va, vb).

    The shared edge is the one direction both faces can measure, so it is the
    pivot: r = phi_i - phi_j, where phi_k is the angle from u_k to the edge,
    measured in face k's own frame.  Adding r to an angle in fj expresses it in
    fi, which in the doubled representation is a multiplication by exp(2i r).
    """
    e = V[vb] - V[va]
    out = []
    for f in (fi, fj):
        t = e - np.dot(e, normals[f]) * normals[f]
        out.append(np.arctan2(float(t @ v_frame[f]), float(t @ u_frame[f])))
    return out[0] - out[1]


# ── the field as one sparse solve ────────────────────────────────────────────
# E(z) = sum_ij w_ij |z_i - exp(2i r_ij) z_j|^2  over adjacent face pairs, with
# z_f = exp(2i theta_f) the doubled (pi-periodic) representation and w_ij the
# length of the shared edge, so the energy does not depend on how finely the
# mesh is divided.  The cable and guide faces are held at their target value, so
# the unknowns are the free faces alone and the energy is a quadratic form in
# them: L_ff z_f = -L_fc z_c.  This replaces 300 Jacobi sweeps with one exact
# solve, and removes the constraint weight, which was a soft pull whose value
# had to be chosen.
#
# L is complex Hermitian, NOT real-decoupled: exp(2i r_ij) is complex whenever
# the frames differ by something other than a multiple of 90 deg, so it mixes
# the real and imaginary parts and the system has to be solved over C.
rows, cols, vals = [], [], []
for (va, vb), fl in edge_to_faces.items():
    if len(fl) != 2:
        continue
    fi, fj = fl
    w = float(np.linalg.norm(V[vb] - V[va]))
    R = np.exp(2j * edge_transport(fi, fj, va, vb))
    rows += [fi, fj, fi, fj]
    cols += [fi, fj, fj, fi]
    vals += [w, w, -w * R, -w * np.conj(R)]
L = sp.coo_matrix((vals, (rows, cols)), shape=(n_f, n_f), dtype=complex).tocsr()

free = np.flatnonzero(~face_fixed)
fixed = np.flatnonzero(face_fixed)
z = np.exp(2j * angles)                      # constrained faces hold the target
print(f"Solving {len(free)} free faces ({len(fixed)} constrained) ...", flush=True)
rhs = -L[free][:, fixed] @ z[fixed]
z_free = spla.spsolve(L[free][:, free].tocsc(), rhs)
z[free] = z_free

# |z| carries no direction, and the Dirichlet minimiser is not unit-norm; only
# the argument is used.  A face whose neighbours disagree completely can solve to
# z ~ 0, where the angle is meaningless, so report the worst case.
mag = np.abs(z[free])
print(f"  |z| on free faces: min {mag.min():.3f}  median {np.median(mag):.3f}")
angles = np.angle(z) / 2.0

d1 = np.cos(angles)[:, None] * u_frame + np.sin(angles)[:, None] * v_frame
d2 = np.cross(normals, d1)
d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-10)

# knit_dir_deg convention: the wale direction's angle in the global xy plane,
# mod 180 (a knit direction has no sense).
knit_face = np.degrees(np.arctan2(d1[:, 1], d1[:, 0])) % 180.0
json.dump({"n_faces": int(n_f),
           "knit_dir_deg_face": knit_face.tolist(),
           "d1": d1.tolist(), "d2": d2.tolist(),
           "centroid": centroids.tolist()},
          open(OUT_JSON, "w"))
print(f"Saved {OUT_JSON}")

fig, ax = plt.subplots(figsize=(7, 7))
s = 0.018
segs = [[c[:2] - s * d[:2], c[:2] + s * d[:2]] for c, d in zip(centroids, d1)]
ax.add_collection(LineCollection(segs, colors="#2a78d6", linewidths=0.7))
for c in cables:
    p = V[c]
    ax.plot(p[:, 0], p[:, 1], color="#e34948", lw=1.8)
if AXIS_GUIDES:
    R = np.linalg.norm(V[:, :2], axis=1).max()
    for a in AXIS_DIRS:
        a = np.array(a, dtype=float) / np.linalg.norm(a)
        ax.plot([-R * a[0], R * a[0]], [-R * a[1], R * a[1]],
                color="#f2a93b", lw=1.8, zorder=1)
    ax.scatter(centroids[face_on_guide, 0], centroids[face_on_guide, 1],
               s=3, color="#f2a93b", alpha=0.45, zorder=0)
ax.set_aspect("equal"); ax.autoscale()
ax.set_title("4part directional field: wale d1 (blue), cables (red), "
             "axis guides (orange)", fontsize=10)
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
