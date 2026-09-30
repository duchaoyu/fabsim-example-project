"""Directional (cross) field and 4-region map for pattern_smooth.

Both are consumed by FDM/optimise_pattern_smooth.py:

  optimisation/pattern_smooth_4region_map.json
      {"face_regions": [...nF ints...], "face_knit_dirs_deg": [...nF floats...]}
      fem_batch_nregion reads face_knit_dirs_deg when its length equals nF.
  data/pattern/directional_field_pattern_smooth.json
      per-face d1 / d2 / centroid, same schema as directional_field_4part.json

THE FIELD is the construction of directional_field_4part.py: d1 (wale) is the
tangent of the nearest cable segment, projected into the face; faces carrying
two or more cable vertices are held at that direction, and every other face is
the minimiser of the transported Dirichlet energy of the doubled complex field,
one sparse solve.  The constraint is ALL seven cables of
cable_paths_pattern_smooth.json — the four free-edge cables as well as the three
interior ones, since the free edges are where the highest force densities are.
There are no axis guides: the shape has no mirror line.  KNIT DIRECTION IS NOT
AN OPTIMISATION VARIABLE: it is read off this field and frozen.

THE REGIONS are the four patches the interior cables cut the surface into:
faces are flood-filled across every interior edge except a cable edge.  C00
(the crease) splits west from east; C02 splits the west side, C01 the east.

    python3 FDM/field_regions_pattern_smooth.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pattern")
OFF_PATH = os.path.join(DATA, "pattern_smooth_tri_m.off")
CABLE_JSON = os.path.join(DATA, "cable_paths_pattern_smooth.json")
OUT_FIELD = os.path.join(DATA, "directional_field_pattern_smooth.json")
OUT_MAP = os.path.join(HERE, "optimisation", "pattern_smooth_4region_map.json")
OUT_PNG = os.path.join(DATA, "pattern_smooth_field_regions.png")


def load_off(path):
    with open(path) as f:
        lines = [l for l in f if l.strip() and not l.startswith("#")]
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def main():
    V, F = load_off(OFF_PATH)
    n_f = len(F)
    cab = json.load(open(CABLE_JSON))
    cables = [cab[k] for k in sorted(cab)]
    interior = [cab[k] for k in sorted(cab) if k.startswith("C")]
    print(f"Mesh: {len(V)} verts, {n_f} faces;  cables {sorted(cab)}")

    centroids = V[F].mean(axis=1)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    u_frame = e1 - (e1 * normals).sum(axis=1, keepdims=True) * normals
    u_frame /= np.maximum(np.linalg.norm(u_frame, axis=1, keepdims=True), 1e-10)
    v_frame = np.cross(normals, u_frame)

    # ── the field ─────────────────────────────────────────────────────────────
    p0, p1 = [], []
    for path in cables:
        for a, b in zip(path[:-1], path[1:]):
            p0.append(V[a]); p1.append(V[b])
    p0, p1 = np.array(p0), np.array(p1)
    tan = (p1 - p0) / np.linalg.norm(p1 - p0, axis=1, keepdims=True)
    mids = 0.5 * (p0 + p1)
    nearest = np.argmin(np.linalg.norm(centroids[:, None] - mids[None], axis=2), axis=1)
    d1_init = tan[nearest] - (tan[nearest] * normals).sum(1, keepdims=True) * normals
    d1_init /= np.linalg.norm(d1_init, axis=1, keepdims=True)

    cable_verts = {v for c in cables for v in c}
    fixed_mask = np.array([len(set(f.tolist()) & cable_verts) >= 2 for f in F])
    print(f"Cable-adjacent (constrained) faces: {fixed_mask.sum()}")

    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(F):
        for k in range(3):
            edge_to_faces[tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))].append(fi)

    angles = np.arctan2((d1_init * v_frame).sum(1), (d1_init * u_frame).sum(1))

    def transport(fi, fj, va, vb):
        e = V[vb] - V[va]
        out = []
        for f in (fi, fj):
            t = e - np.dot(e, normals[f]) * normals[f]
            out.append(np.arctan2(float(t @ v_frame[f]), float(t @ u_frame[f])))
        return out[0] - out[1]

    rows, cols, vals = [], [], []
    for (va, vb), fl in edge_to_faces.items():
        if len(fl) != 2:
            continue
        fi, fj = fl
        w = float(np.linalg.norm(V[vb] - V[va]))
        R = np.exp(2j * transport(fi, fj, va, vb))
        rows += [fi, fj, fi, fj]
        cols += [fi, fj, fj, fi]
        vals += [w, w, -w * R, -w * np.conj(R)]
    L = sp.coo_matrix((vals, (rows, cols)), shape=(n_f, n_f), dtype=complex).tocsr()
    free = np.flatnonzero(~fixed_mask)
    fixed = np.flatnonzero(fixed_mask)
    z = np.exp(2j * angles)
    z[free] = spla.spsolve(L[free][:, free].tocsc(), -L[free][:, fixed] @ z[fixed])
    mag = np.abs(z[free])
    print(f"  |z| on free faces: min {mag.min():.3f}  median {np.median(mag):.3f}")
    angles = np.angle(z) / 2.0
    d1 = np.cos(angles)[:, None] * u_frame + np.sin(angles)[:, None] * v_frame
    d2 = np.cross(normals, d1)
    d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-10)
    knit_face = np.degrees(np.arctan2(d1[:, 1], d1[:, 0])) % 180.0

    # ── the regions: flood fill, cable edges are walls ────────────────────────
    walls = {frozenset(e) for p in interior for e in zip(p[:-1], p[1:])}
    adj = defaultdict(list)
    for e, fl in edge_to_faces.items():
        if len(fl) == 2 and frozenset(e) not in walls:
            adj[fl[0]].append(fl[1]); adj[fl[1]].append(fl[0])
    region = -np.ones(n_f, int)
    comps = []
    for f0 in range(n_f):
        if region[f0] >= 0:
            continue
        stack, comp = [f0], []
        region[f0] = len(comps)
        while stack:
            f = stack.pop(); comp.append(f)
            for g in adj[f]:
                if region[g] < 0:
                    region[g] = region[f0]; stack.append(g)
        comps.append(comp)
    # order the regions W -> E by centroid x, so the numbering is stable
    order = np.argsort([centroids[c, 0].mean() for c in comps])
    remap = np.empty(len(comps), int); remap[order] = np.arange(len(comps))
    region = remap[region]
    counts = np.bincount(region)
    print(f"Regions: {len(comps)}, face counts {counts.tolist()}")
    if len(comps) != 4:
        raise SystemExit(f"expected 4 regions, got {len(comps)} — the interior "
                         f"cables do not close the patches")
    for r in range(4):
        m = region == r
        zc = np.exp(2j * np.radians(knit_face[m])).mean()
        print(f"  R{r}: {m.sum():4d} faces, centroid "
              f"({centroids[m, 0].mean():+.3f}, {centroids[m, 1].mean():+.3f}), "
              f"knit mean {np.degrees(np.angle(zc)) / 2 % 180:6.1f} deg, "
              f"coherence |<z>| {abs(zc):.2f}")

    os.makedirs(os.path.dirname(OUT_MAP), exist_ok=True)
    json.dump({"face_regions": region.tolist(),
               "face_knit_dirs_deg": knit_face.tolist()}, open(OUT_MAP, "w"))
    json.dump({"n_faces": int(n_f), "knit_dir_deg_face": knit_face.tolist(),
               "d1": d1.tolist(), "d2": d2.tolist(), "centroid": centroids.tolist()},
              open(OUT_FIELD, "w"))
    print(f"Saved {os.path.relpath(OUT_MAP, HERE)}\n      {os.path.relpath(OUT_FIELD, HERE)}")

    # ── picture ───────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    cols4 = ["#8fb8e8", "#f2c28b", "#a8d5a2", "#d9a8d6"]
    for ax in axs:
        ax.set_aspect("equal"); ax.set_axis_off()
    axs[0].add_collection(PolyCollection([V[f][:, :2] for f in F],
                                         facecolors=[cols4[r] for r in region],
                                         edgecolors="white", linewidths=0.2))
    for r in range(4):
        m = region == r
        axs[0].text(centroids[m, 0].mean(), centroids[m, 1].mean(),
                    f"R{r}\n{m.sum()} f", ha="center", va="center", fontsize=10)
    s = 0.014
    axs[1].add_collection(LineCollection(
        [[c[:2] - s * d[:2], c[:2] + s * d[:2]] for c, d in zip(centroids, d1)],
        colors="#2a78d6", linewidths=0.7))
    for ax in axs:
        for k in sorted(cab):
            p = V[cab[k]]
            ax.plot(p[:, 0], p[:, 1], color="#e34948" if k.startswith("C") else "#f0a0a0",
                    lw=2.0)
        ax.autoscale()
    axs[0].set_title("4 regions cut by the interior cables", fontsize=11)
    axs[1].set_title("wale direction d1 (blue), fixed from the cable-guided field",
                     fontsize=11)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"Saved {os.path.relpath(OUT_PNG, HERE)}")


if __name__ == "__main__":
    main()
