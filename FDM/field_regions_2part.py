"""Directional (cross) field and region map for the 2-part (middle-crease) shape.

Two things are produced, both consumed by optimise_2part.py:

  optimisation/2part_12region_map.json
      {"face_regions": [...1100 ints...],
       "face_knit_dirs_deg": [...1100 floats...]}
      fem_batch_nregion reads face_knit_dirs_deg when its length equals nF and
      then ignores the region-level knit_dir_deg entirely.
  data/2part/directional_field_2part.json
      per-face d1 / d2 / centroid, same schema as directional_field_C5.json,
      so export_C5_field_lines.py-style OBJ export works unchanged.

THE FIELD is derived exactly as in directional_field_D5.py: d1 (wale) is the
tangent of the nearest cable edge, projected into each face's tangent plane;
faces carrying two or more cable vertices hold that direction as a weighted soft
constraint; everything else is relaxed by a complex-valued face-adjacency
Laplacian with parallel transport and pi-ambiguity resolution.  The constraint
here is the extracted cable set (data/2part/cable_paths_2part.json), which is
the crease band plus the southern fan.  KNIT DIRECTION IS NOT AN OPTIMISATION
VARIABLE: it is read off this field and frozen.

THE REGION LAYOUT follows the symmetry the mesh actually has, which was tested
rather than assumed.  Reflecting 2parts_smooth_tri_m.off about x = 0 reproduces
it to 0.000 mm; every other candidate fails badly (mirror y and the 180 deg
rotation both 78 mm, the diagonal swap 173 mm, C3/C4/C6/C8 rotations 128-173 mm).
So the group is a single mirror plane at x = 0 — the crease plane — and nothing
else.  In particular the shape is NOT symmetric front-to-back in y: the force
fan sits at the southern edge only.

The layout is therefore 12 regions = 3 bands in |x| x 2 sides in sign(x) x 2
halves in sign(y):

    band 0  |x| <  0.12   the crease trough, where the cables and all the
                          force concentration are
    band 1  0.12 - 0.40   the lobe flanks and crowns (crowns at |x| = 0.287)
    band 2  |x| >  0.40   the outer skirt running down to the springing

Mirror-x pairs a region with its reflection exactly (face counts come out
87/87, 62/62, 127/127, 124/124, 72/72, 78/78), so the optimiser can tie each
pair to one parameter set and search 6 sf pairs instead of 12.  The y split is
NOT tied, because the geometry is not y-symmetric.

    python3 FDM/field_regions_2part.py
"""
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
MESH = os.path.join(HERE, "data", "2part", "2parts_smooth_tri_m.off")
CABLES = os.path.join(HERE, "data", "2part", "cable_paths_2part.json")
OUT_MAP = os.path.join(HERE, "optimisation", "2part_12region_map.json")
OUT_FIELD = os.path.join(HERE, "data", "2part", "directional_field_2part.json")
OUT_PNG = os.path.join(HERE, "data", "2part", "2part_regions_field.png")

X_BANDS = (0.12, 0.40)     # |x| cuts, see the module docstring
N_REGIONS = 12
SMOOTH_ITERS = 300
CABLE_WEIGHT = 10.0


def load_off(path):
    L = [l for l in open(path).read().split("\n") if l.strip()]
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def region_of(cx, cy):
    band = int(np.digitize(abs(cx), X_BANDS))     # 0, 1, 2
    return band * 4 + (1 if cx >= 0 else 0) * 2 + (1 if cy >= 0 else 0)


REGION_NAMES = [f"{'crease' if b == 0 else 'lobe' if b == 1 else 'skirt'}"
                f"_{'E' if s else 'W'}{'N' if y else 'S'}"
                for b in range(3) for s in range(2) for y in range(2)]


def mirror_region(r):
    """The region that r maps to under x -> -x."""
    b, s, y = r // 4, (r % 4) // 2, r % 2
    return b * 4 + (1 - s) * 2 + y


def main():
    V, F = load_off(MESH)
    n_f = len(F)
    print(f"Mesh: {len(V)} verts, {n_f} faces")

    # ── symmetry, tested not assumed ─────────────────────────────────────────
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    cands = {"mirror x": np.diag([-1.0, 1, 1]), "mirror y": np.diag([1.0, -1, 1]),
             "rot 180": np.diag([-1.0, -1, 1]),
             "swap xy": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])}
    for k in (3, 4, 6, 8):
        a = 2 * np.pi / k
        cands[f"rot {360 // k}"] = np.array([[np.cos(a), -np.sin(a), 0],
                                             [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    print("symmetry test (max vertex deviation under the map):")
    for nm, M in cands.items():
        dev = tree.query(V @ M.T)[0].max()
        print(f"  {nm:>9}: {dev * 1000:8.3f} mm{'   <- exact' if dev < 1e-9 else ''}")

    centroids = V[F].mean(axis=1)
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    u_frame = e1 - (e1 * normals).sum(1, keepdims=True) * normals
    u_frame /= np.maximum(np.linalg.norm(u_frame, axis=1, keepdims=True), 1e-12)
    v_frame = np.cross(normals, u_frame)

    # ── cable tangents = the field's hard data ───────────────────────────────
    cables = json.load(open(CABLES))
    seg_mid, seg_tan, cable_verts = [], [], set()
    for name, path in sorted(cables.items()):
        cable_verts.update(path)
        for a, b in zip(path[:-1], path[1:]):
            d = V[b] - V[a]
            L = np.linalg.norm(d)
            if L > 1e-10:
                seg_mid.append(0.5 * (V[a] + V[b]))
                seg_tan.append(d / L)
    seg_mid, seg_tan = np.array(seg_mid), np.array(seg_tan)
    print(f"Cables: {len(cables)} polylines, {len(seg_mid)} edges, "
          f"{len(cable_verts)} distinct vertices")

    nearest = np.argmin(np.linalg.norm(centroids[:, None, :] - seg_mid[None], axis=2),
                        axis=1)
    angles = np.empty(n_f)
    for fi in range(n_f):
        t = seg_tan[nearest[fi]]
        t = t - np.dot(t, normals[fi]) * normals[fi]
        L = np.linalg.norm(t)
        t = t / L if L > 1e-10 else u_frame[fi]
        angles[fi] = np.arctan2(float(t @ v_frame[fi]), float(t @ u_frame[fi]))

    on_cable = np.array([len(set(F[fi].tolist()) & cable_verts) >= 2
                         for fi in range(n_f)])
    print(f"Cable-adjacent faces (soft-constrained): {on_cable.sum()}")

    edge_faces = defaultdict(list)
    for fi, tri in enumerate(F):
        for k in range(3):
            edge_faces[tuple(sorted((tri[k], tri[(k + 1) % 3])))].append(fi)
    adj = defaultdict(list)
    for fl in edge_faces.values():
        if len(fl) == 2:
            adj[fl[0]].append(fl[1]); adj[fl[1]].append(fl[0])

    def transport(fi, fj, ang):
        d = np.cos(ang) * u_frame[fi] + np.sin(ang) * v_frame[fi]
        d -= (d @ normals[fj]) * normals[fj]
        L = np.linalg.norm(d)
        if L < 1e-10:
            return ang
        d /= L
        return np.arctan2(float(d @ v_frame[fj]), float(d @ u_frame[fj]))

    print(f"Smoothing {SMOOTH_ITERS} iterations …")
    for it in range(SMOOTH_ITERS):
        new = angles.copy()
        for fi in range(n_f):
            zs = CABLE_WEIGHT * np.exp(1j * angles[fi]) if on_cable[fi] else 0j
            n = CABLE_WEIGHT if on_cable[fi] else 0.0
            for fj in adj[fi]:
                da = transport(fj, fi, angles[fj]) - angles[fi]
                da = (da + np.pi / 2) % np.pi - np.pi / 2   # cross field: d ~ -d
                zs += np.exp(1j * (angles[fi] + da)); n += 1.0
            if n > 0:
                new[fi] = np.angle(zs)
        angles = new
    print("Smoothing done.")

    d1 = np.cos(angles)[:, None] * u_frame + np.sin(angles)[:, None] * v_frame
    d2 = np.cross(normals, d1)
    d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-12)

    # ── the knit angle the FEM binary actually wants ─────────────────────────
    # fem_batch_nregion turns knit_dir_deg into (cos, sin, 0) and projects it
    # onto the face, so the angle is a plan azimuth, mod 180.
    face_deg = np.degrees(np.arctan2(d1[:, 1], d1[:, 0])) % 180.0

    face_region = np.array([region_of(cx, cy) for cx, cy in centroids[:, :2]])
    counts = np.bincount(face_region, minlength=N_REGIONS)
    print(f"\nRegions ({N_REGIONS}):")
    knit = np.zeros(N_REGIONS)
    spread = np.zeros(N_REGIONS)
    for r in range(N_REGIONS):
        a = np.radians(face_deg[face_region == r]) * 2.0   # doubled: mod-180 mean
        z = np.exp(1j * a).mean()
        knit[r] = np.degrees(np.angle(z)) / 2.0 % 180.0
        # circular s.d. of the doubled angle, halved back to the mod-180 angle
        spread[r] = np.degrees(np.sqrt(-2 * np.log(abs(z)))) / 2.0
        print(f"  {r:2d} {REGION_NAMES[r]:10s} {counts[r]:4d} faces   "
              f"knit {knit[r]:6.2f} deg   (within-region s.d. {spread[r]:5.2f} deg)"
              f"   mirror -> {mirror_region(r)}")
    # the mirror-pair check: face counts must match exactly
    bad = [r for r in range(N_REGIONS) if counts[r] != counts[mirror_region(r)]]
    print(f"mirror-pair face counts match: {not bad}" + (f"  (bad: {bad})" if bad else ""))

    os.makedirs(os.path.dirname(OUT_MAP), exist_ok=True)
    with open(OUT_MAP, "w") as f:
        json.dump({"face_regions": [int(r) for r in face_region],
                   "face_knit_dirs_deg": [round(float(a), 4) for a in face_deg]}, f)
    with open(OUT_FIELD, "w") as f:
        json.dump({str(fi): {"d1": d1[fi].tolist(), "d2": d2[fi].tolist(),
                             "centroid": centroids[fi].tolist(),
                             "knit_deg": float(face_deg[fi]),
                             "region": int(face_region[fi])}
                   for fi in range(n_f)}, f)
    meta = {"n_regions": N_REGIONS, "x_bands": list(X_BANDS),
            "region_names": REGION_NAMES,
            "mirror_pairs": [[r, mirror_region(r)] for r in range(N_REGIONS)],
            "face_counts": counts.tolist(),
            "knit_dir_deg": [float(k) for k in knit],
            "knit_within_region_sd_deg": [float(s) for s in spread]}
    with open(OUT_MAP.replace(".json", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT_MAP, HERE)}, "
          f"{os.path.relpath(OUT_FIELD, HERE)}")

    # ── picture ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), facecolor="#fcfcfb")
    cmap = plt.get_cmap("tab20")
    ax = axs[0]
    for fi, tri in enumerate(F):
        ax.add_patch(plt.Polygon(V[tri, :2], facecolor=cmap(face_region[fi] % 20),
                                 edgecolor="none"))
    for xc in X_BANDS:
        for s in (-1, 1):
            ax.axvline(s * xc, color="k", lw=0.7, ls="--")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_aspect("equal"); ax.autoscale(); ax.set_axis_off()
    ax.set_title(f"a  {N_REGIONS} regions = 3 |x| bands x sign(x) x sign(y)\n"
                 "mirror plane x = 0 is the only symmetry of this mesh",
                 fontsize=10, loc="left")
    for r in range(N_REGIONS):
        c = centroids[face_region == r][:, :2].mean(0)
        ax.text(*c, str(r), fontsize=8, ha="center", va="center", weight="bold")

    ax = axs[1]
    L = 0.6 * np.mean(np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1))
    ax.add_collection(LineCollection(
        [[c[:2] - 0.5 * L * d[:2], c[:2] + 0.5 * L * d[:2]]
         for c, d in zip(centroids, d1)], colors="#e34948", lw=0.9))
    ax.add_collection(LineCollection(
        [[c[:2] - 0.5 * L * d[:2], c[:2] + 0.5 * L * d[:2]]
         for c, d in zip(centroids, d2)], colors="#2a78d6", lw=0.5, alpha=0.6))
    for name, path in sorted(cables.items()):
        ax.plot(V[path, 0], V[path, 1], color="#111", lw=1.4, alpha=0.8)
    ax.set_aspect("equal"); ax.autoscale(); ax.set_axis_off()
    ax.set_title("b  the cross field: d1 wale (red), d2 course (blue)\n"
                 "constrained by the extracted cables (black)",
                 fontsize=10, loc="left")

    ax = axs[2]
    tp = ax.tripcolor(V[:, 0], V[:, 1], F, facecolors=face_deg, cmap="twilight",
                      vmin=0, vmax=180)
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.colorbar(tp, ax=ax, shrink=0.75, label="knit (wale) azimuth, deg")
    ax.set_title("c  per-face knit direction, frozen from the field\n"
                 "region values are the circular means of this",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=165, facecolor="#fcfcfb")
    print(f"wrote {os.path.relpath(OUT_PNG, HERE)}")


if __name__ == "__main__":
    main()
