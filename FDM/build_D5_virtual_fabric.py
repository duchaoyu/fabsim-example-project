"""
Close the opening of D5 with a virtual knit patch.

The opening is triangulated in plan (xy) at the shell's edge length, the
patch's xy positions are relaxed and its z is the harmonic interpolation of
the rim heights, i.e. a smooth lid on the rim.  The patch vertices are
appended after the shell's, so every shell index (rim/cable included) is
unchanged.

Output: data/D5/D5_virtual_fabric.off  and  D5_virtual_fabric.json
        ({"n_shell_verts", "n_shell_faces", "patch_faces", "patch_verts"})
"""
import json, os
import numpy as np
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

import optimise_D5_symmetric as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "D5", "D5_virtual_fabric")


def main():
    V, F = S.load_off(S.MESH)
    rim = json.load(open(S.CABLE_J))["vertex_indices"]
    nv, nf = len(V), len(F)
    R = V[rim]
    h = np.linalg.norm(np.diff(np.vstack([R, R[:1]]), axis=0), axis=1).mean()

    # interior points: triangular lattice in plan, away from the rim
    poly = Path(R[:, :2])
    lo, hi = R[:, :2].min(0), R[:, :2].max(0)
    pts = []
    for j, y in enumerate(np.arange(lo[1], hi[1], h * np.sqrt(3) / 2)):
        for x in np.arange(lo[0] + (h / 2) * (j % 2), hi[0], h):
            pts.append((x, y))
    pts = np.array(pts)
    seg_d = np.min(np.linalg.norm(pts[:, None, :] - R[None, :, :2], axis=2), axis=1)
    pts = pts[poly.contains_points(pts) & (seg_d > 0.7 * h)]

    P2 = np.vstack([R[:, :2], pts])
    tri = Delaunay(P2).simplices
    cen = P2[tri].mean(1)
    tri = tri[poly.contains_points(cen)]

    # every rim edge must appear exactly once in the patch
    m = len(rim)
    patch_e = {tuple(sorted(e)) for t in tri for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    missing = [(i, (i + 1) % m) for i in range(m) if tuple(sorted((i, (i + 1) % m))) not in patch_e]
    assert not missing, f"rim edges not recovered by the triangulation: {missing}"

    # global indices: rim -> shell index, interior -> appended
    gid = np.array(rim + list(range(nv, nv + len(pts))))
    PF = gid[tri]
    nP = len(pts)

    # Laplacian relaxation of the interior in plan, then harmonic z
    adj = [set() for _ in range(m + nP)]
    for t in tri:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            adj[a].add(b); adj[b].add(a)
    X = np.zeros((m + nP, 3)); X[:m] = R; X[m:, :2] = pts
    for _ in range(50):
        for i in range(m, m + nP):
            X[i, :2] = X[list(adj[i]), :2].mean(0)
    L = lil_matrix((nP, nP)); b = np.zeros(nP)
    for i in range(m, m + nP):
        L[i - m, i - m] = len(adj[i])
        for j in adj[i]:
            if j >= m: L[i - m, j - m] -= 1
            else: b[i - m] += X[j, 2]
    X[m:, 2] = spsolve(L.tocsr(), b)

    # orient the patch against the shell across each rim edge
    shell_dir = {}
    for t in F:
        for a, bb in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            shell_dir[(a, bb)] = True
    a, bb = rim[0], rim[1]
    for t in PF:
        cyc = [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
        if (a, bb) in cyc or (bb, a) in cyc:
            flip = (a, bb) in cyc if (a, bb) in shell_dir else (bb, a) in cyc
            break
    if flip:
        PF = PF[:, [0, 2, 1]]

    VV = np.vstack([V, X[m:]]); FF = np.vstack([F, PF])
    # closed check: all non-ground edges shared by two faces, consistently oriented
    from collections import Counter
    und = Counter(tuple(sorted(e)) for t in FF for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])))
    dirc = Counter((t[i], t[(i + 1) % 3]) for t in FF for i in range(3))
    assert all(c == 1 for c in dirc.values()), "inconsistent orientation"
    bnd = {v for e, c in und.items() if c == 1 for v in e}
    assert all(VV[v, 2] < 1e-6 for v in bnd), "boundary left outside the ground ring"

    with open(OUT + ".off", "w") as f:
        f.write(f"OFF\n{len(VV)} {len(FF)} 0\n")
        for p in VV: f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for t in FF: f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    json.dump({"n_shell_verts": nv, "n_shell_faces": nf,
               "patch_verts": list(range(nv, len(VV))),
               "patch_faces": list(range(nf, len(FF))), "rim": rim},
              open(OUT + ".json", "w"))
    e = np.vstack([PF[:, [0, 1]], PF[:, [1, 2]], PF[:, [2, 0]]])
    el = np.linalg.norm(VV[e[:, 0]] - VV[e[:, 1]], axis=1)
    print(f"patch: {nP} verts, {len(PF)} faces, edge {el.min()*1000:.0f}-{el.max()*1000:.0f} mm "
          f"(mean {el.mean()*1000:.0f}; shell {h*1000:.0f}), lid z {X[m:,2].min()*1000:.0f}-"
          f"{X[m:,2].max()*1000:.0f} mm -> {OUT}.off")


if __name__ == "__main__":
    main()
