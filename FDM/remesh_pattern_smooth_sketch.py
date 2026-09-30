"""Remesh pattern_smooth so three hand-drawn cables are mesh edges.

Input: data/pattern/remesh/cable_sketch.png — a copy of
optimisation/pattern_smooth_fixSE_fit.png on which three cables were drawn
by hand (pure orange-red strokes, ~RGB 227/39/0) in the "vertical deviation"
panel.  All three run into the south-tip support: west from the NW corner,
middle from the north free edge, east from the NE corner (the north end of
the east line support).

1.  CALIBRATION.  The panel draws the fixed vertices as #2a78d6 dots.  Their
    centroids are matched to the known fixed vertices (23 supports + the E00
    edge) by ICP on a similarity x_px = tx + s x, y_px = ty - s y.  Residual
    median ~0.4 px (1.4 mm) — see sketch_calibration.json.
2.  STROKES.  Stroke pixels are mapped to plan; the tip neighbourhood
    (r < TIP_CUT) is cut out, where the three strokes overlap, and the three
    components are ordered W -> E.  Each is averaged across its width in 4 mm
    y-bins (all three are monotone in y).
3.  CURVES.  Each curve is clipped to the plan footprint, its north end
    snapped to its boundary vertex (NW corner / nearest north-edge vertex /
    NE corner) and its south end joined straight to the tip vertex, then
    resampled at the mean target edge length H.
4.  MESH.  A constrained Delaunay triangulation (triangle, "pq28YY") of the
    ORIGINAL 142 boundary vertices plus the three curves: the boundary and
    the supports are kept exactly, and no Steiner points go on any segment,
    so every cable is a chain of mesh edges.  z comes from the target
    pattern_smooth_tri_m.off by barycentric interpolation in plan (the
    surface is a height field: every face has n_z > 0).

Writes into data/pattern/remesh/:
    pattern_smooth_rm_tri_m.off         remeshed target (FEM rest == target)
    cable_paths_pattern_smooth_rm.json  {C00 west, C01 middle, C02 east, E00-E03}
    cable_paths_pattern_smooth_rm.meta.json   supports + per-cable lengths
    sketch_calibration.json, sketch_curves.json, remesh_check.png

    python3 FDM/remesh_pattern_smooth_sketch.py
"""
import collections
import json
import os

import numpy as np
import triangle
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

import optimise_2part as o2
import extract_cables_pattern_smooth as ecs

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "pattern")
OUT = os.path.join(SRC, "remesh")
SKETCH = os.path.join(OUT, "cable_sketch.png")
TARGET = os.path.join(SRC, "pattern_smooth_tri_m.off")
OLD_CABLES = os.path.join(SRC, "cable_paths_pattern_smooth.json")
OLD_META = os.path.join(SRC, "cable_paths_pattern_smooth.meta.json")

ROI = (slice(150, 500), slice(1100, 1500))   # the vertical-deviation panel, px
TIP_CUT = 0.14      # m: strokes overlap inside this radius of the tip
MIN_ANGLE = 28      # deg, triangle quality bound (not enforced at the tip,
                    # where the input curves themselves meet at ~8-11 deg)


def calibrate(im, V, fixed):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    roi = np.zeros(R.shape, bool); roi[ROI] = True
    dot = (abs(R - 42) < 14) & (abs(G - 120) < 14) & (abs(B - 214) < 14) & roi
    lab, n = ndimage.label(dot)
    cen = np.array(ndimage.center_of_mass(dot, lab, range(1, n + 1)))
    sz = ndimage.sum(dot, lab, range(1, n + 1))
    cen = cen[sz >= 6][:, ::-1]
    P = V[fixed][:, :2]
    s = (np.ptp(cen[:, 0]) / np.ptp(P[:, 0]) + np.ptp(cen[:, 1]) / np.ptp(P[:, 1])) / 2
    tx = cen[:, 0].min() - s * P[:, 0].min()
    ty = cen[:, 1].min() + s * P[:, 1].max()
    for _ in range(30):
        proj = np.c_[tx + s * P[:, 0], ty - s * P[:, 1]]
        _, j = cKDTree(proj).query(cen)
        A = np.c_[np.r_[P[j, 0], -P[j, 1]],
                  np.r_[np.ones(len(j)), np.zeros(len(j))],
                  np.r_[np.zeros(len(j)), np.ones(len(j))]]
        s, tx, ty = np.linalg.lstsq(A, np.r_[cen[:, 0], cen[:, 1]], rcond=None)[0]
    proj = np.c_[tx + s * P[:, 0], ty - s * P[:, 1]]
    d, _ = cKDTree(proj).query(cen)
    return dict(s=float(s), tx=float(tx), ty=float(ty), n_dots=len(cen),
                residual_px_median=float(np.median(d)), residual_px_max=float(d.max()))


def strokes(im, cal, tip_xy):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    roi = np.zeros(R.shape, bool); roi[ROI] = True
    st = (R > 190) & (G < 110) & (B < 35) & roi
    yy, xx = np.nonzero(st)
    X = (xx - cal["tx"]) / cal["s"]
    Y = (cal["ty"] - yy) / cal["s"]
    cut = np.hypot(X - tip_xy[0], Y - tip_xy[1]) < TIP_CUT
    m = st.copy(); m[yy[cut], xx[cut]] = False
    lab, _ = ndimage.label(ndimage.binary_dilation(m, iterations=2))
    lab = lab * m
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    big = ids[np.argsort(-cnt)][:3]
    comps = sorted(big, key=lambda i: X[lab[yy, xx] == i].mean())
    out = []
    for i in comps:
        sel = lab[yy, xx] == i
        x, y = X[sel], Y[sel]
        k = np.digitize(y, np.arange(y.min(), y.max() + 0.004, 0.004))
        pts = np.array([[x[k == b].mean(), y[k == b].mean()] for b in np.unique(k)])
        out.append(pts[np.argsort(-pts[:, 1])])        # north -> south
    return out, int(st.sum())


def point_in_poly(p, poly):
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def resample(pts, h):
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(round(s[-1] / h)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]


def smooth(pts, it=20):
    """Light Laplacian smoothing with fixed ends: the bin-averaged stroke still
    carries pixel jitter, and a jagged constraint makes jagged triangles."""
    p = pts.copy()
    for _ in range(it):
        p[1:-1] = 0.5 * p[1:-1] + 0.25 * (p[:-2] + p[2:])
    return p


def main():
    V, F = o2.load_off(TARGET)
    old = json.load(open(OLD_CABLES))
    meta = json.load(open(OLD_META))
    supports = sorted(meta["supports"])
    fixed = sorted(set(supports) | set(old["E00"]))
    tip = min(supports, key=lambda k: V[k][1])
    im = np.array(Image.open(SKETCH).convert("RGB")).astype(int)

    cal = calibrate(im, V, fixed)
    print(f"calibration: {cal['n_dots']} dots -> {1000 / cal['s']:.2f} mm/px, "
          f"residual median {cal['residual_px_median']:.2f} px "
          f"({cal['residual_px_median'] * 1000 / cal['s']:.1f} mm), "
          f"max {cal['residual_px_max']:.2f} px")
    raw, n_px = strokes(im, cal, V[tip][:2])
    print(f"strokes: {n_px} px -> 3 curves, {[len(c) for c in raw]} bins")

    # ── boundary loop, in order ───────────────────────────────────────────────
    bE = ecs.boundary_edges([list(f) for f in F])
    adj = collections.defaultdict(list)
    for e in bE:
        a, b = tuple(e); adj[a].append(b); adj[b].append(a)
    loop, prev, cur = [tip], None, tip
    while True:
        nxt = [v for v in adj[cur] if v != prev][0]
        if nxt == tip:
            break
        loop.append(nxt); prev, cur = cur, nxt
    poly = V[loop][:, :2]
    bset = set(loop)

    # north-end anchors: NW corner support, nearest north-edge vertex, NE corner
    iso = [s for s in supports if not any(n in supports for n in adj[s])]
    nw = min((s for s in iso if s != tip), key=lambda k: V[k][0] - V[k][1])  # most NW
    line = [s for s in supports if s not in iso]
    ne = max(line, key=lambda k: V[k][1])                                   # top of the line support
    free_b = [v for v in loop if v not in supports]
    kd = cKDTree(V[free_b][:, :2])
    nm = free_b[kd.query(raw[1][0])[1]]
    anchors = [nw, nm, ne]
    print(f"anchors: tip {tip} {V[tip][:2].round(3)}  NW {nw} {V[nw][:2].round(3)}  "
          f"north-edge {nm} {V[nm][:2].round(3)} (stroke began at {raw[1][0].round(3)})  "
          f"NE {ne} {V[ne][:2].round(3)}")

    # mean target edge length
    E_all = {tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in F for k in range(3)}
    H = float(np.mean([np.linalg.norm(V[a] - V[b]) for a, b in E_all]))

    curves = []
    for pts, a in zip(raw, anchors):
        inside = np.array([point_in_poly(p, poly) for p in pts])
        pts = pts[inside]
        # drop stroke samples within H/2 of the anchor, then pin both ends
        pts = pts[np.linalg.norm(pts - V[a][:2], axis=1) > 0.5 * H]
        pts = np.vstack([V[a][:2], pts, V[tip][:2]])
        c = resample(smooth(resample(pts, 0.25 * H)), H)
        c[0], c[-1] = V[a][:2], V[tip][:2]
        curves.append(c)
    for nmx, c in zip(("west", "middle", "east"), curves):
        L = np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))
        print(f"  {nmx:6s}: {len(c)} vertices, plan length {L:.3f} m")

    # ── constrained triangulation ─────────────────────────────────────────────
    pts2 = [tuple(p) for p in poly]
    idx_of = {v: i for i, v in enumerate(loop)}
    segs = [(i, (i + 1) % len(loop)) for i in range(len(loop))]
    curve_idx = []
    for c, a in zip(curves, anchors):
        ids = [idx_of[a]]
        for p in c[1:-1]:
            ids.append(len(pts2)); pts2.append(tuple(p))
        ids.append(idx_of[tip])
        segs += list(zip(ids[:-1], ids[1:]))
        curve_idx.append(ids)
    amax = 0.5 * H * H * np.sqrt(3) / 4 * 2.0
    T = triangle.triangulate({"vertices": np.array(pts2), "segments": np.array(segs)},
                             f"pq{MIN_ANGLE}a{amax:.8f}YY")
    P2, T2 = T["vertices"], T["triangles"]
    assert np.allclose(P2[:len(pts2)], np.array(pts2)), "triangle moved input vertices"

    # ── lift to 3D from the target surface ────────────────────────────────────
    V3 = np.zeros((len(P2), 3)); V3[:, :2] = P2
    for i, v in enumerate(loop):
        V3[i] = V[v]                                       # boundary kept exactly
    tri2 = V[F][:, :, :2]
    cen = tri2.mean(1)
    kd_f = cKDTree(cen)
    for i in range(len(loop), len(P2)):
        p = P2[i]
        for fi in kd_f.query(p, k=12)[1]:
            a, b, c = tri2[fi]
            M = np.array([b - a, c - a]).T
            l1, l2 = np.linalg.solve(M, p - a)
            if l1 >= -1e-9 and l2 >= -1e-9 and l1 + l2 <= 1 + 1e-9:
                V3[i, 2] = (1 - l1 - l2) * V[F[fi, 0], 2] + l1 * V[F[fi, 1], 2] + l2 * V[F[fi, 2], 2]
                break
        else:
            raise SystemExit(f"vertex {i} at {p} is outside the target footprint")

    # orientation: every normal up, as the target
    n = np.cross(V3[T2[:, 1]] - V3[T2[:, 0]], V3[T2[:, 2]] - V3[T2[:, 0]])
    flip = n[:, 2] < 0
    T2[flip] = T2[flip][:, ::-1]

    # ── quality ───────────────────────────────────────────────────────────────
    def angles(t):
        a, b, c = V3[t]
        out = []
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            u, w = q - p, r - p
            out.append(np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
        return out
    ang = np.array([angles(t) for t in T2])
    area = 0.5 * np.linalg.norm(np.cross(V3[T2[:, 1]] - V3[T2[:, 0]], V3[T2[:, 2]] - V3[T2[:, 0]]), axis=1)
    near_tip = np.linalg.norm(V3[T2][:, :, :2].mean(1) - V[tip][:2], axis=1) < TIP_CUT + H
    print(f"mesh: {len(V3)} v / {len(T2)} f (target mesh {len(V)} v / {len(F)} f); "
          f"min angle {ang.min():.1f} deg overall, {ang[~near_tip].min():.1f} deg away from the "
          f"tip; min area {area.min()*1e6:.1f} mm^2; faces with n_z <= 0: "
          f"{int((np.cross(V3[T2[:,1]]-V3[T2[:,0]], V3[T2[:,2]]-V3[T2[:,0]])[:,2] <= 0).sum())}")

    # ── cables, supports, files ───────────────────────────────────────────────
    new_of = {v: i for i, v in enumerate(loop)}
    sup_new = sorted(new_of[s] for s in supports)
    cab = {"C00": curve_idx[0], "C01": curve_idx[1], "C02": curve_idx[2]}
    for k in ("E00", "E01", "E02", "E03"):
        cab[k] = [new_of[v] for v in old[k]]
    cab = {k: [int(v) for v in p] for k, p in cab.items()}
    os.makedirs(OUT, exist_ok=True)
    off = os.path.join(OUT, "pattern_smooth_rm_tri_m.off")
    with open(off, "w") as f:
        f.write(f"OFF\n{len(V3)} {len(T2)} 0\n")
        for v in V3:
            f.write("%.10f %.10f %.10f\n" % tuple(v))
        for t in T2:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    json.dump(cab, open(os.path.join(OUT, "cable_paths_pattern_smooth_rm.json"), "w"), indent=1)
    lens = {k: float(np.sum(np.linalg.norm(np.diff(V3[p], axis=0), axis=1))) for k, p in cab.items()}
    json.dump(dict(source=os.path.relpath(SKETCH, HERE), supports=sup_new, tip=int(new_of[tip]),
                   anchors={"C00": int(new_of[nw]), "C01": int(new_of[nm]), "C02": int(new_of[ne])},
                   lengths_m=lens, edge_length_m=H, min_angle_deg=float(ang.min())),
              open(os.path.join(OUT, "cable_paths_pattern_smooth_rm.meta.json"), "w"), indent=1)
    json.dump(cal, open(os.path.join(OUT, "sketch_calibration.json"), "w"), indent=1)
    json.dump({"raw": [c.tolist() for c in raw], "resampled": [c.tolist() for c in curves]},
              open(os.path.join(OUT, "sketch_curves.json"), "w"))
    print("cables: " + ", ".join(f"{k} {len(p)-1} edges {lens[k]:.3f} m" for k, p in cab.items()))

    # ── check picture: sketch overlay + new mesh ───────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.8))
    crop = im[ROI].astype(np.uint8)
    x0, y0 = ROI[1].start, ROI[0].start
    ext = [(x0 - cal["tx"]) / cal["s"], (x0 + crop.shape[1] - cal["tx"]) / cal["s"],
           (cal["ty"] - y0 - crop.shape[0]) / cal["s"], (cal["ty"] - y0) / cal["s"]]
    axs[0].imshow(crop, extent=ext, alpha=0.55)
    axs[0].plot(*np.r_[poly, poly[:1]].T, "k-", lw=0.8)
    for c, col in zip(curves, ("#1f5fbf", "#2a9d4b", "#8a3fbf")):
        axs[0].plot(*c.T, "-", color=col, lw=1.6)
    axs[0].set_title("your sketch (faded) with the extracted curves, in metres", fontsize=10)
    axs[1].add_collection(PolyCollection([V3[t][:, :2] for t in T2], facecolors="#f4f1ea",
                                         edgecolors="#9a968c", linewidths=0.3))
    for k, p in cab.items():
        axs[1].plot(*V3[p][:, :2].T, "-", color="#e34948" if k.startswith("C") else "#f0a0a0",
                    lw=2.0 if k.startswith("C") else 1.5)
        if k.startswith("C"):
            m = p[len(p) // 3]
            axs[1].text(*V3[m][:2], k, fontsize=9, ha="center",
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    axs[1].plot(*V3[sup_new][:, :2].T, "o", color="#2a78d6", ms=3, ls="none")
    axs[1].set_title(f"remeshed: {len(V3)} v / {len(T2)} f, cables are mesh edges", fontsize=10)
    for ax in axs:
        ax.set_aspect("equal"); ax.autoscale()
    fig.savefig(os.path.join(OUT, "remesh_check.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(off, HERE)} and remesh_check.png")


if __name__ == "__main__":
    main()
