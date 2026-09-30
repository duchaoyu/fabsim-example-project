"""Remesh pattern_smooth on the second hand-drawn cable layout.

Input: data/pattern/remesh2/cable_sketch2.png — a copy of
data/pattern/pattern_smooth_fdm_band10_result.png with three cables drawn by
hand (orange-red strokes, ~RGB 227/36/0) on the "force densities q" panel:

    C00 west    straight at x ~ -0.32, north free edge -> bottom-left edge E01
    C01 middle  north free edge at x ~ -0.13 -> junction J -> south-tip support
    C02 east    NE corner (north end of the east line support) -> J

The east stroke is drawn into the middle one at J ~ (-0.123, -0.824) and runs
with it to the tip; here it ENDS at J, a vertex it shares with C01, rather than
duplicating the trunk (two sliding cables on the same edges).

Differences from remesh_pattern_smooth_sketch.py, which read the first sketch:
  * CALIBRATION.  This panel has no support dots; it has x / y axes in metres
    and draws every mesh edge.  sx, sy, tx, ty come from the bounding box of the
    drawn mesh against the target's plan extent (independent sx, sy, since the
    panel is not guaranteed to be equal-aspect; they come out within 0.3 %),
    and 94 % of the boundary vertices then land on drawn pixels (+-1 px).
  * SPLIT.  Middle and east strokes touch, so they are one pixel component.
    They are separated by alternately fitting x = f(y) cubics to each and
    reassigning pixels to the nearer fit, starting from the clearly separate
    part above y = -0.62; J is where the two fits meet.

The meshing, the lift to 3D and the checks are those of the first script:
original 142 boundary vertices kept exactly, triangle "pq28YY", z by
barycentric interpolation of pattern_smooth_tri_m.off.

    python3 FDM/remesh_pattern_smooth_sketch2.py
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
from matplotlib.collections import PolyCollection

import optimise_2part as o2
import extract_cables_pattern_smooth as ecs
from remesh_pattern_smooth_sketch import resample, smooth, point_in_poly, MIN_ANGLE

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "pattern")
OUT = os.path.join(SRC, "remesh2")
SKETCH = os.path.join(OUT, "cable_sketch2.png")
TARGET = os.path.join(SRC, "pattern_smooth_tri_m.off")
OLD_CABLES = os.path.join(SRC, "cable_paths_pattern_smooth.json")
OLD_META = os.path.join(SRC, "cable_paths_pattern_smooth.meta.json")
PANEL = (830, 125, 1100, 395)   # the q panel, in 2000-px-wide display coordinates
TAG = "rm2"


def calibrate(im, V, F):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    k = im.shape[1] / 2000
    X0, Y0, X1, Y1 = (int(v * k) for v in PANEL)
    black = (R < 60) & (G < 60) & (B < 60)
    sub = black[Y0:Y1, X0:X1]
    rows = np.where(sub.mean(1) > 0.5)[0]
    cols = np.where(sub.mean(0) > 0.5)[0]
    frame = [int(cols.min() + X0 + 3), int(rows.min() + Y0 + 3),
             int(cols.max() + X0 - 3), int(rows.max() + Y0 - 3)]
    white = (R > 245) & (G > 245) & (B > 245)
    m = ~white & ~black
    m[:frame[1]] = m[frame[3]:] = False
    m[:, :frame[0]] = m[:, frame[2]:] = False
    yy, xx = np.nonzero(m)
    sx = (xx.max() - xx.min()) / np.ptp(V[:, 0])
    sy = (yy.max() - yy.min()) / np.ptp(V[:, 1])
    tx = xx.min() - sx * V[:, 0].min()
    ty = yy.min() + sy * V[:, 1].max()
    bd = sorted(o2.boundary_vertices(F))
    P = np.c_[tx + sx * V[bd, 0], ty - sy * V[bd, 1]].round().astype(int)
    hit = [(~white[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]).any() for x, y in P]
    return dict(sx=float(sx), sy=float(sy), tx=float(tx), ty=float(ty), frame=frame,
                boundary_hit=float(np.mean(hit)))


def bin_y(x, y, step=0.004):
    k = np.digitize(y, np.arange(y.min(), y.max() + step, step))
    pts = np.array([[x[k == b].mean(), y[k == b].mean()] for b in np.unique(k)])
    return pts[np.argsort(-pts[:, 1])]                 # north -> south


def strokes(im, cal):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    fx0, fy0, fx1, fy1 = cal["frame"]
    st = (R > 190) & (G < 80) & (B < 40)
    st[:fy0] = st[fy1:] = False
    st[:, :fx0] = st[:, fx1:] = False
    lab, _ = ndimage.label(ndimage.binary_dilation(st, iterations=2))
    lab = lab * st
    yy, xx = np.nonzero(st)
    X = (xx - cal["tx"]) / cal["sx"]
    Y = (cal["ty"] - yy) / cal["sy"]
    L = lab[yy, xx]
    ids, cnt = np.unique(L[L > 0], return_counts=True)
    if len(ids) < 2:
        raise SystemExit(f"expected 2 stroke components, got {len(ids)}")
    two = ids[np.argsort(-cnt)][:2]
    west_id = min(two, key=lambda i: X[L == i].mean())
    merged_id = [i for i in two if i != west_id][0]
    west = bin_y(X[L == west_id], Y[L == west_id])

    x, y = X[L == merged_id], Y[L == merged_id]
    mid = (np.abs(x + 0.13) < 0.03) & (y > -0.62)
    ea = (x > -0.08) & (y > -0.62)
    for _ in range(6):
        pm = np.polyfit(y[mid], x[mid], 3)
        pe = np.polyfit(y[ea], x[ea], 3)
        dm = np.abs(x - np.polyval(pm, y))
        de = np.abs(x - np.polyval(pe, y))
        mid = (dm <= de) & (y > -0.80)
        ea = (de < dm) & (y > -0.80)
    yg = np.linspace(-0.95, -0.6, 351)
    gap = np.abs(np.polyval(pm, yg) - np.polyval(pe, yg))
    yj = float(yg[gap.argmin()])
    J = np.array([np.polyval(pm, yj), yj])
    upper = bin_y(x[mid & (y > yj)], y[mid & (y > yj)])
    trunk = bin_y(x[y < yj], y[y < yj])
    east = bin_y(x[ea & (y > yj)], y[ea & (y > yj)])
    return west, upper, trunk, east, J, float(gap.min())


def main():
    V, F = o2.load_off(TARGET)
    old = json.load(open(OLD_CABLES))
    supports = sorted(json.load(open(OLD_META))["supports"])
    tip = min(supports, key=lambda k: V[k][1])
    im = np.array(Image.open(SKETCH).convert("RGB")).astype(int)

    cal = calibrate(im, V, F)
    print(f"calibration: sx {cal['sx']:.2f} sy {cal['sy']:.2f} px/m "
          f"({1000 / cal['sx']:.2f} mm/px, aspect {cal['sx'] / cal['sy']:.4f}); "
          f"{cal['boundary_hit'] * 100:.0f}% of boundary vertices on drawn pixels")
    west, upper, trunk, east, J, jgap = strokes(im, cal)
    print(f"strokes: west {len(west)}, middle {len(upper)} + trunk {len(trunk)}, "
          f"east {len(east)} bins; junction J = ({J[0]:.3f}, {J[1]:.3f}), fits meet "
          f"to {jgap * 1e3:.1f} mm")

    # boundary loop
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
    iso = [s for s in supports if not any(n in supports for n in adj[s])]
    line = [s for s in supports if s not in iso]
    ne = max(line, key=lambda k: V[k][1])
    free_b = [v for v in loop if v not in supports]
    kd = cKDTree(V[free_b][:, :2])
    snap = lambda p: free_b[kd.query(p)[1]]
    w_top, w_bot, m_top = snap(west[0]), snap(west[-1]), snap(upper[0])
    in_E01 = w_bot in old["E01"]
    print(f"anchors: west {w_top} {V[w_top][:2].round(3)} -> {w_bot} "
          f"{V[w_bot][:2].round(3)} ({'on E01' if in_E01 else 'NOT on E01'}); "
          f"middle {m_top} {V[m_top][:2].round(3)} -> tip {tip}; east NE {ne} "
          f"{V[ne][:2].round(3)} -> J")

    E_all = {tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))) for f in F for k in range(3)}
    H = float(np.mean([np.linalg.norm(V[a] - V[b]) for a, b in E_all]))

    def clean(pts, a_xy, b_xy):
        pts = pts[[point_in_poly(p, poly) for p in pts]]
        pts = pts[(np.linalg.norm(pts - a_xy, axis=1) > 0.5 * H) &
                  (np.linalg.norm(pts - b_xy, axis=1) > 0.5 * H)]
        c = resample(smooth(resample(np.vstack([a_xy, pts, b_xy]), 0.25 * H)), H)
        c[0], c[-1] = a_xy, b_xy
        return c

    c_west = clean(west, V[w_top][:2], V[w_bot][:2])
    c_up = clean(upper, V[m_top][:2], J)
    c_tr = clean(trunk, J, V[tip][:2])
    c_east = clean(east, V[ne][:2], J)

    pts2 = [tuple(p) for p in poly]
    idx_of = {v: i for i, v in enumerate(loop)}
    segs = [(i, (i + 1) % len(loop)) for i in range(len(loop))]
    j_idx = len(pts2); pts2.append(tuple(J))

    def chain(c, a_idx, b_idx):
        ids = [a_idx]
        for p in c[1:-1]:
            ids.append(len(pts2)); pts2.append(tuple(p))
        ids.append(b_idx)
        segs.extend(zip(ids[:-1], ids[1:]))
        return ids

    C00 = chain(c_west, idx_of[w_top], idx_of[w_bot])
    up = chain(c_up, idx_of[m_top], j_idx)
    tr = chain(c_tr, j_idx, idx_of[tip])
    C01 = up + tr[1:]
    C02 = chain(c_east, idx_of[ne], j_idx)

    amax = 0.5 * H * H * np.sqrt(3) / 4 * 2.0
    T = triangle.triangulate({"vertices": np.array(pts2), "segments": np.array(segs)},
                             f"pq{MIN_ANGLE}a{amax:.8f}YY")
    P2, T2 = T["vertices"], T["triangles"]
    assert np.allclose(P2[:len(pts2)], np.array(pts2)), "triangle moved input vertices"

    V3 = np.zeros((len(P2), 3)); V3[:, :2] = P2
    for i, v in enumerate(loop):
        V3[i] = V[v]
    tri2 = V[F][:, :, :2]
    kd_f = cKDTree(tri2.mean(1))
    for i in range(len(loop), len(P2)):
        p = P2[i]
        for fi in kd_f.query(p, k=12)[1]:
            a, b, c = tri2[fi]
            l1, l2 = np.linalg.solve(np.array([b - a, c - a]).T, p - a)
            if l1 >= -1e-9 and l2 >= -1e-9 and l1 + l2 <= 1 + 1e-9:
                V3[i, 2] = ((1 - l1 - l2) * V[F[fi, 0], 2] + l1 * V[F[fi, 1], 2]
                            + l2 * V[F[fi, 2], 2])
                break
        else:
            raise SystemExit(f"vertex {i} at {p} is outside the target footprint")
    n = np.cross(V3[T2[:, 1]] - V3[T2[:, 0]], V3[T2[:, 2]] - V3[T2[:, 0]])
    T2[n[:, 2] < 0] = T2[n[:, 2] < 0][:, ::-1]

    def min_angle(t):
        a, b, c = V3[t]
        out = []
        for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
            u, w = q - p, r - p
            out.append(np.degrees(np.arccos(np.clip(
                u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1))))
        return min(out)
    ang = np.array([min_angle(t) for t in T2])
    area = 0.5 * np.linalg.norm(np.cross(V3[T2[:, 1]] - V3[T2[:, 0]],
                                         V3[T2[:, 2]] - V3[T2[:, 0]]), axis=1)
    worst = np.argsort(ang)[:4]
    print(f"mesh: {len(V3)} v / {len(T2)} f; min angle {ang.min():.1f} deg, "
          f"{(ang < 15).sum()} faces < 15 deg, min area {area.min() * 1e6:.1f} mm^2; worst at "
          + ", ".join(f"({V3[T2[i]].mean(0)[0]:.2f},{V3[T2[i]].mean(0)[1]:.2f}) {ang[i]:.1f}"
                      for i in worst))

    new_of = {v: i for i, v in enumerate(loop)}
    # The middle cable is two cables, cut at the junction J where C02 meets it,
    # so the part above J and the trunk below it carry their own rest lengths.
    cab = {"C00": C00, "C01a": up, "C01b": tr, "C02": C02}
    for k in ("E00", "E01", "E02", "E03"):
        cab[k] = [new_of[v] for v in old[k]]
    # The top free-edge cable is three cables, not one: it is cut where the
    # west and middle cables meet it, so each section between anchor points
    # carries its own rest length.  E03a: NW corner -> C00, E03b: C00 -> C01,
    # E03c: C01 -> NE corner (the order along E03 is checked, not assumed).
    top = cab.pop("E03")
    cuts = sorted(top.index(p[0]) for p in (C00, C01) if p[0] in top)
    if len(cuts) != 2:
        raise SystemExit("C00 / C01 do not both start on the top edge cable E03")
    if V3[top[0]][0] > V3[top[-1]][0]:         # run W -> E so a/b/c read left to right
        top = top[::-1]
        cuts = sorted(len(top) - 1 - c for c in cuts)
    for name, (i, j) in zip(("E03a", "E03b", "E03c"),
                            ((0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], len(top) - 1))):
        cab[name] = top[i:j + 1]
    cab = {k: [int(v) for v in p] for k, p in cab.items()}
    os.makedirs(OUT, exist_ok=True)
    off = os.path.join(OUT, f"pattern_smooth_{TAG}_tri_m.off")
    with open(off, "w") as f:
        f.write(f"OFF\n{len(V3)} {len(T2)} 0\n")
        for v in V3:
            f.write("%.10f %.10f %.10f\n" % tuple(v))
        for t in T2:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
    with open(os.path.join(OUT, f"pattern_smooth_{TAG}.obj"), "w") as f:
        f.write("# pattern_smooth remeshed on the second hand-drawn cable layout\n")
        for v in V3:
            f.write("v %.10f %.10f %.10f\n" % tuple(v))
        for t in T2:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
    json.dump(cab, open(os.path.join(OUT, f"cable_paths_pattern_smooth_{TAG}.json"), "w"),
              indent=1)
    lens = {k: float(np.sum(np.linalg.norm(np.diff(V3[p], axis=0), axis=1)))
            for k, p in cab.items()}
    json.dump(dict(source=os.path.relpath(SKETCH, HERE),
                   supports=sorted(new_of[s] for s in supports), tip=int(new_of[tip]),
                   junction=int(j_idx), junction_xy=J.tolist(),
                   lengths_m=lens, edge_length_m=H, min_angle_deg=float(ang.min())),
              open(os.path.join(OUT, f"cable_paths_pattern_smooth_{TAG}.meta.json"), "w"),
              indent=1)
    json.dump(cal, open(os.path.join(OUT, "sketch_calibration.json"), "w"), indent=1)
    print("cables: " + ", ".join(f"{k} {len(p) - 1} edges {lens[k]:.3f} m"
                                 for k, p in cab.items()))

    fig, axs = plt.subplots(1, 2, figsize=(15, 6.8))
    fx0, fy0, fx1, fy1 = cal["frame"]
    crop = im[fy0:fy1, fx0:fx1].astype(np.uint8)
    ext = [(fx0 - cal["tx"]) / cal["sx"], (fx1 - cal["tx"]) / cal["sx"],
           (cal["ty"] - fy1) / cal["sy"], (cal["ty"] - fy0) / cal["sy"]]
    axs[0].imshow(crop, extent=ext, alpha=0.5)
    axs[0].plot(*np.r_[poly, poly[:1]].T, "k-", lw=0.8)
    for c, col in ((c_west, "#1f5fbf"), (np.vstack([c_up, c_tr[1:]]), "#2a9d4b"),
                   (c_east, "#8a3fbf")):
        axs[0].plot(*c.T, "-", color=col, lw=1.6)
    axs[0].plot(*J, "o", color="k", ms=4)
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
    sup_new = sorted(new_of[s] for s in supports)
    axs[1].plot(*V3[sup_new][:, :2].T, "o", color="#2a78d6", ms=3, ls="none")
    axs[1].set_title(f"remeshed: {len(V3)} v / {len(T2)} f, cables are mesh edges",
                     fontsize=10)
    for ax in axs:
        ax.set_aspect("equal"); ax.autoscale()
    fig.savefig(os.path.join(OUT, "remesh_check.png"), dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(off, HERE)} and remesh_check.png")


if __name__ == "__main__":
    main()
