"""
Straight, uncrossed cables for the 2-part smooth shape, and a mesh whose edges
follow them.

Two problems with FDM/route_continuous_cables_2part.py, both visible in
data/2part/2part_cables_continuous.png:

  * the cables CROSS, because that script paired each top anchor with the
    OPPOSITE bottom corner (top-left -> bottom-right).  Pairing each top anchor
    with the bottom anchor on its OWN side removes every crossing.
  * the cables ZIGZAG, because a shortest path is confined to mesh edges and can
    only turn in the directions the triangulation offers.

So here the route is only a guide.  It is smoothed with a spline (endpoints
pinned to their boundary anchors), the left pair is built as the exact mirror of
the right pair, and the result is then used as a CONSTRAINT in a fresh
triangulation, so the new mesh has real edges along each cable instead of a
staircase near it.

Pipeline: route on q -> smooth -> mirror -> triangle PSLG -> lift z from the
target surface -> write .off + cable vertex paths.

    .venv/bin/python FDM/remesh_2part_cables.py [--smooth 0.02] [--target-edge 0.045]
"""
import os, json, heapq, collections, argparse
import numpy as np
from scipy.interpolate import splprep, splev, LinearNDInterpolator
import triangle as tr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--smooth", type=float, default=0.02, help="spline smoothing factor")
ap.add_argument("--target-edge", type=float, default=0.045, help="target edge length, m")
args = ap.parse_args()

OUT_OFF   = os.path.join(DATA, "2part_cablemesh.off")
OUT_JSON  = os.path.join(DATA, "cable_paths_2part_cablemesh.json")
OUT_PNG   = os.path.join(DATA, "2part_cablemesh.png")

# ── the original surface ─────────────────────────────────────────────────────
L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])

fdm = json.load(open(os.path.join(DATA, "mesh_out_2part_smooth_latest.json")))
gg = fdm.get("data", fdm)
qe = {}
for k, d in gg["edgedata"].items():
    u, v = eval(k)
    qv = d.get("qpre", 1.0)
    qe[tuple(sorted((int(u), int(v))))] = float(qv[0] if isinstance(qv, (list, tuple)) else qv)

cnt = collections.Counter()
for t in F:
    for k in range(3):
        cnt[tuple(sorted((t[k], t[(k + 1) % 3])))] += 1
bdry_edges = [e for e, n in cnt.items() if n == 1]
bdry = {v for e in bdry_edges for v in e}

# ordered boundary loop
nxt = collections.defaultdict(list)
for u, v in bdry_edges:
    nxt[u].append(v); nxt[v].append(u)
loop = [min(bdry)]
while True:
    cand = [w for w in nxt[loop[-1]] if len(loop) < 2 or w != loop[-2]]
    if not cand or cand[0] == loop[0]:
        break
    loop.append(cand[0])
print(f"boundary loop: {len(loop)} of {len(bdry)} vertices")

# The right pair is routed inside x >= X_KEEPOUT only.  Its mirror then lives in
# x <= -X_KEEPOUT, so the two halves cannot meet, let alone cross.  Routing on the
# full mesh and mirroring afterwards is what produced the 4 crossings: the routed
# curve wanders across x = 0 in the crease and its mirror wanders back.
X_KEEPOUT = 0.015

adj = collections.defaultdict(list)
for (u, v) in cnt:
    if V[u, 0] < X_KEEPOUT or V[v, 0] < X_KEEPOUT:
        continue
    ln = float(np.linalg.norm(V[u] - V[v]))
    w = ln / (qe.get((u, v), 0.0) + args.eps) ** args.alpha
    adj[u].append((v, w)); adj[v].append((u, w))


def route(src, dst):
    dist = {src: 0.0}; prev = {}; pq = [(0.0, src)]; seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w in adj[u]:
            if d + w < dist.get(v, np.inf):
                dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


# ── right-hand pair only; the left pair is its mirror ────────────────────────
# Only ONE curve is routed.  The second right-hand cable is a lateral OFFSET of
# it, because two independently routed cables both chase the same q ridge and
# swap order along it - that is what produced the 4 crossings, and crossing
# segments are invalid PSLG input, which is why triangle then refined to 11k
# vertices.  An offset of a simple curve cannot cross its parent.
SPINE = ("R0", 365, 464)
OFFSET = 0.055          # m, lateral spacing between the two right-hand cables


def smooth_curve(idx):
    P = V[idx][:, :2]
    # spline through the routed points, ends pinned
    tck, _ = splprep([P[:, 0], P[:, 1]], s=args.smooth, k=3)
    n = max(12, int(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / args.target_edge))
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    C = np.column_stack([x, y])
    C[0], C[-1] = P[0], P[-1]          # exact anchors
    C[:, 0] = np.maximum(C[:, 0], X_KEEPOUT)   # the spline may undercut the keep-out
    C[0], C[-1] = P[0], P[-1]
    return C


def offset_curve(C, d, bpts):
    """Offset C by d along its plan normal, then snap the ends back onto the
    boundary so the cable stays anchored."""
    t = np.gradient(C, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    O = C + d * n
    for k in (0, -1):
        O[k] = bpts[np.argmin(np.linalg.norm(bpts - O[k], axis=1))]
    O[:, 0] = np.maximum(O[:, 0], X_KEEPOUT)
    return O


bpts = V[loop][:, :2]
nm, a, b = SPINE
C0 = smooth_curve(route(a, b))
C1 = offset_curve(C0, OFFSET, bpts)

curves = {"R0": C0, "R1": C1}
for k in ("R0", "R1"):
    M = curves[k].copy(); M[:, 0] *= -1.0      # exact mirror
    curves[k.replace("R", "L")] = M

# straightness / smoothness report
def turning(C):
    d = np.diff(C, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1, 1)))
    return ang


print(f"\n{'cable':6s} {'pts':>4s} {'len_m':>7s} {'chord':>7s} {'len/chord':>9s} "
      f"{'turn mean':>9s} {'turn max':>8s}")
for nm, C in curves.items():
    ln = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    ch = float(np.linalg.norm(C[-1] - C[0]))
    t = turning(C)
    print(f"{nm:6s} {len(C):4d} {ln:7.3f} {ch:7.3f} {ln/ch:9.3f} "
          f"{t.mean():9.2f} {t.max():8.2f}")

# crossing check
def segs_cross(p, q, r, s):
    d1 = np.cross(s - r, p - r); d2 = np.cross(s - r, q - r)
    d3 = np.cross(q - p, r - p); d4 = np.cross(q - p, s - p)
    return ((d1 * d2 < 0) and (d3 * d4 < 0))


names = list(curves)
nx = 0
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        A, B = curves[names[i]], curves[names[j]]
        for a in range(len(A) - 1):
            for b in range(len(B) - 1):
                if segs_cross(A[a], A[a + 1], B[b], B[b + 1]):
                    nx += 1
print(f"\ncable-cable crossings: {nx}")

# ── PSLG and triangulation ───────────────────────────────────────────────────
pts, segs = [], []


def add_pt(p, tol=1e-7):
    for i, q in enumerate(pts):
        if abs(q[0] - p[0]) < tol and abs(q[1] - p[1]) < tol:
            return i
    pts.append([float(p[0]), float(p[1])])
    return len(pts) - 1


marks = []
bl = [add_pt(V[i, :2]) for i in loop]
for i in range(len(bl)):
    segs.append([bl[i], bl[(i + 1) % len(bl)]]); marks.append(1)

CABLE_MARK = {nm: 10 + i for i, nm in enumerate(sorted(curves))}
for nm, C in curves.items():
    ids = [add_pt(p) for p in C]
    for i in range(len(ids) - 1):
        segs.append([ids[i], ids[i + 1]]); marks.append(CABLE_MARK[nm])

A = {"vertices": np.array(pts), "segments": np.array(segs),
     "segment_markers": np.array(marks)}
max_area = (args.target_edge ** 2) * np.sqrt(3) / 4.0
out = tr.triangulate(A, f"pq32a{max_area:.8f}")
P2, T = out["vertices"], out["triangles"]
out_seg = out["segments"]; out_mark = out["segment_markers"].ravel()
print(f"\nremeshed: {len(P2)} vertices, {len(T)} faces "
      f"(original 581 v / 1100 f), target edge {args.target_edge} m")

# ── lift z from the target surface ───────────────────────────────────────────
interp = LinearNDInterpolator(V[:, :2], V[:, 2])
z = interp(P2[:, 0], P2[:, 1])
if np.isnan(z).any():
    from scipy.interpolate import NearestNDInterpolator
    nn = NearestNDInterpolator(V[:, :2], V[:, 2])
    bad = np.isnan(z)
    z[bad] = nn(P2[bad, 0], P2[bad, 1])
    print(f"  {bad.sum()} vertices outside the convex hull, filled by nearest")
V2 = np.column_stack([P2, z])

# cable vertex indices in the NEW mesh
# Triangle splits each constrained segment and carries the marker onto every
# subsegment, so the cable path is recovered by chaining the subsegments with a
# given marker - not by matching coordinates, which misses the Steiner points
# triangle inserts along the way.
cables_new = {}
for nm, mk in CABLE_MARK.items():
    sub = [tuple(int(x) for x in out_seg[i]) for i in range(len(out_seg))
           if out_mark[i] == mk]
    nb = collections.defaultdict(list)
    for u, v in sub:
        nb[u].append(v); nb[v].append(u)
    ends = [v for v, l in nb.items() if len(l) == 1]
    assert len(ends) == 2, f"{nm}: {len(ends)} endpoints, not a simple path"
    start = min(ends, key=lambda v: -P2[v, 1])      # begin at the top anchor
    path = [start]
    while True:
        nxts = [w for w in nb[path[-1]] if len(path) < 2 or w != path[-2]]
        if not nxts:
            break
        path.append(nxts[0])
    assert len(path) == len(sub) + 1, f"{nm}: chain {len(path)} vs {len(sub)+1}"
    cables_new[nm] = [int(i) for i in path]

# do the cable vertices actually form mesh edges now?
E2 = set()
for t in T:
    for k in range(3):
        E2.add(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))))
tot = miss = 0
for nm, p in cables_new.items():
    for i in range(len(p) - 1):
        tot += 1
        if tuple(sorted((p[i], p[i + 1]))) not in E2:
            miss += 1
print(f"cable segments that are real mesh edges: {tot - miss}/{tot}")

with open(OUT_OFF, "w") as f:
    f.write("OFF\n%d %d 0\n" % (len(V2), len(T)))
    for p in V2:
        f.write("%.17g %.17g %.17g\n" % tuple(p))
    for t in T:
        f.write("3 %d %d %d\n" % tuple(int(x) for x in t))
json.dump(cables_new, open(OUT_JSON, "w"), indent=1)
print(f"Saved {OUT_OFF}\nSaved {OUT_JSON}")

# ── figure ───────────────────────────────────────────────────────────────────
COL = {"R0": "#e34948", "R1": "#f2a93b", "L0": "#2a78d6", "L1": "#3fa46a"}
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

ax = axes[0]
ax.triplot(V[:, 0], V[:, 1], F, color="0.85", lw=0.3)
old = json.load(open(os.path.join(DATA, "cable_paths_2part_continuous.json")))
for p in old.values():
    ax.plot(V[p][:, 0], V[p][:, 1], color="#c0392b", lw=2.0, alpha=0.85)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("before — routed on mesh edges\ncrossed and zigzag", fontsize=10)

ax = axes[1]
ax.triplot(V[:, 0], V[:, 1], F, color="0.9", lw=0.3)
for nm, C in curves.items():
    ax.plot(C[:, 0], C[:, 1], color=COL[nm], lw=2.6, label=nm)
    ax.scatter(C[[0, -1], 0], C[[0, -1], 1], s=40, color=COL[nm], ec="k", lw=0.6, zorder=5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=8, loc="lower left")
ax.set_title(f"after — smoothed, mirrored, same-side pairing\n"
             f"{nx} crossings", fontsize=10)

ax = axes[2]
ax.triplot(V2[:, 0], V2[:, 1], T, color="0.82", lw=0.35)
for nm, p in cables_new.items():
    ax.plot(V2[p][:, 0], V2[p][:, 1], color=COL[nm], lw=2.6)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"the conforming remesh\n{len(V2)} v / {len(T)} f, "
             f"cables are mesh edges", fontsize=10)

fig.suptitle("2-part smooth — straightened cables and a mesh built around them",
             fontsize=13, y=1.04)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
