"""
Continuous cables for the 2-part smooth shape: each cable is ONE simple path,
anchored on the boundary at BOTH ends, running from the top edge down the crease
and out to the bottom-left / bottom-right edge.

This replaces the 24 extracted chains, which are out-and-back walks with free
ends (see FDM/build_middle_Y_2part.py).  Routing is a shortest path on the mesh
edge graph with

    weight(e) = length(e) / (q(e) + EPS) ** ALPHA

so the path is pulled onto the high-q crease but is still free to leave it where
the force concentration stops.  ALPHA trades the two off: 0 gives the plain
geodesic, large values hug the q ridge at any length cost.

    .venv/bin/python FDM/route_continuous_cables_2part.py [--alpha 2.0]
"""
import os, sys, json, heapq, collections, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")

ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--congestion", type=float, default=6.0,
                help="multiply the weight of an edge already used by a previous "
                     "cable, so the cables spread across the crease instead of "
                     "collapsing onto the single cheapest ridge")
args = ap.parse_args()

OUT_JSON = os.path.join(DATA, "cable_paths_2part_continuous.json")
OUT_PNG  = os.path.join(DATA, "2part_cables_continuous.png")

L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])

# ── q per edge, from the FDM result ──────────────────────────────────────────
fdm = json.load(open(os.path.join(DATA, "mesh_out_2part_smooth_latest.json")))
g = fdm.get("data", fdm)
qe = {}
for k, d in g["edgedata"].items():
    u, v = eval(k)
    qv = d.get("qpre", 1.0)
    qe[tuple(sorted((int(u), int(v))))] = float(qv[0] if isinstance(qv, (list, tuple)) else qv)

cnt = collections.Counter()
for t in F:
    for k in range(3):
        cnt[tuple(sorted((t[k], t[(k + 1) % 3])))] += 1
edges = list(cnt)
bdry = {v for e, n in cnt.items() if n == 1 for v in e}

base_w = {}
for (u, v) in edges:
    ln = float(np.linalg.norm(V[u] - V[v]))
    base_w[(u, v)] = ln / (qe.get((u, v), 0.0) + args.eps) ** args.alpha

used = collections.Counter()


def build_adj():
    a = collections.defaultdict(list)
    for (u, v), w in base_w.items():
        ww = w * (args.congestion ** used[(u, v)])
        a[u].append((v, ww)); a[v].append((u, ww))
    return a


adj = build_adj()


def route(src, dst):
    global adj
    dist = {src: 0.0}; prev = {}; pq = [(0.0, src)]; seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if dst not in seen:
        return None
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


# top anchors near the crease, paired to the far bottom corner so the cables
# cross at the fan, the way the sketch has them
PAIRS = [("K0", 99,  494),   # top left  -> bottom right
         ("K1", 81,  464),
         ("K2", 365, 185),
         ("K3", 378, 214)]   # top right -> bottom left

cables, rows = {}, []
for name, a, b in PAIRS:
    adj = build_adj()
    p = route(a, b)
    assert p is not None, f"{name}: no path"
    for i in range(len(p) - 1):
        used[tuple(sorted((p[i], p[i + 1])))] += 1
    assert len(set(p)) == len(p), f"{name}: path revisits a vertex"
    P = V[p]
    ln = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    qs = [qe.get(tuple(sorted((p[i], p[i + 1]))), 0.0) for i in range(len(p) - 1)]
    cables[name] = [int(i) for i in p]
    rows.append((name, a, b, len(p), ln, float(np.mean(qs)), float(np.min(qs))))

print(f"alpha={args.alpha}  eps={args.eps}")
print(f"{'id':4s} {'from':>6s} {'to':>6s} {'verts':>5s} {'len_m':>7s} {'q_mean':>7s} {'q_min':>6s}  both ends on boundary")
for nm, a, b, n, ln, qm, qn in rows:
    print(f"{nm:4s} {'v%d'%a:>6s} {'v%d'%b:>6s} {n:5d} {ln:7.3f} {qm:7.3f} {qn:6.3f}  "
          f"{a in bdry and b in bdry}")
total = sum(r[4] for r in rows)
print(f"total cable length {total:.3f} m  (24-chain set: 15.6 m of walks, ~8 m of distinct path)")

json.dump(cables, open(OUT_JSON, "w"), indent=1)
print("Saved", OUT_JSON)

# ── figure ───────────────────────────────────────────────────────────────────
tri = Triangulation(V[:, 0], V[:, 1], F)
qv = np.zeros(nv); wv = np.zeros(nv)
for (u, v), q in qe.items():
    qv[u] += q; qv[v] += q; wv[u] += 1; wv[v] += 1
qv /= np.maximum(wv, 1)

COL = ["#e34948", "#f2a93b", "#2a78d6", "#3fa46a"]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))

ax = axes[0]
ax.tripcolor(tri, qv, cmap="magma_r", shading="gouraud")
ax.triplot(tri, color="w", lw=0.15, alpha=0.5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("force density q (vertex mean) — what the routing follows", fontsize=10)

ax = axes[1]
ax.triplot(tri, color="0.9", lw=0.3)
for (nm, p), c in zip(cables.items(), COL):
    P = V[p]
    ax.plot(P[:, 0], P[:, 1], color=c, lw=2.6, label=f"{nm}  {len(p)}v", zorder=3)
    ax.scatter(P[[0, -1], 0], P[[0, -1], 1], s=45, color=c, ec="k", lw=0.6, zorder=5)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.set_title(f"{len(cables)} continuous cables, {total:.2f} m\n"
             f"every end anchored on the boundary", fontsize=10)

fig.suptitle(f"2-part smooth — continuous cable routing (alpha={args.alpha})", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print("Saved", OUT_PNG)
