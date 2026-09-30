"""Extract cable trajectories from the pattern_smooth FDM result (band10 run).

Reuses the routing in figure_cable_extraction.py — the mesh read as a weighted
graph with

    w_e = (1 / q_e^2) * (1 + lambda * sin^2(theta_e))

routed with Dijkstra over DIRECTED edges (THETA_MODE = "hybrid") — as
extract_cables_2part.py does, and writes the cables out as vertex-index
polylines for the FEM optimiser.

What differs from the 2-part shape is the support condition, and it changes
how the cables are found.

1.  Supports are NOT the whole perimeter.  Only the z = 0 vertices are anchored
    (fofin_pattern_smooth.py): three point supports (NW corner, SW corner,
    south tip) and one line support along the east edge.  The rest of the
    boundary is a free edge, and a free edge in an FDM net under pressure IS a
    cable — it carries the highest q in the net (p50 2.6, up to 12).  So the
    edge cables are taken topologically, not by threshold: every run of free
    boundary edges between two supports is one cable.  Thresholding them would
    only break them where q dips (the west edge drops below 2 near y = -0.35).

2.  The interior cables are routed on the INTERIOR edges only.  The boundary
    edges are removed from the routing graph so they cannot enter the
    high-force set, and a route terminates on any support or edge-cable vertex
    at cost ANCHOR_COST — force handed to an edge cable reaches a support.
    The routing itself is run() below, not fce.extract: see its docstring for
    why (chain ends already on the boundary, and TIES_OUTWARD).

Q_THRESHOLD is a fixed q on the interior edges.  Interior q runs 0.01 - 12.2
with a median of 0.07; the p95 used on the 2-part net (0.73) picks up a dozen
scattered 1-3 edge fragments across the lobes.  The --scan over q gives one
wide plateau, q = 1.4 - 2.1, on which the retained set (70 edges) is identical
at lambda = 5, 10 and 20; mu = 1 - 20 does not change it either.
Q_THRESHOLD = 2.0 sits inside it.

What comes out: the crease (C00, q up to 12) from the south-tip support to the
north edge cable, and two weaker ties routed through q ~ 1 - 2 edges — one
from the NE line support diagonally into the crease (C01 + its 3-edge stub
C03), one parallel to the crease on its west side (C02).

    python3 FDM/extract_cables_pattern_smooth.py            # extract + write + plot
    python3 FDM/extract_cables_pattern_smooth.py --scan     # the q / lambda scan
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_cable_extraction as fce
import extract_cables_2part as x2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pattern")
RESULT = os.path.join(DATA, "mesh_out_pattern_smooth_band10_latest.json")
OUTJ = os.path.join(DATA, "cable_paths_pattern_smooth.json")
OUTM = os.path.join(DATA, "cable_paths_pattern_smooth.meta.json")
OUTP = os.path.join(DATA, "pattern_smooth_cables.png")

LAMBDA = 10.0
ANCHOR_COST = 5.0    # = LAMBDA / 2, the minimum the coupling rule allows
Q_THRESHOLD = 2.0    # interior edges only, see the module docstring
MIN_CABLE_EDGES = 3  # a 1-2 edge stub is not a cable worth detailing


def load():
    x2.RESULT = RESULT
    return x2.load()


def boundary_edges(F):
    seen = collections.Counter()
    for f in F:
        for a, b in zip(f, f[1:] + f[:1]):
            seen[frozenset((a, b))] += 1
    return {e for e, c in seen.items() if c == 1}


def edge_cables(V, bE, supports):
    """Split the boundary loop into runs of free edges between supports."""
    adj = collections.defaultdict(list)
    for e in bE:
        a, b = tuple(e)
        adj[a].append(b)
        adj[b].append(a)
    # walk the loop once, starting on a support
    start = min(supports & set(adj))
    loop, prev, cur = [start], None, start
    while True:
        nxt = [v for v in adj[cur] if v != prev][0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt
    k = next(i for i, v in enumerate(loop)
             if v in supports and loop[(i + 1) % len(loop)] not in supports)
    loop = loop[k:] + loop[:k] + [loop[k]]     # rotate: begin as a free run leaves a support
    runs, cur = [], None
    for a, b in zip(loop[:-1], loop[1:]):
        if a in supports and b in supports:    # an edge of the line support
            continue
        if cur is None:
            cur = [a]
        cur.append(b)
        if b in supports:
            runs.append(cur)
            cur = None
    return runs


def run(V, E_in, s2, term, lam, mu, qth, apex):
    """fce.extract, with the two changes this support layout needs.

    * A chain end already on a support or an edge cable is not routed — it is
      terminated.  fce.extract routes every end, and an end on the boundary
      then walks back along its own chain to the far termination, doubling it.
    * No TIES_OUTWARD.  "Outward" is distance from the apex, and here the apex
      is the east crown, so the rule sends the crease ends and the SW corner
      chain the long way round the net.
    Routes run on the interior edges and may not use the chain's own edges;
    they end on the boundary (cost mu) or on an already-retained interior
    cable (free), strongest chain first as in fce.extract.
    """
    q_of = {frozenset((u, w)): qq for u, w, qq in E_in}
    mesh_adj = collections.defaultdict(set)
    for u, w, _ in E_in:
        mesh_adj[u].add(w)
        mesh_adj[w].add(u)
    hi = [e for e in E_in if e[2] >= qth]
    hi_edges = {frozenset((u, w)) for u, w, _ in hi}
    hi_comps, hi_adj = fce.graph_components(hi_edges)
    order = sorted(hi_comps, key=lambda c: -max(
        qq for u, w, qq in hi if u in c and w in c))
    routes, tip_routes, cable_v = set(), {}, set()
    for comp in order:
        own = {e for e in hi_edges if tuple(e)[0] in comp}
        adj = {v: {n for n in nb if frozenset((v, n)) not in own}
               for v, nb in mesh_adj.items()}
        adj = collections.defaultdict(set, adj)
        prices = {b: mu for b in term}
        prices.update({v: 0.0 for v in cable_v})
        grown = set()
        for t in [v for v in comp if len(hi_adj[v]) == 1] or sorted(comp)[:1]:
            if t in term:
                continue
            walk = fce.turn_route(V, adj, q_of, s2, lam, t, prices) or [t]
            for a, b in zip(walk[:-1], walk[1:]):
                routes.add(frozenset((a, b)))
            grown.update(walk)
            tip_routes[t] = walk
        cable_v |= comp | grown
    edges = hi_edges | routes
    comps, _ = fce.graph_components(edges)
    return dict(edges=edges, comps=comps, hi=hi, hi_edges=hi_edges,
                hi_comps=hi_comps, routes=routes, tip_routes=tip_routes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--mu", type=float, default=ANCHOR_COST)
    ap.add_argument("--q", type=float, default=Q_THRESHOLD)
    args = ap.parse_args()

    V, F, E = load()
    bE = boundary_edges(F)
    supports = {k for k in V if abs(V[k][2]) < 1e-9}
    q_of = {frozenset((u, w)): qq for u, w, qq in E}
    E_in = [e for e in E if frozenset(e[:2]) not in bE]
    apex = fce.apex_point(V)
    s2 = {frozenset((u, w)): fce.sin2_theta(V, apex, u, w) for u, w, _ in E}

    edges_c = edge_cables(V, bE, supports)
    term = supports | {v for c in edges_c for v in c}
    q_in = np.array([e[2] for e in E_in])
    print(f"{len(V)} vertices  {len(F)} faces  {len(E)} edges  "
          f"{len(bE)} boundary edges  {len(supports)} supports")
    print(f"interior q: min {q_in.min():.4f}  median {np.median(q_in):.4f}  "
          f"max {q_in.max():.4f}  -> q >= {args.q:g}: {(q_in >= args.q).sum()} edges")
    print(f"apex (xy) = ({apex[0]:+.4f}, {apex[1]:+.4f})")

    if args.scan:
        for lam in (5.0, 10.0, 20.0):
            prev = None
            print(f"\nq plateaux at lambda = {lam:g}, mu = {lam / 2:g}:")
            for qth in np.round(np.arange(1.0, 4.01, 0.1), 2):
                r = run(V, E_in, s2, term, lam, lam / 2, qth, apex)
                s = x2._sig(r)
                if s != prev:
                    print(f"  q >= {qth:.1f}  {len(r['hi']):3d} hi edges  "
                          f"{len(r['hi_comps'])} chains  {len(r['edges'])} retained  "
                          f"[{s}]")
                    prev = s
        return

    res = run(V, E_in, s2, term, args.lam, args.mu, args.q, apex)
    ends_in = sorted({v for e in res["edges"] for v in tuple(e)} & term)
    print(f"\nlambda={args.lam:g}  mu={args.mu:g}: {len(res['hi'])} high-force "
          f"interior edges in {len(res['hi_comps'])} chains -> "
          f"{len(res['edges'])} retained edges, {len(res['comps'])} components, "
          f"{len(ends_in)} terminations on the boundary")
    inner = fce.trace_cables(V, res["edges"], q_of, res["hi_edges"],
                             res["tip_routes"])
    inner = [c for c in inner if c["edges"] >= MIN_CABLE_EDGES]

    def describe(path):
        qs = [q_of[frozenset((a, b))] for a, b in zip(path[:-1], path[1:])]
        return dict(path=[int(v) for v in path], edges=len(path) - 1,
                    q_min=min(qs), q_max=max(qs), q_mean=float(np.mean(qs)),
                    length=float(sum(np.linalg.norm(V[b] - V[a])
                                     for a, b in zip(path[:-1], path[1:]))),
                    closed=False)

    cables = ([("E%02d" % i, "edge", describe(p)) for i, p in enumerate(edges_c)]
              + [("C%02d" % i, "interior", c) for i, c in enumerate(inner)])

    def end_kind(v):
        return "support" if v in supports else ("edge" if v in term else "cable")

    print(f"{'name':>5} {'kind':>8} {'edges':>5} {'length':>8}  {'q range':>12}  ends")
    for nm, kind, c in cables:
        a, b = c["path"][0], c["path"][-1]
        print(f"{nm:>5} {kind:>8} {c['edges']:>5} {c['length']:>7.3f} m  "
              f"{c['q_min']:>5.2f}-{c['q_max']:<6.2f} {end_kind(a)} -> {end_kind(b)}")

    with open(OUTJ, "w") as f:
        json.dump({nm: c["path"] for nm, _, c in cables}, f, indent=1)
    with open(OUTM, "w") as f:
        json.dump(dict(source=os.path.relpath(RESULT, HERE), lam=args.lam,
                       anchor_cost=args.mu, q_threshold=args.q,
                       apex_xy=[float(apex[0]), float(apex[1])],
                       supports=sorted(int(v) for v in supports),
                       cables=[dict(name=nm, kind=kind, **c)
                               for nm, kind, c in cables]), f, indent=1)
    print(f"wrote {os.path.relpath(OUTJ, HERE)} and {os.path.relpath(OUTM, HERE)}")

    # ── the picture ───────────────────────────────────────────────────────────
    mesh_lines = [[V[u][:2], V[w][:2]] for u, w, _ in E]
    fig = plt.figure(figsize=(15.0, 5.4), facecolor=fce.SURFACE)
    ax = fig.add_axes([0.02, 0.06, 0.29, 0.86])
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.add_collection(LineCollection(mesh_lines, colors=fce.MESH, lw=0.45))
    hi = [(u, w) for u, w, qq in E if qq >= args.q]
    ax.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w in hi],
                                     colors=fce.SERIES_1, lw=2.4, capstyle="round"))
    ax.plot(*np.array([V[v][:2] for v in supports]).T, "o", color=fce.SERIES_2,
            ms=3.6, ls="none")
    ax.autoscale()
    ax.set_title(f"a  the force concentration: q $\\geq$ {args.q:g}\n"
                 f"{len(hi)} edges ({len(res['hi'])} interior); supports in blue",
                 fontsize=10, color=fce.INK, loc="left")

    ax2 = fig.add_axes([0.34, 0.06, 0.29, 0.86])
    ax2.set_aspect("equal"); ax2.set_axis_off()
    ax2.add_collection(LineCollection(mesh_lines, colors=fce.MESH, lw=0.45))
    for nm, kind, c in cables:
        p = c["path"]
        ax2.add_collection(LineCollection(
            [[V[a][:2], V[b][:2]] for a, b in zip(p[:-1], p[1:])],
            colors=fce.ROUTE_LIGHT if kind == "edge" else fce.SERIES_1,
            lw=2.2 if kind == "edge" else 2.6, capstyle="round"))
    ax2.plot(*np.array([V[v][:2] for v in supports]).T, "o", color=fce.SERIES_2,
             ms=3.6, ls="none")
    for nm, kind, c in cables:
        k = len(c["path"]) // 2
        ax2.text(*V[c["path"][k]][:2], nm, fontsize=7, color=fce.INK,
                 ha="center", va="center", zorder=9,
                 bbox=dict(fc=fce.SURFACE, ec="none", alpha=0.9, pad=1.0))
    ax2.autoscale()
    n_e = sum(k == "edge" for _, k, _ in cables)
    ax2.set_title(f"b  routed ($\\lambda$ = {args.lam:g}, $\\mu$ = {args.mu:g})\n"
                  f"{n_e} free-edge cables, {len(cables) - n_e} interior",
                  fontsize=10, color=fce.INK, loc="left")

    ax3 = fig.add_axes([0.655, 0.02, 0.34, 0.94], projection="3d",
                       computed_zorder=False)
    ax3.set_proj_type("ortho")
    ax3.add_collection3d(Poly3DCollection([[V[i] for i in f] for f in F],
                                          facecolors=fce.SURFACE, alpha=0.7,
                                          edgecolors="#cac8c2", linewidths=0.3))
    for nm, kind, c in cables:
        p = c["path"]
        ax3.add_collection3d(Line3DCollection(
            [[V[a], V[b]] for a, b in zip(p[:-1], p[1:])],
            colors=fce.ROUTE_LIGHT if kind == "edge" else fce.SERIES_1,
            linewidths=2.4, capstyle="round"))
    P = np.array([V[k] for k in V])
    c0 = 0.5 * (P.min(0) + P.max(0))
    ax3.set_xlim(c0[0] - 0.62, c0[0] + 0.62); ax3.set_ylim(c0[1] - 0.62, c0[1] + 0.62)
    ax3.set_zlim(0, 0.42)
    ax3.set_box_aspect((1, 1, 0.42), zoom=1.25)
    ax3.view_init(elev=26, azim=-62)
    ax3.set_axis_off()
    ax3.set_title("c  the cable trajectories on the surface", fontsize=10,
                  color=fce.INK, loc="left")

    fig.savefig(OUTP, dpi=170, facecolor=fce.SURFACE)
    print(f"wrote {os.path.relpath(OUTP, HERE)}")


if __name__ == "__main__":
    main()
