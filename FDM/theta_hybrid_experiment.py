"""EXPERIMENT, not the method in the figure: an alternative definition of theta.

Hybrid theta: radial from the apex for a path's FIRST edge (which has no
predecessor), turn-from-the-previous-edge for every edge after it.

Because the reference is now path-dependent, the search runs over DIRECTED EDGES
(the line graph, 2 x 336 = 672 states) rather than over vertices. Dijkstra is still
exact; only the state space changes.
"""
import collections
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import figure_cable_extraction as fce

V, F, E = fce.load()
boundary = fce.boundary_vertices(F)
apex = fce.apex_point(V)
q_of = {frozenset((u, w)): qq for u, w, qq in E}
s2_radial = {frozenset((u, w)): fce.sin2_theta(V, apex, u, w) for u, w, _ in E}

adj = collections.defaultdict(set)
for u, w, _ in E:
    adj[u].add(w)
    adj[w].add(u)


def unit(a, b):
    d = V[b][:2] - V[a][:2]
    return d / np.linalg.norm(d)


def sin2_turn(t, u, v):
    """sin^2 of the turn between edge (t,u) and edge (u,v), in plan."""
    d1, d2 = unit(t, u), unit(u, v)
    return float(np.clip(abs(d1[0] * d2[1] - d1[1] * d2[0]), 0.0, 1.0) ** 2)


def edge_cost(u, v, s2):
    return (1.0 / q_of[frozenset((u, v))] ** 2) * (1.0 + LAM * s2)


def route_from_tip(tip, prices):
    """Least-cost turn-aware route from tip to any priced termination.
    Returns the ordered vertex path, or None."""
    # state = (prev, cur): we arrived at cur along the edge prev->cur
    dist, parent = {}, {}
    pq = []
    for v in adj[tip]:                       # first edge: radial reference
        c = edge_cost(tip, v, s2_radial[frozenset((tip, v))])
        st = (tip, v)
        dist[st] = c
        parent[st] = None
        heapq.heappush(pq, (c, st))
    while pq:
        d, st = heapq.heappop(pq)
        if d > dist.get(st, np.inf) + 1e-12:
            continue
        p, cur = st
        for nxt in adj[cur]:
            if nxt == p:
                continue
            c = d + edge_cost(cur, nxt, sin2_turn(p, cur, nxt))
            st2 = (cur, nxt)
            if c < dist.get(st2, np.inf) - 1e-12:
                dist[st2] = c
                parent[st2] = st
                heapq.heappush(pq, (c, st2))
    best, best_st = np.inf, None
    for st, d in dist.items():
        end = st[1]
        if end in prices and d + prices[end] < best:
            best, best_st = d + prices[end], st
    if best_st is None:
        return None
    path = [best_st[1]]
    st = best_st
    while st is not None:
        path.append(st[0])
        st = parent[st]
    return path[::-1]


def extract_hybrid(lam, q_threshold, mu):
    global LAM
    LAM = lam
    hi = [e for e in E if e[2] >= q_threshold]
    hi_edges = {frozenset((u, w)) for u, w, _ in hi}
    hi_comps, hi_adj = fce.graph_components(hi_edges)
    order = sorted(hi_comps,
                   key=lambda c: -max(qq for u, w, qq in hi if u in c and w in c))
    routes, tip_routes, cable_v = set(), {}, set()
    for comp in order:
        tips = [v for v in comp if len(hi_adj[v]) == 1] or sorted(comp)
        prices = {b: mu for b in boundary}
        for v in cable_v:
            prices[v] = 0.0
        grown = set()
        for t in tips:
            path = route_from_tip(t, prices)
            if not path:
                continue
            for a, b in zip(path[:-1], path[1:]):
                routes.add(frozenset((a, b)))
                grown.update((a, b))
            tip_routes[t] = path
        cable_v |= comp | grown
    edges = hi_edges | routes
    comps, _ = fce.graph_components(edges)
    return dict(edges=edges, hi_edges=hi_edges, tip_routes=tip_routes,
                comps=comps, hi_comps=hi_comps)


if __name__ == "__main__":
    corners = [v for v in boundary
               if min(np.linalg.norm(V[v][:2] - np.array(c))
                      for c in [(0, 0), (0, 1.2), (1.2, 0), (1.2, 1.2)]) < 0.06]
    print(f"{'mode':>7} {'lam':>4} {'edges':>5} {'comp':>4} {'anch':>4} {'corn':>4}  "
          f"cables (lengths)")
    for lam in [0.0, 2.0, 5.0, 10.0, 20.0]:
        for mode in ("radial", "hybrid"):
            r = (fce.extract(V, E, {frozenset((u, w)): s2_radial[frozenset((u, w))]
                                    for u, w, _ in E}, boundary, lam, 2.0,
                             anchor_cost=5.0)
                 if mode == "radial" else extract_hybrid(lam, 2.0, 5.0))
            vs = {x for e in r["edges"] for x in tuple(e)}
            anc = vs & boundary
            cab = fce.trace_cables(V, r["edges"], q_of, r["hi_edges"],
                                   r["tip_routes"])
            lens = sorted(round(c["length"], 3) for c in cab)
            print(f"{mode:>7} {lam:>4.0f} {len(r['edges']):>5} {len(r['comps']):>4} "
                  f"{len(anc):>4} {sum(1 for c in corners if c in vs):>4}  "
                  f"{len(cab)} {lens}")

    print("\nhybrid: where is the good regime? (4 corners, 1 component, 4 anchors)")
    good = []
    for lam in [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100, 300, 1000]:
        r = extract_hybrid(float(lam), 2.0, 5.0)
        vs = {x for e in r["edges"] for x in tuple(e)}
        anc = len(vs & boundary)
        corn = sum(1 for c in corners if c in vs)
        ok = corn == 4 and len(r["comps"]) == 1 and anc == 4
        good.append(lam) if ok else None
        print(f"  lam={lam:>5} {len(r['edges']):>3}e  comp={len(r['comps'])} "
              f"anch={anc:>2} corners={corn}/4  {'OK' if ok else ''}")
    print(f"  good for lambda in {min(good)} – {max(good)}" if good else "  none")
