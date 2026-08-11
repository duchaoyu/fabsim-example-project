"""Figure: extracting cable trajectories from the force-density network.

Implements the method as written: the mesh is read as a weighted graph with

    w_e = (1 / q_e^2) * (1 + lambda * sin^2(theta_e))

where q_e is the edge force density and theta_e the angle to a reference direction:
radial from the apex for a route's FIRST edge, which has no predecessor, and the
previous edge for every edge after it (THETA_MODE = "hybrid"). Because that reference
is path-dependent, routing runs over directed edges rather than vertices; Dijkstra is
still exact, on a state space of 2|E| instead of |V|. THETA_MODE = "radial" restores
the all-radial definition, where w_e is a constant per edge.

Edges with q >= Q_THRESHOLD form the chains,
retained strongest first; Dijkstra then routes each chain END to whichever termination
is cheapest — an already-retained cable for nothing, or a new boundary anchor at a
cost ANCHOR_COST. Only components reaching a boundary vertex, directly or through
another retained cable, are kept.

LAMBDA and ANCHOR_COST are not independent. Raising lambda makes tangential travel
expensive, and an arch end can only reach a diagonal by travelling tangentially,
while escaping straight out to the boundary is radial and cheap. So past lambda ~ 8
at ANCHOR_COST = 5 the arches detach from the diagonals and re-anchor on the
boundary, which is the opposite of what ANCHOR_COST is there to achieve. Keeping all
four arches tied needs ANCHOR_COST to grow with lambda, roughly ANCHOR_COST >= lambda/2.

ANCHOR_COST is what decides whether the arches hang off the boundary or off the
diagonals. At 0 each arch end takes the nearer termination, which is the boundary
one edge away: 5 separate components on 12 anchors. From about 2 the arches start
tying into the diagonals, and from 3 upward the result is one connected network on
the 4 corners, unchanged all the way to 40.

Seeding at the chain ends rather than at every high-force vertex matters: seeding
everywhere ties each arch back at every node (a comb of ties), and dropping the
high-force edges from the union altogether leaves the arches as disconnected stubs,
since each arch seed then routes radially outward without ever traversing the arch.

Panels:
  a  the force-density factor 1/q^2 (the part of the cost that IS per-edge)
  b  lambda = 0: force-density guidance alone, the path drifts onto the arches
  c  lambda = LAMBDA: the penalty restored, chains cross the surface
  d  lambda = LAMBDA_HIGH: a second value in the band, giving the identical network
  e  the assembled subgraph and the boundary-anchor test
  f  the retained cable trajectories in 3D

    python FDM/figure_cable_extraction.py
"""
import collections
import heapq
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
# Vector export for Illustrator. fonttype 42 embeds TrueType rather than Type 3, so
# labels arrive as editable text instead of outlines; svg.fonttype "none" leaves SVG
# text as <text> referencing the font by name.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "data", "crossvault", "mesh_out_cross_20240404130204.json")
OUT = os.path.join(HERE, "figures", "cable_extraction.png")

THETA_MODE = "hybrid"   # "hybrid": a path's FIRST edge is referenced to the outward
                        # radial direction from the apex (it has no predecessor), and
                        # every edge after it to the turn from its predecessor. Since
                        # the reference is then path-dependent, routing runs over
                        # directed edges. "radial": theta is the angle to the radial
                        # direction for every edge, so w_e is a constant per edge and
                        # a vertex-space Dijkstra suffices. Both give the same network
                        # here; hybrid holds it over a far wider band of lambda.
LAMBDA = 15.0       # mid-band. Measured good range: 7 - 30 hybrid, 4 - 7.5 radial
LAMBDA_HIGH = 20.0  # a second value inside the band, to show nothing moves
LAMBDA_OFF = 0.0    # the comparison case: pure force-density guidance
Q_THRESHOLD = 2.0   # force concentrations = edges with q >= this. A fixed value, not
                    # a percentile: the selected set is the same anywhere in
                    # 2.0 <= q <= 2.2, and it cuts all four arches at the same place.
                    # The top decile (p90 = 2.23) sits just above that plateau and
                    # trims two of the four arches by one edge, which shows up as an
                    # asymmetric tie layout. Same threshold as the stiffener figure.
ANCHOR_COST = 5.0      # what a NEW boundary anchor costs a route, against 0 for
                       # tying into an already-retained cable. At 0 a route takes
                       # whichever termination is nearer and each arch ties to the
                       # boundary (5 components, 12 anchors); from about 2 the arches
                       # tie into the diagonals instead, and from 5 upward the result
                       # is one connected network on 4 corner anchors.
TIES_OUTWARD = True    # a tie may not terminate closer to the apex than the chain
                       # end it leaves. Off, whether an arch ties outward (toward the
                       # corners) or inward (toward the crown) goes to whichever is
                       # marginally cheaper, and since the target surface is only
                       # 2-fold symmetric that flips between x and y: two arches tie
                       # outward, two inward, and the four come out 1.050-1.124 m.
                       # On, all four tie outward and land in 1.043-1.056 m.
TIE_TO_CABLES = True   # let a chain end on an already-retained cable, not only
                       # on the boundary — this is what ties the arches into the
                       # diagonals. Set False for one route per chain end straight
                       # out to a support.

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
SURFACE = "#fcfcfb"
MESH = "#e2e0da"
RED = "#e34948"        # the palette's red
SERIES_1 = RED         # cables
SERIES_2 = "#2a78d6"   # apex, anchors, the theta construction — blue against the red
                       # cables: the documented diverging pair, so the two never read
                       # as the same family


def _blend(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)


# one-hue red ramp, light -> dark, stepped off the palette's red by blending toward
# the surface at one end and toward ink at the other so no step is eyeballed.
# Reversed for the cost field so that *cheap to traverse* reads dark.
SEQ_RED = LinearSegmentedColormap.from_list("seq_red", [
    _blend(SURFACE, RED, 0.16), _blend(SURFACE, RED, 0.32),
    _blend(SURFACE, RED, 0.50), _blend(SURFACE, RED, 0.70), RED,
    _blend(RED, INK, 0.22), _blend(RED, INK, 0.45),
])
COST_CMAP = SEQ_RED.reversed()
ROUTE_LIGHT = _blend(SURFACE, RED, 0.42)   # the tie routes, a lighter step of the same hue

FIGW, FIGH = 16.5, 13.0
PLAN_LIM = (-0.075, 1.275)


# --------------------------------------------------------------------- the mesh
def load():
    d = json.load(open(RESULT))
    V = {int(k): np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)])
         for k, v in d["vertex"].items()}
    F = [d["face"][k] for k in sorted(d["face"], key=int)]
    E = []
    for key, attr in d["edgedata"].items():
        u, w = (int(t) for t in key.strip("()").split(","))
        E.append((u, w, float(attr["qpre"][0])))
    return V, F, E


def boundary_vertices(F):
    """A boundary edge belongs to exactly one face."""
    seen = collections.Counter()
    for f in F:
        for a, b in zip(f, f[1:] + f[:1]):
            seen[frozenset((a, b))] += 1
    return {v for e, c in seen.items() if c == 1 for v in e}


def apex_point(V):
    """The apex as a point: centroid of the crown plateau (top 5 % of z). theta only
    needs an outward radial direction, so it need not land on a vertex."""
    z = np.array([V[k][2] for k in V])
    cut = z.max() - 0.05 * (z.max() - z.min())
    return np.mean([V[k][:2] for k in V if V[k][2] >= cut], axis=0)


# ------------------------------------------------------------------- the weights
def sin2_theta(V, apex, u, w):
    """sin^2 of the angle between edge (u,w) and the outward radial direction."""
    r = 0.5 * (V[u][:2] + V[w][:2]) - apex
    d = V[w][:2] - V[u][:2]
    nr, nd = np.linalg.norm(r), np.linalg.norm(d)
    if nr < 1e-9 or nd < 1e-9:
        return 0.0
    r, d = r / nr, d / nd
    return float(np.clip(abs(d[0] * r[1] - d[1] * r[0]), 0.0, 1.0) ** 2)


def costs(E, s2, lam):
    return np.array([(1.0 / qq ** 2) * (1.0 + lam * s2[frozenset((u, w))])
                     for u, w, qq in E])


# ------------------------------------------------------------------- the cables
def graph_components(edges):
    """Connected components of a set of frozenset edges, with the adjacency."""
    adj = collections.defaultdict(set)
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for v0 in adj:
        if v0 in seen:
            continue
        stack, comp = [v0], set()
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            seen.add(c)
            stack.extend(adj[c] - comp)
        comps.append(comp)
    return comps, adj


def sin2_turn(V, t, u, v):
    """sin^2 of the turn between edge (t,u) and edge (u,v), in plan."""
    d1 = V[u][:2] - V[t][:2]
    d2 = V[v][:2] - V[u][:2]
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    return float(np.clip(abs(d1[0] * d2[1] - d1[1] * d2[0]), 0.0, 1.0) ** 2)


def turn_route(V, adj, q_of, s2, lam, tip, prices):
    """Least-cost route from a chain end to any priced termination, with theta read
    as the turn from the preceding edge.

    Because the reference direction is now path-dependent, the cost of entering an
    edge depends on how the path arrived, so the search runs over DIRECTED EDGES —
    state (prev, cur) means 'arrived at cur along prev->cur'. Dijkstra is still
    exact on that state space; only its size changes, from |V| to 2|E|. The first
    edge has no predecessor and is referenced radially instead, which is the one
    place the apex still enters.
    """
    def cost(u, v, s2_val):
        return (1.0 / q_of[frozenset((u, v))] ** 2) * (1.0 + lam * s2_val)

    dist, parent, pq = {}, {}, []
    for v in adj[tip]:                                  # first edge: radial
        st = (tip, v)
        dist[st] = cost(tip, v, s2[frozenset((tip, v))])
        parent[st] = None
        heapq.heappush(pq, (dist[st], st))
    while pq:
        d, st = heapq.heappop(pq)
        if d > dist.get(st, np.inf) + 1e-12:
            continue
        p, cur = st
        for nxt in adj[cur]:
            if nxt == p:
                continue
            c = d + cost(cur, nxt, sin2_turn(V, p, cur, nxt))
            st2 = (cur, nxt)
            if c < dist.get(st2, np.inf) - 1e-12:
                dist[st2] = c
                parent[st2] = st
                heapq.heappush(pq, (c, st2))
    best, best_st = np.inf, None
    for st, d in sorted(dist.items()):                  # sorted: ties break the same
        if st[1] in prices and d + prices[st[1]] < best - 1e-12:
            best, best_st = d + prices[st[1]], st
    if best_st is None:
        return None
    path, st = [best_st[1]], best_st
    while st is not None:
        path.append(st[0])
        st = parent[st]
    return path[::-1]


def extract(V, E, s2, boundary, lam, q_threshold, tie_to_cables=TIE_TO_CABLES,
            anchor_cost=ANCHOR_COST, theta_mode=None,
            ties_outward=TIES_OUTWARD, apex=None):
    """The high-force chains, plus the least-cost route that carries each chain END
    out to a support, then filtered on boundary contact. Seeding at the chain ends
    rather than at every high-force vertex is what keeps one tie per chain end
    instead of a tie at every node along a chain.

    tie_to_cables decides what a route is allowed to terminate on. With it off, the
    only targets are boundary vertices, so every chain runs its own way out to a
    support. With it on, the chains are retained in order of dominance (strongest
    first) and each later chain may also stop on an already-retained cable, since
    force handed to an anchored cable still reaches the support. On this vault that
    is what ties the four arches into the two diagonals instead of sending each arch
    end out to the boundary on its own."""
    mode = theta_mode or THETA_MODE
    apex = apex_point(V) if apex is None else apex
    n = max(V) + 1
    w = costs(E, s2, lam)
    rows = [e[0] for e in E] + [e[1] for e in E]
    cols = [e[1] for e in E] + [e[0] for e in E]
    q_of = {frozenset((u, w_)): qq for u, w_, qq in E}
    mesh_adj = collections.defaultdict(set)
    for u, w_, _ in E:
        mesh_adj[u].add(w_)
        mesh_adj[w_].add(u)

    hi = [e for e in E if e[2] >= q_threshold]
    hi_edges = {frozenset((u, w_)) for u, w_, _ in hi}
    hi_comps, hi_adj = graph_components(hi_edges)
    # strongest chain first, so the diagonals are retained before the arches and are
    # therefore available as a target for them
    order = sorted(hi_comps, key=lambda c: -max(
        qq for u, w_, qq in hi if u in c and w_ in c))

    chains = set(hi_edges)
    routes, seeds, tip_routes = set(), [], {}
    cable_v = set()                 # vertices already retained as cable
    for comp in order:
        tips = [v for v in comp if len(hi_adj[v]) == 1] or sorted(comp)
        seeds += tips
        # A virtual super-node holds the two ways a route may terminate: on an
        # already-retained cable (free) or on the boundary (costing anchor_cost, a
        # new support to detail and build). Dijkstra from it then gives every vertex
        # its cheapest termination, and anchor_cost is the knob that trades a new
        # anchor against tying into a cable that is already anchored.
        ends = {b: anchor_cost for b in boundary}
        if tie_to_cables:
            for v in cable_v:
                ends[v] = 0.0
        grown = set()
        for t in tips:
            # ties_outward: a route may not terminate closer to the apex than the
            # chain end it leaves from. Without it, whether a tie reaches its
            # diagonal outward or inward is decided by which is marginally cheaper,
            # and on a target surface that is only 2-fold symmetric that flips
            # between the x and y directions — two arches tie outward toward the
            # corners, two tie inward toward the crown, and the four come out
            # unequal. A tie running back toward the crown also carries force away
            # from the supports, so outward is the structurally sensible choice.
            ends_t = ends
            if ties_outward:
                r0 = np.linalg.norm(V[t][:2] - apex)
                ends_t = {v: p for v, p in ends.items()
                          if np.linalg.norm(V[v][:2] - apex) >= r0 - 1e-9}
                if not ends_t:                  # never leave an end unanchored
                    ends_t = ends
            if mode == "hybrid":
                walk = turn_route(V, mesh_adj, q_of, s2, lam, t, ends_t) or [t]
                for a, b in zip(walk[:-1], walk[1:]):
                    routes.add(frozenset((a, b)))
                    grown.update((a, b))
            else:
                r = rows + [n] * len(ends_t) + list(ends_t)
                c_ = cols + list(ends_t) + [n] * len(ends_t)
                vals = np.r_[w, w, list(ends_t.values()), list(ends_t.values())]
                Gv = coo_matrix((vals, (r, c_)), shape=(n + 1, n + 1)).tocsr()
                _d, pred, _s = dijkstra(Gv, indices=[n], min_only=True,
                                        return_predecessors=True)
                cur, walk = t, [t]
                while cur != n and int(pred[cur]) >= 0:
                    p = int(pred[cur])
                    if p != n:                  # skip the virtual termination edge
                        routes.add(frozenset((p, cur)))
                        grown.update((p, cur))
                        walk.append(p)
                    cur = p
            # the ordered route off this chain end, kept so a cable can later be
            # assembled as route + chain + route rather than by geometry alone
            tip_routes[t] = walk
        cable_v |= comp | grown
    chains |= routes

    comps, _ = graph_components(chains)
    kept = [c for c in comps if c & boundary]
    dropped = [c for c in comps if not (c & boundary)]
    keep = {v for c in kept for v in c}
    keep_edges = {e for e in chains if tuple(e)[0] in keep}
    return dict(chains=chains, edges=keep_edges, comps=comps, kept=kept,
                dropped=dropped, seeds=seeds, hi=hi, hi_edges=hi_edges,
                routes=routes, hi_comps=hi_comps, tip_routes=tip_routes)


def trace_cables(V, edges, q_of, chain_edges, tip_routes,
                 turn_limit_deg=70.0):
    """Resolve the retained edge set into ordered cable polylines.

    An edge set says which edges are cable; it does not say what a cable IS. This is
    done in two steps. The high-force chains are traced geometrically, continuing at
    each junction along whichever neighbour turns least (stopping past
    turn_limit_deg) — that is what splits the twelve edges meeting at the apex into
    two through-diagonals instead of four stubs. Each traced chain is then extended
    along the routes its own ends generated, by role rather than by angle, so every
    cable runs from one termination to the other.

    Returns polylines as ordered vertex lists, ranked by peak force density.
    """
    adj = collections.defaultdict(set)
    for e in edges:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    cos_limit = np.cos(np.deg2rad(turn_limit_deg))

    def direction(a, b):
        d = V[b] - V[a]
        return d / np.linalg.norm(d)

    def step(prev, cur, unused):
        """The straightest unused continuation through cur, or None."""
        d_in = direction(prev, cur)
        best, best_dot = None, cos_limit
        for nxt in adj[cur]:
            if frozenset((cur, nxt)) not in unused:
                continue
            dot = float(d_in @ direction(cur, nxt))
            if dot > best_dot:
                best, best_dot = nxt, dot
        return best

    # Step 1: trace WITHIN the high-force chains only. Geometry decides here, which
    # is what splits the twelve edges at the apex into two through-diagonals.
    unused = set(chain_edges)
    polylines = []
    while unused:
        seed_edge = max(unused, key=lambda e: (q_of[e], sorted(tuple(e))))
        v, nxt = sorted(tuple(seed_edge))
        unused.discard(frozenset((v, nxt)))
        path = [v, nxt]
        while True:                              # forward
            nx = step(path[-2], path[-1], unused)
            if nx is None:
                break
            unused.discard(frozenset((path[-1], nx)))
            path.append(nx)
        while True:                              # and backward from the seed
            nx = step(path[1], path[0], unused)
            if nx is None:
                break
            unused.discard(frozenset((path[0], nx)))
            path.insert(0, nx)
        polylines.append(path)

    # Step 2: extend each traced chain along the route that its own end generated.
    # Role, not angle, decides this: a route exists precisely to carry that end to a
    # termination, so it belongs to that cable however sharply it leaves. Letting the
    # turn angle decide is what made two arches swallow their ties while the other
    # two did not — the tie leaves near-collinear on one pair and at almost 90° on
    # the other, purely because the mesh is not symmetric.
    for path in polylines:
        for end in (0, -1):
            route = tip_routes.get(path[end])
            if not route or len(route) < 2:
                continue
            tail = [int(v) for v in route[1:]]   # route[0] is the chain end itself
            if end == 0:
                path[:0] = tail[::-1]
            else:
                path.extend(tail)

    # anything the chains never claimed (there should be nothing on this vault)
    claimed = {frozenset((a, b)) for p in polylines for a, b in zip(p[:-1], p[1:])}
    leftover = set(edges) - claimed
    if leftover:
        print(f"  note: {len(leftover)} retained edges belong to no cable")

    out = []
    for path in polylines:
        qs = [q_of[frozenset((a, b))] for a, b in zip(path[:-1], path[1:])]
        length = float(sum(np.linalg.norm(V[b] - V[a])
                           for a, b in zip(path[:-1], path[1:])))
        out.append(dict(path=[int(v) for v in path], edges=len(path) - 1,
                        q_min=min(qs), q_max=max(qs), q_mean=float(np.mean(qs)),
                        length=length, closed=path[0] == path[-1]))
    return sorted(out, key=lambda c: -c["q_max"])


# ------------------------------------------------------------------- the drawing
def square(x, y, h):
    return [x, y, h * FIGH / FIGW, h]


def frame(ax):
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.set_xlim(*PLAN_LIM)
    ax.set_ylim(*PLAN_LIM)
    ax.set_axis_off()


def draw_mesh(ax, V, E, color=MESH, lw=0.5):
    ax.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w, _ in E],
                                     colors=color, linewidths=lw, zorder=0))


def draw_cables(ax, V, edges, color=SERIES_1, lw=2.3, zorder=3):
    ax.add_collection(LineCollection([[V[a][:2], V[b][:2]]
                                      for a, b in (tuple(e) for e in edges)],
                                     colors=color, linewidths=lw, zorder=zorder,
                                     capstyle="round"))


def panel_title(fig, x, y, tag, title, subtitle):
    fig.text(x, y, tag, fontsize=12, fontweight="bold", color=INK, va="baseline")
    fig.text(x + 0.017, y, title, fontsize=11.5, color=INK, va="baseline")
    fig.text(x, y - 0.011, subtitle, fontsize=8.8, color=INK_2, va="top",
             linespacing=1.55)


def theta_diagram(fig, rect):
    """A schematic definition of theta, on its own surface at a size where the four
    labels fit, rather than squeezed on top of the network."""
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1.66)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")      # the angle has to be drawn true
    ax.set_anchor("NW")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_color("#e4e2dc")
        sp.set_linewidth(0.7)

    def rot(v, deg):
        t = np.deg2rad(deg)
        return np.array([v[0]*np.cos(t) - v[1]*np.sin(t),
                         v[0]*np.sin(t) + v[1]*np.cos(t)])

    def arc(at, v1, v2, r, label, dy=0.0):
        ax.add_patch(matplotlib.patches.Arc(
            at, r, r, angle=0, color=INK_2, lw=1.0,
            theta1=np.rad2deg(np.arctan2(v1[1], v1[0])),
            theta2=np.rad2deg(np.arctan2(v2[1], v2[0]))))
        bis = (v1 + v2) / np.linalg.norm(v1 + v2)
        ax.text(*(at + bis * (r * 0.72) + np.array([0.0, dy])), label, fontsize=10,
                color=INK, ha="center", va="center", zorder=8,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.95, pad=1.0))

    apex = np.array([0.10, 0.34])
    p0 = np.array([0.62, 0.46])               # the chain end a route starts from
    radial = (p0 - apex) / np.linalg.norm(p0 - apex)
    d1 = rot(radial, 30.0)
    p1 = p0 + d1 * 0.38
    d2 = rot(d1, -35.0)
    p2 = p1 + d2 * 0.40

    # the radial reference, through the chain end
    ax.annotate("", xy=p0 + radial * 0.26, xytext=apex,
                arrowprops=dict(arrowstyle="-|>", color=SERIES_2, lw=1.2,
                                ls=(0, (4, 2.5)), shrinkA=0, shrinkB=0))
    ax.plot(*apex, "o", color=SERIES_2, ms=6)
    # the two edges of the route
    ax.plot(*zip(p0, p1), color=SERIES_1, lw=3.2, solid_capstyle="round")
    ax.plot(*zip(p1, p2), color=SERIES_1, lw=3.2, solid_capstyle="round")
    # and the first edge's direction carried past p1, the reference for the second
    ax.plot(*zip(p1, p1 + d1 * 0.26), color=INK_3, lw=1.0, ls=(0, (3, 2.5)))

    arc(p0, radial, d1, 0.28, r"$\theta_1$")
    arc(p1, d2, d1, 0.24, r"$\theta_2$")

    bb = dict(fc=SURFACE, ec="none", alpha=0.95, pad=1.4)
    ax.text(*(apex + np.array([0.15, -0.11])), "apex", fontsize=8, color=SERIES_2,
            ha="center", va="center", bbox=bb)
    ax.text(*(apex + radial * 0.52 + np.array([0.0, -0.12])), "radial", fontsize=8,
            color=SERIES_2, ha="center", va="center", bbox=bb)
    ax.text(*(0.5 * (p1 + p2) + np.array([0.0, -0.14])), "route", fontsize=8,
            color=SERIES_1, ha="center", va="center", bbox=bb)
    return ax


def note(ax, text, xy, xytext, ha="center"):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8.2, color=INK_2, ha=ha,
                va="center", linespacing=1.4,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.85, pad=1.6),
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.8,
                                shrinkA=1, shrinkB=2))


def main():
    V, F, E = load()
    boundary = boundary_vertices(F)
    apex = apex_point(V)
    s2 = {frozenset((u, w)): sin2_theta(V, apex, u, w) for u, w, _ in E}
    q = np.array([e[2] for e in E])
    q_thr = float(Q_THRESHOLD)

    off = extract(V, E, s2, boundary, LAMBDA_OFF, q_thr)
    on = extract(V, E, s2, boundary, LAMBDA, q_thr)
    w_on = costs(E, s2, LAMBDA)

    corners = [v for v in boundary
               if min(np.linalg.norm(V[v][:2] - np.array(c))
                      for c in [(0, 0), (0, 1.2), (1.2, 0), (1.2, 1.2)]) < 0.06]

    def corners_hit(res):
        vs = {x for e in res["edges"] for x in tuple(e)}
        return sum(1 for c in corners if c in vs)

    def tangential(res):
        return sum(1 for e in res["edges"] if s2[e] > 0.5)

    # over what range of lambda is the extracted network unchanged?
    plateau = [lam for lam in np.arange(1.0, 45.5, 0.5)
               if extract(V, E, s2, boundary, float(lam), q_thr)["edges"] == on["edges"]]
    lam_lo, lam_hi = min(plateau), max(plateau)
    print(f"network identical to lambda={LAMBDA:g} for lambda in "
          f"{lam_lo:g} – {lam_hi:g} (scanned 1 – 45 in steps of 0.5)")

    mu_plateau = [mu for mu in np.arange(0.5, 40.5, 0.5)
                  if extract(V, E, s2, boundary, LAMBDA, q_thr,
                             anchor_cost=float(mu))["edges"] == on["edges"]]
    mu_lo, mu_hi = min(mu_plateau), max(mu_plateau)
    print(f"network identical to mu={ANCHOR_COST:g} for mu in {mu_lo:g} – {mu_hi:g} "
          f"(scanned 0.5 – 40 in steps of 0.5)")
    high = extract(V, E, s2, boundary, LAMBDA_HIGH, q_thr)
    broke = extract(V, E, s2, boundary, 100.0, q_thr)
    print(f"lambda={LAMBDA_HIGH:g}: {len(high['edges'])} edges, "
          f"identical to lambda={LAMBDA:g}: {high['edges'] == on['edges']}")
    loose = extract(V, E, s2, boundary, LAMBDA, q_thr, anchor_cost=0.0)
    l_anc = {x for e in loose["edges"] for x in tuple(e)} & boundary
    print(f"for comparison, mu=0: {len(loose['edges'])} edges, "
          f"{len(loose['comps'])} components, {len(l_anc)} anchors")

    print(f"q threshold = {q_thr:.2f} -> {len(on['hi'])} high-force edges in "
          f"{len(on['hi_comps'])} chains, {len(on['seeds'])} chain-end seeds")
    print(f"cost range at lambda={LAMBDA:g}: {w_on.min():.3f} – {w_on.max():.1f}")
    for name, res in [("lambda=0", off), (f"lambda={LAMBDA:g}", on)]:
        print(f"{name:>10}: {len(res['edges']):3d} edges  {len(res['comps'])} components "
              f"({len(res['kept'])} anchored, {len(res['dropped'])} discarded)  "
              f"corners reached {corners_hit(res)}/4  tangential edges {tangential(res)}")

    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    X1, X2, X3, X4 = 0.045, 0.280, 0.515, 0.750
    ROW1, H1 = 0.506, 0.269        # 3.5 in squares, as in the
    T1, T2 = 0.848, 0.405          # stiffener figure

    # ------------------------------------------------- a  the weighted graph
    ax_a = fig.add_axes(square(X1, ROW1, H1))
    frame(ax_a)
    # With theta read as a turn, the directional factor depends on how a path
    # arrives, so it is not a property of the edge and cannot be drawn on one. What
    # can be drawn is the force-density factor 1/q^2, the part of the cost that is
    # fixed per edge. The slack web puts a long tail on it, so the ramp is cut at 20
    # and the tail folded into the top step.
    w_static = np.array([1.0 / qq ** 2 for _, _, qq in E])
    norm = LogNorm(vmin=max(w_static.min(), 1e-2), vmax=20.0)
    pref = -np.log10(np.clip(w_static, 1e-3, None))
    lw = 0.4 + 2.2 * (pref - pref.min()) / np.ptp(pref)
    ax_a.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w, _ in E],
                                       array=w_on, cmap=COST_CMAP, norm=norm,
                                       linewidths=lw, capstyle="round", zorder=1))
    ax_a.plot(*apex, "o", color=SERIES_2, ms=5.5, zorder=6)
    ax_a.text(*(apex + np.array([0.15, 0.055])), "apex", fontsize=8.2, color=SERIES_2,
              ha="center", va="center",
              bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=1.4), zorder=7)
    panel_title(fig, X1, T1, "a", "the mesh as a weighted graph",
                f"the force-density factor 1/q²: dark and thick is cheap.\n"
                f"Squaring q makes the preference steep rather than gentle,\n"
                f"so a route is pulled hard onto the strongest edges. The\n"
                f"directional factor is not drawn — it depends on how a\n"
                f"path arrives, not on the edge.")

    theta_diagram(fig, [0.815, 0.355, 0.137, 0.105])
    fig.text(0.760, 0.490,
             "how $\\theta$ is measured: a route's first edge from\nthe radial, "
             "every edge after it from the one before",
             fontsize=8.6, color=INK_2, va="top", linespacing=1.5)

    # drawn as a mesh rather than with fig.colorbar, which renders the ramp as an
    # embedded raster image — this keeps the whole page live vector for Illustrator
    cax = fig.add_axes([X1, 0.468, 0.16, 0.009])
    ramp = np.geomspace(norm.vmin, norm.vmax, 240)
    cax.pcolormesh(ramp, np.array([0.0, 1.0]), ramp[None, :-1], cmap=COST_CMAP,
                   norm=norm, shading="flat", linewidth=0)
    cax.set_xscale("log")
    cax.set_xlim(norm.vmin, norm.vmax)
    cax.set_yticks([])
    cax.tick_params(labelsize=8, colors=INK_2, length=2, pad=2)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.text(X1, 0.452, "force-density factor $1/q_e^2$   (log scale, cheap on the "
             f"left; the top step holds everything ≥ {norm.vmax:g})",
             fontsize=8.8, color=INK_2, ha="left", va="top")

    # ------------------------------------------------- b  lambda = 0
    ax_b = fig.add_axes(square(X2, ROW1, H1))
    frame(ax_b)
    draw_mesh(ax_b, V, E)
    draw_cables(ax_b, V, off["edges"])
    ax_b.plot(*apex, "o", color=SERIES_2, ms=5, zorder=5)
    panel_title(fig, X2, T1, "b", "λ = 0, force density alone",
                f"{len(off['edges'])} edges, and the diagonals never arrive: each one\n"
                f"reaches an arch of comparable force, turns onto it and\n"
                f"follows the perimeter instead. {corners_hit(off)} of the 4 corners\n"
                f"are reached.")
    hook = min(off["edges"],
               key=lambda e: np.linalg.norm(np.mean([V[x][:2] for x in tuple(e)], axis=0)
                                            - np.array([0.30, 0.95])))
    note(ax_b, "the route turns off the\ndiagonal onto the arch",
         np.mean([V[x][:2] for x in tuple(hook)], axis=0), (0.66, 1.09))

    # ------------------------------------------------- c  lambda = 5
    ax_c = fig.add_axes(square(X3, ROW1, H1))
    frame(ax_c)
    draw_mesh(ax_c, V, E)
    draw_cables(ax_c, V, on["edges"])
    ax_c.plot(*apex, "o", color=SERIES_2, ms=5, zorder=5)
    for cv in corners:
        ax_c.plot(*V[cv][:2], "o", color=SERIES_2, ms=4, zorder=5)
    panel_title(fig, X3, T1, "c", f"λ = {LAMBDA:g}, with the turn penalty",
                f"{len(on['edges'])} edges. The diagonals now cross the surface to\n"
                f"all {corners_hit(on)} corners, and the arches survive anyway on force\n"
                f"alone, each tied into a diagonal rather than to the\n"
                f"boundary.")

    # ------------------------------------------------- d  a higher lambda
    ax_hi = fig.add_axes(square(X4, ROW1, H1))
    frame(ax_hi)
    draw_mesh(ax_hi, V, E)
    draw_cables(ax_hi, V, high["edges"])
    ax_hi.plot(*apex, "o", color=SERIES_2, ms=5, zorder=5)
    for cv in corners:
        ax_hi.plot(*V[cv][:2], "o", color=SERIES_2, ms=4, zorder=5)
    same = high["edges"] == on["edges"]
    panel_title(fig, X4, T1, "d", f"λ = {LAMBDA_HIGH:g}, further up the band",
                (f"the same {len(high['edges'])} edges as c, edge for edge. Nothing moves\n"
                 f"anywhere in λ = {lam_lo:g} – {lam_hi:g}, so λ picks the regime, not the\n"
                 f"layout within it. Past the band the arches detach: at\n"
                 f"λ = 100, {len(broke['comps'])} components on "
                 f"{len({x for e in broke['edges'] for x in tuple(e)} & boundary)} "
                 f"anchors."
                 if same else
                 f"{len(high['edges'])} edges — NOT identical to c, so the\n"
                 f"plateau claim needs re-checking."))

    # ------------------------------------------------- e  assembled + filtered
    ax_d = fig.add_axes(square(X1, 0.055, 0.280))
    frame(ax_d)
    draw_mesh(ax_d, V, E)
    hi_e = on["hi_edges"]
    draw_cables(ax_d, V, on["edges"] - hi_e, color=ROUTE_LIGHT, lw=2.0, zorder=2)
    draw_cables(ax_d, V, on["edges"] & hi_e, color=SERIES_1, lw=2.8, zorder=3)
    anchors = sorted({x for e in on["edges"] for x in tuple(e)} & boundary)

    # the retained edge set, resolved into ordered cables and ranked
    q_of = {frozenset((u, w_)): qq for u, w_, qq in E}
    cables = trace_cables(V, on["edges"], q_of, on["hi_edges"],
                          on["tip_routes"])
    n_aa = sum(1 for c in cables
               if c["path"][0] in boundary and c["path"][-1] in boundary)
    arch_len = sorted(c["length"] for c in cables
                      if not (c["path"][0] in boundary and c["path"][-1] in boundary))
    arch_spread = (arch_len[-1] - arch_len[0]) / np.mean(arch_len) * 100 if arch_len else 0
    print(f"the {len(arch_len)} cable-to-cable arches: "
          + ", ".join(f"{x:.3f}" for x in arch_len)
          + f" m — spread {arch_spread:.1f} %")
    print(f"\n{len(cables)} cables traced from {len(on['edges'])} retained edges "
          f"({n_aa} anchor-to-anchor, {len(cables) - n_aa} cable-to-cable)")
    print(f"{'#':>2}  {'edges':>5}  {'length':>7}  {'q range':>12}  {'ends':>17}  path")
    for i, c in enumerate(cables, 1):
        a, b = c["path"][0], c["path"][-1]
        ends = f"{'anchor' if a in boundary else 'cable':>7} → " \
               f"{'anchor' if b in boundary else 'cable'}"
        path = " ".join(str(v) for v in c["path"])
        print(f"{i:>2}  {c['edges']:>5}  {c['length']:>6.3f} m  "
              f"{c['q_min']:>5.2f}–{c['q_max']:<5.2f}  {ends:>17}  "
              f"{path if len(path) <= 60 else path[:57] + '…'}")
    with open(os.path.join(HERE, "data", "crossvault", "cables_ordered.json"), "w") as f:
        json.dump(dict(lam=LAMBDA, anchor_cost=ANCHOR_COST, q_threshold=q_thr,
                       apex=[float(apex[0]), float(apex[1])],
                       vertices={str(k): [float(x) for x in V[k]] for k in V},
                       cables=cables), f, indent=1)
    print("wrote data/crossvault/cables_ordered.json")
    ax_d.plot([V[a][0] for a in anchors], [V[a][1] for a in anchors], "o",
              color=SERIES_2, ms=4.6, zorder=6, ls="none")
    panel_title(fig, X1, T2, "e", "assembled, then filtered on anchorage",
                f"the {len(on['hi'])} high-force edges (dark) form {len(on['hi_comps'])} "
                f"chains, retained strongest first, so the diagonals are\nalready in place "
                f"when the arches look for a termination (light). {len(on['comps'])} "
                f"component, {len(on['kept'])} anchored, "
                f"{len(on['dropped'])} discarded,\n{len(anchors)} anchors instead of 12. "
                f"Traced into {len(cables)} cables: {n_aa} run anchor to anchor, "
                f"{len(cables) - n_aa} diagonal to diagonal.\nTies run outward only, so "
                f"those {len(arch_len)} come out within {arch_spread:.1f} % of each other "
                f"rather than {7.0:.0f} %.")
    for i, c in enumerate(cables, 1):
        # a quarter along, not the midpoint: both diagonals share vertex 70 at their
        # centre, so midpoint labels would sit on top of each other
        k = len(c["path"]) // 4 if c["edges"] >= 10 else len(c["path"]) // 2
        pos = V[c["path"][k]][:2]
        radial = pos - apex
        nr = np.linalg.norm(radial)
        off = radial / nr * 0.075 if nr > 1e-3 else np.array([0.0, 0.075])
        ax_d.text(*(pos + off), str(i), fontsize=8.6, color=INK, ha="center",
                  va="center", fontweight="semibold", zorder=8,
                  bbox=dict(fc=SURFACE, ec="none", alpha=0.9, pad=1.3))

    leg = fig.legend(handles=[
        plt.Line2D([], [], color=SERIES_1, lw=2.8, label="high-force chain"),
        plt.Line2D([], [], color=ROUTE_LIGHT, lw=2.0, label="route to a support"),
        plt.Line2D([], [], color=SERIES_2, marker="o", ls="none", ms=4.6,
                   label=f"boundary anchor ({len(anchors)})")],
        loc="lower left", bbox_to_anchor=(X1, 0.012), ncol=3, fontsize=8.6,
        handlelength=1.6, borderpad=0.4, columnspacing=1.6, frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # ------------------------------------------------- e  the cables in 3D
    ax_e = fig.add_axes([0.400, 0.080, 0.575, 0.300], projection="3d",
                        computed_zorder=False)
    ax_e.set_proj_type("ortho")
    ax_e.patch.set_visible(False)
    ax_e.add_collection3d(Poly3DCollection([[V[i] for i in f] for f in F],
                                           facecolors=SURFACE, alpha=0.72,
                                           edgecolors="#cac8c2", linewidths=0.4,
                                           zorder=1))
    ax_e.add_collection3d(Line3DCollection(
        [[V[a], V[b]] for a, b in (tuple(e) for e in on["edges"])],
        colors=SERIES_1, linewidths=2.6, capstyle="round", zorder=3))
    ax_e.scatter([V[a][0] for a in anchors], [V[a][1] for a in anchors],
                 [V[a][2] for a in anchors], color=SERIES_2, s=14, zorder=4,
                 depthshade=False)
    ax_e.set_xlim(0.02, 1.18)
    ax_e.set_ylim(0.02, 1.18)
    ax_e.set_zlim(0.0, 0.42)
    ax_e.set_box_aspect((1, 1, 0.40), zoom=1.32)
    ax_e.view_init(elev=24, azim=-58)
    ax_e.set_axis_off()
    panel_title(fig, 0.430, T2, "f", "the cable trajectories",
                f"two diagonals crossing at the apex carry every arch end that reaches\n"
                f"them, so the four arches hang off the diagonals rather than off the\n"
                f"boundary, and the whole network lands on just {len(anchors)} corner anchors.")

    # -------------------------------------------------------------- framing
    fig.text(X1, 0.972, "Cable locations, read off the force-density network",
             fontsize=15.5, color=INK, fontweight="semibold", va="baseline")
    fig.text(X1, 0.952,
             "The mesh is treated as a weighted graph in which an edge is cheap to "
             "traverse where the force density is high and where the path\ndoes not "
             "turn. Dijkstra then returns the dominant tensile load paths, and the "
             "chains it finds are kept only where they reach a\nsupport — a chain ends "
             "on an already-retained cable for nothing, or on the boundary at the price "
             "of a new anchor.",
             fontsize=9.5, color=INK_2, va="top", linespacing=1.6)
    fig.text(X1, 0.890,
             r"$w_e \; = \; \frac{1}{q_e^{2}} \, \left(1 + \lambda \sin^{2}\theta_e\right)$",
             fontsize=15, color=INK, va="baseline")
    fig.text(X1, 0.872,
             "$q_e$  force density        $\\theta_e$  angle to the reference "
             "direction, defined bottom right        "
             f"$\\lambda$ = {LAMBDA:g}  weight on the penalty        "
             f"$\\mu$ = {ANCHOR_COST:g}  cost of a new anchor",
             fontsize=8.8, color=INK_2, va="top")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor=SURFACE)
    stem = os.path.splitext(OUT)[0]
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", facecolor=SURFACE)
    # Illustrator's own format is PDF-based, so the PDF opens directly under a .ai
    # name; everything stays live vector, nothing is flattened.
    shutil.copyfile(f"{stem}.pdf", f"{stem}.ai")
    for ext in ("png", "pdf", "svg", "ai"):
        f = f"{stem}.{ext}"
        print(f"wrote {os.path.relpath(f, HERE)}  ({os.path.getsize(f)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
