"""Extract cable trajectories from the 2-part (middle-crease) FDM result.

Reuses the routing in figure_cable_extraction.py — the mesh read as a weighted
graph with

    w_e = (1 / q_e^2) * (1 + lambda * sin^2(theta_e))

routed with Dijkstra over DIRECTED edges (THETA_MODE = "hybrid": a route's first
edge is referenced to the outward radial direction, every edge after it to the
turn from its predecessor) — but reads the 2-part FDM network instead of the
crossvault and writes the retained cables out as vertex-index polylines for the
FEM optimiser.

Two things differ from the crossvault and are worth stating plainly.

1.  Q_THRESHOLD.  The crossvault uses a fixed q >= 2.0.  On this network q runs
    0.010 - 2.485 with a median of 0.153, so 2.0 selects almost nothing; the
    95th percentile (q >= 1.224, 84 edges) is used instead.  That set is the
    crease band along x ~ 0 plus the fan at the southern edge — visibly the
    force concentration, see the PNG.

2.  The apex.  apex_point() takes the centroid of the top 5 % of z.  This shape
    has TWO lobes (crowns at x = +-0.29 m, z = 0.255 m) separated by a crease at
    x = 0 (z = 0.157 m), so that centroid lands at (0.000, 0.018) — in the
    CREASE, a saddle, not a crown.  It is kept anyway, and it is defensible
    here, but not for the reason it is on a vault:
      * the load path that matters IS the crease, and it runs through that
        point, so "radial from the apex" is very nearly "along the crease" for
        every crease edge — sin^2(theta) ~ 0 there, which is exactly the
        cheap direction we want;
      * the supports are the whole outer perimeter, so distance from that point
        is a sound monotone measure of "outward" for the TIES_OUTWARD rule.
    Referencing each edge to the nearer LOBE apex instead was tried; it changes
    nothing structural (the lobes carry almost no force density) and gives a
    more fragmented network — 5 components / 126 edges against 3 / 153.

LAMBDA / ANCHOR_COST are not free.  The module docstring of
figure_cable_extraction.py requires ANCHOR_COST >= lambda/2, and the value is
picked by a plateau scan (--scan) rather than inherited: with mu = 10 the
extracted edge set is identical for lambda = 16 - 23, the widest plateau in
1 - 45, so LAMBDA = 20 sits mid-band; with lambda = 20 the set is identical for
mu = 6 - 40.

    python3 FDM/extract_cables_2part.py            # extract + write + plot
    python3 FDM/extract_cables_2part.py --scan     # the lambda / mu plateau scan
"""
import argparse
import collections
import hashlib
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

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "data", "2part", "mesh_out_2part_smooth_latest.json")
OUTJ = os.path.join(HERE, "data", "2part", "cable_paths_2part.json")
OUTM = os.path.join(HERE, "data", "2part", "cable_paths_2part.meta.json")
OUTP = os.path.join(HERE, "data", "2part", "2part_cables.png")

LAMBDA = 20.0        # mid-plateau, see the module docstring
ANCHOR_COST = 10.0   # = LAMBDA / 2, the minimum the coupling rule allows
Q_PCT = 95.0         # percentile, not a fixed q: this network's q is 10x smaller
MIN_CABLE_EDGES = 3  # a 1-2 edge stub is not a cable worth detailing


def load():
    d = json.load(open(RESULT))
    d = d["data"] if "data" in d and "vertex" in d["data"] else d
    V = {int(k): np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)])
         for k, v in d["vertex"].items()}
    F = [d["face"][k] for k in sorted(d["face"], key=int)]
    E = []
    for key, attr in d["edgedata"].items():
        u, w = (int(t) for t in key.strip("()").split(","))
        q = attr.get("qpre", attr.get("q"))
        E.append((u, w, float(q[0] if isinstance(q, (list, tuple)) else q)))
    return V, F, E


def _sig(res):
    return hashlib.md5(str(sorted(tuple(sorted(e)) for e in res["edges"]))
                       .encode()).hexdigest()[:8]


def scan(V, E, s2, bd, apex, qth):
    """The plateau test: over what band of lambda (and of mu) is the extracted
    edge set literally unchanged?  Picking the midpoint of the widest band is
    what makes the choice defensible rather than inherited."""
    for mode, vals in (("lambda", np.arange(1.0, 45.5, 1.0)),
                       ("mu", np.arange(1.0, 40.5, 1.0))):
        prev, runs = None, []
        for v in vals:
            kw = (dict(lam=float(v), anchor_cost=ANCHOR_COST) if mode == "lambda"
                  else dict(lam=LAMBDA, anchor_cost=float(v)))
            r = fce.extract(V, E, s2, bd, kw["lam"], qth, apex=apex,
                            anchor_cost=kw["anchor_cost"])
            s = _sig(r)
            an = len({x for e in r["edges"] for x in tuple(e)} & bd)
            if s != prev:
                runs.append([v, v, s, len(r["edges"]), len(r["comps"]), an])
                prev = s
            else:
                runs[-1][1] = v
        held = "mu = %g" % ANCHOR_COST if mode == "lambda" else "lambda = %g" % LAMBDA
        print(f"\n{mode} plateaux ({held} held fixed, q >= {qth:.4f}):")
        for a, b, s, ne, nc, an in runs:
            print(f"  {mode:>6} {a:5.1f} - {b:5.1f}  width {b - a + 1:4.1f}  "
                  f"{ne:3d} edges  {nc} components  {an} anchors  [{s}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true",
                    help="run the lambda / mu plateau scan and stop")
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--mu", type=float, default=ANCHOR_COST)
    ap.add_argument("--q-pct", type=float, default=Q_PCT)
    args = ap.parse_args()

    V, F, E = load()
    bd = fce.boundary_vertices(F)
    q = np.array([e[2] for e in E])
    qth = float(np.percentile(q, args.q_pct))
    apex = fce.apex_point(V)

    print(f"{len(V)} vertices  {len(F)} faces  {len(E)} edges  {len(bd)} boundary")
    print(f"q: min {q.min():.4f}  median {np.median(q):.4f}  max {q.max():.4f}  "
          f"-> p{args.q_pct:g} = {qth:.4f}  ({(q >= qth).sum()} edges)")
    z = np.array([V[k][2] for k in V])
    print(f"apex (xy) = ({apex[0]:+.4f}, {apex[1]:+.4f})  z range {z.min():.4f} - "
          f"{z.max():.4f}   [NB: the centroid of the top 5 % of z lands in the "
          f"crease, not on a lobe crown — see the module docstring]")

    s2 = {frozenset((u, w)): fce.sin2_theta(V, apex, u, w) for u, w, _ in E}

    if args.scan:
        scan(V, E, s2, bd, apex, qth)
        return

    res = fce.extract(V, E, s2, bd, args.lam, qth, apex=apex,
                      anchor_cost=args.mu)
    anchors = sorted({x for e in res["edges"] for x in tuple(e)} & bd)
    print(f"\nlambda={args.lam:g}  mu={args.mu:g}: "
          f"{len(res['hi'])} high-force edges in {len(res['hi_comps'])} chains -> "
          f"{len(res['edges'])} retained edges, {len(res['comps'])} components "
          f"({len(res['kept'])} anchored, {len(res['dropped'])} discarded), "
          f"{len(anchors)} boundary anchors")

    q_of = {frozenset((u, w)): qq for u, w, qq in E}
    cables = fce.trace_cables(V, res["edges"], q_of, res["hi_edges"],
                              res["tip_routes"])
    kept = [c for c in cables if c["edges"] >= MIN_CABLE_EDGES]
    print(f"traced {len(cables)} cables, {len(kept)} with >= {MIN_CABLE_EDGES} "
          f"edges (the rest are stubs and are dropped)")

    names = [f"C{i:02d}" for i in range(len(kept))]
    print(f"{'name':>5} {'edges':>5} {'length':>8}  {'q range':>12}  {'ends':>17}")
    for nm, c in zip(names, kept):
        a, b = c["path"][0], c["path"][-1]
        ends = (f"{'anchor' if a in bd else 'cable':>7} -> "
                f"{'anchor' if b in bd else 'cable'}")
        print(f"{nm:>5} {c['edges']:>5} {c['length']:>7.3f} m  "
              f"{c['q_min']:>5.2f}-{c['q_max']:<5.2f}  {ends:>17}")

    # {name: [vertex indices]} — the same shape as data/C5/cable_paths_C5_obj.json,
    # which optimise_C5_16region.py reads with sorted(json.load(f).items()).
    with open(OUTJ, "w") as f:
        json.dump({nm: [int(v) for v in c["path"]] for nm, c in zip(names, kept)},
                  f, indent=1)
    with open(OUTM, "w") as f:
        json.dump(dict(source=os.path.relpath(RESULT, HERE),
                       lam=args.lam, anchor_cost=args.mu,
                       q_percentile=args.q_pct, q_threshold=qth,
                       apex_xy=[float(apex[0]), float(apex[1])],
                       n_anchors=len(anchors), anchors=anchors,
                       cables=[dict(name=nm, **{k: v for k, v in c.items()})
                               for nm, c in zip(names, kept)]), f, indent=1)
    print(f"wrote {os.path.relpath(OUTJ, HERE)} and "
          f"{os.path.relpath(OUTM, HERE)}")

    # ── the picture ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15.0, 5.4), facecolor=fce.SURFACE)
    ax = fig.add_axes([0.02, 0.06, 0.29, 0.86])
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w, _ in E],
                                     colors=fce.MESH, lw=0.45))
    hi = [(u, w) for u, w, qq in E if qq >= qth]
    ax.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w in hi],
                                     colors=fce.SERIES_1, lw=2.4,
                                     capstyle="round"))
    ax.autoscale()
    ax.set_title(f"a  the force concentration: q $\\geq$ p{args.q_pct:g} = "
                 f"{qth:.2f}\n{len(hi)} edges in {len(res['hi_comps'])} chains",
                 fontsize=10, color=fce.INK, loc="left")

    ax2 = fig.add_axes([0.34, 0.06, 0.29, 0.86])
    ax2.set_aspect("equal"); ax2.set_axis_off()
    ax2.add_collection(LineCollection([[V[u][:2], V[w][:2]] for u, w, _ in E],
                                      colors=fce.MESH, lw=0.45))
    ax2.add_collection(LineCollection(
        [[V[a][:2], V[b][:2]] for a, b in (tuple(e) for e in
                                           res["edges"] - res["hi_edges"])],
        colors=fce.ROUTE_LIGHT, lw=2.0, capstyle="round"))
    ax2.add_collection(LineCollection(
        [[V[a][:2], V[b][:2]] for a, b in (tuple(e) for e in
                                           res["edges"] & res["hi_edges"])],
        colors=fce.SERIES_1, lw=2.6, capstyle="round"))
    ax2.plot([V[a][0] for a in anchors], [V[a][1] for a in anchors], "o",
             color=fce.SERIES_2, ms=4.4, ls="none")
    ax2.plot(*apex, "o", color=fce.SERIES_2, ms=6)
    for nm, c in zip(names, kept):
        k = len(c["path"]) // 2
        ax2.text(*V[c["path"][k]][:2], nm, fontsize=7, color=fce.INK,
                 ha="center", va="center", zorder=9,
                 bbox=dict(fc=fce.SURFACE, ec="none", alpha=0.9, pad=1.0))
    ax2.autoscale()
    ax2.set_title(f"b  routed and anchored  ($\\lambda$ = {args.lam:g}, "
                  f"$\\mu$ = {args.mu:g})\n{len(kept)} cables, "
                  f"{len(anchors)} boundary anchors",
                  fontsize=10, color=fce.INK, loc="left")

    ax3 = fig.add_axes([0.655, 0.02, 0.34, 0.94], projection="3d",
                       computed_zorder=False)
    ax3.set_proj_type("ortho")
    ax3.add_collection3d(Poly3DCollection([[V[i] for i in f] for f in F],
                                          facecolors=fce.SURFACE, alpha=0.7,
                                          edgecolors="#cac8c2", linewidths=0.3))
    ax3.add_collection3d(Line3DCollection(
        [[V[a], V[b]] for a, b in (tuple(e) for e in res["edges"])],
        colors=fce.SERIES_1, linewidths=2.4, capstyle="round"))
    ax3.scatter([V[a][0] for a in anchors], [V[a][1] for a in anchors],
                [V[a][2] for a in anchors], color=fce.SERIES_2, s=13,
                depthshade=False)
    ax3.set_xlim(-0.62, 0.62); ax3.set_ylim(-0.62, 0.62); ax3.set_zlim(0, 0.42)
    ax3.set_box_aspect((1, 1, 0.42), zoom=1.25)
    ax3.view_init(elev=26, azim=-62)
    ax3.set_axis_off()
    ax3.set_title("c  the cable trajectories on the surface", fontsize=10,
                  color=fce.INK, loc="left")

    fig.savefig(OUTP, dpi=170, facecolor=fce.SURFACE)
    print(f"wrote {os.path.relpath(OUTP, HERE)}")


if __name__ == "__main__":
    main()
