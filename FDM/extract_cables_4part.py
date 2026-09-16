"""Extract cable trajectories from the 4-part FDM result.

Reuses the routing in figure_cable_extraction.py (weighted-graph Dijkstra over
directed edges, w_e = 1/q_e^2 * (1 + lambda sin^2 theta_e)) but reads the 4-part
FDM network, and writes the retained cables out as vertex-index polylines for
the FEM optimiser (FDM/optimise_4part.py).

Two things are NOT taken from the crossvault defaults:

  * Q_THRESHOLD.  The crossvault figure uses a fixed 2.0, which is far above the
    whole q range of this network.  The threshold is taken from a percentile of
    THIS network's q instead (--qpct, default 95).

  * LAMBDA.  figure_cable_extraction's docstring documents a plateau in lambda
    over which the retained network does not change, and the coupling
    ANCHOR_COST >= lambda/2 needed to keep chain ends tied to each other rather
    than escaping radially to the boundary.  This script scans lambda, prints the
    plateau, and picks the middle of the widest one, with
    ANCHOR_COST = max(5, lambda/2).

    python FDM/extract_cables_4part.py [--lam L] [--qpct P]
"""
import os, sys, json, argparse, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_cable_extraction as fce

HERE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(HERE, "data", "4part")
RESULT = os.path.join(DATA, "mesh_out_4part_latest.json")
OUTJ   = os.path.join(DATA, "cable_paths_4part.json")
OUTP   = os.path.join(DATA, "4part_cables.png")

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=None, help="skip the scan, use this lambda")
ap.add_argument("--qpct", type=float, default=97.0)
ap.add_argument("--qpct-force", action="store_true",
                help="use the --qpct percentile instead of the plateau scan")
ap.add_argument("--min-len", type=float, default=0.200,
                help="drop traced cables shorter than this [m]; p95 on the 2-part "
                     "network left short 3-vertex stubs that are noise, not cable")
ap.add_argument("--qth", type=float, default=None)
args = ap.parse_args()


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


V, F, E = load()
bd = fce.boundary_vertices(F)
q  = np.array([e[2] for e in E])
print(f"{len(V)} vertices  {len(F)} faces  {len(E)} edges  {len(bd)} boundary")
print(f"q: min {q.min():.4f}  median {np.median(q):.4f}  mean {q.mean():.4f}  max {q.max():.4f}")
for p in (80, 90, 93, 95, 97, 99):
    print(f"   p{p} q = {np.percentile(q, p):.4f}  ({(q >= np.percentile(q, p)).sum()} edges)")
print(f"   crossvault default Q_THRESHOLD = {fce.Q_THRESHOLD} would select "
      f"{(q >= fce.Q_THRESHOLD).sum()} of {len(E)} edges -> "
      f"{'UNUSABLE' if (q >= fce.Q_THRESHOLD).sum() < 10 else 'ok'}")


def _hi_set(t):
    return frozenset(frozenset((u, w)) for u, w, qq in E if qq >= t)


def pick_qthreshold():
    """Middle of the widest plateau over which the selected edge set is constant."""
    cand = np.unique(np.r_[np.linspace(q.min(), q.max(), 200),
                           [np.percentile(q, p) for p in (90, 95, 97, 99)]])
    cand = cand[(cand > np.percentile(q, 85)) & (cand < q.max())]
    runs, cur = [], [cand[0]]
    sig = _hi_set(cand[0])
    for t in cand[1:]:
        s_ = _hi_set(t)
        if s_ == sig and len(s_) >= 8:
            cur.append(t)
        else:
            runs.append((cur, sig)); cur, sig = [t], s_
    runs.append((cur, sig))
    runs = [r for r in runs if len(r[1]) >= 8]
    best = max(runs, key=lambda r: r[0][-1] - r[0][0])
    print(f"\nq-threshold plateau scan:")
    for r, s_ in sorted(runs, key=lambda r: -(r[0][-1] - r[0][0]))[:6]:
        print(f"  q in [{r[0]:.3f}, {r[-1]:.3f}]  -> {len(s_)} edges"
              f"{'   <== widest' if r is best[0] else ''}")
    return float(0.5 * (best[0][0] + best[0][-1]))


qth = args.qth if args.qth is not None else (
    float(np.percentile(q, args.qpct)) if args.qpct_force else pick_qthreshold())

apex = fce.apex_point(V)
s2 = {frozenset((u, w)): fce.sin2_theta(V, apex, u, w) for u, w, _ in E}
r_apex = np.hypot(apex[0], apex[1])
r_max  = max(np.hypot(V[k][0], V[k][1]) for k in V)
print(f"\napex (xy) = ({apex[0]:+.4f}, {apex[1]:+.4f})  |apex| = {r_apex:.4f} m "
      f"({100*r_apex/r_max:.1f}% of max radius)")
print("  NOTE: with four lobes the 'crown plateau' is the top 5% of z spread over "
      "all four lobes, so the apex centroid lands near the centre of the plan.")
print("  That IS a meaningful outward-radial reference here (the anchor ring is a "
      "circle centred on the same point and every cable runs centre->boundary), but "
      "it is NOT the top of any single lobe: it is the saddle between them.")
print(f"  q_threshold = {qth:.4f}  "
      f"({'p%g' % args.qpct if (args.qpct_force or args.qth is not None) else 'plateau midpoint'}; "
      f"{(q >= qth).sum()} edges)")


def run(lam):
    ac = max(5.0, lam / 2.0)
    res = fce.extract(V, E, s2, bd, lam, qth, anchor_cost=ac, apex=apex)
    q_of = {frozenset((u, w)): qq for u, w, qq in E}

    # A route exists to carry a chain END out to a support.  On the crossvault the
    # arch ends are interior, so every end needs one.  Here the four high-force
    # creases already run all the way out to the anchor ring, so every chain end is
    # ALREADY a boundary vertex — and because tie_to_cables lets a route terminate
    # for free on an already-retained cable, routing them anyway sent each end on a
    # 25-43 vertex excursion right across the dome to find another crease.  Drop the
    # routes off any chain component that already touches the boundary.
    comps_hi, _ = fce.graph_components(res["hi_edges"])
    anchored = {v for c in comps_hi if c & bd for v in c}
    res["tip_routes"] = {t: r for t, r in res["tip_routes"].items()
                         if t not in anchored}
    keep_routes = {e for t, r in res["tip_routes"].items()
                   for e in (frozenset((a, b)) for a, b in zip(r[:-1], r[1:]))}
    res["edges"] = set(res["hi_edges"]) | keep_routes
    res["n_dropped_routes"] = len(res["routes"]) - len(keep_routes)
    # trace_cables() is what turns the retained EDGE SET into ordered polylines;
    # the raw res["edges"] set on its own is fragmented and is not what belongs in
    # cable_paths.
    cables = fce.trace_cables(V, res["edges"], q_of, res["hi_edges"],
                              res["tip_routes"])
    n_raw = len(cables)
    cables = [c for c in cables if c["length"] >= args.min_len]
    n_anch = len({v for e in res["edges"] for v in tuple(e)} & bd)
    return res, cables, n_anch, ac, n_raw


# ── lambda plateau scan ──────────────────────────────────────────────────────
if args.lam is None:
    print("\nlambda plateau scan (ANCHOR_COST = max(5, lambda/2)):")
    print("  lam   anchcost  cables  anchors  kept  edges  signature")
    rows = []
    for lam in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 40.0]:
        res, cables, n_anch, ac, _nr = run(lam)
        sig = hash(frozenset(res["edges"]))
        rows.append((lam, sig, len(cables), n_anch, len(res["kept"]), len(res["edges"])))
        print(f"  {lam:5.1f} {ac:8.1f} {len(cables):7d} {n_anch:8d} "
              f"{len(res['kept']):5d} {len(res['edges']):6d}   {sig & 0xffffff:06x}")
    # widest run of identical signatures
    best, cur = None, [rows[0]]
    for r in rows[1:]:
        if r[1] == cur[-1][1]:
            cur.append(r)
        else:
            if best is None or len(cur) > len(best): best = cur
            cur = [r]
    if best is None or len(cur) > len(best): best = cur
    LAM = float(np.median([r[0] for r in best]))
    print(f"  -> widest plateau: lambda {best[0][0]:g} .. {best[-1][0]:g} "
          f"({len(best)} samples), taking lambda = {LAM:g}")
else:
    LAM = args.lam

res, cables, n_anch, ANCHOR_COST, n_raw = run(LAM)
print(f"\nlambda = {LAM:g}, anchor_cost = {ANCHOR_COST:g}")
print(f"routes: {res.get('n_dropped_routes', 0)} route edges dropped because their "
      f"chain already reaches the boundary")
print(f"min-length cut {args.min_len*1000:.0f} mm dropped {n_raw - len(cables)} "
      f"of {n_raw} traced polylines as stubs")
print(f"{len(cables)} cables, {len(res['edges'])} retained edges, {n_anch} boundary anchors, "
      f"{len(res['kept'])} connected components kept, {len(res['dropped'])} dropped")

Varr = np.array([V[k] for k in sorted(V)])
cen  = np.array([0.0, 0.0])


def cable_az(path):
    p = np.array([V[v][:2] for v in path])
    r = np.hypot(p[:, 0], p[:, 1])
    far = p[np.argmax(r)]
    return float(np.degrees(np.arctan2(far[1], far[0])) % 360)


for i, c in enumerate(cables):
    print(f"  cable {i:2d}: {c['edges']:3d} edges  len {c['length']:.3f} m  "
          f"q {c['q_min']:.3f}..{c['q_max']:.3f}  az {cable_az(c['path']):6.1f} deg  "
          f"ends {c['path'][0]}->{c['path'][-1]}  "
          f"{'(anchored both ends)' if c['path'][0] in bd and c['path'][-1] in bd else ''}")

# ── 4-fold symmetry of the cable SET (geometric, not index-wise) ─────────────
print("\n4-fold symmetry check of the extracted cable set:")


def resample(path, n=40):
    p = np.array([V[v][:2] for v in path])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]


samples = [resample(c["path"]) for c in cables]
for ang in (90.0, 180.0, 270.0):
    c_, s_ = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    R = np.array([[c_, -s_], [s_, c_]])
    worst = 0.0
    for S in samples:
        Sr = S @ R.T
        # one-sided Hausdorff from the rotated cable to the nearest cable
        dists = []
        for T in samples:
            d1 = np.linalg.norm(Sr[:, None, :] - T[None, :, :], axis=2)
            dists.append(float(np.max(np.min(d1, axis=1))))
        worst = max(worst, min(dists))
    print(f"  rotate {ang:5.0f} deg: worst cable maps onto the set to within "
          f"{1000*worst:.1f} mm")
print("  (the cable set is NOT forced to be index-wise symmetric: the triangulation "
      "is not D4, so forcing it would produce polylines whose consecutive vertices "
      "are not mesh edges.  D4 is imposed in optimise_4part.py at the PARAMETER "
      "level: cables in the same 90-deg orbit share one rest-scale.)")

# ── save ─────────────────────────────────────────────────────────────────────
out = {}
for i, c in enumerate(sorted(cables, key=lambda c: cable_az(c["path"]))):
    out[f"C{i}_{cable_az(c['path']):.0f}deg"] = [int(v) for v in c["path"]]
json.dump(out, open(OUTJ, "w"), indent=1)
json.dump({"lambda": LAM, "anchor_cost": ANCHOR_COST, "q_threshold": qth,
           "q_percentile": args.qpct,
           "apex_xy": [float(apex[0]), float(apex[1])],
           "n_cables": len(cables), "n_anchors": n_anch,
           "cables": [{"name": k, "n": len(v)} for k, v in out.items()]},
          open(OUTJ.replace(".json", "_meta.json"), "w"), indent=1)
print(f"\nSaved: {OUTJ}  ({len(out)} cables)")

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(131)
segs = [[V[u][:2], V[w][:2]] for u, w, _ in E]
ax.add_collection(LineCollection(segs, colors="#e2e0da", linewidths=0.4))
qn = (q - q.min()) / max(float(np.ptp(q)), 1e-9)
ax.add_collection(LineCollection(segs, colors=plt.cm.plasma(qn), linewidths=0.8))
ax.plot(*apex, "o", color="#2a78d6", ms=7, zorder=5)
ax.set_aspect("equal"); ax.autoscale()
ax.set_title(f"force densities q  (apex $\\bullet$)", fontsize=9)

ax2 = fig.add_subplot(132)
ax2.add_collection(LineCollection(segs, colors="#e2e0da", linewidths=0.4))
hi = [[V[u][:2], V[w][:2]] for u, w, qq in E if qq >= qth]
ax2.add_collection(LineCollection(hi, colors="#8a8983", linewidths=1.6))
for c in cables:
    p = np.array([V[v][:2] for v in c["path"]])
    ax2.plot(p[:, 0], p[:, 1], color="#e34948", lw=1.8)
anchs = np.array([V[v][:2] for v in ({vv for e in res["edges"] for vv in tuple(e)} & bd)])
if len(anchs):
    ax2.plot(anchs[:, 0], anchs[:, 1], "o", color="#2a78d6", ms=4, zorder=6)
ax2.set_aspect("equal"); ax2.autoscale()
ax2.set_title(f"{len(cables)} cables (red), q>={qth:.2f} chains (grey), "
              f"{n_anch} anchors", fontsize=9)

ax3 = fig.add_subplot(133, projection="3d")
vk = sorted(V)
vi = {v: i for i, v in enumerate(vk)}
Vf = np.array([V[v] for v in vk])
ax3.add_collection3d(Poly3DCollection([Vf[[vi[a] for a in f]] for f in F],
                                      alpha=0.15, facecolor="#cfd6dd",
                                      edgecolor="#e2e0da", linewidths=0.1))
ax3.add_collection3d(Line3DCollection(
    [[V[a], V[b]] for c in cables for a, b in zip(c["path"][:-1], c["path"][1:])],
    colors="#e34948", linewidths=2.0))
ax3.set_box_aspect([1, 1, 0.5]); ax3.view_init(elev=32, azim=-60)
ax3.set_xlim(Vf[:, 0].min(), Vf[:, 0].max()); ax3.set_ylim(Vf[:, 1].min(), Vf[:, 1].max())
ax3.set_zlim(0, Vf[:, 2].max())
ax3.set_title("cable trajectories in 3D", fontsize=9)

fig.suptitle(f"4part cable extraction — lambda={LAM:g}, anchor_cost={ANCHOR_COST:g}, "
             f"q_threshold={qth:.3f}, {len(cables)} cables", fontsize=10)
fig.tight_layout()
fig.savefig(OUTP, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTP}")
