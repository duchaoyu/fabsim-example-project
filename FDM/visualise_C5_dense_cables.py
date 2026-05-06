"""Extract 8 D8 spokes on the dense C5 FDM result and render cables on it.

The dense .obj has different vertex IDs from the .off, so cable_paths_C5.json
indices don't apply. We rebuild the 8 cables by greedy edge-walks from the apex
along each ridge angle θ = 11.25° + k·45° on the form-found mesh.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh

HERE     = os.path.dirname(os.path.abspath(__file__))
RESULT   = os.path.join(HERE, "data", "C5", "mesh_out_C5_dense_latest.json")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_dense_cables.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "cable_paths_C5_dense.json")

APEX_RADIUS_TOL = 1e-3      # apex = unique vertex with smallest xy radius

# Spoke angles are detected from the FDM result itself (see code below) — no
# longer hardcoded to 11.25° + k·45° (which was the .off remesher's choice).


# ── Load form-found dense mesh ────────────────────────────────────────────────
mesh = Mesh.from_json(RESULT)
verts_keys = list(mesh.vertices())
V = np.array([mesh.vertex_coordinates(v) for v in verts_keys])
key_to_idx = {k: i for i, k in enumerate(verts_keys)}

# Adjacency (per-vertex neighbour key list)
adj = {k: list(mesh.vertex_neighbors(k)) for k in verts_keys}

# Boundary set
bdry_keys = set(mesh.vertices_on_boundary())

print(f"Loaded {RESULT}")
print(f"  {len(V)} verts, {mesh.number_of_faces()} faces, {len(bdry_keys)} boundary")
print(f"  x[{V[:,0].min():.4f},{V[:,0].max():.4f}]  z[{V[:,2].min():.4f},{V[:,2].max():.4f}]")


# ── Apex vertex (smallest xy radius, ideally exactly at origin) ───────────────
r_xy = np.linalg.norm(V[:, :2], axis=1)
apex_idx = int(np.argmin(r_xy))
apex_key = verts_keys[apex_idx]
print(f"Apex: vert {apex_key}, xy_r={r_xy[apex_idx]:.5f}, z={V[apex_idx,2]:.4f}")


# ── Force-density-driven cable extraction ────────────────────────────────────
# Cables = paths from apex to boundary that maximise force-density. We use
# Dijkstra from apex with edge weight = 1/q (so high-q edges are short),
# pick 8 distinct boundary endpoints with smallest distance, then take their
# Dijkstra paths back to apex. This recovers exactly the radial high-q
# "spokes" that the FDM optimisation produced.
import heapq

edges_list = list(mesh.edges())
edge_q = {tuple(sorted([u, v])): float(mesh.edge_attribute((u, v), "qpre"))
          for u, v in edges_list}
print(f"FDM q range: [{min(edge_q.values()):.3f}, {max(edge_q.values()):.3f}], "
      f"mean={np.mean(list(edge_q.values())):.3f}")


# ── Dijkstra from apex with weight = 1/q² (stronger high-q preference) ──────
def dijkstra_from_apex():
    dist = {k: np.inf for k in verts_keys}
    prev = {k: None for k in verts_keys}
    dist[apex_key] = 0.0
    pq = [(0.0, apex_key)]
    while pq:
        d, uk = heapq.heappop(pq)
        if d > dist[uk]: continue
        for vk in adj[uk]:
            qe = edge_q[tuple(sorted([uk, vk]))]
            w = 1.0 / max(qe, 1e-6) ** 2        # quadratic: strongly prefers high q
            nd = d + w
            if nd < dist[vk]:
                dist[vk] = nd
                prev[vk] = uk
                heapq.heappush(pq, (nd, vk))
    return dist, prev

dist, prev = dijkstra_from_apex()
def reconstruct(target):
    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

# Pick 8 boundary endpoints: smallest-distance per 45° angular bucket
def angle_diff_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)

apex_xy = V[apex_idx, :2]
bdry_list = list(bdry_keys)
bdry_th = []
for bk in bdry_list:
    bv = V[key_to_idx[bk], :2] - apex_xy
    bdry_th.append(np.degrees(np.arctan2(bv[1], bv[0])) % 360.0)
bdry_th = np.array(bdry_th)
bdry_dist = np.array([dist[k] for k in bdry_list])

# Best cable terminus = boundary vertex whose highest-q incident edge is largest.
# That is the vertex where the FDM-found radial cable actually anchors.
bdry_max_q = []
for bk in bdry_list:
    qs_b = [edge_q[tuple(sorted([bk, n]))] for n in adj[bk]]
    bdry_max_q.append(max(qs_b))
bdry_max_q = np.array(bdry_max_q)

# Score combines anchor-q (high = good cable terminus) and apex distance
# (small = reachable on a high-q chain). Normalise both to [0,1] and add.
inv_dist = 1.0 / (bdry_dist + 1e-6)
score = bdry_max_q / bdry_max_q.max() + inv_dist / inv_dist.max()

N_CABLES    = 8
MIN_SEP_DEG = 25.0
order = np.argsort(-score)                      # descending
selected, selected_th = [], []
for i in order:
    if len(selected) == N_CABLES:
        break
    th_i = float(bdry_th[i])
    if all(angle_diff_deg(th_i, t) > MIN_SEP_DEG for t in selected_th):
        selected.append(bdry_list[i])
        selected_th.append(th_i)
# Sort by angle for nicer printing
ord_th = np.argsort(selected_th)
selected    = [selected[i] for i in ord_th]
selected_th = [selected_th[i] for i in ord_th]
SPOKE_ANGLES = list(selected_th)
print(f"\n8 cable endpoints (angle, dist) — by min-q-distance from apex:")
for k, (bk, th) in enumerate(zip(selected, selected_th)):
    print(f"  cable {k}: vertex {bk}  θ={th:6.2f}°  dist={dist[bk]:.3f}")

cables = {}
print("\nReconstructing high-q paths:")
for k, (bk, theta) in enumerate(zip(selected, selected_th)):
    path = reconstruct(bk)
    name = f"S{k}_{theta:.2f}deg"
    cables[name] = [int(v) for v in path]
    end_xy = V[key_to_idx[path[-1]], :2]
    end_th = np.degrees(np.arctan2(end_xy[1], end_xy[0]))
    end_r  = np.linalg.norm(end_xy)
    # mean q along path
    qs = [edge_q[tuple(sorted([path[i], path[i+1]]))] for i in range(len(path)-1)]
    print(f"  {name}: {len(path)} verts, end θ={end_th:6.2f}°, r={end_r:.4f}, "
          f"mean q={np.mean(qs):.3f}, min q={min(qs):.3f}")

# ── Circular (hoop) cable: high-q INNER circumferential ring ─────────────────
# The FDM force flow shows a clear inner polygonal ring (around r ≈ 0.2 for
# the C5 dome) connecting the bases of the radial spokes. We want THAT ring,
# not the outer boundary.
#
# Strategy:
#   1. consider only non-boundary edges (boundary = outer ring r ≈ 0.6).
#   2. keep edges that are mostly tangential (|Δr|/L small) AND high-q.
#   3. cluster them by radius and pick the *innermost* substantial cluster
#      (>= 6 edges) — that is the inner hoop.
#   4. walk the cluster's subgraph as a closed loop in CCW order.
print("\nExtracting INNER circular hoop cable...")

R_max_xy = float(np.linalg.norm(V[:, :2], axis=1).max())
R_BOUNDARY_EXCLUDE = 0.85 * R_max_xy           # ignore anything ≥ 85% of outer radius

edge_circ = {}
qs_arr = np.array(list(edge_q.values()))
q_thr  = float(np.percentile(qs_arr, 70))      # top 30% of q (inner hoop has lower q than spokes)
for u, v in edges_list:
    # exclude any edge touching boundary or sitting near outer rim
    if u in bdry_keys or v in bdry_keys:
        continue
    iu, iv = key_to_idx[u], key_to_idx[v]
    pu, pv = V[iu, :2], V[iv, :2]
    ru, rv = np.linalg.norm(pu), np.linalg.norm(pv)
    r_mean = 0.5 * (ru + rv)
    if r_mean > R_BOUNDARY_EXCLUDE:
        continue
    L = np.linalg.norm(pv - pu) + 1e-9
    dr = abs(rv - ru)
    qe = edge_q[tuple(sorted([u, v]))]
    if qe < q_thr:
        continue
    circ_score = 1.0 - dr / L                    # 1 = pure tangent, 0 = pure radial
    if circ_score > 0.6:
        edge_circ[tuple(sorted([u, v]))] = (qe, r_mean)

print(f"  candidate inner circumferential edges (q>{q_thr:.3f}, r<{R_BOUNDARY_EXCLUDE:.3f}): {len(edge_circ)}")

# Group by radius. Score each cluster by sum of q across its edges and pick
# the strongest cluster (the brightest hoop in the force-flow plot, e.g. the
# r≈0.2 heptagon visible in the C5 result), not just the innermost one.
hoop_added = False
if edge_circ:
    bin_w = 0.04
    bins = np.arange(0, R_BOUNDARY_EXCLUDE + bin_w, bin_w)
    # Per-bin: edge count and sum of q.
    bin_count = np.zeros(len(bins) - 1)
    bin_qsum  = np.zeros(len(bins) - 1)
    for (qe, r) in edge_circ.values():
        b = int(r // bin_w)
        if 0 <= b < len(bin_count):
            bin_count[b] += 1
            bin_qsum[b]  += qe
    # candidates with enough edges to form a closed ring around the apex
    candidate_bins = [b for b in range(len(bin_count)) if bin_count[b] >= 6]
    if candidate_bins:
        # Pick by largest q-sum (strongest hoop).
        peak_bin = max(candidate_bins, key=lambda b: bin_qsum[b])
        r_lo, r_hi = bins[peak_bin], bins[peak_bin + 1]
        # Widen to neighbouring bins if also populous, so a hoop straddling
        # a bin boundary still forms a single closed loop.
        if peak_bin + 1 < len(bin_count) and bin_count[peak_bin + 1] >= 4:
            r_hi = bins[peak_bin + 2]
        if peak_bin > 0 and bin_count[peak_bin - 1] >= 4:
            r_lo = bins[peak_bin - 1]
        ring_edges = [e for e, (qe, r) in edge_circ.items() if r_lo <= r < r_hi]
        # Diagnostic: report all candidates so we see why this one won.
        print("  candidate hoop bins (radius bin, edges, sum-q):")
        for b in candidate_bins:
            mark = " *" if b == peak_bin else "  "
            print(f"    {mark} r∈[{bins[b]:.3f},{bins[b+1]:.3f}]  "
                  f"edges={int(bin_count[b])}  Σq={bin_qsum[b]:.2f}")

        ring_adj = {}
        for u, v in ring_edges:
            ring_adj.setdefault(u, []).append(v)
            ring_adj.setdefault(v, []).append(u)

        # Greedy CCW walk starting at smallest-angle vertex.
        start = min(ring_adj.keys(),
                    key=lambda k: np.arctan2(V[key_to_idx[k], 1], V[key_to_idx[k], 0]))
        path, prev_v, cur = [start], None, start
        while True:
            nbs = [n for n in ring_adj[cur] if n != prev_v and n not in path]
            if not nbs:
                if start in ring_adj[cur] and len(path) > 4:
                    path.append(start)
                break
            cur_th = np.arctan2(V[key_to_idx[cur], 1], V[key_to_idx[cur], 0])
            def ccw_step(n):
                nth = np.arctan2(V[key_to_idx[n], 1], V[key_to_idx[n], 0])
                return (nth - cur_th) % (2 * np.pi)
            nxt = min(nbs, key=ccw_step)
            path.append(nxt)
            prev_v, cur = cur, nxt
            if cur == start: break
            if len(path) > 200: break

        cables["H0_inner_hoop"] = [int(v) for v in path]
        hoop_added = True
        print(f"  inner hoop: {len(path)} verts at r ≈ [{r_lo:.3f}, {r_hi:.3f}], "
              f"closed = {path[0] == path[-1]}")
    else:
        print(f"  no inner ring found (max bin count {int(hist.max()) if len(hist) else 0})")

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"Saved cable JSON: {OUT_JSON}")


# ── Plot ──────────────────────────────────────────────────────────────────────
r_xy_all = np.linalg.norm(V[:, :2], axis=1)
F = np.array([[key_to_idx[k] for k in mesh.face_vertices(f)[:3]]
              for f in mesh.faces()])
face_verts = [[V[F[i, j]] for j in range(3)] for i in range(len(F))]
face_z_mean = V[F].mean(axis=1)[:, 2]
zmax = max(V[:, 2].max(), 1e-6)

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
poly = Poly3DCollection(face_verts, alpha=0.7,
                        facecolor=cm.viridis(face_z_mean / zmax),
                        edgecolor="white", linewidth=0.08)
ax1.add_collection3d(poly)

for name, path in cables.items():
    pts = np.array([V[key_to_idx[v]] for v in path])
    is_hoop = name.startswith("H")
    col   = "lime"   if is_hoop else "red"
    msize = 10       if is_hoop else 14
    ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
             color=col, linewidth=2.0, alpha=0.95)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                color="yellow" if is_hoop else "orange",
                s=msize, edgecolors=col, linewidths=0.4, zorder=5)

ax1.set_title(f"C5 dense FDM result — {len(V)}v / {len(F)}f  "
              f"(red = 8 spokes, lime = inner hoop)",
              color="white", fontsize=11)
ax1.set_xlabel("x", color="white", fontsize=8)
ax1.set_ylabel("y", color="white", fontsize=8)
ax1.set_zlabel("z", color="white", fontsize=8)
ax1.tick_params(colors="white", labelsize=7)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
xr = float(np.abs(V[:, :2]).max()) * 1.05
ax1.set_xlim(-xr, xr); ax1.set_ylim(-xr, xr); ax1.set_zlim(0, zmax * 1.2)
ax1.view_init(elev=25, azim=-50)

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

# Top-down: edges coloured & sized by q (force-density flow). Cables overlay.
edges_xy = [(V[key_to_idx[u], :2], V[key_to_idx[v], :2]) for u, v in edges_list]
qs = np.array([edge_q[tuple(sorted([u, v]))] for u, v in edges_list])
q_norm = matplotlib.colors.LogNorm(vmin=max(qs.min(), 0.01), vmax=qs.max())
order = np.argsort(qs)
for ei in order:
    p0, p1 = edges_xy[ei]
    qq = qs[ei]
    lw = 0.2 + 4 * (np.log10(qq + 0.01) - np.log10(0.01)) / (
            np.log10(qs.max()) - np.log10(0.01))
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]],
             color=cm.plasma(q_norm(qq)), linewidth=lw, alpha=0.7,
             solid_capstyle='round')

R_max = r_xy_all.max() * 1.05

bdry_pts = np.array([V[key_to_idx[k]] for k in bdry_keys])
ax2.scatter(bdry_pts[:, 0], bdry_pts[:, 1], c="cyan", s=18, zorder=4,
            label=f"boundary ({len(bdry_keys)})")

for name, path in cables.items():
    is_hoop = name.startswith("H")
    pts = np.array([V[key_to_idx[v]] for v in path])
    line_col = "lime" if is_hoop else "red"
    pt_col   = "yellow" if is_hoop else "orange"
    ax2.plot(pts[:, 0], pts[:, 1], color=line_col, linewidth=2.4, alpha=0.95)
    ax2.scatter(pts[:, 0], pts[:, 1], color=pt_col, s=18,
                edgecolors=line_col, linewidths=0.5, zorder=5)
    if is_hoop:
        continue
    end = pts[-1]
    angle = float(name.split("_")[1].rstrip("deg"))
    ax2.scatter(end[0], end[1], c="yellow", s=90, zorder=6, marker="*",
                edgecolors="red", linewidths=1)
    ax2.annotate(f"{angle:.1f}°", xy=(end[0], end[1]),
                 xytext=(end[0]*1.13, end[1]*1.13), color="red", fontsize=8,
                 ha="center", va="center")

ax2.set_title("Top-down — edge colour/thickness ∝ q (FDM force flow), "
              "red = 8 cables along high-q paths from apex",
              color="white", fontsize=9)
ax2.set_xlabel("x", color="white", fontsize=8)
ax2.set_ylabel("y", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)

fig.suptitle(f"C5 dense FDM result with stiffener spokes",
             color="white", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved PNG: {OUT_PNG}")
