"""
Dijkstra-based cable path generator.

Given a triangulated mesh and a starting angle on the boundary, finds the
smoothest path (minimum total turning angle) to the opposite boundary point.
Smoothness weight = turning angle at each interior vertex (straight = 0).
"""

import heapq
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MESH_PATH


def load_off(path: str):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    assert lines[0] == "OFF"
    nv, nf, _ = map(int, lines[1].split())
    V = np.array([list(map(float, lines[2 + i].split())) for i in range(nv)])
    F = np.array([list(map(int, lines[2 + nv + i].split()))[1:] for i in range(nf)])
    return V, F


def build_adjacency(V, F):
    """Return edge adjacency: adj[v] = list of neighbour vertex indices."""
    n = len(V)
    adj = [[] for _ in range(n)]
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            adj[a].append(b)
            adj[b].append(a)
    # Deduplicate
    adj = [list(set(nb)) for nb in adj]
    return adj


def find_boundary_vertices(F):
    from collections import Counter
    edge_count = Counter()
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            edge_count[tuple(sorted([a, b]))] += 1
    boundary = set()
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            boundary.add(a)
            boundary.add(b)
    return sorted(boundary)


def boundary_vertex_at_angle(V, boundary_verts, angle_deg: float) -> int:
    """Return the boundary vertex closest to the given polar angle (degrees)."""
    bv = np.array(boundary_verts)
    coords = V[bv, :2]
    centroid = coords.mean(axis=0)
    coords_c = coords - centroid
    angles = np.degrees(np.arctan2(coords_c[:, 1], coords_c[:, 0])) % 360.0
    target = angle_deg % 360.0
    diffs = np.abs(angles - target)
    diffs = np.minimum(diffs, 360.0 - diffs)
    return int(bv[np.argmin(diffs)])


def opposite_boundary_vertex(V, boundary_verts, source_idx: int) -> int:
    """Return the boundary vertex roughly opposite to source_idx."""
    bv = np.array(boundary_verts)
    centroid = V[bv, :2].mean(axis=0)
    src_vec = V[source_idx, :2] - centroid
    src_angle = math.atan2(src_vec[1], src_vec[0])
    opposite_angle = src_angle + math.pi

    coords_c = V[bv, :2] - centroid
    angles = np.arctan2(coords_c[:, 1], coords_c[:, 0])
    # Wrap the angular difference into [0, pi].  The previous
    # np.minimum(d, 2*pi - d) went *negative* whenever d > 2*pi, which happens
    # because opposite_angle reaches 3*pi/2 while arctan2 returns (-pi, pi];
    # argmin then locked onto that negative entry and returned a vertex ~90 deg
    # from the true antipode for every source except the one near 0 deg.  The
    # wale cable (90 deg) was a 95 deg chord of 0.878 m rather than a 1.208 m
    # diameter because of this.
    delta = angles - opposite_angle
    diffs = np.abs(np.arctan2(np.sin(delta), np.cos(delta)))
    return int(bv[np.argmin(diffs)])


def turning_angle(V, u: int, v: int, w: int) -> float:
    """Turning angle (radians) at vertex v when coming from u, going to w."""
    d1 = V[v] - V[u]
    d2 = V[w] - V[v]
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = np.clip(np.dot(d1 / n1, d2 / n2), -1.0, 1.0)
    return math.acos(cos_a)


def dijkstra_smooth_path(V, adj, boundary_set, source, target):
    """
    Dijkstra with weight = cumulative turning angle along the path.
    Returns ordered list of vertex indices from source to target.
    """
    # State: (cost, current_vertex, previous_vertex)
    INF = float('inf')
    dist = {source: 0.0}
    prev = {source: None}
    prev_vertex = {source: -1}

    heap = [(0.0, source, -1)]   # (cost, node, came_from)

    while heap:
        cost, u, from_v = heapq.heappop(heap)
        if cost > dist.get(u, INF):
            continue
        if u == target:
            break
        for w in adj[u]:
            if from_v >= 0:
                angle = turning_angle(V, from_v, u, w)
            else:
                angle = 0.0
            new_cost = cost + angle
            if new_cost < dist.get(w, INF):
                dist[w] = new_cost
                prev[w] = u
                prev_vertex[w] = from_v
                heapq.heappush(heap, (new_cost, w, u))

    # Reconstruct path
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path[0] == source else []


def generate_cable_path(cable_angle_deg, mesh_path=MESH_PATH):
    """
    Return an ordered list of vertex indices for a smooth cable path across
    the mesh at the given polar angle.

    Projects vertices onto the source→target axis, keeps those within a band of
    ≈2 edge lengths, then takes the one closest to the axis in each axial bin of
    one edge length.  The result advances monotonically along the axis, so its
    arc length is the span of the chord (plus mesh roughness) rather than a
    zigzag across the band.
    """
    V, F = load_off(mesh_path)
    boundary = find_boundary_vertices(F)

    source = boundary_vertex_at_angle(V, boundary, cable_angle_deg)
    target = opposite_boundary_vertex(V, boundary, source)

    # ── Axis projection ────────────────────────────────────────────────────
    p0 = V[source, :2]
    p1 = V[target, :2]
    axis     = p1 - p0
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-12:
        raise RuntimeError("Source and target boundary vertices are coincident.")
    axis_dir = axis / axis_len

    pts_xy = V[:, :2]
    t_vals = (pts_xy - p0) @ axis_dir                          # parameter ∈ [0, L]
    perp   = pts_xy - (p0[None] + t_vals[:, None] * axis_dir)
    dist   = np.linalg.norm(perp, axis=1)

    # Band width ≈ 2 median edge lengths
    edge_lens = [np.linalg.norm(V[F[i, j]] - V[F[i, (j+1) % 3]])
                 for i in range(min(200, len(F))) for j in range(3)]
    edge = float(np.median(edge_lens))
    band = 2.0 * edge

    in_band = (dist < band) & (t_vals >= -1e-6) & (t_vals <= axis_len + 1e-6)
    cand = np.where(in_band)[0]

    # One node per axial station, nearest the axis.  Keeping *every* vertex in
    # the band and sorting by t made the path weave from one side of the band to
    # the other: 88% of its length was transverse and the polyline ran 7-8x the
    # true span (6.32 m across a 1.21 m disc).  As a single SlidingCable with one
    # tension, that zigzag could absorb any rest-length change by wiggling
    # sideways, so the cable barely constrained the dome at all.
    nbins  = max(2, int(round(axis_len / edge)))
    bin_of = np.clip((t_vals[cand] / axis_len * nbins).astype(int), 0, nbins - 1)
    path = []
    for b in range(nbins):
        sel = cand[bin_of == b]
        if len(sel):
            path.append(int(sel[np.argmin(dist[sel])]))

    # Guarantee the exact boundary endpoints, and drop duplicates in order
    path = [p for p in path if p not in (source, target)]
    seen, uniq = set(), []
    for p in path:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    path = [source] + uniq + [target]

    if len(path) < 2:
        raise RuntimeError(
            f"Cable path too short ({len(path)} vertices) — check mesh or angle."
        )
    return path


if __name__ == "__main__":
    path = generate_cable_path(cable_angle_deg=45.0)
    print(f"Cable path: {len(path)} vertices")
    print(f"  start={path[0]}, end={path[-1]}")


# ── Fixed cable directions for sensitivity analysis ───────────────────────────
# Wale direction = vertical (90 deg polar), course = horizontal (0 deg polar).
WALE_CABLE_ANGLE   = 90.0
COURSE_CABLE_ANGLE =  0.0


def cable_path_length(path: list, V) -> float:
    """Arc length of a cable path (metres) on the reference mesh vertices V."""
    import numpy as np
    return float(sum(
        np.linalg.norm(V[path[k + 1]] - V[path[k]])
        for k in range(len(path) - 1)
    ))
