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
    diffs = np.abs(angles - opposite_angle)
    diffs = np.minimum(diffs, 2 * math.pi - diffs)
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

    Uses straight-line axis projection: projects all vertices onto the
    source→target axis and selects those within a band of ≈2 edge-lengths,
    giving a smooth cross-section path rather than a zigzagging graph walk.
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
    band = 2.0 * float(np.median(edge_lens))

    in_band = (dist < band) & (t_vals >= -1e-6) & (t_vals <= axis_len + 1e-6)
    in_band[source] = True
    in_band[target] = True

    selected = np.where(in_band)[0]
    order    = np.argsort(t_vals[selected])
    path     = selected[order].tolist()

    # Guarantee source at front, target at back
    if source in path:
        path.remove(source)
    if target in path:
        path.remove(target)
    path = [source] + path + [target]

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
