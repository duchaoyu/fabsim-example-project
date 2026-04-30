"""
Discrete curvature computation for deformed FEM meshes.

Reads the original mesh connectivity (OFF file) and replaces vertex positions
with those from a deformed FEA output (_verts.csv), then computes per-vertex
Gaussian curvature K, mean curvature H, and principal curvatures k1, k2 using
the cotangent Laplacian.

Boundary vertices are excluded from all statistics (they are fixed in the FEM).
"""

import os
import numpy as np
import pandas as pd
# ── mesh IO ───────────────────────────────────────────────────────────────────

def read_off(path):
    """Return (verts, faces) as float64 and int32 arrays."""
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert lines[0] == "OFF"
    nv, nf, _ = map(int, lines[1].split())
    verts = np.array([list(map(float, lines[2 + i].split())) for i in range(nv)])
    faces = np.array(
        [list(map(int, lines[2 + nv + i].split()[1:])) for i in range(nf)],
        dtype=np.int32,
    )
    return verts, faces


def boundary_vertices(faces, n_verts):
    """Return set of vertex indices that lie on the mesh boundary."""
    edge_count = {}
    for f in faces:
        for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            key = (min(a, b), max(a, b))
            edge_count[key] = edge_count.get(key, 0) + 1
    bdry = set()
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            bdry.add(a)
            bdry.add(b)
    return bdry


# ── curvature via cotangent Laplacian ─────────────────────────────────────────

def compute_curvatures(verts, faces):
    """
    Per-vertex discrete curvatures using the cotangent Laplacian (Meyer et al.
    2003) for mean curvature H and angle-deficit for Gaussian curvature K.

    Δ_S x_i = (1/A_i) Σ_j ((cot α_ij + cot β_ij)/2) (x_j - x_i) = 2 H_i n_i
    K_i     = (2π - Σ θ_j) / A_i   (angle deficit)

    Returns dict with per-vertex arrays:
        H  – unsigned mean curvature  (m⁻¹)
        K  – Gaussian curvature       (m⁻²)
        k1 – max principal curvature  (m⁻¹)
        k2 – min principal curvature  (m⁻¹)
    """
    V = len(verts)
    areas        = np.zeros(V)
    Lx           = np.zeros((V, 3))   # accumulates Σ_j cot_opp * (x_j - x_i)
    angle_sum    = np.zeros(V)        # sum of face angles at each vertex

    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        pa, pb, pc = verts[a], verts[b], verts[c]

        # Edge vectors leaving each vertex
        ab, ac = pb - pa, pc - pa
        ba, bc = pa - pb, pc - pb
        ca, cb = pa - pc, pb - pc

        # Triangle area
        cross = np.cross(ab, ac)
        A = 0.5 * np.linalg.norm(cross)
        if A < 1e-15:
            continue

        # Interior angles (clipped for numerical safety)
        def _angle(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-12 or nv < 1e-12:
                return 0.0
            return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))

        ang_a = _angle(ab, ac)
        ang_b = _angle(ba, bc)
        ang_c = _angle(ca, cb)

        cot_a = np.cos(ang_a) / max(np.sin(ang_a), 1e-10)
        cot_b = np.cos(ang_b) / max(np.sin(ang_b), 1e-10)
        cot_c = np.cos(ang_c) / max(np.sin(ang_c), 1e-10)

        # 1/3 barycentric area per vertex
        areas[a] += A / 3.0
        areas[b] += A / 3.0
        areas[c] += A / 3.0

        # Angle deficit accumulation
        angle_sum[a] += ang_a
        angle_sum[b] += ang_b
        angle_sum[c] += ang_c

        # Cotangent Laplacian: for edge (a,b) opposite angle is at c → cot_c
        Lx[a] += cot_c * (pb - pa)
        Lx[b] += cot_c * (pa - pb)
        # for edge (b,c) opposite angle at a → cot_a
        Lx[b] += cot_a * (pc - pb)
        Lx[c] += cot_a * (pb - pc)
        # for edge (c,a) opposite angle at b → cot_b
        Lx[c] += cot_b * (pa - pc)
        Lx[a] += cot_b * (pc - pa)

    safe_area = np.where(areas > 1e-15, areas, 1.0)

    # Δ_S x_i = Lx_i / (2 A_i),  H_i = |Δ_S x_i| / 2 = |Lx_i| / (4 A_i)
    H = np.linalg.norm(Lx, axis=1) / (4.0 * safe_area)

    # Gaussian curvature: K = (2π - Σθ) / A
    K = (2.0 * np.pi - angle_sum) / safe_area

    # Clamp K at boundary vertices (angle sum ≠ 2π there by construction)
    disc = np.maximum(H ** 2 - K, 0.0)
    k1   = H + np.sqrt(disc)
    k2   = H - np.sqrt(disc)

    return {"H": H, "K": K, "k1": k1, "k2": k2}


# ── aggregated statistics ─────────────────────────────────────────────────────

def curvature_stats(H, K, k1, k2, interior_mask):
    """Aggregate curvature arrays to scalar statistics (interior only)."""
    H_in  = np.abs(H[interior_mask])
    K_in  =        K[interior_mask]
    k1_in =        k1[interior_mask]
    k2_in =        k2[interior_mask]

    def safe(arr):
        return arr[np.isfinite(arr)]

    return {
        "H_mean":  np.mean(safe(H_in))  if len(safe(H_in))  else np.nan,
        "H_max":   np.max(safe(H_in))   if len(safe(H_in))  else np.nan,
        "K_mean":  np.mean(safe(K_in))  if len(safe(K_in))  else np.nan,
        "K_max":   np.max(safe(np.abs(K_in)))  if len(safe(K_in)) else np.nan,
        "k1_mean": np.mean(safe(k1_in)) if len(safe(k1_in)) else np.nan,
        "k2_mean": np.mean(safe(k2_in)) if len(safe(k2_in)) else np.nan,
    }


# ── batch processing ──────────────────────────────────────────────────────────

def enrich_results(results_csv, mesh_off, data_dir, out_csv=None):
    """
    Load results.csv, compute curvature statistics for every sample that has
    a _verts.csv file, and return an enriched DataFrame.

    Parameters
    ----------
    results_csv : path to sensitivity_analysis/data/results.csv
    mesh_off    : path to circular_flat.off (provides face connectivity)
    data_dir    : directory containing <sample_id>_verts.csv files
    out_csv     : if given, saves enriched DataFrame there

    Returns
    -------
    pd.DataFrame with new columns: H_mean, H_max, K_mean, K_max, k1_mean, k2_mean
    """
    print("Loading mesh connectivity...")
    rest_verts, faces = read_off(mesh_off)
    bdry = boundary_vertices(faces, len(rest_verts))
    interior_mask = np.array([i not in bdry for i in range(len(rest_verts))])
    print(f"  {len(rest_verts)} vertices, {len(faces)} faces, "
          f"{interior_mask.sum()} interior vertices")

    df = pd.read_csv(results_csv)

    curv_rows = []
    n_done = 0
    for _, row in df.iterrows():
        sid   = int(row["sample_id"])
        vpath = os.path.join(data_dir, f"{sid:05d}_verts.csv")
        if not os.path.exists(vpath):
            curv_rows.append({k: np.nan for k in
                              ["H_mean", "H_max", "K_mean", "K_max", "k1_mean", "k2_mean"]})
            continue

        vdf   = pd.read_csv(vpath).sort_values("vid")
        verts = vdf[["x", "y", "z"]].values

        try:
            curv = compute_curvatures(verts, faces)
            stats = curvature_stats(**curv, interior_mask=interior_mask)
        except Exception as e:
            print(f"  [{sid:05d}] curvature failed: {e}")
            stats = {k: np.nan for k in
                     ["H_mean", "H_max", "K_mean", "K_max", "k1_mean", "k2_mean"]}

        curv_rows.append(stats)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(df)} done...")

    curv_df = pd.DataFrame(curv_rows, index=df.index)
    enriched = pd.concat([df, curv_df], axis=1)

    if out_csv:
        enriched.to_csv(out_csv, index=False)
        print(f"Saved enriched results → {out_csv}")

    return enriched


if __name__ == "__main__":
    from config import DATA_DIR, MESH_PATH
    enrich_results(
        results_csv=os.path.join(DATA_DIR, "results.csv"),
        mesh_off=MESH_PATH,
        data_dir=DATA_DIR,
        out_csv=os.path.join(DATA_DIR, "results_with_curvature.csv"),
    )
