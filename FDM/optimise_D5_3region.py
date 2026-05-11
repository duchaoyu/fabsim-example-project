"""
D5 FEM inverse optimisation — 3-region per-region sf.

Regions built by simultaneous BFS from 3 seed faces (one per seed vertex).
Each region gets its own sf_wale and sf_course.
Cable rest-length scale fixed at 0.95 (global optimum).
Knit directions from directional field (per-region circular mean).

6 free parameters:
  p = [sf_wale_0, sf_wale_1, sf_wale_2,
       sf_course_0, sf_course_1, sf_course_2]

Warm-start: sf_wale=1.1526, sf_course=1.0725 (D5 global optimum).

Usage:
    python3 optimise_D5_3region.py [--maxiter 300] [--out-prefix d5_3r]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from collections import deque
import numpy as np
from scipy.optimize import minimize

HERE    = os.path.dirname(os.path.abspath(__file__))
MESH    = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
TARGET  = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
FIELD_J = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
BINARY  = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
OUT_DIR = os.path.join(HERE, "optimisation")

N_REGIONS    = 3
SEED_VERTS   = [781, 140, 95]   # one seed vertex per region
CABLE_EA     = 157000.0
PRESSURE     = 1000.0
CABLE_SCALE  = 0.95             # fixed from global optimisation


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


# ── Face adjacency ────────────────────────────────────────────────────────────
def build_face_adj(F):
    from collections import defaultdict
    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        for e in [(min(a,b), max(a,b)),
                  (min(b,c), max(b,c)),
                  (min(a,c), max(a,c))]:
            edge_to_faces[e].append(fi)
    adj = [[] for _ in range(len(F))]
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            adj[faces[0]].append(faces[1])
            adj[faces[1]].append(faces[0])
    return adj


# ── 3-region map: simultaneous BFS from seed faces ────────────────────────────
def build_region_map(V, F):
    with open(CABLE_J) as f:
        cable = json.load(f)
    with open(FIELD_J) as f:
        field = json.load(f)

    nF  = len(F)
    adj = build_face_adj(F)

    # For each seed vertex, pick the first face that contains it
    seed_faces = []
    for sv in SEED_VERTS:
        for fi, face in enumerate(F):
            if sv in face:
                seed_faces.append(fi)
                break
    assert len(seed_faces) == N_REGIONS, "Seed face not found for a seed vertex"

    # Simultaneous multi-source BFS — each face claimed by whichever seed reaches it first
    region = [-1] * nF
    queue  = deque()
    for r, sf in enumerate(seed_faces):
        region[sf] = r
        queue.append(sf)

    while queue:
        fi = queue.popleft()
        r  = region[fi]
        for nb in adj[fi]:
            if region[nb] == -1:
                region[nb] = r
                queue.append(nb)

    assert -1 not in region, "Some faces not reached by BFS"

    # Per-region mean knit direction from directional field
    d1      = np.array([field[str(fi)]["d1"] for fi in range(nF)])
    angles  = np.arctan2(d1[:, 1], d1[:, 0])
    knit_dirs = []
    region_arr = np.array(region)
    for r in range(N_REGIONS):
        mask = region_arr == r
        a2   = 2 * angles[mask]
        mean = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
        knit_dirs.append(float(np.degrees(mean) % 180))

    cable_path = cable["vertex_indices"] + [cable["vertex_indices"][0]]
    return region, knit_dirs, cable_path, seed_faces


# ── Validity gate ─────────────────────────────────────────────────────────────
_target_crown = [None]


def _check_valid(verts, V_rest):
    if not np.all(np.isfinite(verts)):
        return False, "NaN/Inf"
    max_disp = float(np.max(np.linalg.norm(verts - V_rest, axis=1)))
    if max_disp < 1e-8:
        return False, f"max_disp={max_disp:.2e} — rest shape returned"
    t = _target_crown[0]
    if t is not None:
        crown = float(verts[:, 2].max())
        if crown < 0.3 * t or crown > 3.0 * t:
            return False, f"crown={crown:.4f} out of range"
        if float(verts[:, 2].min()) < -0.01 * t:
            return False, "mesh folded"
    return True, "OK"


# ── FEM runner ────────────────────────────────────────────────────────────────
_call_count = [0]
_out_prefix  = ["d5_3r"]


def run_fem(sf_wale, sf_course, knit_dirs, face_region, cable_path, V_rest):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    scales = [float(CABLE_SCALE)] * (len(cable_path) - 1)
    params = {
        "pressure":          PRESSURE,
        "motif":             1,
        "cable_ea":          CABLE_EA,
        "cable_paths":       [cable_path],
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(N_REGIONS)],
        "cable_rest_scales": scales,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    region_map_path = os.path.join(OUT_DIR, "D5_3region_map.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": face_region}, f)

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH, region_map_path, params_path, prefix]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        scalars_path = prefix + "_scalars.csv"
        verts_path   = prefix + "_verts.csv"
        if not os.path.exists(scalars_path):
            return None
        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        out = {k: float(v) for k, v in row.items()}
        if os.path.exists(verts_path):
            out["verts"] = np.loadtxt(verts_path, delimiter=",", skiprows=1)[:, 1:]
            ok, reason = _check_valid(out["verts"], V_rest)
            if not ok:
                print(f"  [{_call_count[0]:4d}] FEM INVALID: {reason}")
                return None
        return out
    except Exception as e:
        print(f"  [{_call_count[0]:4d}] FEM exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter",    type=int,   default=300)
    parser.add_argument("--out-prefix", type=str,   default="d5_3r")
    parser.add_argument("--sf0-wale",   type=float, default=1.1526,
                        help="Warm-start sf_wale (D5 global optimum)")
    parser.add_argument("--sf0-course", type=float, default=1.0725,
                        help="Warm-start sf_course (D5 global optimum)")
    args = parser.parse_args()
    _out_prefix[0] = args.out_prefix

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, lbl in [(BINARY,"binary"),(MESH,"mesh"),(CABLE_J,"cable"),(FIELD_J,"field")]:
        if not os.path.exists(path):
            print(f"Not found: {lbl} {path}"); sys.exit(1)

    V, F         = load_off(MESH)
    V_rest       = V.copy()
    V_target, _  = load_off(TARGET)
    bdry_mask    = np.hypot(V_target[:, 0], V_target[:, 1]) > \
                   np.hypot(V_target[:, 0], V_target[:, 1]).max() * 0.98
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:, 2].max())
    _target_crown[0] = t_crown

    print(f"FEM mesh : {len(V)} verts, {len(F)} faces")
    print(f"Target   : {len(interior_idx)} interior verts, crown={t_crown:.4f} m")

    face_region, knit_dirs, cable_path, seed_faces = build_region_map(V, F)
    counts = np.bincount(face_region, minlength=N_REGIONS)
    print(f"Seed verts  : {SEED_VERTS}  →  seed faces: {seed_faces}")
    print(f"Region sizes: {counts.tolist()}")
    print(f"Knit dirs   : {[round(d,1) for d in knit_dirs]}°")
    print(f"Cable       : {len(cable_path)-1} segments, fixed scale={CABLE_SCALE}")
    print(f"Pressure    : {PRESSURE} Pa  |  motif: 1")

    # ── Warm-start check ──────────────────────────────────────────────────────
    sf_w0 = np.full(N_REGIONS, args.sf0_wale)
    sf_c0 = np.full(N_REGIONS, args.sf0_course)
    print(f"\nWarm-start check (sf_w={args.sf0_wale}, sf_c={args.sf0_course}) …")
    out0 = run_fem(sf_w0, sf_c0, knit_dirs, face_region, cable_path, V_rest)
    if out0 is None:
        print("ERROR: FEM failed at warm-start"); sys.exit(1)
    d0    = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(d0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0:.4f} m")
    if args.maxiter == 0:
        print("maxiter=0 — warm-start check only."); return

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(p):
        sf_w = p[:N_REGIONS]
        sf_c = p[N_REGIONS:]
        out  = run_fem(sf_w, sf_c, knit_dirs, face_region, cable_path, V_rest)
        if out is None or "verts" not in out:
            return 1e3
        diff = out["verts"][interior_idx] - V_target[interior_idx]
        loss = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        if _call_count[0] % 5 == 0:
            print(f"  [{_call_count[0]:4d}]  RMSE={loss:.4f} m  "
                  f"sf_w=[{','.join(f'{v:.4f}' for v in sf_w)}]  "
                  f"sf_c=[{','.join(f'{v:.4f}' for v in sf_c)}]")
        return loss

    p0     = np.concatenate([sf_w0, sf_c0])
    bounds = [(0.80, 1.50)] * N_REGIONS + [(0.80, 1.50)] * N_REGIONS

    print(f"\nOptimising {2*N_REGIONS} params (3×sf_wale + 3×sf_course), "
          f"maxiter={args.maxiter} …")
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-9,
                            "gtol": 1e-5, "eps": 0.002})

    print(f"\nConverged: {res.success}  |  {res.message}")
    print(f"Final RMSE: {res.fun:.4f} m   FEM calls: {_call_count[0]}")

    p_opt    = res.x
    sf_w_opt = p_opt[:N_REGIONS]
    sf_c_opt = p_opt[N_REGIONS:]

    print(f"\n=== Optimal 3-region parameters ===")
    print(f"{'Region':>6}  {'seed_v':>7}  {'seed_f':>7}  {'faces':>6}  "
          f"{'knit°':>6}  {'sf_wale':>8}  {'sf_course':>9}")
    for r in range(N_REGIONS):
        print(f"{r:6d}  {SEED_VERTS[r]:7d}  {seed_faces[r]:7d}  "
              f"{counts[r]:6d}  {knit_dirs[r]:6.1f}  "
              f"{sf_w_opt[r]:8.4f}  {sf_c_opt[r]:9.4f}")

    # ── Final FEM ─────────────────────────────────────────────────────────────
    print("\nRunning final FEM …")
    out_f = run_fem(sf_w_opt, sf_c_opt, knit_dirs, face_region, cable_path, V_rest)
    if out_f and "verts" in out_f:
        d = out_f["verts"][interior_idx] - V_target[interior_idx]
        rmse_f = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
        print(f"  crown={out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")
        print(f"  RMSE ={rmse_f:.4f} m")

    # ── Save result JSON ───────────────────────────────────────────────────────
    result = {
        "geometry": "D5", "n_regions": N_REGIONS,
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "cable_scale_fixed": CABLE_SCALE,
        "seed_vertices": SEED_VERTS, "seed_faces": seed_faces,
        "converged": bool(res.success),
        "rmse_m": float(res.fun), "n_calls": _call_count[0],
        "regions": [{"region_id": r, "seed_vertex": SEED_VERTS[r],
                     "seed_face": seed_faces[r], "n_faces": int(counts[r]),
                     "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w_opt[r]),
                     "sf_course": float(sf_c_opt[r])} for r in range(N_REGIONS)],
    }
    out_json = os.path.join(OUT_DIR, "D5_3region_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
