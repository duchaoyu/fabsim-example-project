"""
D5 FEM inverse optimisation — N equal bands along directional field,
with Laplacian smoothness regularisation between adjacent regions.

Region assignment: face centroids projected onto mean d1 direction,
divided into N equal-size bands (iso-course lines across the dome).

Objective: RMSE + lambda_smooth * Σ_{adj (i,j)} [(sf_w_i-sf_w_j)² + (sf_c_i-sf_c_j)²]

Usage:
    python3 optimise_D5_laplacian.py [--n-regions 10] [--lambda-smooth 0.01]
                                     [--maxiter 500] [--out-prefix d5_lap]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

HERE    = os.path.dirname(os.path.abspath(__file__))
MESH    = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
TARGET  = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
FIELD_J = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
BINARY  = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
OUT_DIR = os.path.join(HERE, "optimisation")

CABLE_EA     = 157000.0
PRESSURE     = 1000.0
CABLE_SCALE  = 0.95
CABLE2_VERTS = [89, 88]
CABLE2_SCALE = 1.0

SF_W0 = 1.1526
SF_C0 = 1.0725


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


# ── Region assignment: equal bands along mean d1 direction ────────────────────
def build_region_map(V, F, field, n_regions):
    d1 = np.array([field[str(fi)]["d1"] for fi in range(len(F))])

    # Mean d1 direction (doubling trick for headless vectors)
    d1_2 = 2 * np.arctan2(d1[:, 1], d1[:, 0])
    mean_angle = np.arctan2(np.sin(d1_2).mean(), np.cos(d1_2).mean()) / 2
    mean_d1 = np.array([np.cos(mean_angle), np.sin(mean_angle), 0.0])

    # Project face centroids onto mean d1
    centroids = V[F].mean(axis=1)
    proj = centroids @ mean_d1

    # Divide into n_regions equal-size bands by sorted projection
    order = np.argsort(proj)
    face_region = np.zeros(len(F), dtype=int)
    chunk = len(F) / n_regions
    for fi, idx in enumerate(order):
        face_region[idx] = min(int(fi / chunk), n_regions - 1)

    # Per-region mean knit direction from field
    angles = np.arctan2(d1[:, 1], d1[:, 0])
    knit_dirs = []
    for r in range(n_regions):
        mask = face_region == r
        a2 = 2 * angles[mask]
        mean_a = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
        knit_dirs.append(float(np.degrees(mean_a) % 180))

    return face_region, knit_dirs


# ── Region adjacency graph (for Laplacian penalty) ────────────────────────────
def build_region_adj(F, face_region):
    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(F):
        for e in [tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((a, c)))]:
            edge_to_faces[e].append(fi)
    adj = set()
    for faces in edge_to_faces.values():
        if len(faces) == 2:
            r0, r1 = face_region[faces[0]], face_region[faces[1]]
            if r0 != r1:
                adj.add((min(r0, r1), max(r0, r1)))
    return list(adj)


# ── Validity gate ─────────────────────────────────────────────────────────────
_target_crown = [None]

def _check_valid(verts, V_rest):
    if not np.all(np.isfinite(verts)):
        return False, "NaN/Inf"
    if float(np.max(np.linalg.norm(verts - V_rest, axis=1))) < 1e-8:
        return False, "rest shape returned"
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
_out_prefix = ["d5_lap"]
_rmap_path  = [None]

def run_fem(sf_wale, sf_course, knit_dirs, face_region, n_regions, V_rest):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          PRESSURE,
        "motif":             1,
        "cable_ea":          CABLE_EA,
        "cable_paths":       [_cable_path[0], CABLE2_VERTS],
        "cable_rest_scales": [float(CABLE_SCALE), float(CABLE2_SCALE)],
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(n_regions)],
    }

    with open(_rmap_path[0], "w") as f:
        json.dump({"face_regions": face_region.tolist()}, f)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH, _rmap_path[0], params_path, prefix]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        scalars_path = prefix + "_scalars.csv"
        verts_path   = prefix + "_verts.csv"
        if not os.path.exists(scalars_path):
            return None
        with open(scalars_path) as f:
            out = {k: float(v) for k, v in next(csv.DictReader(f)).items()}
        if os.path.exists(verts_path):
            out["verts"] = np.loadtxt(verts_path, delimiter=",", skiprows=1)[:, 1:]
            ok, reason = _check_valid(out["verts"], V_rest)
            if not ok:
                print(f"  [{_call_count[0]:4d}] INVALID: {reason}")
                return None
        return out
    except Exception as e:
        print(f"  [{_call_count[0]:4d}] exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass


_cable_path = [None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-regions",     type=int,   default=10)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--maxiter",       type=int,   default=500)
    parser.add_argument("--out-prefix",    type=str,   default="d5_lap")
    args = parser.parse_args()

    _out_prefix[0] = args.out_prefix
    _rmap_path[0]  = os.path.join(OUT_DIR, f"{args.out_prefix}_map.json")
    n_regions      = args.n_regions
    lam            = args.lambda_smooth

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, lbl in [(BINARY,"binary"),(MESH,"mesh"),(CABLE_J,"cable"),(FIELD_J,"field")]:
        if not os.path.exists(path):
            print(f"Not found: {lbl} {path}"); sys.exit(1)

    V, F        = load_off(MESH)
    V_rest      = V.copy()
    V_target, _ = load_off(TARGET)
    bdry_mask   = np.hypot(V_target[:,0], V_target[:,1]) > \
                  np.hypot(V_target[:,0], V_target[:,1]).max() * 0.98
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:,2].max())
    _target_crown[0] = t_crown

    with open(FIELD_J) as f: field = json.load(f)
    with open(CABLE_J) as f: cable = json.load(f)
    _cable_path[0] = cable["vertex_indices"] + [cable["vertex_indices"][0]]

    face_region, knit_dirs = build_region_map(V, F, field, n_regions)
    region_adj = build_region_adj(F, face_region)

    counts = np.bincount(face_region, minlength=n_regions)
    print(f"FEM mesh   : {len(V)} verts, {len(F)} faces")
    print(f"Target     : {len(interior_idx)} interior verts, crown={t_crown:.4f} m")
    print(f"Regions    : {n_regions} bands along d1  {counts.tolist()}")
    print(f"KnitDirs   : {[round(d,1) for d in knit_dirs]}°")
    print(f"Adj pairs  : {len(region_adj)}")
    print(f"λ_smooth   : {lam}")
    print(f"Cables     : inner loop (sc={CABLE_SCALE}) + v{CABLE2_VERTS[0]}→v{CABLE2_VERTS[1]} (sc={CABLE2_SCALE})")

    # ── Sanity check ──────────────────────────────────────────────────────────
    sf_w0 = np.full(n_regions, SF_W0)
    sf_c0 = np.full(n_regions, SF_C0)
    print(f"\nSanity check …")
    out0 = run_fem(sf_w0, sf_c0, knit_dirs, face_region, n_regions, V_rest)
    if out0 is None:
        print("ERROR: FEM failed at sanity check"); sys.exit(1)
    d0   = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(d0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0*1000:.2f} mm")

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(p):
        sf_w = p[:n_regions]
        sf_c = p[n_regions:]
        out  = run_fem(sf_w, sf_c, knit_dirs, face_region, n_regions, V_rest)
        if out is None or "verts" not in out:
            return 1e3
        diff = out["verts"][interior_idx] - V_target[interior_idx]
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        lap  = sum((sf_w[i]-sf_w[j])**2 + (sf_c[i]-sf_c[j])**2
                   for i, j in region_adj)
        loss = rmse + lam * lap
        if _call_count[0] % 10 == 0:
            print(f"  [{_call_count[0]:4d}]  RMSE={rmse*1000:.2f} mm  lap={lap:.4f}  "
                  f"loss={loss:.6f}  "
                  f"sf_w=[{','.join(f'{v:.3f}' for v in sf_w)}]")
        return loss

    p0     = np.concatenate([sf_w0, sf_c0])
    bounds = [(0.80, 1.30)] * n_regions + [(0.80, 1.30)] * n_regions

    print(f"\nOptimising {2*n_regions} params (λ={lam}), maxiter={args.maxiter} …")
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-10,
                            "gtol": 1e-5, "eps": 0.002})

    print(f"\nConverged: {res.success}  |  {res.message}")
    p_opt = res.x
    sf_w_opt = p_opt[:n_regions]
    sf_c_opt = p_opt[n_regions:]

    # ── Final FEM (RMSE only, no penalty) ─────────────────────────────────────
    print("\nRunning final FEM …")
    out_f = run_fem(sf_w_opt, sf_c_opt, knit_dirs, face_region, n_regions, V_rest)
    rmse_f = None
    if out_f and "verts" in out_f:
        d = out_f["verts"][interior_idx] - V_target[interior_idx]
        rmse_f = float(np.sqrt(np.mean(np.sum(d**2, axis=1)))) * 1000
        print(f"  crown={out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")
        print(f"  RMSE ={rmse_f:.2f} mm")

    lap_f = sum((sf_w_opt[i]-sf_w_opt[j])**2 + (sf_c_opt[i]-sf_c_opt[j])**2
                for i, j in region_adj)
    print(f"  Laplacian penalty (unweighted): {lap_f:.6f}")
    print(f"\nPer-region params:")
    print(f"  {'R':>3}  {'faces':>6}  {'knit°':>6}  {'sf_wale':>8}  {'sf_course':>9}")
    for r in range(n_regions):
        print(f"  {r:3d}  {counts[r]:6d}  {knit_dirs[r]:6.1f}  "
              f"{sf_w_opt[r]:8.4f}  {sf_c_opt[r]:9.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "geometry": "D5", "n_regions": n_regions,
        "lambda_smooth": lam,
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "converged": bool(res.success), "message": res.message,
        "rmse_mm": rmse_f, "n_calls": _call_count[0],
        "region_adj": [[int(i), int(j)] for i, j in region_adj],
        "regions": [{"region_id": r, "n_faces": int(counts[r]),
                     "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w_opt[r]),
                     "sf_course": float(sf_c_opt[r])}
                    for r in range(n_regions)],
    }
    out_json = os.path.join(OUT_DIR, f"{args.out_prefix}_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
