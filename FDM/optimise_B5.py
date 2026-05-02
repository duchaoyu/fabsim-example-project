"""
B5 9-region FEM inverse optimisation.

Pipeline:
  1. FEM rest-shape mesh = B5_flat.off (B5.obj topology, z=0) → fast Newton convergence
     FEM target          = B5_shared.off (B5.obj 3D dome) → vertex-by-vertex comparison
  2. 12 cable sections (4 cables × 3 sections each, split at 4 intersection nodes)
     dividing the mesh into 9 knit regions
  3. Optimise per-region (sf_wale, sf_course) via scipy L-BFGS-B
     — knit_dir fixed from cable geometry
     — objective: RMSE of FEM inflated shape vs B5 target vertices (interior only)
  4. Save result JSON

Usage:
    python3 optimise_B5.py [--motif 1] [--pressure 1000] [--maxiter 200]
"""
import argparse, csv, json, os, sys, subprocess, tempfile
import numpy as np
from scipy.optimize import minimize

HERE   = os.path.dirname(os.path.abspath(__file__))

# FEM mesh and target: B5_remeshed (3D dome, 497v/929f, feature-preserving remesh)
# Starting from dome with sf>1 → rest shape < dome → elastic inward force
# balances pressure at dome equilibrium (sf≈2.25 gives crown≈3.86m at p=1000)
MESH_PATH   = os.environ.get("FEM_MESH",
    os.path.join(HERE, "..", "data", "B5_remeshed_shared.off"))
TARGET_OFF  = os.environ.get("FEM_TARGET",
    os.path.join(HERE, "..", "data", "B5_remeshed_shared.off"))

BINARY           = os.environ.get("FEM_BINARY_NREGION",
    os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))
CABLE_PATHS_FILE = os.environ.get("FEM_CABLE_PATHS",
    os.path.join(HERE, "..", "data", "cable_paths_B5_remeshed.json"))
OUT_DIR = os.path.join(HERE, "optimisation")

# ── Knit directions per region (row, col) ─────────────────────────────────────
KNIT_DIR = {
    # uniform 0° everywhere (wale along x-axis, aligned with cables C/D)
    (0,0): 0, (0,1): 0, (0,2): 0,
    (1,0): 0, (1,1): 0, (1,2): 0,
    (2,0): 0, (2,1): 0, (2,2): 0,
}

# ── Load OFF mesh ─────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

# ── Region assignment (3×3 grid) ─────────────────────────────────────────────
def make_region_map(V, F, x_lo, x_hi, y_lo, y_hi):
    face_cx = V[F[:,0],0]/3 + V[F[:,1],0]/3 + V[F[:,2],0]/3
    face_cy = V[F[:,0],1]/3 + V[F[:,1],1]/3 + V[F[:,2],1]/3
    regions = []
    for cx, cy in zip(face_cx, face_cy):
        col = 0 if cx < x_lo else (1 if cx < x_hi else 2)
        row = 0 if cy < y_lo else (1 if cy < y_hi else 2)
        regions.append(row * 3 + col)
    return regions

def region_knit_dir(region_id):
    row, col = divmod(region_id, 3)
    return KNIT_DIR[(row, col)]

# ── Load cable sections ───────────────────────────────────────────────────────
def load_cable_paths(path):
    if not os.path.exists(path):
        print(f"WARNING: cable paths not found: {path}")
        return []
    with open(path) as f:
        d = json.load(f)
    return list(d.values()) if isinstance(d, dict) else d

# ── Run FEM binary ─────────────────────────────────────────────────────────────
_call_count = [0]

def run_fem(sf_wale_per_region, sf_course_per_region, knit_dirs,
            pressure, motif, region_map_path, cable_paths=None, cable_ea=157000.0,
            cable_rest_scales=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":    float(pressure),
        "motif":       int(motif),
        "cable_ea":    float(cable_ea),
        "cable_paths": cable_paths if cable_paths else [],
        "regions":  [
            {
                "sf_wale":      float(sf_wale_per_region[r]),
                "sf_course":    float(sf_course_per_region[r]),
                "knit_dir_deg": float(knit_dirs[r]),
            }
            for r in range(9)
        ]
    }
    if cable_rest_scales is not None:
        params["cable_rest_scales"] = [float(x) for x in cable_rest_scales]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"run_{_call_count[0]:05d}")
    cmd = [BINARY, MESH_PATH, region_map_path, params_path, prefix]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  FEM error: {result.stderr[:200]}")
            return None

        scalars_path = prefix + "_scalars.csv"
        verts_path   = prefix + "_verts.csv"
        if not os.path.exists(scalars_path):
            return None

        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        out = {k: float(v) for k, v in row.items()}

        if os.path.exists(verts_path):
            vdata = np.loadtxt(verts_path, delimiter=",", skiprows=1)
            out["verts"] = vdata[:, 1:]   # (nV, 3)

        return out
    except Exception as e:
        print(f"  FEM exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass

# ── Objective (vertex-by-vertex RMSE vs B5 target) ────────────────────────────
def make_objective(V_target, interior_idx, region_map_path, knit_dirs,
                   pressure, motif, cable_paths=None, cable_ea=157000.0):
    t_crown = float(V_target[:,2].max())
    history = []

    def objective(p):
        sf_w   = p[:9]
        sf_c   = p[9:18]
        scales = p[18:30] if len(p) >= 30 else None
        out  = run_fem(sf_w, sf_c, knit_dirs, pressure, motif, region_map_path,
                       cable_paths, cable_ea, cable_rest_scales=scales)
        if out is None:
            return 1e6

        if "verts" in out:
            diff = out["verts"][interior_idx] - V_target[interior_idx]
            loss = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        else:
            loss = abs(out["crown_height"] - t_crown) / (t_crown + 1e-6)

        history.append({"call": _call_count[0], "loss": loss,
                        "crown_height": out.get("crown_height", 0.0)})
        if _call_count[0] % 10 == 0 or _call_count[0] == 1:
            print(f"  [{_call_count[0]:4d}]  rmse={loss:.4f} m  "
                  f"crown={out.get('crown_height', 0):.4f} m  "
                  f"(target {t_crown:.4f})")
        return loss

    return objective, history

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",    type=int,   default=1)
    parser.add_argument("--pressure", type=float, default=1000.0)
    parser.add_argument("--maxiter",  type=int,   default=200)
    parser.add_argument("--mesh", type=str, default=None,
                        help="Override FEM mesh (env: FEM_MESH)")
    args = parser.parse_args()

    global MESH_PATH
    if args.mesh:
        MESH_PATH = args.mesh

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(BINARY):
        print(f"FEM binary not found: {BINARY}")
        sys.exit(1)

    # FEM mesh (3D dome)
    V, F = load_off(MESH_PATH)
    fem_span = V[:,0].max() - V[:,0].min()
    print(f"FEM mesh:   {len(V)} verts, {len(F)} faces, span={fem_span:.2f} m")

    # Target (3D dome — same file; FEM output should match this)
    V_target, _ = load_off(TARGET_OFF)
    bdry_mask    = np.abs(V_target[:,2]) < 0.01
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:,2].max())
    print(f"Target:     {len(V_target)} verts, {len(interior_idx)} interior, crown={t_crown:.4f} m")

    # 12 cable sections (4 cables × 3 sections, split at intersection nodes)
    cable_paths = load_cable_paths(CABLE_PATHS_FILE)
    cable_ea    = 157000.0   # N  (1 mm steel cable)
    print(f"Cables:     {len(cable_paths)} sections, EA={cable_ea:.0f} N")

    # 3×3 region map — split at ±R/3 in x and y (the cable lines)
    R     = fem_span / 2
    x_lo, x_hi = -R/3, R/3
    y_lo, y_hi = -R/3, R/3
    region_ids = make_region_map(V, F, x_lo, x_hi, y_lo, y_hi)
    knit_dirs  = [region_knit_dir(r) for r in range(9)]

    region_map_path = os.path.join(OUT_DIR, "region_map.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": region_ids}, f)
    counts = np.bincount(region_ids, minlength=9)
    print(f"Regions:    {counts.tolist()}")
    print(f"Knit dirs:  {knit_dirs}")
    print(f"Pressure:   {args.pressure} Pa  |  motif: {args.motif}")

    # Initial guess: sf=1.041 + cable_rest_scale=1.0 (no cable pre-stress)
    n_cables = len(cable_paths)
    p0 = np.concatenate([np.full(18, 1.041), np.full(n_cables, 1.0)])
    print(f"\nSanity check (sf=1.041 uniform, cable_rest_scale=1.0)...")
    out0 = run_fem(p0[:9], p0[9:18], knit_dirs, args.pressure, args.motif,
                   region_map_path, cable_paths, cable_ea,
                   cable_rest_scales=p0[18:18+n_cables])
    if out0 is None:
        print("ERROR: FEM failed at sf=1.041 — check binary and mesh")
        sys.exit(1)
    print(f"  crown={out0['crown_height']:.4f} m  (target {t_crown:.4f})")
    if "verts" in out0:
        diff0 = out0["verts"][interior_idx] - V_target[interior_idx]
        rmse0 = float(np.sqrt(np.mean(np.sum(diff0**2, axis=1))))
        print(f"  RMSE = {rmse0:.4f} m")

    # Optimise — sf bounds (0.9, 2.0); cable_rest_scale bounds (0.85, 1.05)
    # scale<1: cable pre-tensioned (pulls dome inward); scale>1: slack (no force)
    bounds = [(0.9, 2.0)] * 18 + [(0.85, 1.05)] * n_cables
    n_params = 18 + n_cables
    print(f"\nOptimising {n_params} params (9×sf_wale + 9×sf_course "
          f"+ {n_cables}×cable_rest_scale), maxiter={args.maxiter}...")
    print(f"  L-BFGS-B, sf bounds (0.9, 2.0), cable bounds (0.85, 1.05), FD eps=0.05\n")

    objective, history = make_objective(
        V_target, interior_idx, region_map_path, knit_dirs,
        args.pressure, args.motif, cable_paths, cable_ea)

    result = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-10, "gtol": 1e-6, "eps": 0.05},
    )

    print(f"\nConverged: {result.success}  |  {result.message}")
    print(f"Final RMSE: {result.fun:.4f} m   calls: {_call_count[0]}")

    p_opt = result.x
    sf_w  = p_opt[:9]
    sf_c  = p_opt[9:18]
    scales = p_opt[18:18+n_cables]

    print("\n=== Optimal parameters per region ===")
    print(f"{'Region':>6}  {'row':>4}  {'col':>4}  {'knit_dir':>8}  "
          f"{'sf_wale':>8}  {'sf_course':>9}")
    for r in range(9):
        row, col = divmod(r, 3)
        print(f"{r:6d}  {row:4d}  {col:4d}  {knit_dirs[r]:8.0f}°  "
              f"{sf_w[r]:8.4f}  {sf_c[r]:9.4f}")

    # Cable names (e.g. A1, A2, ... D3) if the JSON is a dict
    cable_names = [f"C{i}" for i in range(n_cables)]
    if os.path.exists(CABLE_PATHS_FILE):
        with open(CABLE_PATHS_FILE) as f:
            cd = json.load(f)
        if isinstance(cd, dict):
            cable_names = list(cd.keys())[:n_cables]
    print("\n=== Optimal cable rest-length scales ===")
    for i, name in enumerate(cable_names):
        print(f"  {name:>4}  scale={scales[i]:.4f}  (1.0=no pre-stress)")

    # Final FEM run
    print("\nRunning final FEM simulation...")
    out_final = run_fem(sf_w, sf_c, knit_dirs, args.pressure, args.motif,
                        region_map_path, cable_paths, cable_ea,
                        cable_rest_scales=scales)
    if out_final:
        print(f"  crown = {out_final['crown_height']:.4f} m  (target {t_crown:.4f})")
        if "verts" in out_final:
            diff_f = out_final["verts"][interior_idx] - V_target[interior_idx]
            print(f"  RMSE  = {np.sqrt(np.mean(np.sum(diff_f**2, axis=1))):.4f} m")

    results = {
        "motif":    args.motif,
        "pressure": args.pressure,
        "fem_mesh": MESH_PATH,
        "target_crown_m": float(t_crown),
        "converged": bool(result.success),
        "loss_rmse_m": float(result.fun),
        "n_calls":  _call_count[0],
        "regions":  [
            {"region_id": r, "row": r//3, "col": r%3,
             "knit_dir_deg": knit_dirs[r],
             "sf_wale": float(sf_w[r]), "sf_course": float(sf_c[r])}
            for r in range(9)
        ],
        "cables": [
            {"name": cable_names[i], "rest_scale": float(scales[i])}
            for i in range(n_cables)
        ],
        "history": history[-30:],
    }
    out_json = os.path.join(OUT_DIR, "B5_optimised_params.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
