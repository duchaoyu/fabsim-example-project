"""
FDM-guided 9-region FEM inverse optimisation for B5.

Pipeline:
  1. Load FDM result → extract B5 target surface (scaled to FEM mesh units)
  2. Assign 9 regions to FEM circular mesh faces (3×3 grid from cable positions)
  3. Optimise per-region (sf_wale, sf_course) via scipy L-BFGS-B
     — knit_dir per region is FIXED from cable geometry (FDM-derived)
     — objective: match FEM deformed shape to B5 target (crown_height + curvature)
  4. Save result JSON + visualise convergence

Usage:
    python3 optimise_B5.py [--motif 1] [--pressure 500] [--jobs 1]
"""
import argparse, csv, json, os, sys, subprocess, tempfile, shutil
import numpy as np
from scipy.optimize import minimize

HERE   = os.path.dirname(os.path.abspath(__file__))
SA_DIR = os.path.join(HERE, "..", "sensitivity_analysis")
sys.path.insert(0, SA_DIR)

BINARY      = os.environ.get("FEM_BINARY_NREGION",
    os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))
MESH_PATH   = os.environ.get("FEM_MESH",
    os.path.join(HERE, "..", "data", "circular_flat_fdm.off"))
CABLE_PATHS_FILE = os.environ.get("FEM_CABLE_PATHS",
    os.path.join(HERE, "..", "data", "cable_paths_B5.json"))
FDM_JSON    = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
OUT_DIR     = os.path.join(HERE, "optimisation")

# ── FEM mesh geometry ─────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

# ── Region assignment for FEM faces (3×3 grid) ────────────────────────────────
# Knit directions per region (row, col) — from FDM cable analysis
KNIT_DIR = {
    (0,0): 45, (0,1):  0, (0,2): 45,
    (1,0): 90, (1,1): 45, (1,2): 90,
    (2,0): 45, (2,1):  0, (2,2): 45,
}

def make_region_map(V, F, x_lo, x_hi, y_lo, y_hi):
    """Assign each face a region id 0-8 based on 3×3 x/y grid."""
    face_cx = V[F[:,0],0]/3 + V[F[:,1],0]/3 + V[F[:,2],0]/3
    face_cy = V[F[:,0],1]/3 + V[F[:,1],1]/3 + V[F[:,2],1]/3
    regions = []
    for cx, cy in zip(face_cx, face_cy):
        col = 0 if cx < x_lo else (1 if cx < x_hi else 2)
        row = 0 if cy < y_lo else (1 if cy < y_hi else 2)
        regions.append(row * 3 + col)   # 0-8
    return regions

def region_knit_dir(region_id):
    row, col = divmod(region_id, 3)
    return KNIT_DIR[(row, col)]

# ── Extract B5 targets from FDM result (scaled to FEM mesh) ──────────────────
def extract_fdm_targets(fdm_json, fem_V):
    # Load COMPAS JSON manually (Mesh.from_json fails on this format version)
    with open(fdm_json) as f:
        d = json.load(f)

    vid_keys   = sorted(d['vertex'].keys(), key=int)
    vpts       = np.array([[d['vertex'][k]['x'], d['vertex'][k]['y'],
                             d['vertex'][k]['z']] for k in vid_keys])

    # Boundary: vertices with z=0 (constrained nodes in FDM)
    bdry_mask  = np.abs(vpts[:, 2]) < 1e-6
    free_mask  = ~bdry_mask

    fdm_span   = 20.0                         # B5 footprint span (m)
    fem_span   = fem_V[:,0].max() - fem_V[:,0].min()
    scale      = fem_span / fdm_span

    fdm_crown  = float(vpts[free_mask, 2].max())
    fem_crown_target = fdm_crown * scale

    # Curvature along x=0 and y=0 slices — approximate via central-difference
    # on the FDM dome vertices closest to those planes.
    curv_scale = fdm_span / fem_span
    R    = fdm_span / 2
    band = 0.08 * R   # 0.8m band around x=0 / y=0

    def mean_curvature_in_band(axis):
        other = 1 if axis == 0 else 0
        in_band = (np.abs(vpts[:, axis]) < band) & free_mask
        if in_band.sum() < 3:
            return None
        pts = vpts[in_band]
        pts_sorted = pts[np.argsort(pts[:, other])]
        z = pts_sorted[:, 2]
        s = pts_sorted[:, other]
        ds = np.diff(s)
        dz = np.diff(z)
        if len(ds) < 2 or np.any(ds < 1e-8):
            return None
        d2z = np.diff(dz / ds) / (0.5 * (ds[:-1] + ds[1:]))
        kappa = float(np.mean(np.abs(d2z)))
        return kappa * curv_scale

    H_x0 = mean_curvature_in_band(0)
    H_y0 = mean_curvature_in_band(1)

    return {
        "crown_height": fem_crown_target,
        "H_mean_x0":    H_x0,
        "H_mean_y0":    H_y0,
        "scale":        scale,
        "fdm_crown":    fdm_crown,
    }

# ── Cable paths ───────────────────────────────────────────────────────────────
def load_cable_paths(path):
    if not os.path.exists(path):
        print(f"WARNING: cable paths file not found: {path}")
        return []
    with open(path) as f:
        d = json.load(f)
    # Accept either {A:[...], B:[...]} or [[...], [...]]
    if isinstance(d, dict):
        return list(d.values())
    return d

# ── Run FEM binary ─────────────────────────────────────────────────────────────
_call_count = [0]

def run_fem(sf_wale_per_region, sf_course_per_region, knit_dirs,
            pressure, motif, region_map_path, cable_paths=None, cable_ea=157000.0):
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"run_{_call_count[0]:05d}")
    cmd = [BINARY, MESH_PATH, region_map_path, params_path, prefix]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        scalars_path = prefix + "_scalars.csv"
        if not os.path.exists(scalars_path):
            return None
        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        return {k: float(v) for k, v in row.items()}
    except Exception:
        return None
    finally:
        os.unlink(params_path)

# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(targets, region_map_path, knit_dirs, pressure, motif,
                   cable_paths=None, cable_ea=157000.0):
    t_crown = targets["crown_height"]
    t_Hx    = targets["H_mean_x0"]
    t_Hy    = targets["H_mean_y0"]

    # Weights: crown is most reliable; curvature noisier
    w_crown, w_Hx, w_Hy = 1.0, 0.3, 0.3

    history = []

    def objective(p):
        # p: 18 values — sf_wale[0..8] then sf_course[0..8]
        sf_w = p[:9]
        sf_c = p[9:]
        out  = run_fem(sf_w, sf_c, knit_dirs, pressure, motif, region_map_path,
                       cable_paths, cable_ea)
        if out is None:
            return 1e6

        loss  = w_crown * ((out["crown_height"] - t_crown) / t_crown) ** 2
        if t_Hx is not None and "H_mean_x0" in out and out["H_mean_x0"] > 0:
            loss += w_Hx * ((out["H_mean_x0"] - t_Hx) / (t_Hx + 1e-6)) ** 2
        if t_Hy is not None and "H_mean_y0" in out and out["H_mean_y0"] > 0:
            loss += w_Hy * ((out["H_mean_y0"] - t_Hy) / (t_Hy + 1e-6)) ** 2

        history.append({
            "call":         _call_count[0],
            "loss":         loss,
            "crown_height": out["crown_height"],
            "target_crown": t_crown,
        })
        if _call_count[0] % 5 == 0 or _call_count[0] == 1:
            print(f"  [{_call_count[0]:4d}]  loss={loss:.6f}  "
                  f"h={out['crown_height']:.4f} (target {t_crown:.4f})")

        return loss

    return objective, history

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",    type=int,   default=1)
    parser.add_argument("--pressure", type=float, default=500.0)
    parser.add_argument("--maxiter",  type=int,   default=200)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Check binary
    if not os.path.exists(BINARY):
        print(f"FEM binary not found: {BINARY}")
        sys.exit(1)

    # Load FEM mesh
    V, F = load_off(MESH_PATH)
    fem_span = V[:,0].max() - V[:,0].min()
    print(f"FEM mesh: {len(V)} verts, {len(F)} faces, span={fem_span:.4f} m")

    # Load cable paths
    cable_paths = load_cable_paths(CABLE_PATHS_FILE)
    cable_ea = 157000.0   # N  (1mm diameter steel cable)
    print(f"Cables: {len(cable_paths)} paths, EA={cable_ea:.0f} N")

    # Region splits at ±1/6 of span (i.e., at ±R/3 mapped to FEM scale)
    R     = fem_span / 2
    x_lo, x_hi = -R/3, R/3
    y_lo, y_hi = -R/3, R/3
    region_ids = make_region_map(V, F, x_lo, x_hi, y_lo, y_hi)
    knit_dirs  = [region_knit_dir(r) for r in range(9)]

    # Save region map
    region_map_path = os.path.join(OUT_DIR, "region_map.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": region_ids}, f)
    counts = np.bincount(region_ids, minlength=9)
    print(f"Region face counts: {counts.tolist()}")
    print(f"Knit dirs: {knit_dirs}")

    # Extract targets from FDM
    print("\nExtracting B5 targets from FDM result...")
    targets = extract_fdm_targets(FDM_JSON, V)
    print(f"  FDM crown:  {targets['fdm_crown']:.4f} m  (B5 scale)")
    print(f"  FEM target: {targets['crown_height']:.4f} m")
    if targets["H_mean_x0"]: print(f"  H_mean_x0:  {targets['H_mean_x0']:.4f}")
    if targets["H_mean_y0"]: print(f"  H_mean_y0:  {targets['H_mean_y0']:.4f}")

    # Initial parameters: sf=1.0 for all regions
    p0     = np.ones(18)
    bounds = [(0.75, 1.5)] * 18

    # Sanity-check: run FEM with p0
    print("\nSanity check (all sf=1.0)...")
    sf0 = p0[:9]; sc0 = p0[9:]
    out0 = run_fem(sf0, sc0, knit_dirs, args.pressure, args.motif, region_map_path,
                   cable_paths, cable_ea)
    if out0 is None:
        print("ERROR: FEM failed at sf=1.0 — check binary and mesh path")
        sys.exit(1)
    print(f"  crown_height = {out0['crown_height']:.4f} m  "
          f"(target {targets['crown_height']:.4f})")

    # Optimise
    print(f"\nOptimising {len(p0)} parameters (9×sf_wale + 9×sf_course)...")
    print(f"  L-BFGS-B, max {args.maxiter} iterations, finite-difference gradient\n")

    objective, history = make_objective(
        targets, region_map_path, knit_dirs, args.pressure, args.motif,
        cable_paths, cable_ea)

    result = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-9, "gtol": 1e-6,
                 "eps": 0.01,   # finite-difference step for gradient
                 },
    )

    print(f"\nConverged: {result.success}  |  {result.message}")
    print(f"Final loss: {result.fun:.6f}  calls: {_call_count[0]}")

    p_opt  = result.x
    sf_w   = p_opt[:9]
    sf_c   = p_opt[9:]

    print("\n=== Optimal parameters per region ===")
    print(f"{'Region':>6}  {'row':>4}  {'col':>4}  {'knit_dir':>8}  "
          f"{'sf_wale':>8}  {'sf_course':>9}")
    for r in range(9):
        row, col = divmod(r, 3)
        print(f"{r:6d}  {row:4d}  {col:4d}  {knit_dirs[r]:8.0f}°  "
              f"{sf_w[r]:8.4f}  {sf_c[r]:9.4f}")

    # Final FEM run to get output files
    print("\nRunning final FEM simulation...")
    out_final = run_fem(sf_w, sf_c, knit_dirs, args.pressure, args.motif, region_map_path,
                        cable_paths, cable_ea)
    if out_final:
        print(f"  crown_height = {out_final['crown_height']:.4f} m  "
              f"(target {targets['crown_height']:.4f})")

    # Save results
    results = {
        "motif":    args.motif,
        "pressure": args.pressure,
        "target":   targets,
        "converged": bool(result.success),
        "loss":     float(result.fun),
        "n_calls":  _call_count[0],
        "regions":  [
            {
                "region_id":    r,
                "row":          r // 3,
                "col":          r  % 3,
                "knit_dir_deg": knit_dirs[r],
                "sf_wale":      float(sf_w[r]),
                "sf_course":    float(sf_c[r]),
            }
            for r in range(9)
        ],
        "history":  history[-20:],   # last 20 evaluations
    }
    out_json = os.path.join(OUT_DIR, "B5_optimised_params.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
