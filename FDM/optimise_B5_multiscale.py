"""
B5 scale comparison: 9-region FEM inverse optimisation at 1.2 m, 1.5 m, 2.0 m, 3.0 m.

For each diameter the base B5_remeshed_shared.off is uniformly scaled, written to
data/B5_remeshed_shared_<tag>.off, and the full L-BFGS-B optimisation is run.
Cable paths (vertex indices) are scale-independent and reused unchanged.

Results are saved to:
  FDM/optimisation/B5_<tag>_optimised_params.json

A summary comparison table is printed at the end.

Usage:
    python3 optimise_B5_multiscale.py [--motif 1] [--pressure 1000] [--maxiter 200]
    python3 optimise_B5_multiscale.py --diameters 1.5 2.0   # subset
"""
import argparse, csv, json, os, sys, subprocess, tempfile
import numpy as np
from scipy.optimize import minimize

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "..", "data")
OUT_DIR   = os.path.join(HERE, "optimisation")

BASE_MESH = os.path.join(DATA_DIR, "B5_remeshed_shared.off")   # 1.2 m dome
CABLE_PATHS_FILE = os.environ.get(
    "FEM_CABLE_PATHS", os.path.join(DATA_DIR, "cable_paths_B5_remeshed.json"))
BINARY = os.environ.get(
    "FEM_BINARY_NREGION", os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))

KNIT_DIR = {(r, c): 0 for r in range(3) for c in range(3)}

ALL_DIAMETERS = [1.2, 1.5, 2.0, 3.0]

# ── OFF helpers ────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

def save_off(path, V, F):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def scaled_mesh_path(diameter):
    tag = diameter_tag(diameter)
    if abs(diameter - 1.2) < 1e-6:
        return BASE_MESH      # already 1.2 m
    return os.path.join(DATA_DIR, f"B5_remeshed_shared_{tag}.off")

def diameter_tag(d):
    return f"{d:.1f}m".replace(".", "p")

def ensure_scaled_mesh(diameter):
    path = scaled_mesh_path(diameter)
    if os.path.exists(path):
        return path
    V0, F0 = load_off(BASE_MESH)
    base_span = V0[:,0].max() - V0[:,0].min()
    scale = diameter / base_span
    V_scaled = V0 * scale
    save_off(path, V_scaled, F0)
    print(f"  Created {os.path.basename(path)}  (scale={scale:.4f})")
    return path

# ── Region map ────────────────────────────────────────────────────────────────
def make_region_map(V, F, x_lo, x_hi, y_lo, y_hi):
    face_cx = V[F[:,0],0]/3 + V[F[:,1],0]/3 + V[F[:,2],0]/3
    face_cy = V[F[:,0],1]/3 + V[F[:,1],1]/3 + V[F[:,2],1]/3
    regions = []
    for cx, cy in zip(face_cx, face_cy):
        col = 0 if cx < x_lo else (1 if cx < x_hi else 2)
        row = 0 if cy < y_lo else (1 if cy < y_hi else 2)
        regions.append(row * 3 + col)
    return regions

# ── Cable paths ───────────────────────────────────────────────────────────────
def load_cable_paths(path):
    if not os.path.exists(path):
        print(f"WARNING: cable paths not found: {path}")
        return []
    with open(path) as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {str(i): v for i, v in enumerate(d)}

# ── FEM runner ────────────────────────────────────────────────────────────────
_call_count = [0]

def run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
            region_map_path, cable_paths_list, cable_ea,
            cable_rest_scales=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":    float(pressure),
        "motif":       int(motif),
        "cable_ea":    float(cable_ea),
        "cable_paths": cable_paths_list if cable_paths_list else [],
        "regions": [
            {
                "sf_wale":      float(sf_w[r]),
                "sf_course":    float(sf_c[r]),
                "knit_dir_deg": float(knit_dirs[r]),
            }
            for r in range(9)
        ],
    }
    if cable_rest_scales is not None:
        params["cable_rest_scales"] = [float(x) for x in cable_rest_scales]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"run_{_call_count[0]:05d}")
    cmd = [BINARY, mesh_path, region_map_path, params_path, prefix]

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
            out["verts"] = vdata[:, 1:]

        return out
    except Exception as e:
        print(f"  FEM exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass

# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(mesh_path, V_target, interior_idx, region_map_path,
                   knit_dirs, pressure, motif, cable_paths_list, cable_ea):
    t_crown = float(V_target[:,2].max())
    history = []

    def objective(p):
        sf_w   = p[:9]
        sf_c   = p[9:18]
        scales = p[18:] if len(p) > 18 else None
        out = run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
                      region_map_path, cable_paths_list, cable_ea,
                      cable_rest_scales=scales)
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
            print(f"    [{_call_count[0]:4d}]  rmse={loss:.4f} m  "
                  f"crown={out.get('crown_height',0):.4f} m  (target {t_crown:.4f})")
        return loss

    return objective, history

# ── Per-diameter optimisation ──────────────────────────────────────────────────
def optimise_one(diameter, motif, pressure, maxiter, cable_paths_dict):
    tag = diameter_tag(diameter)
    print(f"\n{'='*60}")
    print(f"  Diameter = {diameter} m  (tag={tag})")
    print(f"{'='*60}")

    mesh_path = ensure_scaled_mesh(diameter)
    V, F = load_off(mesh_path)
    fem_span = V[:,0].max() - V[:,0].min()
    print(f"  Mesh: {len(V)}v / {len(F)}f  span={fem_span:.4f} m")

    # Target = same file (3D dome, FEM matches to it)
    V_target = V.copy()
    bdry_mask    = np.abs(V_target[:,2]) < 0.01 * diameter
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:,2].max())
    print(f"  Target crown={t_crown:.4f} m  interior={len(interior_idx)} verts")

    # Cable paths (vertex indices, scale-independent)
    cable_names = list(cable_paths_dict.keys())
    cable_paths_list = list(cable_paths_dict.values())
    n_cables = len(cable_paths_list)
    print(f"  Cables: {n_cables} sections  EA={157000.0:.0f} N  pressure={pressure} Pa")

    # Region map
    R = fem_span / 2
    region_ids = make_region_map(V, F, -R/3, R/3, -R/3, R/3)
    knit_dirs  = [KNIT_DIR[(r//3, r%3)] for r in range(9)]

    region_map_path = os.path.join(OUT_DIR, f"region_map_{tag}.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": region_ids}, f)
    counts = np.bincount(region_ids, minlength=9)
    print(f"  Regions: {counts.tolist()}")

    # Initial guess
    p0 = np.concatenate([np.full(18, 1.041), np.full(n_cables, 1.0)])
    print(f"\n  Sanity check (sf=1.041, cable_rest_scale=1.0)...")
    out0 = run_fem(mesh_path, p0[:9], p0[9:18], knit_dirs, pressure, motif,
                   region_map_path, cable_paths_list, 157000.0,
                   cable_rest_scales=p0[18:])
    if out0 is None:
        print("  ERROR: FEM failed at initial point — skipping this diameter")
        return None
    diff0 = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(diff0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0:.4f} m")

    # Optimise
    bounds = [(0.9, 2.0)] * 18 + [(0.85, 1.05)] * n_cables
    print(f"\n  Optimising {18+n_cables} params, maxiter={maxiter}...")
    objective, history = make_objective(
        mesh_path, V_target, interior_idx, region_map_path,
        knit_dirs, pressure, motif, cable_paths_list, 157000.0)

    result = minimize(
        objective, p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "eps": 0.05},
    )

    p_opt  = result.x
    sf_w   = p_opt[:9]
    sf_c   = p_opt[9:18]
    scales = p_opt[18:]

    print(f"\n  Converged: {result.success}  |  {result.message}")
    print(f"  Final RMSE: {result.fun:.4f} m   calls: {_call_count[0]}")
    print(f"  Normalised RMSE / diameter: {result.fun / diameter:.4f}")

    # Final FEM run for verification
    out_final = run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
                        region_map_path, cable_paths_list, 157000.0,
                        cable_rest_scales=scales)
    final_crown = out_final.get("crown_height", float("nan")) if out_final else float("nan")

    result_dict = {
        "diameter_m":    diameter,
        "tag":           tag,
        "motif":         motif,
        "pressure_pa":   pressure,
        "fem_mesh":      mesh_path,
        "target_crown_m": float(t_crown),
        "final_crown_m": float(final_crown),
        "converged":     bool(result.success),
        "loss_rmse_m":   float(result.fun),
        "rmse_normalised": float(result.fun / diameter),
        "n_calls":       _call_count[0],
        "regions": [
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

    out_json = os.path.join(OUT_DIR, f"B5_{tag}_optimised_params.json")
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  Saved: {out_json}")
    return result_dict

# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison(all_results):
    print(f"\n{'='*75}")
    print(f"  B5 SCALE COMPARISON  (pressure=1000 Pa, motif=1)")
    print(f"{'='*75}")
    hdr = (f"{'Diam':>6}  {'Conv':>5}  {'RMSE[m]':>9}  {'RMSE/D':>8}  "
           f"{'Crown_t':>8}  {'Crown_f':>8}  {'sf_w_mean':>9}  {'sf_c_mean':>9}")
    print(hdr)
    print("-"*75)
    for r in all_results:
        if r is None:
            continue
        sf_w_m = np.mean([reg["sf_wale"]  for reg in r["regions"]])
        sf_c_m = np.mean([reg["sf_course"] for reg in r["regions"]])
        print(f"  {r['diameter_m']:>4.1f}m  "
              f"{'Y' if r['converged'] else 'N':>5}  "
              f"{r['loss_rmse_m']:>9.4f}  "
              f"{r['rmse_normalised']:>8.4f}  "
              f"{r['target_crown_m']:>8.4f}  "
              f"{r['final_crown_m']:>8.4f}  "
              f"{sf_w_m:>9.4f}  "
              f"{sf_c_m:>9.4f}")
    print(f"{'='*75}")

    print("\n  Per-region sf_wale and sf_course by diameter:")
    print(f"  {'Region':>6}  " + "  ".join(f"{'d='+str(r['diameter_m'])+'m':>14}"
                                             for r in all_results if r))
    for reg_id in range(9):
        row_s = f"  {reg_id:>6}  "
        for r in all_results:
            if r is None:
                row_s += " "*16
                continue
            reg = r["regions"][reg_id]
            row_s += f"  w={reg['sf_wale']:.3f}/c={reg['sf_course']:.3f}"
        print(row_s)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",     type=int,   default=1)
    parser.add_argument("--pressure",  type=float, default=1000.0)
    parser.add_argument("--maxiter",   type=int,   default=200)
    parser.add_argument("--diameters", type=float, nargs="+",
                        default=ALL_DIAMETERS,
                        help="Diameters to run (default: 1.2 1.5 2.0 3.0)")
    args = parser.parse_args()

    if not os.path.exists(BINARY):
        print(f"FEM binary not found: {BINARY}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    cable_paths_dict = load_cable_paths(CABLE_PATHS_FILE)
    if not cable_paths_dict:
        print("ERROR: no cable paths loaded")
        sys.exit(1)
    print(f"Loaded {len(cable_paths_dict)} cable sections from {CABLE_PATHS_FILE}")

    all_results = []
    for d in sorted(args.diameters):
        res = optimise_one(d, args.motif, args.pressure, args.maxiter, cable_paths_dict)
        all_results.append(res)

    if any(r is not None for r in all_results):
        print_comparison(all_results)

    # Save combined summary
    summary = [r for r in all_results if r is not None]
    summary_path = os.path.join(OUT_DIR, "B5_multiscale_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

if __name__ == "__main__":
    main()
