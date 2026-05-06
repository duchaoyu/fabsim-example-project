"""
C5 D8-symmetric FEM inverse optimisation.

Pipeline:
  1. FEM mesh = FDM/data/C5_remeshed.off (D8-projected dome, 961v / 1856f)
     used as both rest shape and target — sf>1 makes the rest shape smaller
     than the dome so the elastic-inward force balances pressure at dome
     equilibrium (same trick as B5).
  2. 8 cables along ridge spokes from data/cable_paths_C5.json,
     dividing the dome into 8 D8-equivalent wedge regions.
  3. D8 parameter sharing reduces 8 region triples + 8 cable scales (32 vars)
     to:
        sf_wale, sf_course        — shared across all 8 regions
        knit_dir_relative         — knit angle relative to each wedge spoke;
                                     absolute knit_dir per region = wedge_centre
                                     angle + knit_dir_relative
        cable_rest_scale          — shared across all 8 cables
     → 4 free variables (vs B5's 30).
  4. Objective: vertex RMSE of FEM-inflated shape vs C5 target (interior verts).

Usage:
    python3 optimise_C5.py [--motif 1] [--pressure 1000] [--maxiter 200]
"""
import argparse, csv, json, os, sys, subprocess, tempfile
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))

# Inputs
MESH_PATH        = os.environ.get("FEM_MESH",
    os.path.join(HERE, "data", "C5_remeshed.off"))
TARGET_OFF       = os.environ.get("FEM_TARGET",
    os.path.join(HERE, "data", "C5_remeshed.off"))
CABLE_PATHS_FILE = os.environ.get("FEM_CABLE_PATHS",
    os.path.join(HERE, "..", "data", "cable_paths_C5.json"))
BINARY           = os.environ.get("FEM_BINARY_NREGION",
    os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))

OUT_DIR = os.path.join(HERE, "optimisation")

# D8 geometry: 8 wedges, with cable spokes at θ = 11.25° + k·45°
N_REGIONS    = 8
SPOKE_OFFSET = 11.25                           # first spoke angle (deg)
WEDGE_WIDTH  = 360 / N_REGIONS                 # 45 deg per wedge
WEDGE_CENTRES = [SPOKE_OFFSET + WEDGE_WIDTH * (k + 0.5) for k in range(N_REGIONS)]
# i.e. wedge k spans [SPOKE_OFFSET + k·45°, SPOKE_OFFSET + (k+1)·45°],
# with bisector at SPOKE_OFFSET + (k+0.5)·45°  (between two spokes)


# ── Load OFF ──────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2 + nv]])
    F = np.array([[int(x) for x in l.split()[1:4]] for l in lines[2 + nv:2 + nv + nf]])
    return V, F


# ── Wedge assignment (8 regions split at the cable spokes) ────────────────────
def make_region_map_d8(V, F):
    cx = (V[F[:, 0], 0] + V[F[:, 1], 0] + V[F[:, 2], 0]) / 3
    cy = (V[F[:, 0], 1] + V[F[:, 1], 1] + V[F[:, 2], 1]) / 3
    theta_deg = (np.degrees(np.arctan2(cy, cx)) - SPOKE_OFFSET) % 360
    return (theta_deg // WEDGE_WIDTH).astype(int).tolist()


# ── Load cable paths ──────────────────────────────────────────────────────────
def load_cable_paths(path):
    with open(path) as f:
        d = json.load(f)
    if isinstance(d, dict):
        names = list(d.keys())
        return list(d.values()), names
    return d, [f"S{i}" for i in range(len(d))]


# ── Run FEM binary ────────────────────────────────────────────────────────────
_call_count = [0]


def run_fem(sf_wale, sf_course, knit_relative_deg, pressure, motif,
            region_map_path, cable_paths, cable_ea, cable_rest_scale):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    # Same sf for all 8 wedges; knit_dir rotates per wedge to preserve D8
    regions = [
        {
            "sf_wale":      float(sf_wale),
            "sf_course":    float(sf_course),
            "knit_dir_deg": float((WEDGE_CENTRES[r] + knit_relative_deg) % 180),
        }
        for r in range(N_REGIONS)
    ]
    params = {
        "pressure":          float(pressure),
        "motif":             int(motif),
        "cable_ea":          float(cable_ea),
        "cable_paths":       cable_paths,
        "regions":           regions,
        "cable_rest_scales": [float(cable_rest_scale)] * len(cable_paths),
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
def make_objective(V_target, interior_idx, region_map_path, pressure, motif,
                   cable_paths, cable_ea):
    t_crown = float(V_target[:, 2].max())
    history = []

    def objective(p):
        sf_w, sf_c, knit_rel, cable_scale = p
        out = run_fem(sf_w, sf_c, knit_rel, pressure, motif,
                      region_map_path, cable_paths, cable_ea, cable_scale)
        if out is None:
            return 1e6

        if "verts" in out:
            diff = out["verts"][interior_idx] - V_target[interior_idx]
            loss = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        else:
            loss = abs(out["crown_height"] - t_crown) / (t_crown + 1e-6)

        history.append({"call": _call_count[0], "loss": loss,
                        "crown_height": out.get("crown_height", 0.0),
                        "sf_wale": float(sf_w), "sf_course": float(sf_c),
                        "knit_rel": float(knit_rel),
                        "cable_scale": float(cable_scale)})
        if _call_count[0] % 5 == 0 or _call_count[0] == 1:
            print(f"  [{_call_count[0]:4d}]  rmse={loss:.4f} m  "
                  f"crown={out.get('crown_height', 0):.4f} (target {t_crown:.4f})  "
                  f"sf=({sf_w:.3f},{sf_c:.3f})  knit_rel={knit_rel:6.2f}°  "
                  f"cable={cable_scale:.4f}")
        return loss

    return objective, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",    type=int,   default=1)
    parser.add_argument("--pressure", type=float, default=1000.0)
    parser.add_argument("--maxiter",  type=int,   default=200)
    parser.add_argument("--mesh",     type=str,   default=None)
    args = parser.parse_args()

    global MESH_PATH
    if args.mesh:
        MESH_PATH = args.mesh

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(BINARY):
        print(f"FEM binary not found: {BINARY}"); sys.exit(1)
    if not os.path.exists(MESH_PATH):
        print(f"Mesh not found: {MESH_PATH}"); sys.exit(1)

    V, F = load_off(MESH_PATH)
    span = V[:, 0].max() - V[:, 0].min()
    print(f"FEM mesh:   {len(V)} verts, {len(F)} faces, span={span:.2f} m")

    V_target, _   = load_off(TARGET_OFF)
    bdry_mask     = np.abs(V_target[:, 2]) < 0.01
    interior_idx  = np.where(~bdry_mask)[0]
    t_crown       = float(V_target[:, 2].max())
    print(f"Target:     {len(V_target)} verts, {len(interior_idx)} interior, crown={t_crown:.4f} m")

    cable_paths, cable_names = load_cable_paths(CABLE_PATHS_FILE)
    cable_ea = 157000.0
    print(f"Cables:     {len(cable_paths)} spokes ({', '.join(cable_names)})")
    print(f"            EA={cable_ea:.0f} N  (1 mm steel)")

    region_ids = make_region_map_d8(V, F)
    region_map_path = os.path.join(OUT_DIR, "region_map.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": region_ids}, f)
    counts = np.bincount(region_ids, minlength=N_REGIONS)
    print(f"Regions:    {counts.tolist()} (D8 wedges)")
    print(f"Wedge centres: {[round(c, 2) for c in WEDGE_CENTRES]}°")
    print(f"Pressure:   {args.pressure} Pa  |  motif: {args.motif}")

    # sf≈2.3 gives crown≈5.4m (target≈5.05m) with C5 mesh at p=1000Pa
    p0 = np.array([2.3, 2.3, 0.0, 1.0])           # sf_w, sf_c, knit_rel, cable_scale
    print(f"\nSanity check at sf=2.3, knit_rel=0°, cable_rest=1.0 ...")
    out0 = run_fem(p0[0], p0[1], p0[2], args.pressure, args.motif,
                   region_map_path, cable_paths, cable_ea, p0[3])
    if out0 is None:
        print("ERROR: FEM failed at initial guess"); sys.exit(1)
    print(f"  crown={out0['crown_height']:.4f} m  (target {t_crown:.4f})")
    if "verts" in out0:
        diff0 = out0["verts"][interior_idx] - V_target[interior_idx]
        print(f"  RMSE = {np.sqrt(np.mean(np.sum(diff0 ** 2, axis=1))):.4f} m")

    bounds = [(1.0, 5.0),       # sf_wale
              (1.0, 5.0),       # sf_course
              (-90.0, 90.0),    # knit_rel  (knit fibre direction relative to spoke, mod 180°)
              (0.85, 1.05)]     # cable_rest_scale
    print(f"\nOptimising 4 D8-shared params, maxiter={args.maxiter}...")
    print(f"  L-BFGS-B, bounds = {bounds}, FD eps = 0.05\n")

    objective, history = make_objective(
        V_target, interior_idx, region_map_path,
        args.pressure, args.motif, cable_paths, cable_ea)

    result = minimize(
        objective, p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-8, "gtol": 1e-6, "eps": 0.05},
    )

    print(f"\nConverged: {result.success}  |  {result.message}")
    print(f"Final RMSE: {result.fun:.4f} m   calls: {_call_count[0]}")

    sf_w, sf_c, knit_rel, cable_scale = result.x
    print(f"\n=== Optimal D8 parameters ===")
    print(f"  sf_wale            = {sf_w:.4f}")
    print(f"  sf_course          = {sf_c:.4f}")
    print(f"  knit_rel (per spoke) = {knit_rel:.3f}°")
    print(f"  cable_rest_scale   = {cable_scale:.4f}  (1.0 = no pre-stress)")

    print("\nRunning final FEM simulation...")
    out_final = run_fem(sf_w, sf_c, knit_rel, args.pressure, args.motif,
                        region_map_path, cable_paths, cable_ea, cable_scale)
    if out_final:
        print(f"  crown = {out_final['crown_height']:.4f} m  (target {t_crown:.4f})")
        if "verts" in out_final:
            diff_f = out_final["verts"][interior_idx] - V_target[interior_idx]
            print(f"  RMSE  = {np.sqrt(np.mean(np.sum(diff_f ** 2, axis=1))):.4f} m")

    results = {
        "geometry":       "C5",
        "symmetry":       "D8",
        "motif":          args.motif,
        "pressure":       args.pressure,
        "fem_mesh":       MESH_PATH,
        "target_crown_m": float(t_crown),
        "converged":      bool(result.success),
        "loss_rmse_m":    float(result.fun),
        "n_calls":        _call_count[0],
        "optimum": {
            "sf_wale":          float(sf_w),
            "sf_course":        float(sf_c),
            "knit_rel_deg":     float(knit_rel),
            "cable_rest_scale": float(cable_scale),
        },
        "wedge_centres_deg": WEDGE_CENTRES,
        "cable_names":       cable_names,
        "history":           history[-30:],
    }
    out_json = os.path.join(OUT_DIR, "C5_optimised_params.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
