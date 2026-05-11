"""
D5 FEM inverse optimisation.

3 parameters:
  p = [sf_wale, sf_course, scale_cable]

8 regions (angular sectors around inner cable centroid), knit directions
from the directional field.  1 cable: inner boundary loop (42 verts).

Usage:
    python3 optimise_D5.py [--maxiter 300] [--out-prefix d5_opt]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
import numpy as np
from scipy.optimize import minimize

HERE     = os.path.dirname(os.path.abspath(__file__))
MESH     = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
TARGET   = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J  = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
FIELD_J  = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
BINARY   = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
OUT_DIR  = os.path.join(HERE, "optimisation")

N_REGIONS    = 8
CABLE_EA     = 157000.0
PRESSURE     = 1000.0
CABLE2_VERTS = [89, 88]   # cross-cable spanning inner opening to fix crown height
CABLE2_SCALE = 1.0        # no pre-tension; constrains distance only


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f: lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F


# ── Build region map from 8 angular sectors around inner cable centroid ───────
def build_region_map(V, F):
    with open(CABLE_J) as f: cable = json.load(f)
    with open(FIELD_J) as f: field = json.load(f)

    ci = np.array(cable["vertices"]).mean(axis=0)
    centroids = V[F].mean(axis=1)
    dx = centroids[:, 0] - ci[0]
    dy = centroids[:, 1] - ci[1]
    az = np.degrees(np.arctan2(dy, dx)) % 360
    face_region = (az // (360 / N_REGIONS)).astype(int).tolist()

    d1 = np.array([field[str(fi)]["d1"] for fi in range(len(F))])
    angles = np.arctan2(d1[:, 1], d1[:, 0])

    knit_dirs = []
    for s in range(N_REGIONS):
        mask = np.array(face_region) == s
        if mask.sum() == 0:
            knit_dirs.append(0.0)
            continue
        a2 = 2 * angles[mask]
        mean_a = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
        knit_dirs.append(float(np.degrees(mean_a) % 180))

    cable_path = cable["vertex_indices"] + [cable["vertex_indices"][0]]  # closed loop
    return face_region, knit_dirs, cable_path


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
_out_prefix  = ["d5_opt"]


def run_fem(sf_wale, sf_course, scale_cable,
            knit_dirs, face_region, cable_path, V_rest):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          PRESSURE,
        "motif":             1,
        "cable_ea":          CABLE_EA,
        "cable_paths":       [cable_path, CABLE2_VERTS],
        "regions":           [{"sf_wale":      float(sf_wale),
                               "sf_course":    float(sf_course),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(N_REGIONS)],
        "cable_rest_scales": [float(scale_cable), float(CABLE2_SCALE)],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    region_map_path = os.path.join(OUT_DIR, "D5_region_map.json")
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
    parser.add_argument("--out-prefix", type=str,   default="d5_opt")
    parser.add_argument("--sf0-wale",   type=float, default=1.032)
    parser.add_argument("--sf0-course", type=float, default=1.042)
    parser.add_argument("--sc0-cable",  type=float, default=0.95)
    args = parser.parse_args()
    _out_prefix[0] = args.out_prefix

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, lbl in [(BINARY,"binary"),(MESH,"mesh"),(CABLE_J,"cable"),(FIELD_J,"field")]:
        if not os.path.exists(path): print(f"Not found: {lbl} {path}"); sys.exit(1)

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

    face_region, knit_dirs, cable_path = build_region_map(V, F)
    counts = np.bincount(face_region, minlength=N_REGIONS)
    print(f"Regions  : {counts.tolist()}")
    print(f"KnitDirs : {[round(d,1) for d in knit_dirs]}°")
    print(f"Cable    : {len(cable_path)-1} segments (inner loop) + cross-cable v{CABLE2_VERTS[0]}→v{CABLE2_VERTS[1]} (scale={CABLE2_SCALE})")
    print(f"Pressure : {PRESSURE} Pa  |  motif: 1")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print(f"\nSanity check (sf_w={args.sf0_wale}, sf_c={args.sf0_course}, sc={args.sc0_cable}) …")
    out0 = run_fem(args.sf0_wale, args.sf0_course, args.sc0_cable,
                   knit_dirs, face_region, cable_path, V_rest)
    if out0 is None:
        print("ERROR: FEM failed at warm-start"); sys.exit(1)
    d0 = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(d0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0:.4f} m")
    if args.maxiter == 0:
        print("maxiter=0 — sanity check only."); return

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(p):
        sf_w, sf_c, sc = p
        out = run_fem(sf_w, sf_c, sc, knit_dirs, face_region, cable_path, V_rest)
        if out is None or "verts" not in out:
            return 1e3
        diff = out["verts"][interior_idx] - V_target[interior_idx]
        loss = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        if _call_count[0] % 5 == 0:
            print(f"  [{_call_count[0]:4d}]  RMSE={loss:.4f} m  "
                  f"p=[sf_w={sf_w:.4f}, sf_c={sf_c:.4f}, sc={sc:.4f}]")
        return loss

    p0     = np.array([args.sf0_wale, args.sf0_course, args.sc0_cable])
    bounds = [(0.80, 1.20), (0.80, 1.20), (0.70, 1.05)]

    print(f"\nOptimising 3 params, maxiter={args.maxiter} …")
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-9,
                            "gtol": 1e-5, "eps": 0.002})

    print(f"\nConverged: {res.success}  |  {res.message}")
    print(f"Final RMSE: {res.fun:.4f} m   FEM calls: {_call_count[0]}")
    p_opt = res.x
    print(f"  sf_wale={p_opt[0]:.4f}  sf_course={p_opt[1]:.4f}  scale_cable={p_opt[2]:.4f}")

    # ── Final FEM ─────────────────────────────────────────────────────────────
    print("\nRunning final FEM …")
    out_f = run_fem(p_opt[0], p_opt[1], p_opt[2],
                    knit_dirs, face_region, cable_path, V_rest)
    if out_f and "verts" in out_f:
        d = out_f["verts"][interior_idx] - V_target[interior_idx]
        rmse_f = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
        print(f"  crown={out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")
        print(f"  RMSE ={rmse_f:.4f} m")

    # ── Save result JSON ───────────────────────────────────────────────────────
    result = {
        "geometry": "D5", "n_regions": N_REGIONS,
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "converged": bool(res.success),
        "rmse_m": float(res.fun), "n_calls": _call_count[0],
        "params": {"sf_wale": float(p_opt[0]),
                   "sf_course": float(p_opt[1]),
                   "scale_cable": float(p_opt[2])},
        "regions": [{"region_id": r, "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(p_opt[0]),
                     "sf_course": float(p_opt[1])} for r in range(N_REGIONS)],
    }
    out_json = os.path.join(OUT_DIR, "D5_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
