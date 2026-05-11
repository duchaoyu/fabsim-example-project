"""
D5 FEM inverse optimisation — 4 regions.

Region layout derived from the 1-region optimal result:
  R0, R1, R2 — local high-deviation patches grown by BFS from seed vertices
                781, 140, 95 (the three largest-error spots in the 1-region fit).
                BFS stops where per-face mean deviation falls below the p60
                threshold (~6.6 mm), so each patch covers the "problem" area.
  R3          — everything else (the bulk that already fits well).

8 free parameters:
  p = [sf_wale_0..3, sf_course_0..3]

Warm-start: all regions at 1-region global optimum
  sf_wale=1.1526, sf_course=1.0725, cable_scale=0.95.

Usage:
    python3 optimise_D5_4region.py [--maxiter 300] [--out-prefix d5_4r]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from collections import deque, defaultdict
import numpy as np
from scipy.optimize import minimize

HERE    = os.path.dirname(os.path.abspath(__file__))
MESH    = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
TARGET  = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
FIELD_J = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
ONEREGION_VERTS = os.path.join(HERE, "optimisation", "d5_opt_00130_verts.csv")
BINARY  = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
OUT_DIR = os.path.join(HERE, "optimisation")

N_REGIONS   = 4
SEED_VERTS  = [781, 140, 95]      # seeds for the 3 local patches
DEV_PCT     = 60                  # p60 face-deviation threshold
CABLE_EA    = 157000.0
PRESSURE    = 1000.0
CABLE_SCALE = 0.95
SF_W0       = 1.1526              # global optimum warm-start
SF_C0       = 1.0725


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


# ── 4-region map ──────────────────────────────────────────────────────────────
def build_region_map(V, F):
    with open(CABLE_J) as f: cable = json.load(f)
    with open(FIELD_J) as f: field = json.load(f)

    nF  = len(F)
    adj = build_face_adj(F)

    # Per-vertex deviation from 1-region optimal result
    v1r  = np.loadtxt(ONEREGION_VERTS, delimiter=",", skiprows=1)[:, 1:]
    V_target, _ = load_off(TARGET)
    vert_dev = np.linalg.norm(v1r - V_target, axis=1) * 1000  # mm

    # Per-face mean deviation
    face_dev  = np.array([vert_dev[F[fi]].mean() for fi in range(nF)])
    threshold = float(np.percentile(face_dev, DEV_PCT))
    print(f"  Dev threshold (p{DEV_PCT}): {threshold:.2f} mm")

    # Seed faces (first face containing each seed vertex)
    seed_faces = []
    for sv in SEED_VERTS:
        for fi, face in enumerate(F):
            if sv in face:
                seed_faces.append(fi); break
    assert len(seed_faces) == 3

    # Multi-source BFS: claim face only if face_dev >= threshold
    region = [-1] * nF
    queue  = deque()
    for r, sf in enumerate(seed_faces):
        region[sf] = r
        queue.append(sf)

    while queue:
        fi = queue.popleft()
        r  = region[fi]
        for nb in adj[fi]:
            if region[nb] == -1 and face_dev[nb] >= threshold:
                region[nb] = r
                queue.append(nb)

    # Region 3 = everything not claimed by the 3 local patches
    for fi in range(nF):
        if region[fi] == -1:
            region[fi] = 3

    # Per-region mean knit direction (circular mean)
    d1      = np.array([field[str(fi)]["d1"] for fi in range(nF)])
    angles  = np.arctan2(d1[:, 1], d1[:, 0])
    region_arr = np.array(region)
    knit_dirs  = []
    for r in range(N_REGIONS):
        mask = region_arr == r
        a2   = 2 * angles[mask]
        mean = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
        knit_dirs.append(float(np.degrees(mean) % 180))

    cable_path = cable["vertex_indices"] + [cable["vertex_indices"][0]]
    counts     = np.bincount(region, minlength=N_REGIONS)
    return region, knit_dirs, cable_path, seed_faces, counts, threshold


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
_out_prefix = ["d5_4r"]


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

    region_map_path = os.path.join(OUT_DIR, "D5_4region_map.json")
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
    parser.add_argument("--out-prefix", type=str,   default="d5_4r")
    parser.add_argument("--sf0-wale",   type=float, default=SF_W0)
    parser.add_argument("--sf0-course", type=float, default=SF_C0)
    args = parser.parse_args()
    _out_prefix[0] = args.out_prefix

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, lbl in [(BINARY,"binary"),(MESH,"mesh"),(CABLE_J,"cable"),
                      (FIELD_J,"field"),(ONEREGION_VERTS,"1-region verts")]:
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

    print(f"FEM mesh : {len(V)} verts, {len(F)} faces")
    print(f"Target   : {len(interior_idx)} interior verts, crown={t_crown:.4f} m")

    face_region, knit_dirs, cable_path, seed_faces, counts, threshold = \
        build_region_map(V, F)

    print(f"Seed verts  : {SEED_VERTS}  →  seed faces: {seed_faces}")
    print(f"Region sizes: R0={counts[0]} R1={counts[1]} R2={counts[2]} R3(rest)={counts[3]}")
    print(f"Knit dirs   : {[round(d,1) for d in knit_dirs]}°")
    print(f"Cable       : {len(cable_path)-1} segments, fixed scale={CABLE_SCALE}")

    # ── Warm-start ────────────────────────────────────────────────────────────
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

    print(f"\nOptimising {2*N_REGIONS} params (4×sf_wale + 4×sf_course), "
          f"maxiter={args.maxiter} …")
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-9,
                            "gtol": 1e-5, "eps": 0.002})

    print(f"\nConverged: {res.success}  |  {res.message}")
    print(f"Final RMSE: {res.fun:.4f} m   FEM calls: {_call_count[0]}")

    p_opt    = res.x
    sf_w_opt = p_opt[:N_REGIONS]
    sf_c_opt = p_opt[N_REGIONS:]

    print(f"\n=== Optimal 4-region parameters ===")
    labels = [f"R{r} (seed {SEED_VERTS[r]})" for r in range(3)] + ["R3 (rest)"]
    print(f"{'Region':>20}  {'faces':>6}  {'knit°':>6}  {'sf_wale':>8}  {'sf_course':>9}")
    for r in range(N_REGIONS):
        print(f"{labels[r]:>20}  {counts[r]:6d}  {knit_dirs[r]:6.1f}  "
              f"{sf_w_opt[r]:8.4f}  {sf_c_opt[r]:9.4f}")

    # ── Final FEM ─────────────────────────────────────────────────────────────
    print("\nRunning final FEM …")
    out_f = run_fem(sf_w_opt, sf_c_opt, knit_dirs, face_region, cable_path, V_rest)
    if out_f and "verts" in out_f:
        d     = out_f["verts"][interior_idx] - V_target[interior_idx]
        rmse_f = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
        print(f"  crown={out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")
        print(f"  RMSE ={rmse_f:.4f} m")

    # ── Save result JSON ───────────────────────────────────────────────────────
    result = {
        "geometry": "D5", "n_regions": N_REGIONS,
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "cable_scale_fixed": CABLE_SCALE,
        "dev_threshold_mm": threshold, "dev_percentile": DEV_PCT,
        "seed_vertices": SEED_VERTS, "seed_faces": seed_faces,
        "converged": bool(res.success),
        "rmse_m": float(res.fun), "n_calls": _call_count[0],
        "regions": [{"region_id": r,
                     "label": labels[r],
                     "n_faces": int(counts[r]),
                     "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w_opt[r]),
                     "sf_course": float(sf_c_opt[r])}
                    for r in range(N_REGIONS)],
    }
    out_json = os.path.join(OUT_DIR, "D5_4region_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
