"""
C5 16-region FEM inverse optimisation.

56 free variables:
  16 × sf_wale       — wale (radial) stretch factor per region
  16 × sf_course     — course (circumferential) stretch factor per region
  24 × cable_rest_scale — 8 inner spokes + 8 outer spokes + 8 hoop arcs

Knit directions are fixed from the radial directional field:
  knit_dir_deg = azimuth(region centroid) % 180   (wale = radial)

FEM mesh + cable paths: FDM/data/C5_remeshed.off + data/cable_paths_C5.json
  Full 16-vert spoke paths are split at the hoop ring (r ≈ 5.33 m);
  hoop arcs are built from the 64-vertex hoop ring in the OFF mesh.

Usage:
    python3 optimise_C5_16region.py [--motif 1] [--pressure 1000] [--maxiter 300]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))

MESH_PATH        = os.environ.get("FEM_MESH",
    os.path.join(HERE, "data", "C5_remeshed.off"))
TARGET_OFF       = os.environ.get("FEM_TARGET",
    os.path.join(HERE, "data", "C5_remeshed.off"))
SPOKE_PATHS_FILE = os.environ.get("FEM_CABLE_PATHS",
    os.path.join(HERE, "..", "data", "cable_paths_C5.json"))
BINARY           = os.environ.get("FEM_BINARY_NREGION",
    os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))
OUT_DIR          = os.path.join(HERE, "optimisation")

N_REGIONS = 16   # 8 inner + 8 outer wedges
N_CABLES  = 24   # 8 inner spokes + 8 outer spokes + 8 hoop arcs


# ── Mesh loader ───────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


# ── Build 24 cable sections from existing spoke paths ─────────────────────────
def build_cable_paths(V):
    r_xy = np.hypot(V[:, 0], V[:, 1])

    with open(SPOKE_PATHS_FILE) as f:
        spokes = json.load(f)

    # Sort spokes by name (S0 < S1 < … < S7)
    spoke_list = sorted(spokes.items())

    # Split each 16-vert spoke at its hoop crossing (vertex closest to r_hoop).
    # First pass: find all split vertices to compute a consistent r_hoop.
    r_guess   = 5.33
    split_idx = [int(np.argmin(np.abs(r_xy[path] - r_guess)))
                 for _, path in spoke_list]
    hoop_junctions = [spoke_list[k][1][split_idx[k]] for k in range(8)]
    r_hoop = float(np.mean(r_xy[hoop_junctions]))

    # Build inner / outer spoke sections
    inner_paths, outer_paths = [], []
    spoke_names = []
    for k, (name, path) in enumerate(spoke_list):
        si  = split_idx[k]
        inner_paths.append(path[:si + 1])
        outer_paths.append(path[si:])
        spoke_names.append(name)

    # Hoop ring: all vertices at r_hoop ± 0.4 m, sorted by azimuth
    ring_mask = np.abs(r_xy - r_hoop) < 0.4
    ring_idx  = np.where(ring_mask)[0]
    ring_az   = np.arctan2(V[ring_idx, 1], V[ring_idx, 0])
    hoop_ring = ring_idx[np.argsort(ring_az)].tolist()   # 64 verts, CCW

    def ring_arc(v_start, v_end):
        """Extract the short arc of hoop_ring from v_start to v_end (CCW)."""
        n = len(hoop_ring)
        i0 = hoop_ring.index(v_start)
        i1 = hoop_ring.index(v_end)
        if i1 >= i0:
            return hoop_ring[i0:i1 + 1]
        return hoop_ring[i0:] + hoop_ring[:i1 + 1]   # wrap-around

    # 8 hoop arcs between consecutive spoke junctions (in spoke angular order)
    junc_az  = np.degrees(np.arctan2(V[hoop_junctions, 1],
                                      V[hoop_junctions, 0])) % 360
    order    = np.argsort(junc_az).tolist()   # sort junctions by azimuth
    hoop_arc_paths = []
    hoop_arc_names = []
    for i in range(8):
        k0 = order[i]
        k1 = order[(i + 1) % 8]
        arc = ring_arc(hoop_junctions[k0], hoop_junctions[k1])
        hoop_arc_paths.append(arc)
        hoop_arc_names.append(f"Ha{i}")

    # Concatenate: Si0,So0, Si1,So1, …, Ha0…Ha7
    paths, names = [], []
    for k in range(8):
        paths.append(inner_paths[k]); names.append(f"Si{k}")
        paths.append(outer_paths[k]); names.append(f"So{k}")
    paths.extend(hoop_arc_paths); names.extend(hoop_arc_names)

    return paths, names, hoop_junctions, r_hoop


# ── 16-region map on OFF mesh ─────────────────────────────────────────────────
def build_region_map(V, F, hoop_junctions, r_hoop):
    centroids = V[F].mean(axis=1)
    cx, cy    = centroids[:, 0], centroids[:, 1]
    r_face    = np.hypot(cx, cy)
    az_face   = np.degrees(np.arctan2(cy, cx)) % 360

    # Sector boundaries = midpoints between consecutive spoke junction azimuths
    junc_az = np.sort(np.degrees(
        np.arctan2(V[hoop_junctions, 1], V[hoop_junctions, 0])) % 360)
    mid_az  = []
    for k in range(8):
        a1, a2 = junc_az[k], junc_az[(k + 1) % 8]
        if a2 < a1: a2 += 360
        mid_az.append((a1 + a2) / 2 % 360)
    mid_az = np.sort(mid_az)

    def sector(az):
        return int(np.searchsorted(mid_az, az) % 8)

    face_region = [sector(az) + (0 if r_face[fi] < r_hoop else 8)
                   for fi, az in enumerate(az_face)]
    return face_region, mid_az


# ── Per-region knit direction (wale = radial, circular mean) ──────────────────
def region_knit_dirs(V, F, face_region):
    centroids = V[F].mean(axis=1)
    cx, cy    = centroids[:, 0], centroids[:, 1]
    sin_sum   = np.zeros(N_REGIONS)
    cos_sum   = np.zeros(N_REGIONS)
    for fi, r in enumerate(face_region):
        az = np.arctan2(cy[fi], cx[fi])
        sin_sum[r] += np.sin(az)
        cos_sum[r] += np.cos(az)
    return [float(np.degrees(np.arctan2(sin_sum[r], cos_sum[r])) % 180)
            for r in range(N_REGIONS)]


# ── Run FEM binary ─────────────────────────────────────────────────────────────
_call_count = [0]
_out_prefix  = ["c5_16r"]   # mutable default; overridden in main()

# Minimum displacement from rest shape for a valid FEM result (m).
# If max ||V_sim − V_rest|| < this, the solver returned the undeformed mesh.
_MIN_DISPLACEMENT = 0.05

def run_fem(sf_wale, sf_course, knit_dirs,
            pressure, motif, region_map_path,
            cable_paths, cable_ea, cable_rest_scales, V_rest):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          float(pressure),
        "motif":             int(motif),
        "cable_ea":          float(cable_ea),
        "cable_paths":       cable_paths,
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(N_REGIONS)],
        "cable_rest_scales": [float(s) for s in cable_rest_scales],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH_PATH, region_map_path, params_path, prefix]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            print(f"  [{_call_count[0]:4d}] FEM error (rc={res.returncode}): "
                  f"{res.stderr[:200]}")
            return None
        scalars_path = prefix + "_scalars.csv"
        verts_path   = prefix + "_verts.csv"
        if not os.path.exists(scalars_path):
            print(f"  [{_call_count[0]:4d}] FEM error: no scalars output")
            return None
        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        out = {k: float(v) for k, v in row.items()}
        if os.path.exists(verts_path):
            out["verts"] = np.loadtxt(verts_path, delimiter=",", skiprows=1)[:, 1:]
            # Validity gate: reject if solver returned the undeformed rest shape
            max_disp = float(np.max(np.linalg.norm(out["verts"] - V_rest, axis=1)))
            if max_disp < _MIN_DISPLACEMENT:
                print(f"  [{_call_count[0]:4d}] FEM INVALID: "
                      f"max disp={max_disp:.4f} m — Newton did not converge")
                return None
        return out
    except Exception as e:
        print(f"  [{_call_count[0]:4d}] FEM exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass


# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(V_target, V_rest, interior_idx, knit_dirs,
                   region_map_path, pressure, motif, cable_paths, cable_ea):
    t_crown = float(V_target[:, 2].max())
    history = []

    def objective(p):
        sf_w   = p[:N_REGIONS]
        sf_c   = p[N_REGIONS:2 * N_REGIONS]
        scales = p[2 * N_REGIONS:]
        out = run_fem(sf_w, sf_c, knit_dirs, pressure, motif,
                      region_map_path, cable_paths, cable_ea, scales, V_rest)
        if out is None or "verts" not in out:
            loss = 1e3   # penalise invalid FEM result heavily
        else:
            diff = out["verts"][interior_idx] - V_target[interior_idx]
            loss = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        history.append(loss)
        if len(history) % 10 == 0:
            print(f"  [{_call_count[0]:4d}]  RMSE={loss:.4f} m")
        return loss

    return objective, history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",      type=int,   default=1)
    parser.add_argument("--pressure",   type=float, default=1000.0)
    parser.add_argument("--maxiter",    type=int,   default=300)
    parser.add_argument("--out-prefix", type=str,   default="c5_16r",
                        help="Prefix for output files in optimisation/")
    args = parser.parse_args()
    _out_prefix[0] = args.out_prefix

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, label in [(BINARY, "FEM binary"), (MESH_PATH, "FEM mesh"),
                         (SPOKE_PATHS_FILE, "spoke paths")]:
        if not os.path.exists(path):
            print(f"{label} not found: {path}"); sys.exit(1)

    V, F        = load_off(MESH_PATH)
    V_rest      = V.copy()           # undeformed rest shape for validity checks
    V_target, _ = load_off(TARGET_OFF)
    bdry_mask    = np.hypot(V_target[:, 0], V_target[:, 1]) > \
                   (np.hypot(V_target[:, 0], V_target[:, 1]).max() * 0.98)
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:, 2].max())

    print(f"FEM mesh : {len(V)} verts, {len(F)} faces")
    print(f"Target   : {len(V_target)} verts, {len(interior_idx)} interior, "
          f"crown={t_crown:.4f} m")

    cable_paths, cable_names, hoop_junctions, r_hoop = build_cable_paths(V)
    cable_ea = 157000.0
    print(f"Cables   : {len(cable_paths)} sections  (inner+outer spokes + hoop arcs)")
    print(f"           lens = {[len(p) for p in cable_paths]}")
    print(f"Hoop r   : {r_hoop:.3f} m")

    face_region, mid_az = build_region_map(V, F, hoop_junctions, r_hoop)
    knit_dirs            = region_knit_dirs(V, F, face_region)

    region_map_path = os.path.join(OUT_DIR, "C5_16region_map.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": face_region}, f)

    counts = np.bincount(face_region, minlength=N_REGIONS)
    print(f"Regions  : inner={counts[:8].tolist()}")
    print(f"           outer={counts[8:].tolist()}")
    print(f"Knit dirs: {[round(d, 1) for d in knit_dirs]}°")
    print(f"Pressure : {args.pressure} Pa  |  motif: {args.motif}")

    # sf≈2.35 is the uniform starting point where crown ≈ target (5.055 m).
    # The FEM Newton solver diverges below sf≈1.3, so bounds start at 2.0.
    SF0 = 2.35
    p0 = np.concatenate([
        np.full(N_REGIONS, SF0),
        np.full(N_REGIONS, SF0),
        np.full(N_CABLES,  1.000),
    ])

    print(f"\nSanity check at sf={SF0}, cable_scale=1.0 …")
    out0 = run_fem(p0[:N_REGIONS], p0[N_REGIONS:2*N_REGIONS], knit_dirs,
                   args.pressure, args.motif, region_map_path,
                   cable_paths, cable_ea, p0[2*N_REGIONS:], V_rest)
    if out0 is None:
        print("ERROR: FEM failed at initial guess"); sys.exit(1)
    print(f"  crown={out0['crown_height']:.4f} m  (target {t_crown:.4f} m)")
    if "verts" in out0:
        d = out0["verts"][interior_idx] - V_target[interior_idx]
        print(f"  RMSE = {np.sqrt(np.mean(np.sum(d**2, axis=1))):.4f} m")

    if args.maxiter == 0:
        print("maxiter=0 — sanity check only, skipping optimisation.")
        return

    n_params = len(p0)
    # Lower bound 2.0: FEM Newton diverges below sf≈1.3; safe margin at 2.0.
    bounds = ([(2.0, 5.0)] * N_REGIONS +
              [(2.0, 5.0)] * N_REGIONS +
              [(0.85, 1.05)] * N_CABLES)

    print(f"\nOptimising {n_params} params "
          f"({N_REGIONS}×sf_wale + {N_REGIONS}×sf_course + {N_CABLES}×cable_scale), "
          f"maxiter={args.maxiter} …")

    objective, history = make_objective(
        V_target, V_rest, interior_idx, knit_dirs,
        region_map_path, args.pressure, args.motif, cable_paths, cable_ea)

    result = minimize(
        objective, p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": args.maxiter, "ftol": 1e-8, "gtol": 1e-5, "eps": 0.05},
    )

    print(f"\nConverged: {result.success}  |  {result.message}")
    print(f"Final RMSE: {result.fun:.4f} m   FEM calls: {_call_count[0]}")

    p_opt  = result.x
    sf_w   = p_opt[:N_REGIONS]
    sf_c   = p_opt[N_REGIONS:2*N_REGIONS]
    scales = p_opt[2*N_REGIONS:]

    print("\n=== Optimal per-region parameters ===")
    print(f"{'r':>3}  {'zone':>6}  {'sect':>5}  {'knit°':>6}  "
          f"{'sf_wale':>8}  {'sf_course':>9}")
    for r in range(N_REGIONS):
        print(f"{r:3d}  {'inner' if r<8 else 'outer':>6}  {r%8:5d}  "
              f"{knit_dirs[r]:6.1f}  {sf_w[r]:8.4f}  {sf_c[r]:9.4f}")

    print("\n=== Optimal cable rest-length scales ===")
    for i, name in enumerate(cable_names):
        print(f"  {name:>4}  scale={scales[i]:.4f}")

    print("\nRunning final FEM …")
    out_f = run_fem(sf_w, sf_c, knit_dirs, args.pressure, args.motif,
                    region_map_path, cable_paths, cable_ea, scales, V_rest)
    if out_f:
        print(f"  crown = {out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")
        if "verts" in out_f:
            d = out_f["verts"][interior_idx] - V_target[interior_idx]
            print(f"  RMSE  = {np.sqrt(np.mean(np.sum(d**2,axis=1))):.4f} m")

    results = {
        "geometry": "C5", "n_regions": N_REGIONS, "n_cables": N_CABLES,
        "motif": args.motif, "pressure": args.pressure,
        "fem_mesh": MESH_PATH, "target_crown_m": float(t_crown),
        "converged": bool(result.success),
        "loss_rmse_m": float(result.fun), "n_calls": _call_count[0],
        "regions": [{"region_id": r, "zone": "inner" if r<8 else "outer",
                     "sector": r%8, "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w[r]), "sf_course": float(sf_c[r])}
                    for r in range(N_REGIONS)],
        "cables": [{"name": cable_names[i], "rest_scale": float(scales[i])}
                   for i in range(N_CABLES)],
        "history": history[-50:],
    }
    out_json = os.path.join(OUT_DIR, "C5_16region_optimised.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
