"""
D5 FEM inverse optimisation — 4 regions, adaptive boundaries.

Alternating optimisation (ported from best_fit_stretch_factors_3region_adaptive.cpp):

  (A) Inner L-BFGS — fix region map, optimise sf_wale/sf_course per region
  (B) Outer boundary reassignment — fix sf values, greedily re-assign each
      boundary face to whichever neighbouring region lowers the RMSE loss.
      Connectivity constraint: a face is only moved if removing it from its
      current region leaves that region still connected (BFS check).

Convergence: stop when step (B) swaps zero faces (or max_outer reached).

Starting point:
  Region map   — D5_4region_map.json (built from p60 dev-threshold BFS,
                 same as the fixed-region run)
  sf warm-start — 1-region global optimum: sf_wale=1.1526, sf_course=1.0725

Usage:
    python3 optimise_D5_4region_adaptive.py [--max-outer 8] [--inner-iter 100]
                                            [--out-prefix d5_4ra]
"""
import argparse, csv, json, os, subprocess, sys, tempfile
from collections import deque, defaultdict
import numpy as np
from scipy.optimize import minimize

HERE    = os.path.dirname(os.path.abspath(__file__))
MESH    = os.path.join(HERE, "data", "D5", "D5_remeshed_fem_cable2faces.off")
TARGET  = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
CABLE_J = os.path.join(HERE, "data", "D5", "D5_cable_inner.json")
FIELD_J = os.path.join(HERE, "data", "D5", "directional_field_D5.json")
ONEREGION_VERTS = os.path.join(HERE, "optimisation", "d5_opt_00130_verts.csv")
RMAP_J  = os.path.join(HERE, "optimisation", "D5_4region_map.json")
BINARY  = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
OUT_DIR = os.path.join(HERE, "optimisation")

N_REGIONS   = 4
SEED_VERTS  = [781, 140, 95]
DEV_PCT     = 60
CABLE_EA     = 157000.0
PRESSURE     = 1000.0
CABLE_SCALE  = 0.95
CABLE2_VERTS  = [89, 88]   # cross-cable spanning inner opening to fix crown height
CABLE2_SCALE0 = 0.90       # initial value; optimised during inner L-BFGS
SF_W0       = 1.1526
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


# ── Connectivity check ────────────────────────────────────────────────────────
def is_connected_without(fi, r, face_region, adj):
    """True if region r remains connected when face fi is removed."""
    nF    = len(face_region)
    start = next((g for g in range(nF) if face_region[g] == r and g != fi), None)
    if start is None:
        return False  # would empty the region
    count   = sum(1 for g in range(nF) if face_region[g] == r and g != fi)
    visited = [False] * nF
    queue   = deque([start])
    visited[start] = True
    reached = 1
    while queue:
        g = queue.popleft()
        for nb in adj[g]:
            if not visited[nb] and face_region[nb] == r and nb != fi:
                visited[nb] = True
                reached += 1
                queue.append(nb)
    return reached == count


# ── Build initial region map (from saved JSON or recompute) ───────────────────
def load_or_build_region_map(V, F):
    if os.path.exists(RMAP_J):
        with open(RMAP_J) as f:
            face_region = json.load(f)["face_regions"]
        print(f"  Loaded region map from {RMAP_J}")
    else:
        # recompute from deviation threshold
        v1r      = np.loadtxt(ONEREGION_VERTS, delimiter=",", skiprows=1)[:, 1:]
        V_target, _ = load_off(TARGET)
        vert_dev    = np.linalg.norm(v1r - V_target, axis=1) * 1000
        nF          = len(F)
        face_dev    = np.array([vert_dev[F[fi]].mean() for fi in range(nF)])
        threshold   = float(np.percentile(face_dev, DEV_PCT))
        adj         = build_face_adj(F)
        seed_faces  = []
        for sv in SEED_VERTS:
            for fi, face in enumerate(F):
                if sv in face: seed_faces.append(fi); break
        face_region = [-1] * nF
        queue       = deque()
        for r, sf in enumerate(seed_faces):
            face_region[sf] = r; queue.append(sf)
        while queue:
            fi = queue.popleft(); r = face_region[fi]
            for nb in adj[fi]:
                if face_region[nb] == -1 and face_dev[nb] >= threshold:
                    face_region[nb] = r; queue.append(nb)
        for fi in range(nF):
            if face_region[fi] == -1: face_region[fi] = 3
        print(f"  Recomputed region map (p{DEV_PCT} threshold)")

    region_arr = np.array(face_region)
    counts     = np.bincount(region_arr, minlength=N_REGIONS)
    print(f"  Region sizes: R0={counts[0]} R1={counts[1]} R2={counts[2]} R3={counts[3]}")
    return list(face_region)


# ── Knit directions: per-face from directional field ─────────────────────────
# Per-face angles are written to the region_map JSON so the C++ binary uses
# the actual field direction for every face instead of a single regional mean.
_face_knit_dirs_deg = [None]   # set once in main()

def compute_knit_dirs(face_region, F):
    """Per-region circular mean — kept for params JSON (backward compat)."""
    with open(FIELD_J) as f:
        field = json.load(f)
    nF     = len(F)
    d1_list = [field[str(fi)]["d1"] if str(fi) in field else [1.0, 0.0, 0.0]
               for fi in range(nF)]
    d1     = np.array(d1_list)
    angles = np.arctan2(d1[:, 1], d1[:, 0])
    region_arr = np.array(face_region)
    knit_dirs  = []
    for r in range(N_REGIONS):
        mask = region_arr == r
        a2   = 2 * angles[mask]
        mean = np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2
        knit_dirs.append(float(np.degrees(mean) % 180))
    return knit_dirs

def load_face_knit_dirs(F):
    """Load per-face d1 angles (degrees, [0,180)) from the directional field.
    Faces not in the field (cable pressure faces) default to 0°."""
    with open(FIELD_J) as f:
        field = json.load(f)
    result = []
    for fi in range(len(F)):
        entry = field.get(str(fi))
        if entry is not None:
            d1 = entry["d1"]
            result.append(float(np.degrees(np.arctan2(d1[1], d1[0])) % 180))
        else:
            result.append(0.0)
    return result


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
_call_count  = [0]
_out_prefix  = ["d5_4ra"]
_rmap_path   = [None]

def run_fem(sf_wale, sf_course, knit_dirs, face_region, V_rest, scale_cable2=CABLE2_SCALE0):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          PRESSURE,
        "motif":             1,
        "cable_ea":          CABLE_EA,
        "cable_paths":       [_cable_path[0], CABLE2_VERTS],
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(N_REGIONS)],
        "cable_rest_scales": [float(CABLE_SCALE), float(scale_cable2)],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf); params_path = pf.name

    # write current region map (include per-face knit dirs if available)
    rmap = {"face_regions": face_region}
    if _face_knit_dirs_deg[0] is not None:
        rmap["face_knit_dirs_deg"] = _face_knit_dirs_deg[0]
    with open(_rmap_path[0], "w") as f:
        json.dump(rmap, f)

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH, _rmap_path[0], params_path, prefix]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=40)
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


# ── Inner L-BFGS ─────────────────────────────────────────────────────────────
def inner_optimise(p0, face_region, knit_dirs, interior_idx, V_target, V_rest,
                   maxiter, label=""):
    # p0 = [sf_w x N, sf_c x N, scale_cable2]
    call_ref = [0]
    def objective(p):
        sf_w = p[:N_REGIONS]; sf_c = p[N_REGIONS:2*N_REGIONS]; sc2 = p[2*N_REGIONS]
        out  = run_fem(sf_w, sf_c, knit_dirs, face_region, V_rest, scale_cable2=sc2)
        if out is None or "verts" not in out:
            return 1e3
        diff = out["verts"][interior_idx] - V_target[interior_idx]
        loss = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        call_ref[0] += 1
        if call_ref[0] % 5 == 0:
            print(f"    [{_call_count[0]:4d}] RMSE={loss:.4f} m  sc2={sc2:.4f}  "
                  f"sf_w=[{','.join(f'{v:.4f}' for v in sf_w)}]  "
                  f"sf_c=[{','.join(f'{v:.4f}' for v in sf_c)}]")
        return loss

    bounds = [(0.80, 1.50)] * N_REGIONS + [(0.80, 1.50)] * N_REGIONS + [(0.70, 1.05)]
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-9,
                            "gtol": 1e-5, "eps": 0.002})
    print(f"  {label} inner: {res.message}  RMSE={res.fun:.4f} m  calls={call_ref[0]}")
    return res.x, res.fun


# ── Boundary smoothing ────────────────────────────────────────────────────────
def smooth_boundaries(face_region, adj, n_passes=2):
    """
    Majority-vote smoothing: reassign each face to the region held by the
    majority of its edge-adjacent neighbours, provided connectivity is preserved.
    Purely geometric — no FEM calls. Run for n_passes sweeps.
    """
    nF = len(face_region)
    total_swapped = 0
    for _ in range(n_passes):
        new_region = list(face_region)
        n_swapped  = 0
        for fi in range(nF):
            nbrs = adj[fi]
            if not nbrs:
                continue
            counts = defaultdict(int)
            for nb in nbrs:
                counts[face_region[nb]] += 1
            majority = max(counts, key=counts.get)
            if majority != face_region[fi]:
                if is_connected_without(fi, face_region[fi], face_region, adj):
                    new_region[fi] = majority
                    n_swapped += 1
        face_region[:] = new_region
        total_swapped += n_swapped
        if n_swapped == 0:
            break
    arr = np.bincount(np.array(face_region), minlength=N_REGIONS)
    print(f"  [smooth] {total_swapped} face(s) reassigned  "
          f"→ R0={arr[0]} R1={arr[1]} R2={arr[2]} R3={arr[3]}")
    return face_region


# ── Outer boundary reassignment ───────────────────────────────────────────────
def reassign_boundary_faces(sf_wale, sf_course, scale_cable2, knit_dirs,
                            face_region, adj, interior_idx, V_target, V_rest):
    """
    Greedy Jacobi-style: try moving each boundary face to each neighbour region.
    Accept if RMSE improves and connectivity is preserved.
    Returns (new_face_region, n_swapped).
    """
    nF        = len(face_region)
    base_out  = run_fem(sf_wale, sf_course, knit_dirs, face_region, V_rest, scale_cable2=scale_cable2)
    if base_out is None or "verts" not in base_out:
        print("  [reassign] base FEM failed — skipping")
        return face_region, 0
    base_diff  = base_out["verts"][interior_idx] - V_target[interior_idx]
    base_loss  = float(np.sqrt(np.mean(np.sum(base_diff**2, axis=1))))
    print(f"  [reassign] base RMSE = {base_loss*1000:.2f} mm")

    new_region = list(face_region)
    n_swapped  = 0

    for fi in range(nF):
        r_orig = face_region[fi]
        # collect unique neighbour regions
        nbr_regions = {face_region[nb] for nb in adj[fi] if face_region[nb] != r_orig}
        if not nbr_regions:
            continue  # interior face
        # connectivity guard
        if not is_connected_without(fi, r_orig, face_region, adj):
            continue

        best_loss = base_loss
        best_r    = r_orig

        for r2 in nbr_regions:
            face_region[fi] = r2        # trial
            # recompute knit dir for affected regions only (fast approx: reuse)
            trial_out = run_fem(sf_wale, sf_course, knit_dirs, face_region, V_rest, scale_cable2=scale_cable2)
            face_region[fi] = r_orig    # restore
            if trial_out is None or "verts" not in trial_out:
                continue
            diff  = trial_out["verts"][interior_idx] - V_target[interior_idx]
            trial = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
            if trial < best_loss - 1e-6:
                best_loss = trial; best_r = r2

        new_region[fi] = best_r
        if best_r != r_orig:
            n_swapped += 1

    face_region[:] = new_region   # apply all accepted swaps
    counts = np.bincount(np.array(face_region), minlength=N_REGIONS)
    print(f"  [reassign] {n_swapped} face(s) swapped  "
          f"→ R0={counts[0]} R1={counts[1]} R2={counts[2]} R3={counts[3]}")
    return face_region, n_swapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-outer",    type=int,   default=12)
    parser.add_argument("--inner-iter",   type=int,   default=80)
    parser.add_argument("--out-prefix",   type=str,   default="d5_4ra")
    parser.add_argument("--sf0-wale",     type=float, default=SF_W0)
    parser.add_argument("--sf0-course",   type=float, default=SF_C0)
    parser.add_argument("--init-from-json", type=str, default=None,
                        help="JSON with per-region sf_wale/sf_course to use as initial guess")
    parser.add_argument("--smooth-passes",type=int,   default=2,
                        help="Majority-vote smoothing passes after boundary reassignment (0=off)")
    parser.add_argument("--smooth-every", type=int,   default=10,
                        help="Apply smoothing every N outer iterations")
    args = parser.parse_args()
    _out_prefix[0] = args.out_prefix
    _rmap_path[0]  = os.path.join(OUT_DIR, f"{_out_prefix[0]}_map.json")

    os.makedirs(OUT_DIR, exist_ok=True)
    for path, lbl in [(BINARY,"binary"),(MESH,"mesh"),(CABLE_J,"cable"),(FIELD_J,"field")]:
        if not os.path.exists(path):
            print(f"Not found: {lbl} {path}"); sys.exit(1)

    V, F        = load_off(MESH)
    V_rest      = V.copy()
    V_target, _ = load_off(TARGET)
    bdry_mask   = np.hypot(V_target[:,0], V_target[:,1]) > \
                  np.hypot(V_target[:,0], V_target[:,1]).max() * 0.98
    interior_idx     = np.where(~bdry_mask)[0]
    t_crown          = float(V_target[:,2].max())
    _target_crown[0] = t_crown

    with open(CABLE_J) as f: cable = json.load(f)
    _cable_path[0] = cable["vertex_indices"] + [cable["vertex_indices"][0]]

    print(f"FEM mesh : {len(V)} verts, {len(F)} faces")
    print(f"Target   : {len(interior_idx)} interior verts, crown={t_crown:.4f} m")

    face_region = load_or_build_region_map(V, F)
    adj         = build_face_adj(F)

    # Per-face knit directions from directional field (written to region_map JSON)
    _face_knit_dirs_deg[0] = load_face_knit_dirs(F)
    # Per-region circular mean — still used in params JSON
    knit_dirs = compute_knit_dirs(face_region, F)
    print(f"Knit dirs (region mean, for params): {[round(d,1) for d in knit_dirs]}°")

    # ── Warm-start check ──────────────────────────────────────────────────────
    if args.init_from_json and os.path.exists(args.init_from_json):
        with open(args.init_from_json) as f:
            init_j = json.load(f)
        sf_w0 = np.array([init_j["regions"][r]["sf_wale"]  for r in range(N_REGIONS)])
        sf_c0 = np.array([init_j["regions"][r]["sf_course"] for r in range(N_REGIONS)])
        sc2_0 = float(init_j.get("scale_cable2", CABLE2_SCALE0))
        print(f"  Init from {args.init_from_json}")
    else:
        sf_w0 = np.full(N_REGIONS, args.sf0_wale)
        sf_c0 = np.full(N_REGIONS, args.sf0_course)
        sc2_0 = CABLE2_SCALE0
    print(f"\nWarm-start check …")
    out0 = run_fem(sf_w0, sf_c0, knit_dirs, face_region, V_rest, scale_cable2=sc2_0)
    if out0 is None:
        print("ERROR: FEM failed at warm-start"); sys.exit(1)
    d0    = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(d0**2, axis=1))))
    print(f"  RMSE = {rmse0*1000:.2f} mm")

    # ── Alternating optimisation ──────────────────────────────────────────────
    p_cur     = np.concatenate([sf_w0, sf_c0, [sc2_0]])
    best_rmse = rmse0

    for outer in range(args.max_outer):
        counts = np.bincount(np.array(face_region), minlength=N_REGIONS)
        print(f"\n══ Outer {outer+1}/{args.max_outer}  "
              f"R0={counts[0]} R1={counts[1]} R2={counts[2]} R3={counts[3]} ══")

        # (A) Inner L-BFGS
        print("  (A) Inner L-BFGS …")
        p_cur, rmse_inner = inner_optimise(
            p_cur, face_region, knit_dirs, interior_idx, V_target, V_rest,
            maxiter=args.inner_iter, label=f"outer{outer+1}")
        best_rmse = min(best_rmse, rmse_inner)

        sf_w_cur = p_cur[:N_REGIONS]
        sf_c_cur = p_cur[N_REGIONS:2*N_REGIONS]
        sc2_cur  = float(p_cur[2*N_REGIONS])
        print(f"  After inner: RMSE={rmse_inner*1000:.2f} mm  sc2={sc2_cur:.4f}")
        for r in range(N_REGIONS):
            lbl = f"R{r}" if r < 3 else "R3(rest)"
            print(f"    {lbl}: sf_w={sf_w_cur[r]:.4f}  sf_c={sf_c_cur[r]:.4f}")

        # (B) Outer boundary reassignment
        print("  (B) Boundary reassignment …")
        face_region, n_swapped = reassign_boundary_faces(
            sf_w_cur, sf_c_cur, sc2_cur, knit_dirs,
            face_region, adj, interior_idx, V_target, V_rest)

        # (C) Boundary smoothing (every --smooth-every iterations)
        if args.smooth_passes > 0 and (outer + 1) % args.smooth_every == 0:
            print(f"  (C) Boundary smoothing ({args.smooth_passes} passes) …")
            face_region = smooth_boundaries(face_region, adj, n_passes=args.smooth_passes)

        # save updated region map
        with open(_rmap_path[0], "w") as f:
            json.dump({"face_regions": face_region}, f)

        if n_swapped == 0:
            print("  → No faces swapped. Converged.")
            break

    # ── Final result ──────────────────────────────────────────────────────────
    sf_w_opt = p_cur[:N_REGIONS]
    sf_c_opt = p_cur[N_REGIONS:2*N_REGIONS]
    sc2_opt  = float(p_cur[2*N_REGIONS])

    print(f"\nRunning final FEM …")
    out_f = run_fem(sf_w_opt, sf_c_opt, knit_dirs, face_region, V_rest, scale_cable2=sc2_opt)
    counts = np.bincount(np.array(face_region), minlength=N_REGIONS)
    if out_f and "verts" in out_f:
        d      = out_f["verts"][interior_idx] - V_target[interior_idx]
        rmse_f = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
        print(f"  RMSE  = {rmse_f*1000:.2f} mm")
        print(f"  crown = {out_f['crown_height']:.4f} m  (target {t_crown:.4f} m)")

    labels = [f"R{r} (seed sv{SEED_VERTS[r]})" for r in range(3)] + ["R3 (rest)"]
    print(f"\n=== Optimal 4-region adaptive parameters ===")
    print(f"{'Region':>22}  {'faces':>6}  {'knit°':>6}  {'sf_wale':>8}  {'sf_course':>9}")
    for r in range(N_REGIONS):
        print(f"{labels[r]:>22}  {counts[r]:6d}  {knit_dirs[r]:6.1f}  "
              f"{sf_w_opt[r]:8.4f}  {sf_c_opt[r]:9.4f}")

    result = {
        "geometry": "D5", "n_regions": N_REGIONS,
        "method": "adaptive_boundary_reassignment",
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "cable_scale_fixed": CABLE_SCALE,
        "scale_cable2": sc2_opt,
        "seed_vertices": SEED_VERTS,
        "dev_percentile": DEV_PCT,
        "converged": True,
        "rmse_m": float(rmse_f) if out_f else None,
        "n_calls": _call_count[0],
        "regions": [{"region_id": r, "label": labels[r],
                     "n_faces": int(counts[r]),
                     "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w_opt[r]),
                     "sf_course": float(sf_c_opt[r])}
                    for r in range(N_REGIONS)],
    }
    out_json = os.path.join(OUT_DIR, f"{_out_prefix[0]}_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Total FEM calls: {_call_count[0]}")
    print(f"\nComparison:")
    print(f"  1-region global  : 7.61 mm")
    print(f"  3-region BFS     : 5.94 mm")
    if out_f:
        print(f"  4-region adaptive: {rmse_f*1000:.2f} mm")


if __name__ == "__main__":
    main()
