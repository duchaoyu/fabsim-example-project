"""
D5 FEM inverse optimisation — MIRROR-SYMMETRIC region layout.

D5 is a teardrop/keyhole shell with a single mirror axis (x -> -x, the
vertical y-axis; the opening sits at the bottom).  The plain
`optimise_D5_laplacian.py` divides the dome into N equal-count angular
wedges placed at arbitrary angles, so the optimised material distribution
comes out lopsided even though the geometry is symmetric.

This script enforces the mirror symmetry as a HARD parameter-sharing
constraint:

  * Regions are built symmetrically: `n_pairs` bands per side (banded by
    angle-from-top |beta|), split at the mirror axis -> 2*n_pairs regions
    that come in mirror pairs (h, h + n_pairs).
  * Parameters are tied across each pair: region h and region h+n_pairs
    share (sf_wale, sf_course).  Free params = 2*n_pairs (not 4*n_pairs).
  * Warm-start from a previous asymmetric result by mapping its per-face
    sf onto the new regions and averaging each mirror pair.

Objective: RMSE + lambda_smooth * Σ_{adj (i,j)} [(Δsf_w)² + (Δsf_c)²]
(the Laplacian is evaluated on the full expanded region set, so it also
smooths across the mirror axis).

Usage:
    python3 optimise_D5_symmetric.py [--n-pairs 5] [--lambda-smooth 0.01]
              [--maxiter 500] [--out-prefix d5_sym]
              [--init-from-json optimisation/d5_10lap_v3_optimised.json]
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

CABLE_EA    = 157000.0
PRESSURE    = 1000.0
CABLE_SCALE = 0.95

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


# ── Symmetric region assignment ───────────────────────────────────────────────
def build_symmetric_region_map(V, F, field, cable_idx, n_pairs):
    """Mirror-symmetric regions about the x=0 axis.

    beta = angle of a face centroid from the +y axis (straight up), so the
    mirror x -> -x maps beta -> -beta.  |beta| runs 0 (top) .. pi (bottom /
    opening) and is mirror-invariant, so we band each side independently by
    equal face-count in |beta|.  Region id = band (right, x>=cx) or
    band + n_pairs (left).  Region h and region h+n_pairs are mirror images.

    Knit direction per region = mean d1 angle in that region (placeholder;
    the actual per-face knit dirs come from the directional field).
    """
    n_f       = len(F)
    d1        = np.array([field[str(fi)]["d1"] if str(fi) in field else [1.0, 0.0, 0.0]
                          for fi in range(n_f)])
    centroids = V[F].mean(axis=1)
    cc        = V[cable_idx].mean(axis=0)[:2]

    rel   = centroids[:, :2] - cc
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    beta  = (theta - np.pi / 2 + np.pi) % (2 * np.pi) - np.pi   # from +y, wrapped to (-pi,pi]
    absb  = np.abs(beta)
    side_right = rel[:, 0] >= 0.0                              # x >= cx  ->  right half

    face_region = np.zeros(n_f, dtype=int)
    for right, base in [(True, 0), (False, n_pairs)]:
        idx = np.where(side_right == right)[0]
        order = idx[np.argsort(absb[idx])]
        chunk = len(order) / n_pairs
        for i, fi in enumerate(order):
            face_region[fi] = base + min(int(i / chunk), n_pairs - 1)

    n_regions = 2 * n_pairs
    angles = np.arctan2(d1[:, 1], d1[:, 0])
    knit_dirs = []
    for r in range(n_regions):
        mask = face_region == r
        a2   = 2 * angles[mask]
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
_out_prefix = ["d5_sym"]
_rmap_path  = [None]
_face_knit_dirs_deg = [None]
_cable_path = [None]
_rim = [{"name": "clamped", "params": None, "follower": False}]   # --rim

def run_fem(sf_wale, sf_course, knit_dirs, face_region, n_regions, V_rest):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          PRESSURE,
        "motif":             5,   # wale 12507 / course 5000: stitch structure I (Section 7.2)
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(n_regions)],
    }

    params.update(_rim[0]["params"] or
                  {"cable_ea": CABLE_EA, "cable_paths": [_cable_path[0]],
                   "cable_rest_scales": [float(CABLE_SCALE)]})

    rmap = {"face_regions": face_region.tolist()}
    if _face_knit_dirs_deg[0] is not None:
        rmap["face_knit_dirs_deg"] = _face_knit_dirs_deg[0]
    with open(_rmap_path[0], "w") as f:
        json.dump(rmap, f)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH, _rmap_path[0], params_path, prefix]
    env = dict(os.environ)
    if _rim[0]["follower"]:
        env.update(FEM_PRESSURE="follower", FEM_FOLLOWER_START="volume")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=180 if _rim[0]["follower"] else 30, env=env)
        # A failed Newton solve exits 0 and returns ~the start state (= target):
        # gate on the residual of the last load stage.
        res = [float(l.split("max=")[1].split()[0]) for l in r.stderr.splitlines()
               if l.startswith("SOLVER_RESIDUAL")]
        if res and res[-1] > 1e-3:
            print(f"  [{_call_count[0]:4d}] INVALID: residual {res[-1]:.2e}")
            return None
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


# ── Warm-start seed from a previous (asymmetric) result ───────────────────────
def seed_from_prev(prev_json, V, F, field, cable_idx, face_region, n_pairs):
    """Map a previous per-region result onto the new symmetric regions and
    average each mirror pair.  Reconstructs the previous angular-sector map
    exactly as optimise_D5_laplacian.py did, so the per-face sf is faithful."""
    prev = json.load(open(prev_json))
    prev_regions = prev.get("regions", [])
    n_prev = len(prev_regions)
    if n_prev == 0:
        return None, None
    if prev.get("symmetry") == "mirror_x" and prev.get("n_pairs") == n_pairs:
        # Same symmetric layout: take the pair values directly.
        return (np.array([prev_regions[h]["sf_wale"]   for h in range(n_pairs)]),
                np.array([prev_regions[h]["sf_course"] for h in range(n_pairs)]))

    # Reconstruct previous angular-sector assignment (matches laplacian script).
    centroids = V[F].mean(axis=1)
    cc    = V[cable_idx].mean(axis=0)[:2]
    rel   = centroids[:, :2] - cc
    theta = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(theta)
    prev_fr = np.zeros(len(F), dtype=int)
    chunk   = len(F) / n_prev
    for i, fi in enumerate(order):
        prev_fr[fi] = min(int(i / chunk), n_prev - 1)

    prev_sf_w = np.array([r["sf_wale"]   for r in prev_regions])
    prev_sf_c = np.array([r["sf_course"] for r in prev_regions])
    face_sf_w = prev_sf_w[prev_fr]
    face_sf_c = prev_sf_c[prev_fr]

    # Mean current sf per NEW region, then average mirror pairs (h, h+n_pairs).
    sf_w_pair = np.zeros(n_pairs)
    sf_c_pair = np.zeros(n_pairs)
    for h in range(n_pairs):
        mask = (face_region == h) | (face_region == h + n_pairs)
        sf_w_pair[h] = face_sf_w[mask].mean()
        sf_c_pair[h] = face_sf_c[mask].mean()
    return sf_w_pair, sf_c_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs",       type=int,   default=5,
                        help="Independent mirror pairs; total regions = 2*n_pairs")
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--maxiter",       type=int,   default=500)
    parser.add_argument("--out-prefix",    type=str,   default="d5_sym")
    parser.add_argument("--init-from-json", type=str,
                        default=os.path.join(OUT_DIR, "d5_10lap_v3_optimised.json"),
                        help="Previous result JSON to warm-start from")
    parser.add_argument("--rim", type=str, default="clamped",
                        help="opening rim support: 'clamped' (whole boundary fixed, the "
                             "original model), 'cable' (rim free, inner cable only) or "
                             "'ring<d>' (rim free, GFRP rod of d mm formed to the rim, "
                             "pinned at its ground contact).  Released rims use follower "
                             "pressure.")
    parser.add_argument("--sf-lo", type=float, default=0.80,
                        help="lower bound on the stretch factors (1.001 keeps the "
                             "fabric in tension under the follower load)")
    args = parser.parse_args()

    _out_prefix[0] = args.out_prefix
    _rmap_path[0]  = os.path.join(OUT_DIR, f"{args.out_prefix}_map.json")
    n_pairs        = args.n_pairs
    n_regions      = 2 * n_pairs
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
    if args.rim != "clamped":
        # Release the rim: fix the z = 0 ground ring and the rim's ground contact.
        ci  = cable["vertex_indices"]
        i0  = int(np.argmin(V[ci, 2]))
        rim = ci[i0:] + ci[:i0 + 1]
        fixed = sorted(set(int(v) for v in np.where(V[:, 2] < 1e-6)[0]) | {rim[0]})
        if args.rim == "cable":
            extra = {"cable_ea": CABLE_EA, "cable_paths": [_cable_path[0]],
                     "cable_rest_scales": [float(CABLE_SCALE)]}
        elif args.rim.startswith("ring"):
            d = float(args.rim[4:]) / 1000.0
            extra = {"spline_paths": [rim], "spline_EA": 40e9 * np.pi * d**2 / 4,
                     "spline_EI": 40e9 * np.pi * d**4 / 64, "spline_rest": 1}
        else:
            sys.exit(f"unknown --rim {args.rim}")
        _rim[0] = {"name": args.rim, "follower": True,
                   "params": {**extra, "fixed_vertices": fixed, "newton_reg_max": 1e6}}

    # Per-face knit directions from the directional field (d1 vector -> degrees).
    face_knit_degs = []
    for fi in range(len(F)):
        entry = field.get(str(fi))
        if entry is not None:
            d1 = entry["d1"]
            face_knit_degs.append(float(np.degrees(np.arctan2(d1[1], d1[0])) % 180))
        else:
            face_knit_degs.append(0.0)
    _face_knit_dirs_deg[0] = face_knit_degs

    face_region, knit_dirs = build_symmetric_region_map(
        V, F, field, cable["vertex_indices"], n_pairs)
    region_adj = build_region_adj(F, face_region)

    counts = np.bincount(face_region, minlength=n_regions)
    print(f"FEM mesh   : {len(V)} verts, {len(F)} faces")
    print(f"Target     : {len(interior_idx)} interior verts, crown={t_crown:.4f} m")
    print(f"Symmetry   : mirror x -> -x  ({n_pairs} pairs -> {n_regions} regions)")
    print(f"Regions    : {counts.tolist()}")
    print(f"Pairs      : " + ", ".join(f"({h},{h+n_pairs})" for h in range(n_pairs)))
    print(f"Adj pairs  : {len(region_adj)}")
    print(f"λ_smooth   : {lam}")
    print(f"Rim        : {args.rim}" + (f" (inner cable sc={CABLE_SCALE})"
                                            if args.rim in ("clamped", "cable") else ""))

    # ── Initial parameters (warm-start from previous result) ───────────────────
    sf_w0 = sf_c0 = None
    if args.init_from_json and os.path.exists(args.init_from_json):
        sf_w0, sf_c0 = seed_from_prev(args.init_from_json, V, F, field,
                                      cable["vertex_indices"], face_region, n_pairs)
        if sf_w0 is not None:
            print(f"Init from : {args.init_from_json} (mapped + mirror-averaged)")
            print(f"  seed sf_wale  = [{','.join(f'{v:.3f}' for v in sf_w0)}]")
            print(f"  seed sf_course= [{','.join(f'{v:.3f}' for v in sf_c0)}]")
    if sf_w0 is None:
        sf_w0 = np.full(n_pairs, SF_W0)
        sf_c0 = np.full(n_pairs, SF_C0)
        print(f"Init from : defaults SF_W0={SF_W0} SF_C0={SF_C0}")

    def expand(p):
        """Half params (n_pairs sf_w + n_pairs sf_c) -> full 2*n_pairs regions."""
        sf_w_h = p[:n_pairs]
        sf_c_h = p[n_pairs:]
        sf_w = np.concatenate([sf_w_h, sf_w_h])   # region h and h+n_pairs share
        sf_c = np.concatenate([sf_c_h, sf_c_h])
        return sf_w, sf_c

    # ── Sanity check ──────────────────────────────────────────────────────────
    print(f"\nSanity check …")
    sf_w_s, sf_c_s = expand(np.concatenate([sf_w0, sf_c0]))
    out0 = run_fem(sf_w_s, sf_c_s, knit_dirs, face_region, n_regions, V_rest)
    if out0 is None:
        print("ERROR: FEM failed at sanity check"); sys.exit(1)
    d0    = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(d0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0*1000:.2f} mm")

    # ── Objective ─────────────────────────────────────────────────────────────
    def objective(p):
        sf_w, sf_c = expand(p)
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
                  f"sf_w=[{','.join(f'{v:.3f}' for v in p[:n_pairs])}]")
        return loss

    p0     = np.concatenate([sf_w0, sf_c0])
    bounds = [(args.sf_lo, 1.30)] * (2 * n_pairs)
    p0     = np.clip(p0, args.sf_lo, 1.30)

    print(f"\nOptimising {2*n_pairs} free params (symmetric), λ={lam}, "
          f"maxiter={args.maxiter} …")
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-10,
                            "gtol": 1e-5, "eps": 0.002})

    print(f"\nConverged: {res.success}  |  {res.message}")
    sf_w_opt, sf_c_opt = expand(res.x)

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
    print(f"\nPer-region params (mirror pairs share values):")
    print(f"  {'R':>3}  {'pair':>4}  {'faces':>6}  {'knit°':>6}  {'sf_wale':>8}  {'sf_course':>9}")
    for r in range(n_regions):
        print(f"  {r:3d}  {r % n_pairs:4d}  {counts[r]:6d}  {knit_dirs[r]:6.1f}  "
              f"{sf_w_opt[r]:8.4f}  {sf_c_opt[r]:9.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "geometry": "D5", "symmetry": "mirror_x", "n_pairs": n_pairs,
        "n_regions": n_regions, "lambda_smooth": lam,
        "pressure": PRESSURE, "cable_ea": CABLE_EA,
        "cable_scale_fixed": CABLE_SCALE,
        "rim": args.rim, "sf_lo": args.sf_lo, "rim_params": _rim[0]["params"],
        "converged": bool(res.success), "message": res.message,
        "rmse_mm": rmse_f, "n_calls": _call_count[0],
        "region_adj": [[int(i), int(j)] for i, j in region_adj],
        "pair_of_region": [int(r % n_pairs) for r in range(n_regions)],
        "regions": [{"region_id": r, "pair": int(r % n_pairs),
                     "n_faces": int(counts[r]),
                     "knit_dir_deg": knit_dirs[r],
                     "sf_wale": float(sf_w_opt[r]),
                     "sf_course": float(sf_c_opt[r])}
                    for r in range(n_regions)],
    }
    out_json = os.path.join(OUT_DIR, f"{args.out_prefix}_optimised.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_json}")

    # Also persist the region map for downstream visualisation.
    rmap_out = os.path.join(OUT_DIR, f"{args.out_prefix}_region_map.json")
    with open(rmap_out, "w") as f:
        json.dump({"face_regions": face_region.tolist(),
                   "n_pairs": n_pairs, "n_regions": n_regions}, f)
    print(f"Saved: {rmap_out}")


if __name__ == "__main__":
    main()
