"""
D5 (shell with opening): inverse fit with VIRTUAL FABRIC in the opening.

During inflation the knit is continuous over the opening and is cut out after
the shell has set, so the opening is a pressurised, bulging patch of fabric
that loads the rim, not a free hole.  The published model clamped the rim
instead.  Here:

  * mesh = D5_virtual_fabric.off (build_D5_virtual_fabric.py): the shell plus
    a patch closing the opening; only the z = 0 ground ring (and the rim's
    ground contact) is fixed, so the rim is free and is held by --rim.
  * design variables = the 2*n_pairs symmetric shell stretch factors
    (as optimise_D5_symmetric.py) + one (sf_wale, sf_course) pair for the patch.
  * objective = RMSE over the SHELL vertices only (rim included) + the
    Laplacian smoothing over the shell regions.  The patch is not scored: its
    shape is irrelevant, only its pull on the rim matters.

Every boundary vertex is fixed, so the volume-work pressure is exact
(no follower load needed).

Usage:
    python3 optimise_D5_virtual_fabric.py --rim cable [--out-prefix d5_vf_cable]
    python3 optimise_D5_virtual_fabric.py --rim ring10
"""
import argparse, csv, json, os, subprocess, tempfile
import numpy as np
from scipy.optimize import minimize

import optimise_D5_symmetric as S

HERE   = os.path.dirname(os.path.abspath(__file__))
MESH   = os.path.join(HERE, "data", "D5", "D5_virtual_fabric.off")
INFO_J = os.path.join(HERE, "data", "D5", "D5_virtual_fabric.json")
OUT_DIR = S.OUT_DIR
E_GFRP = 40e9


def rim_params(rim_mode, rim):
    if rim_mode == "cable":
        return {"cable_ea": S.CABLE_EA, "cable_paths": [rim + [rim[0]]],
                "cable_rest_scales": [S.CABLE_SCALE]}
    if rim_mode.startswith("ring"):
        i0 = rim.index(min(rim, key=lambda v: _V[v, 2]))
        path = rim[i0:] + rim[:i0 + 1]
        d = float(rim_mode[4:]) / 1000.0
        return {"spline_paths": [path], "spline_EA": E_GFRP * np.pi * d**2 / 4,
                "spline_EI": E_GFRP * np.pi * d**4 / 64, "spline_rest": 1}
    if rim_mode == "none":
        return {}
    raise SystemExit(f"unknown --rim {rim_mode}")


_V = None
_calls = [0]


def run_fem(regions, base, rmap_path, prefix):
    params = {**base, "regions": regions}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
    try:
        r = subprocess.run([S.BINARY, MESH, rmap_path, pf.name, prefix],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    finally:
        os.unlink(pf.name)
    res = [float(l.split("max=")[1].split()[0]) for l in r.stderr.splitlines()
           if l.startswith("SOLVER_RESIDUAL")]
    if not res or res[-1] > 1e-3:
        return None, f"residual {res[-1] if res else 'n/a'}"
    vp = prefix + "_verts.csv"
    if not os.path.exists(vp):
        return None, "no output"
    X = np.loadtxt(vp, delimiter=",", skiprows=1)[:, 1:]
    if not np.all(np.isfinite(X)):
        return None, "NaN"
    return X, "OK"


def main():
    global _V
    ap = argparse.ArgumentParser()
    ap.add_argument("--rim", default="cable", help="cable | ring<d mm> | none")
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--lambda-smooth", type=float, default=0.01)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--patch-sf0", type=float, default=1.05)
    ap.add_argument("--init-from-json", default=os.path.join(OUT_DIR, "d5_sym_pfix_optimised.json"))
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()
    pre = args.out_prefix or f"d5_vf_{args.rim}"
    n_pairs, lam = args.n_pairs, args.lambda_smooth
    nsr = 2 * n_pairs                          # shell regions; the patch is region nsr

    VV, FF = S.load_off(MESH)
    info = json.load(open(INFO_J))
    nv, nf = info["n_shell_verts"], info["n_shell_faces"]
    _V = VV
    V, F = VV[:nv], FF[:nf]
    rim = info["rim"]
    field = json.load(open(S.FIELD_J))

    face_region, knit_dirs = S.build_symmetric_region_map(V, F, field, rim, n_pairs)
    region_adj = S.build_region_adj(F, face_region)
    ang = [float(np.degrees(np.arctan2(field[str(f)]["d1"][1], field[str(f)]["d1"][0])) % 180)
           if str(f) in field else 0.0 for f in range(nf)]
    # patch knit direction: that of the nearest shell face (the knit is one piece)
    C = VV[FF].mean(1)
    near = [int(np.argmin(np.linalg.norm(C[:nf] - C[f], axis=1))) for f in range(nf, len(FF))]
    rmap = {"face_regions": face_region.tolist() + [nsr] * (len(FF) - nf),
            "face_knit_dirs_deg": ang + [ang[k] for k in near]}
    os.makedirs(OUT_DIR, exist_ok=True)
    rmap_path = os.path.join(OUT_DIR, f"{pre}_map.json")
    json.dump(rmap, open(rmap_path, "w"))
    a2 = 2 * np.radians([ang[k] for k in near])
    patch_dir = float(np.degrees(np.arctan2(np.sin(a2).mean(), np.cos(a2).mean()) / 2) % 180)

    ground = [int(v) for v in np.where(VV[:, 2] < 1e-6)[0]]
    i0 = int(np.argmin(V[rim, 2]))
    fixed = sorted(set(ground) | {rim[i0]})
    base = {"pressure": S.PRESSURE, "motif": 5, "newton_reg_max": 1e6,
            "fixed_vertices": fixed, **rim_params(args.rim, rim)}

    interior = np.where(np.hypot(V[:, 0], V[:, 1]) <= np.hypot(V[:, 0], V[:, 1]).max() * 0.98)[0]
    t_crown = float(V[:, 2].max())

    sf_w0, sf_c0 = S.seed_from_prev(args.init_from_json, V, F, field, rim, face_region, n_pairs)
    if sf_w0 is None:
        sf_w0, sf_c0 = np.full(n_pairs, S.SF_W0), np.full(n_pairs, S.SF_C0)
    p0 = np.concatenate([sf_w0, sf_c0, [args.patch_sf0, args.patch_sf0]])

    def regions_of(p):
        sw = np.concatenate([p[:n_pairs]] * 2)
        sc = np.concatenate([p[n_pairs:2 * n_pairs]] * 2)
        regs = [{"sf_wale": float(sw[r]), "sf_course": float(sc[r]),
                 "knit_dir_deg": float(knit_dirs[r])} for r in range(nsr)]
        regs.append({"sf_wale": float(p[-2]), "sf_course": float(p[-1]),
                     "knit_dir_deg": patch_dir})
        return regs, sw, sc

    best = {"loss": np.inf}

    def evaluate(p, tag=None):
        _calls[0] += 1
        regs, sw, sc = regions_of(p)
        prefix = os.path.join(OUT_DIR, tag or f"{pre}_eval")
        X, why = run_fem(regs, base, rmap_path, prefix)
        if X is not None:
            crown = X[:nv, 2].max()
            if not (0.3 * t_crown < crown < 3 * t_crown):
                X, why = None, f"crown {crown:.3f}"
        if X is None:
            print(f"  [{_calls[0]:4d}] INVALID: {why}")
            return None
        u = X[:nv] - V
        rmse = float(np.sqrt(np.mean(np.sum(u[interior]**2, axis=1))))
        lap = sum((sw[i] - sw[j])**2 + (sc[i] - sc[j])**2 for i, j in region_adj)
        return dict(X=X, rmse=rmse, lap=lap, loss=rmse + lam * lap,
                    rim_max=float(np.linalg.norm(u[rim], axis=1).max()))

    def objective(p):
        r = evaluate(p)
        if r is None:
            return 1e3
        if r["loss"] < best["loss"]:
            best.update(r, p=p.copy())
        if _calls[0] % 10 == 0:
            print(f"  [{_calls[0]:4d}] RMSE {r['rmse']*1000:6.2f} mm  rim max {r['rim_max']*1000:5.1f} mm  "
                  f"patch sf {p[-2]:.3f}/{p[-1]:.3f}   best {best['rmse']*1000:.2f}", flush=True)
        return r["loss"]

    print(f"Mesh   : {len(VV)} verts ({nv} shell + {len(VV)-nv} patch), {len(FF)} faces "
          f"({len(FF)-nf} patch); {len(fixed)} fixed")
    print(f"Rim    : {args.rim};  patch knit dir {patch_dir:.1f} deg;  "
          f"objective over {len(interior)} shell verts")
    r0 = evaluate(p0, f"{pre}_start")
    print(f"Start  : RMSE {r0['rmse']*1000:.2f} mm, rim max {r0['rim_max']*1000:.1f} mm")

    bounds = [(0.80, 1.30)] * (2 * n_pairs) + [(0.80, 1.50)] * 2
    res = minimize(objective, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": args.maxiter, "ftol": 1e-10, "gtol": 1e-5, "eps": 0.002})

    p = best["p"]
    rf = evaluate(p, f"{pre}_best")
    regs, sw, sc = regions_of(p)
    print(f"\n{res.message}\nBest  : RMSE {rf['rmse']*1000:.2f} mm, rim max {rf['rim_max']*1000:.1f} mm, "
          f"patch sf_wale {p[-2]:.4f} sf_course {p[-1]:.4f}  ({_calls[0]} calls)")
    out = {"geometry": "D5", "model": "virtual_fabric", "rim": args.rim,
           "rim_params": {k: v for k, v in base.items() if k not in ("regions",)},
           "symmetry": "mirror_x", "n_pairs": n_pairs, "n_regions": nsr,
           "lambda_smooth": lam, "pressure": S.PRESSURE,
           "converged": bool(res.success), "message": str(res.message),
           "rmse_mm": rf["rmse"] * 1000, "rim_max_mm": rf["rim_max"] * 1000,
           "n_calls": _calls[0], "start_rmse_mm": r0["rmse"] * 1000,
           "patch": {"sf_wale": float(p[-2]), "sf_course": float(p[-1]), "knit_dir_deg": patch_dir},
           "regions": [{"region_id": r, "pair": r % n_pairs, **regs[r]} for r in range(nsr)]}
    json.dump(out, open(os.path.join(OUT_DIR, f"{pre}_optimised.json"), "w"), indent=2)
    print("Saved", os.path.join(OUT_DIR, f"{pre}_optimised.json"))


if __name__ == "__main__":
    main()
