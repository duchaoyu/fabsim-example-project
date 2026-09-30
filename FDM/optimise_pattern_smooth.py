"""
pattern_smooth knit + cable FEM inverse optimisation: 4 regions, 3 interior
cables, 4 free-edge cables.

Adapted from optimise_2part.py.  What differs:

  * Supports.  Only the z = 0 vertices of pattern_smooth_tri_m.off are fixed
    (3 point supports + the east line support, 23 vertices), passed to
    fem_batch_nregion as "fixed_vertices".  The rest of the boundary is a FREE
    EDGE held by the four edge cables E00 - E03.  (Without "fixed_vertices" the
    binary fixes the whole topological boundary, which would clamp the free
    edges onto the target and make them unfittable.)

  * Regions.  4, built by field_regions_pattern_smooth.py: the patches the
    interior cables C00 / C01 / C02 cut the surface into, numbered W -> E.
    There is no symmetry to tie anything with.

  * Knit direction is NOT optimised: per face from the cable-guided field
    ("face_knit_dirs_deg" in the region map).

  * Cables.  7 polylines from extract_cables_pattern_smooth.py, each with its
    own rest-length scale: 3 interior + 4 edge.

  * Material.  Stitch structure 1 as MEASURED (E1 10300, E2 13400 N/m, nu 0.58),
    passed explicitly as E1/E2/nu — the binary's motif 1 table still holds the
    pre-2026-09-19 estimates.

  * Loss.  RMSE over every non-support vertex, free edge included: where the
    free edge ends up is part of the shape.

  * rest == target, so the 5 mm displacement floor of optimise_2part.py is kept.
    Every call is logged to optimisation/<prefix>_calls.jsonl.

Usage:
    python3 FDM/optimise_pattern_smooth.py --phase 0 --sweep-sf 0.96,1.0,1.04
    python3 FDM/optimise_pattern_smooth.py --phase 1      # 3 params, uniform
    python3 FDM/optimise_pattern_smooth.py --phase 2 --seed-json ..._p1_result.json
                                                         # 4+4 sf + 7 cables = 15
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import minimize

import optimise_2part as o2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "optimisation")
DATA = os.path.join(HERE, "data", "pattern")

MESH_PATH = os.path.join(DATA, "pattern_smooth_tri_m.off")
TARGET_OFF = MESH_PATH                                    # rest == target
CABLE_FILE = os.path.join(DATA, "cable_paths_pattern_smooth.json")
CABLE_META = os.path.join(DATA, "cable_paths_pattern_smooth.meta.json")
REGION_MAP = os.path.join(OUT_DIR, "pattern_smooth_4region_map.json")
BINARY = os.environ.get(
    "FEM_BINARY_NREGION", os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))

N_REGIONS = 4
CABLE_EA = 157000.0
MATERIAL = {"E1": 10300.0, "E2": 13400.0, "nu": 0.58}    # stitch structure 1, measured

_call = [0]
_prefix = ["pattern_smooth"]
_log = [None]
_best = [None]
_t0 = [None]
_time_limit = [0.0]
_interior = [None]      # non-boundary vertices, for the fold check


class _TimeUp(Exception):
    pass


def run_fem(sw, sc, knit, pressure, cable_paths, cscales, fixed, V_rest, t_crown,
            min_disp, tag=""):
    _call[0] += 1
    n = _call[0]
    params = {"pressure": float(pressure), "motif": 1, "cable_ea": CABLE_EA,
              **MATERIAL,
              "cable_paths": cable_paths, "fixed_vertices": fixed,
              "regions": [{"sf_wale": float(sw[r]), "sf_course": float(sc[r]),
                           "knit_dir_deg": float(knit[r])} for r in range(N_REGIONS)],
              "cable_rest_scales": [float(s) for s in cscales]}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     dir=OUT_DIR) as pf:
        json.dump(params, pf)
        ppath = pf.name
    prefix = os.path.join(OUT_DIR, f"{_prefix[0]}_{n:05d}")
    out, reason = None, "ok"
    try:
        r = subprocess.run([BINARY, MESH_PATH, REGION_MAP, ppath, prefix],
                           capture_output=True, text=True,
                           timeout=float(os.environ.get("FEM_TIMEOUT", "300")))
        status = next((l.split()[1] for l in r.stderr.splitlines()[::-1]
                       if l.startswith("SOLVER_STATUS")), "?")
        if r.returncode != 0:
            reason = f"rc={r.returncode}: {r.stderr[-160:]}"
        elif status != "success":
            reason = f"solver {status}"
        else:
            with open(prefix + "_scalars.csv") as f:
                out = {k: float(v) for k, v in next(csv.DictReader(f)).items()}
            X = np.loadtxt(prefix + "_verts.csv", delimiter=",", skiprows=1)[:, 1:]
            md = float(np.max(np.linalg.norm(X - V_rest, axis=1)))
            crown = float(X[:, 2].max())
            if not np.all(np.isfinite(X)):
                reason = "NaN/Inf"
            elif md < min_disp:
                reason = f"max_disp {md*1e3:.3f} mm < floor — degenerate"
            elif not (0.3 * t_crown < crown < 3.0 * t_crown):
                reason = f"crown {crown:.4f} outside physical range"
            elif X[_interior[0], 2].min() < -0.05 * t_crown:
                # interior only: a free edge sagging below the support plane is
                # a bad fit, which the loss already charges, not a fold
                reason = f"min interior z {X[_interior[0], 2].min():.4f} — folded"
            if reason != "ok":
                out = None
            else:
                out.update(verts=X, max_disp=md, crown=crown)
    except subprocess.TimeoutExpired:
        reason = "timeout"
    finally:
        os.unlink(ppath)
        for suf in ("_verts.csv", "_scalars.csv", "_stress.csv"):
            if tag != "final" and os.path.exists(prefix + suf):
                os.unlink(prefix + suf)       # per-call CSVs are ~100 kB each
    if out is None:
        print(f"  [{n:5d}] FEM INVALID: {reason}", flush=True)
    rec = {"call": n, "tag": tag, "valid": out is not None, "reason": reason,
           "sf_wale": [round(float(v), 6) for v in sw],
           "sf_course": [round(float(v), 6) for v in sc],
           "cable_rest_scales": [round(float(v), 6) for v in cscales],
           "pressure": float(pressure)}
    if out is not None:
        rec.update(crown=out["crown"], max_stress=out.get("max_stress"),
                   max_disp_mm=out["max_disp"] * 1e3)
    return out, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--pressure", type=float, default=1000.0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--min-disp-mm", type=float, default=5.0)
    ap.add_argument("--sweep-sf", type=str, default="0.96,0.98,1.00,1.02,1.04")
    ap.add_argument("--seed-json", type=str, default=None)
    ap.add_argument("--sf0", type=float, default=1.0)
    ap.add_argument("--cable0", type=float, default=0.98)
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--out-prefix", type=str, default=None)
    args = ap.parse_args()

    prefix = args.out_prefix or f"pattern_smooth_p{args.phase}"
    _prefix[0] = prefix
    _log[0] = os.path.join(OUT_DIR, f"{prefix}_calls.jsonl")
    _time_limit[0] = args.time_limit
    for p in (BINARY, MESH_PATH, REGION_MAP, CABLE_FILE):
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    V, F = o2.load_off(MESH_PATH)
    V_target = V.copy()
    fixed = sorted(json.load(open(CABLE_META))["supports"])
    free_idx = np.array(sorted(set(range(len(V))) - set(fixed)))
    _interior[0] = np.array(sorted(set(range(len(V))) - o2.boundary_vertices(F)))
    t_crown = float(V_target[:, 2].max())
    span = float(np.ptp(V_target[:, 0]))
    rm = json.load(open(REGION_MAP))
    face_region = np.array(rm["face_regions"])
    pf = np.asarray(rm["face_knit_dirs_deg"], float)
    knit = np.array([np.degrees(np.angle(np.exp(2j * np.radians(pf[face_region == r]))
                                         .mean())) / 2 % 180 for r in range(N_REGIONS)])
    cab = json.load(open(CABLE_FILE))
    cable_names = sorted(cab)
    cable_paths = [cab[k] for k in cable_names]

    print(f"Mesh    : {len(V)} verts, {len(F)} faces, {len(fixed)} supports, "
          f"{len(free_idx)} free (fitted)")
    print(f"Target  : crown {t_crown:.4f} m, span {span:.4f} m  (rest == target)")
    print(f"Regions : {N_REGIONS}, faces {np.bincount(face_region).tolist()}; knit per "
          f"face, region means {[round(float(k), 1) for k in knit]} deg — FIXED")
    print(f"Cables  : {cable_names}  EA {CABLE_EA:g}")
    print(f"Material: E1 {MATERIAL['E1']:g}  E2 {MATERIAL['E2']:g}  nu {MATERIAL['nu']}"
          f"   pressure {args.pressure:g} Pa")
    print(f"Log     : {os.path.relpath(_log[0], HERE)}", flush=True)

    def evaluate(sw, sc, cs, tag):
        out, rec = run_fem(sw, sc, knit, args.pressure, cable_paths, cs, fixed,
                           V, t_crown, args.min_disp_mm / 1e3, tag)
        l = 1e3 if out is None else float(np.sqrt(np.mean(np.sum(
            (out["verts"][free_idx] - V_target[free_idx]) ** 2, axis=1))))
        rec["rmse_mm"] = None if out is None else l * 1e3
        with open(_log[0], "a") as f:
            f.write(json.dumps(rec) + "\n")
        return out, l

    def track(l, p):
        if _best[0] is None or l < _best[0][0]:
            _best[0] = (l, np.array(p, float))
        if _time_limit[0] and time.time() - _t0[0] > _time_limit[0]:
            raise _TimeUp()
        return l

    if args.phase == 0:
        for v in [float(s) for s in args.sweep_sf.split(",")]:
            t = time.time()
            out, l = evaluate([v] * 4, [v] * 4, [args.cable0] * 7, f"sweep {v}")
            print(f"  sf={v:.3f}  " + ("INVALID" if out is None else
                  f"RMSE {l*1e3:8.3f} mm  crown {out['crown']:.4f}  "
                  f"({time.time()-t:.1f} s)"), flush=True)
        return

    seed = json.load(open(args.seed_json)) if args.seed_json else None
    if args.phase == 1:
        p0 = np.array([args.sf0, args.sf0, args.cable0])
        bnds = [(0.80, 1.50), (0.80, 1.50), (0.80, 1.05)]

        def expand(p):
            return [p[0]] * 4, [p[1]] * 4, [p[2]] * 7
    else:
        if seed:
            p0 = np.array(list(seed["sf_wale"]) + list(seed["sf_course"]) +
                          list(seed["cable_rest_scales"]))
        else:
            p0 = np.array([args.sf0] * 8 + [args.cable0] * 7)
        bnds = [(0.80, 1.50)] * 8 + [(0.80, 1.05)] * 7

        def expand(p):
            return p[:4], p[4:8], p[8:15]

    def obj(p):
        sw, sc, cs = expand(p)
        _, l = evaluate(sw, sc, cs, f"p{args.phase}")
        if _best[0] is None or l < _best[0][0] or _call[0] % 10 == 0:
            print(f"  [{_call[0]:5d}] RMSE {l*1e3:9.4f} mm  "
                  f"p=[{','.join(f'{v:.4f}' for v in p)}]", flush=True)
        return track(l, p)

    _t0[0] = time.time()
    msg = ""
    try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": args.maxiter, "ftol": 1e-10,
                                "gtol": 1e-6, "eps": 0.002})
        msg = str(res.message)
    except _TimeUp:
        msg = f"stopped at time limit ({_time_limit[0]:.0f} s)"
    best_l, best_p = _best[0]
    elapsed = time.time() - _t0[0]
    print(f"\n{msg}\nBest RMSE {best_l*1e3:.4f} mm after {_call[0]} FEM calls, "
          f"{elapsed/60:.1f} min")

    sw, sc, cs = expand(best_p)
    out, l = evaluate(sw, sc, cs, "final")
    if out is None:
        print("WARNING: the best point does not re-evaluate as valid")
        return
    dev = np.linalg.norm(out["verts"] - V_target, axis=1)
    print(f"  re-evaluated RMSE {l*1e3:.4f} mm ({100*l/span:.2f} % of span)  "
          f"max dev {dev[free_idx].max()*1e3:.2f} mm  crown {out['crown']:.4f} "
          f"(target {t_crown:.4f})  max disp {out['max_disp']*1e3:.2f} mm")
    d = {"geometry": "pattern_smooth", "phase": args.phase, "message": msg,
         "n_calls": _call[0], "elapsed_s": elapsed,
         "mesh": os.path.relpath(MESH_PATH, HERE),
         "region_map": os.path.relpath(REGION_MAP, HERE),
         "cable_file": os.path.relpath(CABLE_FILE, HERE),
         "cable_names": cable_names, "fixed_vertices": fixed,
         "pressure": args.pressure, "material": MATERIAL, "cable_ea": CABLE_EA,
         "rmse_mm": l * 1e3, "max_dev_mm": float(dev[free_idx].max()) * 1e3,
         "crown_m": out["crown"], "target_crown_m": t_crown,
         "max_disp_mm": out["max_disp"] * 1e3, "max_stress": out.get("max_stress"),
         "knit_dir_deg_region_means": knit.tolist(), "knit_is_fixed": True,
         "sf_wale": [float(v) for v in sw], "sf_course": [float(v) for v in sc],
         "cable_rest_scales": [float(v) for v in cs],
         "call_log": os.path.relpath(_log[0], HERE)}
    rj = os.path.join(OUT_DIR, f"{prefix}_result.json")
    json.dump(d, open(rj, "w"), indent=1)
    o2.write_obj(os.path.join(OUT_DIR, f"{prefix}_best_fit.obj"), out["verts"], F,
                 f"pattern_smooth best-fit FEM (phase {args.phase}), RMSE "
                 f"{l*1e3:.3f} mm\nparams: {rj}")
    np.savetxt(os.path.join(OUT_DIR, f"{prefix}_deviation_mm.csv"), dev * 1e3,
               delimiter=",", header="deviation_mm", comments="")
    print(f"Saved {os.path.relpath(rj, HERE)}")


if __name__ == "__main__":
    main()
