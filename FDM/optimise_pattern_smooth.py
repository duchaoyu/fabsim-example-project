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

  * Material.  Motif 1 as in the binary's table, E1 5000, E2 12507 N/m,
    nu 0.198 (the estimates, not the measured stitch structure 1), passed
    explicitly as E1/E2/nu so the choice is visible in every params file.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

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



def set_variant(name):
    """'' = the FDM-extracted cables on the original mesh; 'rm' = the three
    hand-drawn cables on the sketch remesh (remesh_pattern_smooth_sketch.py)."""
    global MESH_PATH, TARGET_OFF, CABLE_FILE, CABLE_META, REGION_MAP
    if name == "rm":
        RM = os.path.join(DATA, "remesh")
        MESH_PATH = TARGET_OFF = os.path.join(RM, "pattern_smooth_rm_tri_m.off")
        CABLE_FILE = os.path.join(RM, "cable_paths_pattern_smooth_rm.json")
        CABLE_META = os.path.join(RM, "cable_paths_pattern_smooth_rm.meta.json")
        REGION_MAP = os.path.join(OUT_DIR, "pattern_smooth_rm_4region_map.json")
    elif name == "rm2":
        RM = os.path.join(DATA, "remesh2")
        MESH_PATH = TARGET_OFF = os.path.join(RM, "pattern_smooth_rm2_tri_m.off")
        CABLE_FILE = os.path.join(RM, "cable_paths_pattern_smooth_rm2.json")
        CABLE_META = os.path.join(RM, "cable_paths_pattern_smooth_rm2.meta.json")
        REGION_MAP = os.path.join(OUT_DIR, "pattern_smooth_rm2_4region_map.json")
    elif name:
        raise ValueError(f"unknown variant {name!r}")


N_REGIONS = 4
CABLE_EA = 157000.0
# motif 1 as the binary's table has it (the pre-2026-09-19 estimates), at the
# user's request; the measured stitch structure 1 is E1 10300, E2 13400, nu 0.58
MATERIAL = {"E1": 5000.0, "E2": 12507.0, "nu": 0.198}
# --material sI: the measured stitch structure I, E1 = wale (along the knit field)
MATERIALS = {"motif1": dict(MATERIAL),
             "sI": {"E1": 12500.0, "E2": 5000.0, "nu": 0.198}}
_splines = [None]      # --spline-edges: {"spline_paths", "spline_EA", "spline_EI", "spline_rest"}
_follower = [False]    # --follower: follower pressure (p n dA on the current surface)

_call = [0]
_prefix = ["pattern_smooth"]
_log = [None]
_best = [None]
_t0 = [None]
_time_limit = [0.0]
_interior = [None]      # non-boundary vertices, for the fold check
_lock = threading.Lock()   # phase 3 runs FEM calls in parallel


class _TimeUp(Exception):
    pass


_reg_max = [0.0]       # --newton-reg-max, passed to the binary when > 0


def run_fem(sw, sc, knit, pressure, cable_paths, cscales, fixed, V_rest, t_crown,
            min_disp, tag=""):
    with _lock:
        _call[0] += 1
        n = _call[0]
    params = {"pressure": float(pressure), "motif": 1, "cable_ea": CABLE_EA,
              **MATERIAL,
              "cable_paths": cable_paths, "fixed_vertices": fixed,
              "regions": [{"sf_wale": float(sw[r]), "sf_course": float(sc[r]),
                           "knit_dir_deg": float(knit[r])} for r in range(N_REGIONS)],
              "cable_rest_scales": [float(s) for s in cscales]}
    if _reg_max[0] > 0:
        params["newton_reg_max"] = _reg_max[0]
    if _splines[0]:
        params.update(_splines[0])
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                     dir=OUT_DIR) as pf:
        json.dump(params, pf)
        ppath = pf.name
    prefix = os.path.join(OUT_DIR, f"{_prefix[0]}_{n:05d}")
    out, reason = None, "ok"
    try:
        env = dict(os.environ)
        if _follower[0]:
            env.update(FEM_PRESSURE="follower", FEM_FOLLOWER_START="volume")
        r = subprocess.run([BINARY, MESH_PATH, REGION_MAP, ppath, prefix],
                           capture_output=True, text=True, env=env,
                           timeout=float(os.environ.get("FEM_TIMEOUT", "300")))
        status = next((l.split()[1] for l in r.stderr.splitlines()[::-1]
                       if l.startswith("SOLVER_STATUS")), "?")
        if r.returncode != 0:
            reason = f"rc={r.returncode}: {r.stderr[-160:]}"
        elif status not in ("success", "follower_converged"):
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


def run_cma(args, p0, bnds, expand, evaluate, track, penalty=lambda sw, sc: 0.0):
    """Phase 3: CMA-ES over the 15 parameters, one generation = --popsize FEM
    calls run --workers at a time.  L-BFGS-B with finite differences does not
    survive this problem: about a quarter of the calls near the start fail in
    the Newton solve, and a 1e3 penalty inside a finite difference is a garbage
    gradient.  CMA-ES only ranks the population, so a failed call is simply
    the worst sample.  Variables are scaled so sigma = 1 is --sigma-sf on a
    stretch factor and --sigma-cable on a cable scale."""
    import cma
    n_sf = 2 * N_REGIONS
    scale = np.array([args.sigma_sf] * n_sf + [args.sigma_cable] * (len(p0) - n_sf))
    lo = (np.array([b[0] for b in bnds]) - p0) / scale
    hi = (np.array([b[1] for b in bnds]) - p0) / scale
    es = cma.CMAEvolutionStrategy(np.zeros(len(p0)), 1.0,
                                  {"bounds": [lo.tolist(), hi.tolist()],
                                   "popsize": args.popsize, "seed": 1,
                                   "maxiter": args.maxiter, "verbose": -9,
                                   "tolfun": 1e-7, "tolx": 1e-4})

    def one(z):
        p = p0 + scale * np.asarray(z)
        sw, sc, cs = expand(p)
        l = evaluate(sw, sc, cs, "p3")[1]
        # the score CMA ranks: RMSE plus the Laplacian penalty (invalid stays 1e3)
        return (l if l >= 1e3 else l + penalty(sw, sc)), p

    gen = 0
    with ThreadPoolExecutor(args.workers) as ex:
        try:
            while not es.stop():
                Z = es.ask()
                res = list(ex.map(one, Z))
                es.tell(Z, [r[0] for r in res])
                gen += 1
                for l, p in res:
                    track(l, p)
                n_bad = sum(r[0] >= 1e3 for r in res)
                print(f"  gen {gen:3d}  calls {_call[0]:5d}  best {_best[0][0]*1e3:8.3f} mm  "
                      f"gen best {min(r[0] for r in res)*1e3:8.3f}  "
                      f"invalid {n_bad}/{len(res)}  sigma {es.sigma:.3f}", flush=True)
        except _TimeUp:
            return f"stopped at time limit ({_time_limit[0]:.0f} s)"
    return f"CMA-ES stop: {dict(es.stop())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[0, 1, 2, 3])
    ap.add_argument("--pressure", type=float, default=1000.0)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--min-disp-mm", type=float, default=5.0)
    ap.add_argument("--sweep-sf", type=str, default="0.96,0.98,1.00,1.02,1.04")
    ap.add_argument("--seed-json", type=str, default=None)
    ap.add_argument("--sf0", type=float, default=1.0)
    ap.add_argument("--cable0", type=float, default=0.98)
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--newton-reg-max", type=float, default=0.0,
                    help="cap on the Newton diagonal regularisation (binary default "
                         "1e4). A compressed StVK region can need more; 1e6 rescued "
                         "10 of 12 regularization_failed calls on the sketch remesh")
    ap.add_argument("--region-map", type=str, default=None,
                    help="region map JSON (face_regions + face_knit_dirs_deg); the "
                         "number of regions is read from it")
    ap.add_argument("--cable-file", type=str, default=None,
                    help="cable paths JSON to use instead of the variant's (e.g. the "
                         "boundary cables only)")
    ap.add_argument("--lambda-smooth", type=float, default=0.0,
                    help="Laplacian penalty on the stretch factors, as in "
                         "optimise_D5_laplacian.py: score = RMSE [m] + lambda * sum over "
                         "adjacent regions of (dsf_wale^2 + dsf_course^2)")
    ap.add_argument("--variant", default="", choices=["", "rm", "rm2"],
                    help="rm / rm2: the first / second hand-drawn cable layout, remeshed")
    ap.add_argument("--fix-edges", type=str, default="",
                    help="comma-separated edge cables (e.g. E00) whose vertices are "
                         "fixed as supports; the cable itself is then dropped")
    ap.add_argument("--workers", type=int, default=16, help="phase 3: parallel FEM calls")
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sigma-sf", type=float, default=0.01,
                    help="phase 3: initial step on the stretch factors")
    ap.add_argument("--sigma-cable", type=float, default=0.005,
                    help="phase 3: initial step on the cable rest scales")
    ap.add_argument("--cable0-edge", type=float, default=None,
                    help="phase 2/3 start: rest scale of the edge cables E*")
    ap.add_argument("--material", choices=sorted(MATERIALS), default="motif1",
                    help="motif1: the pre-2026-09-19 estimates (E1 5000, E2 12507); "
                         "sI: measured stitch structure I (E_wale 12500, E_course 5000)")
    ap.add_argument("--spline-edges", type=float, default=0.0,
                    help="replace the free-edge cables by bending-stiff GFRP rods of "
                         "this diameter (mm, E 40 GPa), formed to the target edge; "
                         "chains of edge cables that share end vertices become one rod")
    ap.add_argument("--sf-lo", type=float, default=0.80,
                    help="lower bound on the stretch factors.  Below 1 the fabric is "
                         "knitted larger than the surface and goes into compression; "
                         "with the follower load that leaves no stable state (1.001 "
                         "made 10/10 random designs converge, 0.80 about 1 in 8)")
    ap.add_argument("--follower", action="store_true",
                    help="follower pressure (correct at free edges) instead of the "
                         "volume-work load")
    args = ap.parse_args()
    set_variant(args.variant)
    MATERIAL.clear(); MATERIAL.update(MATERIALS[args.material])
    _follower[0] = args.follower
    global REGION_MAP, N_REGIONS
    if args.region_map:
        REGION_MAP = args.region_map if os.path.isabs(args.region_map) else \
            os.path.join(HERE, args.region_map) if not os.path.exists(args.region_map) \
            else os.path.abspath(args.region_map)
    N_REGIONS = int(max(json.load(open(REGION_MAP))["face_regions"])) + 1
    global CABLE_FILE
    if args.cable_file:
        CABLE_FILE = args.cable_file if os.path.exists(args.cable_file) else \
            os.path.join(HERE, args.cable_file)
    _reg_max[0] = args.newton_reg_max

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
    # region adjacency: two regions are neighbours when a mesh edge separates them
    e2f = {}
    for fi, t in enumerate(F):
        for k in range(3):
            e2f.setdefault(tuple(sorted((int(t[k]), int(t[(k + 1) % 3])))), []).append(fi)
    adj_pairs = sorted({tuple(sorted((int(face_region[a]), int(face_region[b]))))
                        for fl in e2f.values() if len(fl) == 2
                        for a, b in [fl] if face_region[a] != face_region[b]})
    lam = args.lambda_smooth

    def lap(sw, sc):
        return float(sum((sw[i] - sw[j]) ** 2 + (sc[i] - sc[j]) ** 2 for i, j in adj_pairs))
    knit = np.array([np.degrees(np.angle(np.exp(2j * np.radians(pf[face_region == r]))
                                         .mean())) / 2 % 180 for r in range(N_REGIONS)])
    cab = json.load(open(CABLE_FILE))
    # --fix-edges: those free edges become supports; their cable then lies on
    # fixed vertices, does nothing, and is dropped along with its parameter
    fix_edges = [e for e in args.fix_edges.split(",") if e]
    for e in fix_edges:
        if e not in cab or not e.startswith("E"):
            sys.exit(f"--fix-edges: {e} is not an edge cable of {sorted(cab)}")
        fixed = sorted(set(fixed) | set(cab[e]))
    free_idx = np.array(sorted(set(range(len(V))) - set(fixed)))
    cable_names = [k for k in sorted(cab) if k not in fix_edges]
    spline_names = []
    if args.spline_edges > 0:
        # join edge cables end to end (E03a-E03b-E03c) into continuous rods; at a
        # support each edge keeps its own rod
        fixed_set = set(fixed)
        chains = [list(cab[k]) for k in cable_names if k.startswith("E")]
        spline_names = [k for k in cable_names if k.startswith("E")]
        merged = True
        while merged:
            merged = False
            for i in range(len(chains)):
                for j in range(len(chains)):
                    if i != j and chains[i][-1] == chains[j][0] and \
                            chains[j][0] not in fixed_set:   # never through a support
                        chains[i] += chains[j][1:]; del chains[j]; merged = True; break
                if merged: break
        d = args.spline_edges / 1000.0
        _splines[0] = {"spline_paths": chains, "spline_EA": 40e9 * np.pi * d**2 / 4,
                       "spline_EI": 40e9 * np.pi * d**4 / 64, "spline_rest": 1}
        cable_names = [k for k in cable_names if not k.startswith("E")]
        print(f"Splines : {spline_names} -> {len(chains)} rods "
              f"{[len(c) for c in chains]} verts, GFRP d {args.spline_edges:g} mm, "
              f"EA {_splines[0]['spline_EA']:.3g} N, EI {_splines[0]['spline_EI']:.3g} N m^2")
    cable_paths = [cab[k] for k in cable_names]
    n_c = len(cable_names)
    if fix_edges:
        print(f"Fixed edges: {fix_edges} -> {len(fixed)} fixed vertices")

    print(f"Mesh    : {len(V)} verts, {len(F)} faces, {len(fixed)} supports, "
          f"{len(free_idx)} free (fitted)")
    print(f"Target  : crown {t_crown:.4f} m, span {span:.4f} m  (rest == target)")
    print(f"Regions : {N_REGIONS}, faces {np.bincount(face_region).tolist()}; knit per "
          f"face, region means {[round(float(k), 1) for k in knit]} deg — FIXED")
    print(f"Adjacent: {adj_pairs}   lambda_smooth {lam:g}")
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
        with _lock, open(_log[0], "a") as f:
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
            out, l = evaluate([v] * N_REGIONS, [v] * N_REGIONS, [args.cable0] * n_c,
                              f"sweep {v}")
            print(f"  sf={v:.3f}  " + ("INVALID" if out is None else
                  f"RMSE {l*1e3:8.3f} mm  crown {out['crown']:.4f}  "
                  f"({time.time()-t:.1f} s)"), flush=True)
        return

    seed = json.load(open(args.seed_json)) if args.seed_json else None
    if args.phase == 1:
        p0 = np.array([args.sf0, args.sf0, args.cable0])
        bnds = [(0.80, 1.50), (0.80, 1.50), (0.80, 1.05)]

        def expand(p):
            return [p[0]] * N_REGIONS, [p[1]] * N_REGIONS, [p[2]] * n_c
    else:
        if seed:
            sn = seed.get("cable_names", sorted(cab))
            cs0 = dict(zip(sn, seed["cable_rest_scales"]))
            sw0, sc0 = np.asarray(seed["sf_wale"]), np.asarray(seed["sf_course"])
            if len(sw0) != N_REGIONS:
                # the seed is on a different region layout: each new region
                # starts at the face-weighted mean of the seed over its faces
                src = seed.get("region_map")
                if not src:
                    sys.exit("seed has a different region count and no region_map")
                fr0 = np.array(json.load(open(os.path.join(HERE, src)))["face_regions"])
                sw0 = np.array([sw0[fr0[face_region == r]].mean() for r in range(N_REGIONS)])
                sc0 = np.array([sc0[fr0[face_region == r]].mean() for r in range(N_REGIONS)])
                print(f"Seed    : remapped from {len(seed['sf_wale'])} regions ({src})")
            p0 = np.array(list(sw0) + list(sc0) +
                          [cs0.get(k, args.cable0) for k in cable_names])
        else:
            ce = args.cable0 if args.cable0_edge is None else args.cable0_edge
            p0 = np.array([args.sf0] * (2 * N_REGIONS) +
                          [ce if k.startswith("E") else args.cable0 for k in cable_names])
        bnds = [(args.sf_lo, 1.50)] * (2 * N_REGIONS) + [(0.80, 1.05)] * n_c
        p0 = np.clip(p0, [b[0] for b in bnds], [b[1] for b in bnds])

        def expand(p):
            N = N_REGIONS
            return p[:N], p[N:2 * N], p[2 * N:2 * N + n_c]

    def obj(p):
        sw, sc, cs = expand(p)
        _, l = evaluate(sw, sc, cs, f"p{args.phase}")
        if l < 1e3:
            l = l + lam * lap(sw, sc)
        if _best[0] is None or l < _best[0][0] or _call[0] % 10 == 0:
            print(f"  [{_call[0]:5d}] RMSE {l*1e3:9.4f} mm  "
                  f"p=[{','.join(f'{v:.4f}' for v in p)}]", flush=True)
        return track(l, p)

    _t0[0] = time.time()
    msg = ""
    if args.phase == 3:
        msg = run_cma(args, p0, bnds, expand, evaluate, track,
                      penalty=lambda sw, sc: lam * lap(sw, sc))
    else:
      try:
        res = minimize(obj, p0, method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": args.maxiter, "ftol": 1e-10,
                                "gtol": 1e-6, "eps": 0.002})
        msg = str(res.message)
      except _TimeUp:
        msg = f"stopped at time limit ({_time_limit[0]:.0f} s)"
    best_l, best_p = _best[0]
    elapsed = time.time() - _t0[0]
    print(f"\n{msg}\nBest score {best_l*1e3:.4f} mm-equivalent (RMSE + penalty) after {_call[0]} FEM calls, "
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
    d = {"geometry": "pattern_smooth", "variant": args.variant,
         "phase": args.phase, "message": msg,
         "n_calls": _call[0], "elapsed_s": elapsed,
         "mesh": os.path.relpath(MESH_PATH, HERE),
         "region_map": os.path.relpath(REGION_MAP, HERE),
         "cable_file": os.path.relpath(CABLE_FILE, HERE),
         "cable_names": cable_names, "fixed_vertices": fixed,
         "fixed_edges": fix_edges,
         "pressure": args.pressure, "material": MATERIAL, "cable_ea": CABLE_EA,
         "newton_reg_max": args.newton_reg_max,
         "follower": args.follower, "sf_lo": args.sf_lo, "spline_edges_mm": args.spline_edges,
         "spline_names": spline_names, "splines": _splines[0],
         "n_regions": N_REGIONS, "lambda_smooth": lam,
         "laplacian": lap(sw, sc), "adjacent_regions": adj_pairs,
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
