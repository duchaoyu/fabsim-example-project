"""
2-part (middle-crease) knit + cable FEM inverse optimisation.

Adapted from optimise_C5_16region.py.  The differences that matter:

  * Geometry.  2parts_smooth_tri_m.off, 581 v / 1100 f / 60 boundary, span
    1.20 m, crown 0.2568 m.  Its ONLY symmetry is the mirror plane x = 0
    (tested in field_regions_2part.py: reflection about x = 0 reproduces the
    mesh to 0.000 mm, every other candidate is 78-173 mm off).  So the
    symmetric mode ties mirror pairs and nothing else — there is no rotational
    reduction to exploit and no y symmetry.

  * Regions.  12, built by field_regions_2part.py: 3 bands in |x| (crease,
    lobe, skirt) x sign(x) x sign(y).  Mirror-x pairs them into 6 independent
    parameter sets.

  * Knit direction is NOT optimised.  It is frozen from the directional field
    and delivered PER FACE through "face_knit_dirs_deg" in the region map,
    which fem_batch_nregion prefers over the region-level knit_dir_deg whenever
    its length equals nF.  The region-level values are still written into the
    params (they are what the binary would fall back on) but they do not drive
    the solve.  Per-face matters here: the southern fan makes the within-region
    circular s.d. of the knit angle 44-76 deg in the four southern regions, so
    a single region mean there would be meaningless.

  * Cables.  24 polylines from extract_cables_2part.py.

  * rest == target.  The same OFF is both.  There is no flat rest-shape mesh in
    this pipeline, which is exactly why --min-disp-mm exists: "barely deform"
    scores a near-perfect RMSE, so the search is attracted to a degenerate
    no-deformation optimum.  The floor is kept at 5 mm by default and every
    reported best is checked against it.

  * EVERY CALL IS LOGGED.  optimise_C5_16region.run_fem() deletes its params
    temp file, so per-call parameters were not recoverable afterwards.  Here
    each call appends one line to optimisation/<prefix>_calls.jsonl with the
    full sf / cable-scale vector, the RMSE, the crown and the max displacement.

Usage:
    python3 FDM/optimise_2part.py --phase 0 --sweep-sf 1.00,1.02,1.04,1.06
    python3 FDM/optimise_2part.py --phase 1          # 3 params, locate
    python3 FDM/optimise_2part.py --phase 2          # 15 params, mirror-symmetric
    python3 FDM/optimise_2part.py --phase 3          # 48 params, free
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

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "optimisation")

MESH_PATH = os.environ.get(
    "FEM_MESH", os.path.join(HERE, "data", "2part", "2parts_smooth_tri_m.off"))
TARGET_OFF = os.environ.get("FEM_TARGET", MESH_PATH)   # rest == target
CABLE_FILE = os.environ.get(
    "FEM_CABLE_PATHS", os.path.join(HERE, "data", "2part", "cable_paths_2part.json"))
REGION_MAP = os.environ.get(
    "FEM_REGION_MAP", os.path.join(OUT_DIR, "2part_12region_map.json"))
BINARY = os.environ.get(
    "FEM_BINARY_NREGION", os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))

N_REGIONS = 12
CABLE_EA = 157000.0

# mirror-x pairs of regions, from field_regions_2part.py's layout
#   region = band*4 + side*2 + yhalf,  side 0 = x<0, 1 = x>=0
MIRROR_PAIRS = [(b * 4 + 0 * 2 + y, b * 4 + 1 * 2 + y)
                for b in range(3) for y in range(2)]     # 6 pairs
REGION_NAMES = [f"{'crease' if b == 0 else 'lobe' if b == 1 else 'skirt'}"
                f"_{'E' if s else 'W'}{'N' if y else 'S'}"
                for b in range(3) for s in range(2) for y in range(2)]


# ── I/O ───────────────────────────────────────────────────────────────────────
def load_off(path):
    L = [l for l in open(path).read().split("\n") if l.strip()]
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def boundary_vertices(F):
    """Topological boundary: an edge belonging to exactly one face."""
    from collections import Counter
    c = Counter()
    for f in F:
        for a, b in zip(f, np.roll(f, -1)):
            c[frozenset((int(a), int(b)))] += 1
    return {v for e, n in c.items() if n == 1 for v in e}


def write_obj(path, V, F, header):
    with open(path, "w") as f:
        for line in header.split("\n"):
            f.write(f"# {line}\n")
        for v in V:
            f.write("v %.8f %.8f %.8f\n" % tuple(v))
        for tri in F:
            f.write("f %d %d %d\n" % (tri[0] + 1, tri[1] + 1, tri[2] + 1))


# ── FEM driver ────────────────────────────────────────────────────────────────
_call_count = [0]
_out_prefix = ["2part"]
_target_crown = [None]
_min_disp = [0.0]
_log_path = [None]
_t_start = [None]
_time_limit = [0.0]
_best = [None]          # (loss, params)
_best_meta = [None]     # dict describing the best call


class _TimeUp(Exception):
    pass


def _track(loss, params, meta=None):
    if _best[0] is None or loss < _best[0][0]:
        _best[0] = (float(loss), np.asarray(params, float).copy())
        _best_meta[0] = meta
    if _time_limit[0] and _t_start[0] is not None:
        if time.time() - _t_start[0] > _time_limit[0]:
            raise _TimeUp()
    return loss


class _Result:
    def __init__(self, x, fun, nit, msg):
        self.x, self.fun, self.nit = np.asarray(x), float(fun), nit
        self.success, self.message = False, msg


def _run_minimize(fn, x0, **kw):
    _t_start[0] = time.time()
    _best[0] = None
    try:
        res = minimize(fn, x0, **kw)
        if _best[0] is not None and _best[0][0] < res.fun - 1e-15:
            print(f"  note: scipy reported fun={res.fun:.7f} at its final x, but "
                  f"the best evaluation seen was {_best[0][0]:.7f}; reporting the best.")
            return _Result(_best[0][1], _best[0][0], getattr(res, "nit", -1),
                           "best evaluation (scipy res.fun disagreed)")
        return res
    except _TimeUp:
        el = time.time() - _t_start[0]
        print(f"\n  TIME LIMIT reached after {el:.1f} s — stopping with the best "
              f"point seen (loss {_best[0][0]:.7f}).")
        return _Result(_best[0][1], _best[0][0], -1,
                       f"stopped at time limit ({_time_limit[0]:.0f} s)")


def _check_fem_valid(verts, crown, V_rest):
    """See optimise_C5_16region._check_fem_valid.  The min-disp floor is the
    important one: rest == target, so a solve that barely moves scores a
    near-perfect RMSE and is not an equilibrium of a pressurised membrane."""
    t_crown = _target_crown[0]
    if not np.all(np.isfinite(verts)):
        return False, "NaN/Inf in vertices", None
    max_disp = None
    if V_rest is not None:
        max_disp = float(np.max(np.linalg.norm(verts - V_rest, axis=1)))
        if max_disp < 1e-8:
            return False, f"max_disp={max_disp:.2e} — rest shape returned", max_disp
        if _min_disp[0] > 0.0 and max_disp < _min_disp[0]:
            return False, (f"max_disp={max_disp*1000:.4f} mm < floor "
                           f"{_min_disp[0]*1000:.4f} mm — degenerate "
                           f"no-deformation fit"), max_disp
    if t_crown is not None and (crown < 0.3 * t_crown or crown > 3.0 * t_crown):
        return False, f"crown={crown:.4f} outside physical range", max_disp
    if t_crown is not None and float(verts[:, 2].min()) < -0.01 * t_crown:
        return False, f"min z={verts[:, 2].min():.4f} — mesh folded", max_disp
    return True, "OK", max_disp


def run_fem(sf_wale, sf_course, knit_dirs, pressure, motif,
            cable_paths, cable_rest_scales, V_rest, tag=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1
    n = _call_count[0]

    params = {
        "pressure": float(pressure),
        "motif": int(motif),
        "cable_ea": CABLE_EA,
        "cable_paths": cable_paths,
        "regions": [{"sf_wale": float(sf_wale[r]),
                     "sf_course": float(sf_course[r]),
                     "knit_dir_deg": float(knit_dirs[r])} for r in range(N_REGIONS)],
        "cable_rest_scales": [float(s) for s in cable_rest_scales],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"{_out_prefix[0]}_{n:05d}")
    out, reason = None, "ok"
    try:
        res = subprocess.run([BINARY, MESH_PATH, REGION_MAP, params_path, prefix],
                             capture_output=True, text=True,
                             timeout=float(os.environ.get("FEM_TIMEOUT", "120")))
        if res.returncode != 0:
            reason = f"rc={res.returncode}: {res.stderr[:160]}"
        else:
            sp, vp = prefix + "_scalars.csv", prefix + "_verts.csv"
            if not os.path.exists(sp):
                reason = "no scalars output"
            else:
                with open(sp) as f:
                    row = next(csv.DictReader(f))
                out = {k: float(v) for k, v in row.items()}
                if os.path.exists(vp):
                    out["verts"] = np.loadtxt(vp, delimiter=",", skiprows=1)[:, 1:]
                    ok, reason, md = _check_fem_valid(
                        out["verts"], out.get("crown_height", 0.0), V_rest)
                    out["max_disp"] = md
                    if not ok:
                        print(f"  [{n:5d}] FEM INVALID: {reason}")
                        out = None
    except subprocess.TimeoutExpired:
        reason = "timeout"
        print(f"  [{n:5d}] FEM TIMEOUT")
    except Exception as e:
        reason = f"exception {e}"
        print(f"  [{n:5d}] FEM exception: {e}")
    finally:
        try:
            os.unlink(params_path)
        except OSError:
            pass

    # ── the log the C5 script did not keep ────────────────────────────────────
    if _log_path[0]:
        rec = {"call": n, "tag": tag, "valid": out is not None, "reason": reason,
               "sf_wale": [round(float(v), 6) for v in sf_wale],
               "sf_course": [round(float(v), 6) for v in sf_course],
               "cable_rest_scales": [round(float(v), 6) for v in cable_rest_scales],
               "knit_dir_deg": [round(float(v), 4) for v in knit_dirs],
               "pressure": float(pressure), "motif": int(motif),
               "n_cables": len(cable_paths)}
        if out is not None:
            rec.update(crown=out.get("crown_height"),
                       max_stress=out.get("max_stress"),
                       max_disp_mm=(out["max_disp"] * 1000 if out["max_disp"] else None))
        with open(_log_path[0], "a") as f:
            f.write(json.dumps(rec) + "\n")
    return out


# ── objective ─────────────────────────────────────────────────────────────────
def make_loss(V_target, interior_idx):
    def loss_of(verts):
        d = verts[interior_idx] - V_target[interior_idx]
        return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))
    return loss_of


def append_rmse(rmse, call):
    """Record the RMSE for the call that has just been logged."""
    if not _log_path[0]:
        return
    lines = open(_log_path[0]).read().rstrip("\n").split("\n")
    rec = json.loads(lines[-1])
    if rec.get("call") == call:
        rec["rmse_mm"] = None if rmse is None else float(rmse) * 1000.0
        lines[-1] = json.dumps(rec)
        with open(_log_path[0], "w") as f:
            f.write("\n".join(lines) + "\n")


# ── parameter expansions ──────────────────────────────────────────────────────
def expand_uniform(p):
    """p = [sf_wale, sf_course, cable_scale]"""
    w, c, s = p
    return np.full(N_REGIONS, w), np.full(N_REGIONS, c), s


def expand_mirror(p, n_cable_groups):
    """p = 6 sf_wale + 6 sf_course + n_cable_groups scales, mirror-x tied."""
    sw = np.empty(N_REGIONS)
    sc = np.empty(N_REGIONS)
    for k, (a, b) in enumerate(MIRROR_PAIRS):
        sw[a] = sw[b] = p[k]
        sc[a] = sc[b] = p[6 + k]
    return sw, sc, np.asarray(p[12:12 + n_cable_groups], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=2, choices=[0, 1, 2, 3])
    ap.add_argument("--pressure", type=float, default=1000.0)
    ap.add_argument("--motif", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--method", type=str, default="L-BFGS-B",
                    choices=["L-BFGS-B", "Powell", "Nelder-Mead"])
    ap.add_argument("--min-disp-mm", type=float, default=5.0,
                    help="absolute displacement floor in mm (default 5.0). "
                         "A result whose max displacement from rest falls below "
                         "it is rejected as a degenerate no-deformation fit. "
                         "0 disables it — do not, rest == target here.")
    ap.add_argument("--sweep-sf", type=str, default=None,
                    help="phase 0: comma-separated uniform sf values to evaluate")
    ap.add_argument("--no-cables", action="store_true",
                    help="run without any cables (membrane only), for reference")
    ap.add_argument("--time-limit", type=float, default=0.0)
    ap.add_argument("--out-prefix", type=str, default=None)
    ap.add_argument("--seed-json", type=str, default=None,
                    help="result JSON of an earlier phase to start from")
    ap.add_argument("--sf0-wale", type=float, default=1.03)
    ap.add_argument("--sf0-course", type=float, default=1.03)
    ap.add_argument("--cable0", type=float, default=0.98)
    args = ap.parse_args()

    prefix = args.out_prefix or f"2part_p{args.phase}"
    _out_prefix[0] = prefix
    os.makedirs(OUT_DIR, exist_ok=True)
    _log_path[0] = os.path.join(OUT_DIR, f"{prefix}_calls.jsonl")
    open(_log_path[0], "a").close()
    _time_limit[0] = float(args.time_limit)

    for p, lbl in [(BINARY, "FEM binary"), (MESH_PATH, "mesh"),
                   (REGION_MAP, "region map"), (CABLE_FILE, "cable paths")]:
        if not os.path.exists(p):
            print(f"{lbl} not found: {p}")
            sys.exit(1)

    V, F = load_off(MESH_PATH)
    V_rest = V.copy()
    V_target, _ = load_off(TARGET_OFF)
    bdry = boundary_vertices(F)
    interior_idx = np.array(sorted(set(range(len(V))) - bdry))
    t_crown = float(V_target[:, 2].max())
    _target_crown[0] = t_crown
    _min_disp[0] = args.min_disp_mm / 1000.0
    span = float(np.ptp(V_target[:, 0]))

    rm = json.load(open(REGION_MAP))
    face_region = np.array(rm["face_regions"])
    per_face_knit = rm.get("face_knit_dirs_deg")
    # region-level fall-back values: circular mean (mod 180) of the per-face field
    knit_dirs = np.zeros(N_REGIONS)
    if per_face_knit:
        pf = np.asarray(per_face_knit, float)
        for r in range(N_REGIONS):
            z = np.exp(1j * np.radians(pf[face_region == r]) * 2).mean()
            knit_dirs[r] = np.degrees(np.angle(z)) / 2 % 180.0

    cable_dict = json.load(open(CABLE_FILE))
    cable_names = sorted(cable_dict)
    cable_paths = [] if args.no_cables else [cable_dict[k] for k in cable_names]

    print(f"Mesh    : {len(V)} verts, {len(F)} faces, {len(bdry)} boundary, "
          f"{len(interior_idx)} interior")
    print(f"Target  : crown {t_crown:.4f} m, span {span:.4f} m  (rest == target)")
    print(f"Regions : {N_REGIONS}, counts "
          f"{np.bincount(face_region, minlength=N_REGIONS).tolist()}")
    print(f"Knit    : {'PER FACE from the directional field (region values are the '
                       'fall-back only)' if per_face_knit else 'per region'}; "
          f"region means {[round(k, 1) for k in knit_dirs]} deg  — FIXED, not optimised")
    print(f"Cables  : {len(cable_paths)} "
          f"({'disabled' if args.no_cables else 'lengths ' + str([len(p) for p in cable_paths])})")
    print(f"Floor   : {args.min_disp_mm:.3f} mm max displacement "
          f"({args.min_disp_mm / (span * 1000) * 100:.4f} % of span)"
          if _min_disp[0] > 0 else "Floor   : DISABLED")
    print(f"Log     : {os.path.relpath(_log_path[0], HERE)}")

    loss_of = make_loss(V_target, interior_idx)

    # cable groups, by role, for the mirror-symmetric phase
    def cable_group(path):
        c = V[path].mean(0)
        if abs(c[0]) < 0.12:
            return 0                                # crease band
        return 1 if c[1] < -0.18 else 2             # southern fan / the rest
    groups = np.array([cable_group(p) for p in cable_paths]) if cable_paths \
        else np.zeros(0, int)
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    if len(groups):
        print(f"Cable groups: crease={int((groups == 0).sum())} "
              f"fan={int((groups == 1).sum())} other={int((groups == 2).sum())}")

    def scales_from_groups(g_scales):
        return np.array([g_scales[g] for g in groups]) if len(groups) else np.zeros(0)

    def evaluate(sw, sc, cscales, tag):
        out = run_fem(sw, sc, knit_dirs, args.pressure, args.motif,
                      cable_paths, cscales, V_rest, tag=tag)
        if out is None or "verts" not in out:
            append_rmse(None, _call_count[0])
            return None, 1e3
        l = loss_of(out["verts"])
        append_rmse(l, _call_count[0])
        return out, l

    # ── phase 0: sweep ────────────────────────────────────────────────────────
    if args.phase == 0:
        vals = [float(v) for v in (args.sweep_sf or "0.98,1.00,1.02,1.04,1.06,1.08"
                                   ).split(",")]
        print("\nUniform sf sweep (wale = course, all regions, cable scale "
              f"{args.cable0}):")
        for v in vals:
            sw, sc, s = expand_uniform([v, v, args.cable0])
            out, l = evaluate(sw, sc, scales_from_groups([s] * 3), f"sweep sf={v}")
            if out is None:
                print(f"  sf={v:.4f}  INVALID")
            else:
                print(f"  sf={v:.4f}  RMSE={l*1000:9.4f} mm  "
                      f"crown={out['crown_height']:.5f} (target {t_crown:.5f})  "
                      f"maxdisp={out['max_disp']*1000:8.3f} mm")
        return

    # ── phase 1: 3 params ─────────────────────────────────────────────────────
    if args.phase == 1:
        p0 = np.array([args.sf0_wale, args.sf0_course, args.cable0])
        bnds = [(0.80, 1.50), (0.80, 1.50), (0.80, 1.05)]

        def obj(p):
            sw, sc, s = expand_uniform(p)
            _, l = evaluate(sw, sc, scales_from_groups([s] * 3),
                            "p1 " + ",".join(f"{v:.4f}" for v in p))
            print(f"  [{_call_count[0]:5d}] RMSE={l*1000:9.4f} mm  "
                  f"p=[{','.join(f'{v:.4f}' for v in p)}]", flush=True)
            return _track(l, p)

        res = _run_minimize(obj, p0, method=args.method, bounds=bnds,
                            options={"maxiter": args.maxiter, "ftol": 1e-10,
                                     "gtol": 1e-6, "eps": 0.002})
        save(args, res, prefix, knit_dirs, expand_uniform(res.x)[:2],
             scales_from_groups([res.x[2]] * 3), V, F, V_target, interior_idx,
             cable_names, evaluate, t_crown)
        return

    # ── phase 2 / 3 ───────────────────────────────────────────────────────────
    seed = json.load(open(args.seed_json)) if args.seed_json else None
    if args.phase == 2:
        if seed and "sf_wale" in seed:
            p0 = np.array([seed["sf_wale"][a] for a, _ in MIRROR_PAIRS] +
                          [seed["sf_course"][a] for a, _ in MIRROR_PAIRS] +
                          list(seed.get("cable_group_scales", [args.cable0] * n_groups)))
        else:
            p0 = np.array([args.sf0_wale] * 6 + [args.sf0_course] * 6 +
                          [args.cable0] * n_groups)
        bnds = [(0.80, 1.50)] * 12 + [(0.80, 1.05)] * n_groups

        def obj(p):
            sw, sc, gs = expand_mirror(p, n_groups)
            _, l = evaluate(sw, sc, scales_from_groups(gs), "p2")
            if _best[0] is None or l < _best[0][0] or _call_count[0] % 20 == 0:
                print(f"  [{_call_count[0]:5d}] RMSE={l*1000:9.4f} mm", flush=True)
            return _track(l, p)

        res = _run_minimize(obj, p0, method=args.method, bounds=bnds,
                            options={"maxiter": args.maxiter, "ftol": 1e-10,
                                     "gtol": 1e-6, "eps": 0.002})
        sw, sc, gs = expand_mirror(res.x, n_groups)
        save(args, res, prefix, knit_dirs, (sw, sc), scales_from_groups(gs),
             V, F, V_target, interior_idx, cable_names, evaluate, t_crown,
             extra={"cable_group_scales": [float(v) for v in gs]})
        return

    if args.phase == 3:
        n_c = len(cable_paths)
        if seed and "sf_wale" in seed:
            p0 = np.array(list(seed["sf_wale"]) + list(seed["sf_course"]) +
                          list(seed["cable_rest_scales"]))
        else:
            p0 = np.array([args.sf0_wale] * N_REGIONS +
                          [args.sf0_course] * N_REGIONS + [args.cable0] * n_c)
        bnds = [(0.80, 1.50)] * (2 * N_REGIONS) + [(0.80, 1.05)] * n_c

        def obj(p):
            sw = p[:N_REGIONS]
            sc = p[N_REGIONS:2 * N_REGIONS]
            cs = p[2 * N_REGIONS:]
            _, l = evaluate(sw, sc, cs, "p3")
            if _best[0] is None or l < _best[0][0] or _call_count[0] % 25 == 0:
                print(f"  [{_call_count[0]:5d}] RMSE={l*1000:9.4f} mm", flush=True)
            return _track(l, p)

        res = _run_minimize(obj, p0, method=args.method, bounds=bnds,
                            options={"maxiter": args.maxiter, "ftol": 1e-11,
                                     "gtol": 1e-7, "eps": 0.0015})
        x = res.x
        save(args, res, prefix, knit_dirs,
             (x[:N_REGIONS], x[N_REGIONS:2 * N_REGIONS]), x[2 * N_REGIONS:],
             V, F, V_target, interior_idx, cable_names, evaluate, t_crown)
        return


def save(args, res, prefix, knit_dirs, sfs, cscales, V, F, V_target,
         interior_idx, cable_names, evaluate, t_crown, extra=None):
    sw, sc = sfs
    print(f"\nConverged: {res.success}  |  {res.message}")
    print(f"Best RMSE: {res.fun * 1000:.4f} mm   FEM calls: {_call_count[0]}")

    out, l = evaluate(sw, sc, cscales, "final")
    if out is None:
        print("WARNING: the reported best does not re-evaluate as valid.")
        return
    dev = np.linalg.norm(out["verts"] - V_target, axis=1)
    dev_i = dev[interior_idx]
    max_disp_mm = out["max_disp"] * 1000.0
    floor_mm = _min_disp[0] * 1000.0
    at_floor = floor_mm > 0 and max_disp_mm < 1.5 * floor_mm
    print(f"  re-evaluated RMSE {l*1000:.4f} mm   max interior deviation "
          f"{dev_i.max()*1000:.4f} mm   crown {out['crown_height']:.5f} m "
          f"(target {t_crown:.5f})")
    print(f"  max displacement from rest {max_disp_mm:.3f} mm "
          f"(floor {floor_mm:.3f} mm)"
          + ("   *** PRESSED AGAINST THE FLOOR — this is not a fit ***"
             if at_floor else "   — clear of the floor"))

    res_json = os.path.join(OUT_DIR, f"{prefix}_result.json")
    d = {"geometry": "2part_smooth", "phase": args.phase, "method": args.method,
         "mesh": os.path.relpath(MESH_PATH, HERE),
         "region_map": os.path.relpath(REGION_MAP, HERE),
         "cable_file": os.path.relpath(CABLE_FILE, HERE),
         "cable_names": cable_names, "n_regions": N_REGIONS,
         "region_names": REGION_NAMES,
         "pressure": args.pressure, "motif": args.motif, "cable_ea": CABLE_EA,
         "converged": bool(res.success), "message": str(res.message),
         "n_calls": _call_count[0],
         "rmse_interior_m": float(l),
         "rmse_interior_mm": float(l) * 1000.0,
         "max_interior_deviation_mm": float(dev_i.max()) * 1000.0,
         "crown_m": float(out["crown_height"]), "target_crown_m": float(t_crown),
         "max_disp_from_rest_mm": max_disp_mm,
         "min_disp_floor_mm": floor_mm, "at_floor": bool(at_floor),
         "knit_dir_deg_region_means": [float(k) for k in knit_dirs],
         "knit_is_fixed": True,
         "sf_wale": [float(v) for v in sw], "sf_course": [float(v) for v in sc],
         "cable_rest_scales": [float(v) for v in cscales],
         "call_log": os.path.relpath(_log_path[0], HERE)}
    if extra:
        d.update(extra)
    with open(res_json, "w") as f:
        json.dump(d, f, indent=1)

    obj_path = os.path.join(OUT_DIR, f"{prefix}_best_fit.obj")
    write_obj(obj_path, out["verts"], F,
              f"2-part smooth — best-fit FEM result (phase {args.phase})\n"
              f"interior RMSE     {l*1000:.4f} mm  over {len(interior_idx)} vertices\n"
              f"max interior dev  {dev_i.max()*1000:.4f} mm\n"
              f"crown             {out['crown_height']:.6f} m "
              f"(target {t_crown:.6f} m)\n"
              f"max disp from rest {max_disp_mm:.4f} mm "
              f"(floor {floor_mm:.4f} mm, at_floor={at_floor})\n"
              f"pressure {args.pressure} Pa   motif {args.motif}   "
              f"cable_ea {CABLE_EA}\n"
              f"knit direction fixed from the directional field, not optimised\n"
              f"params: {res_json}")
    np.savetxt(os.path.join(OUT_DIR, f"{prefix}_deviation_mm.csv"),
               dev * 1000.0, delimiter=",", header="deviation_mm", comments="")
    print(f"Saved: {res_json}\n       {obj_path}")


if __name__ == "__main__":
    main()
