"""
Coarse uniform grid for the 2-part smooth FEM fit, run in parallel.

Every region gets the same (sf_wale, sf_course) and every cable the same rest
scale; the point is to locate the basin before optimise_2part.py --phase 2
(Powell) refines it.  The objective is the same interior RMSE against the
target (rest == target) with the same validity gate, including the 5 mm
max-displacement floor.

Material is passed explicitly as E1 = wale (along the knit field), E2 = course,
so the old motif-1 estimate in fem_batch_nregion.cpp is never used.

Usage:
    python3 FDM/grid_2part.py --cable-file FDM/data/2part/cable_paths_2part_continuous.json \
        --out-prefix 2part_K_s1_grid --jobs 18
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optimise_2part as o2   # noqa: E402

OUT_DIR = o2.OUT_DIR


def _frange(spec):
    a, b, s = (float(v) for v in spec.split(":"))
    return [round(v, 4) for v in np.arange(a, b + 1e-9, s)]


def _one(job):
    (i, sw, sc, cs, a) = job
    params = {"pressure": a["pressure"], "motif": 1, "cable_ea": o2.CABLE_EA,
              "E1": a["E_wale"], "E2": a["E_course"], "nu": a["nu"],
              "cable_paths": a["cable_paths"],
              "regions": [{"sf_wale": sw, "sf_course": sc, "knit_dir_deg": float(k)}
                          for k in a["knit_dirs"]],
              "cable_rest_scales": [cs] * len(a["cable_paths"])}
    prefix = os.path.join(a["tmp"], f"g{i:05d}")
    pj = prefix + "_params.json"
    json.dump(params, open(pj, "w"))
    rec = {"call": i, "sf_wale": sw, "sf_course": sc, "cable_scale": cs,
           "valid": False}
    try:
        r = subprocess.run([o2.BINARY, o2.MESH_PATH, o2.REGION_MAP, pj, prefix],
                           capture_output=True, text=True, timeout=a["timeout"])
        sp, vp = prefix + "_scalars.csv", prefix + "_verts.csv"
        if r.returncode != 0 or not os.path.exists(vp):
            rec["reason"] = f"rc={r.returncode}"
        elif "regularization_failed" in r.stdout or "failed" in r.stdout.split("\n")[0]:
            rec["reason"] = "solver failed"
        else:
            row = next(csv.DictReader(open(sp)))
            verts = np.loadtxt(vp, delimiter=",", skiprows=1)[:, 1:]
            V, VT, ii = a["V_rest"], a["V_target"], a["interior"]
            md = float(np.max(np.linalg.norm(verts - V, axis=1)))
            crown = float(row["crown_height"])
            tc = float(VT[:, 2].max())
            if md < a["floor"]:
                rec["reason"] = f"max_disp {md*1000:.2f} mm below floor"
            elif crown < 0.3 * tc or crown > 3 * tc or verts[:, 2].min() < -0.01 * tc:
                rec["reason"] = "crown out of range / folded"
            else:
                d = verts[ii] - VT[ii]
                rec.update(valid=True, reason="OK",
                           rmse_mm=float(np.sqrt(np.mean(np.sum(d ** 2, 1)))) * 1000,
                           crown=crown, max_disp_mm=md * 1000,
                           max_stress=float(row["max_stress"]))
    except subprocess.TimeoutExpired:
        rec["reason"] = "timeout"
    finally:
        for suf in ("_params.json", "_scalars.csv", "_verts.csv", "_stress.csv"):
            try:
                os.unlink(prefix + suf)
            except OSError:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cable-file", default=o2.CABLE_FILE)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--sf-wale", default="0.90:1.30:0.04")
    ap.add_argument("--sf-course", default="0.90:1.30:0.04")
    ap.add_argument("--cable", default="0.88:1.00:0.03")
    ap.add_argument("--pressure", type=float, default=1000.0)
    ap.add_argument("--E-wale", type=float, default=12500.0)
    ap.add_argument("--E-course", type=float, default=5000.0)
    ap.add_argument("--nu", type=float, default=0.198)
    ap.add_argument("--min-disp-mm", type=float, default=5.0)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    V, F = o2.load_off(o2.MESH_PATH)
    VT, _ = o2.load_off(o2.TARGET_OFF)
    interior = np.array(sorted(set(range(len(V))) - o2.boundary_vertices(F)))
    rm = json.load(open(o2.REGION_MAP))
    fr, pf = np.array(rm["face_regions"]), np.asarray(rm["face_knit_dirs_deg"], float)
    knit = [float(np.degrees(np.angle(np.exp(2j * np.radians(pf[fr == r])).mean())) / 2 % 180)
            for r in range(o2.N_REGIONS)]
    cab = json.load(open(args.cable_file))
    tmp = os.path.join(OUT_DIR, f".{args.out_prefix}_tmp")
    os.makedirs(tmp, exist_ok=True)
    shared = dict(pressure=args.pressure, E_wale=args.E_wale, E_course=args.E_course,
                  nu=args.nu, cable_paths=[cab[k] for k in sorted(cab)], knit_dirs=knit,
                  tmp=tmp, timeout=args.timeout, V_rest=V, V_target=VT,
                  interior=interior, floor=args.min_disp_mm / 1000.0)
    pts = list(itertools.product(_frange(args.sf_wale), _frange(args.sf_course),
                                 _frange(args.cable)))
    print(f"{len(pts)} grid points, {len(cab)} cables, E_wale {args.E_wale:.0f} "
          f"E_course {args.E_course:.0f} N/m nu {args.nu}, {args.pressure} Pa", flush=True)
    log = os.path.join(OUT_DIR, f"{args.out_prefix}_calls.jsonl")
    best = None
    with open(log, "w") as fl, ProcessPoolExecutor(args.jobs) as ex:
        jobs = [(i + 1, sw, sc, cs, shared) for i, (sw, sc, cs) in enumerate(pts)]
        for n, rec in enumerate(ex.map(_one, jobs, chunksize=1), 1):
            fl.write(json.dumps(rec) + "\n"); fl.flush()
            if rec["valid"] and (best is None or rec["rmse_mm"] < best["rmse_mm"]):
                best = rec
                print(f"  [{n:4d}/{len(pts)}] best {rec['rmse_mm']:.2f} mm at "
                      f"sw {rec['sf_wale']} sc {rec['sf_course']} cable {rec['cable_scale']} "
                      f"crown {rec['crown']:.4f}", flush=True)
    os.rmdir(tmp)
    nvalid = sum(json.loads(l)["valid"] for l in open(log))
    print(f"done: {nvalid}/{len(pts)} valid; best {best}")
    if best:
        seed = {"sf_wale": [best["sf_wale"]] * o2.N_REGIONS,
                "sf_course": [best["sf_course"]] * o2.N_REGIONS,
                "cable_group_scales": [best["cable_scale"]]}
        json.dump(seed, open(os.path.join(OUT_DIR, f"{args.out_prefix}_seed.json"), "w"))


if __name__ == "__main__":
    main()
