"""Measure the cost of one forward solve, per case-study geometry.

Section 6.5.1 claims that the cost of an objective evaluation is set by the number
of Newton iterations rather than by the size of the mesh, and that its spread over
the design box is wide enough to dominate the wall clock. A single timing at the
optimum cannot support either claim, so this samples a neighbourhood of the
recorded optimum and times every solve.

Sampling is ±10% about the optimum rather than uniform over the bounds, because
that is the neighbourhood a finite-difference L-BFGS-B actually visits: gradient
probes sit within eps of the current iterate, and the line search rarely leaves
the region.

    python3 measure_solve_cost.py [--samples 25] [--out data/solve_cost.json]
"""

import argparse
import json
import os
import random
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BINARY = os.path.join(ROOT, "build-linux", "fem_batch_nregion")
OPT = os.path.join(HERE, "optimisation")

# (label, mesh, region map, source of the optimum, n_regions)
CASES = [
    ("2part (6.4.1)", os.path.join(ROOT, "data", "2part", "2part_opt_simu_m.off"),
     None, None, 1),
    ("B5 (6.4.2)", os.path.join(ROOT, "data", "B5_remeshed_shared.off"),
     os.path.join(OPT, "region_map_1p2m.json"),
     os.path.join(OPT, "B5_optimised_params.json"), 9),
    ("C5 (6.4.3)", os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off"),
     os.path.join(OPT, "C5_16region_map.json"),
     os.path.join(OPT, "C5_16region_optimised_sym.json"), 16),
    ("D5 (6.4.6)", os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off"),
     os.path.join(OPT, "D5_4region_adaptive_map.json"),
     os.path.join(OPT, "D5_4region_adaptive_optimised.json"), 4),
]

# A leftover params file per case supplies the cable topology, which is not stored
# in the optimised-parameter files.
CABLE_DONOR = {9: "tmpm4i21c46.json", 16: "tmpzn4zd3oj.json", 4: "tmpq1jqpczd.json"}

TIMEOUT = 120.0


def mesh_size(path):
    with open(path) as f:
        f.readline()
        nv, nf, _ = map(int, f.readline().split())
    return nv, nf


def base_params(n_regions, opt_path):
    """The optimum, in the JSON shape the forward solver expects."""
    if opt_path is None:                      # 2part: no stored region file
        return dict(pressure=1000.0, motif=1, cable_ea=157000.0, cable_paths=[],
                    regions=[dict(sf_wale=1.1171, sf_course=1.1171,
                                  knit_dir_deg=0.0)],
                    cable_rest_scales=[])
    donor = json.load(open(os.path.join(OPT, CABLE_DONOR[n_regions])))
    opt = json.load(open(opt_path))
    p = dict(donor)
    p["regions"] = [dict(sf_wale=r["sf_wale"], sf_course=r["sf_course"],
                         knit_dir_deg=r["knit_dir_deg"]) for r in opt["regions"]]
    if "cables" in opt:
        p["cable_rest_scales"] = [c["rest_scale"] for c in opt["cables"]]
    return p


def perturb(p, rng, rel=0.10):
    q = json.loads(json.dumps(p))
    for r in q["regions"]:
        r["sf_wale"] *= 1.0 + rng.uniform(-rel, rel)
        r["sf_course"] *= 1.0 + rng.uniform(-rel, rel)
    q["cable_rest_scales"] = [min(1.05, max(0.70, s * (1.0 + rng.uniform(-rel, rel))))
                              for s in q.get("cable_rest_scales", [])]
    return q


def one_solve(mesh, map_path, params, workdir, tag):
    if map_path is None:
        _, nf = mesh_size(mesh)
        map_path = os.path.join(workdir, "map.json")
        with open(map_path, "w") as f:
            json.dump({"face_regions": [0] * nf}, f)
    pp = os.path.join(workdir, "params.json")
    with open(pp, "w") as f:
        json.dump(params, f)
    t0 = time.perf_counter()
    try:
        r = subprocess.run([BINARY, mesh, map_path, pp,
                            os.path.join(workdir, tag)],
                           capture_output=True, text=True, timeout=TIMEOUT)
        dt = time.perf_counter() - t0
        ok = r.returncode == 0 and "OK" in r.stdout
    except subprocess.TimeoutExpired:
        return TIMEOUT, False, True
    return dt, ok, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "solve_cost.json"))
    args = ap.parse_args()

    out = {}
    with tempfile.TemporaryDirectory() as wd:
        for label, mesh, map_path, opt_path, nreg in CASES:
            nv, nf = mesh_size(mesh)
            p0 = base_params(nreg, opt_path)
            rng = random.Random(20260813)

            t_opt, ok, _ = one_solve(mesh, map_path, p0, wd, "at_opt")
            times, n_to = [], 0
            for i in range(args.samples):
                dt, ok_i, timed_out = one_solve(mesh, map_path,
                                                perturb(p0, rng), wd, f"s{i}")
                times.append(dt)
                n_to += int(timed_out)
            times.sort()
            out[label] = dict(n_verts=nv, n_faces=nf, n_regions=nreg,
                              at_optimum=t_opt, samples=times, n_timeout=n_to)
            med = times[len(times) // 2]
            print(f"  {label:16s} {nv:5d} v / {nf:5d} f  "
                  f"optimum {t_opt:6.3f} s   "
                  f"sampled min {times[0]:6.3f}  median {med:6.3f}  "
                  f"max {times[-1]:7.3f} s   spread x{times[-1]/times[0]:5.1f}"
                  + (f"  ({n_to} hit the {TIMEOUT:.0f} s limit)" if n_to else ""))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n  saved: {args.out}")


if __name__ == "__main__":
    main()
