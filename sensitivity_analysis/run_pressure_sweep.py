"""
Pressure sweep of the flat clamped disc (Figure 7.6): crown height against
inflation pressure, in 100 Pa steps, stopped where the maximum membrane stress
reaches the strength limit.

Mesh data/circular_flat.off (1200 mm span), rim clamped, no cable, uniform knit
direction with the wale along x.  Solver build/fem_batch_nregion with the fixed
pressure (740f0189a); E1 is the modulus along the knit direction (wale), so the
measured stitch structures are passed by name:

    structure I : E_wale = 12500, E_course = 5000 N/m, nu = 0.198
    structure II: E_wale =  8000, E_course = 5100 N/m, nu = 0.195

Three pre-strain states per structure: nominal (s_wale = s_course = 1.0, the
"untensioned" specimen), the calibration of Section 7.1.3 (s_wale = 0.960,
s_course = 1.018), and the Section 7.2 reference (s_wale = s_course = 1.1).

The pressure where the max von Mises stress (2nd Piola-Kirchhoff, N/m) reaches
the structure's limit (I 3500, II 4000 N/m) is found by bisection to 1 Pa and stored as the row at_limit = True.
The sweep continues past it to --p-max so the out-of-range part can be drawn.

Outputs:
    data/pressure_sweep.csv           one row per (structure, prestrain, pressure)
    data/pressure_sweep/<tag>_*.csv   per-run vertices, stress, scalars

Usage:
    python3 run_pressure_sweep.py [--jobs 16] [--limit 3500] [--step 100] [--prestrain reference] [--p-max 12000]
"""
import argparse, json, os, subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MESH = os.path.join(ROOT, "data", "circular_flat.off")
BIN = os.environ.get("FEM_NREGION_BINARY", os.path.join(ROOT, "build", "fem_batch_nregion"))
OUT = os.path.join(HERE, "data", "pressure_sweep")
CSV = os.path.join(HERE, "data", "pressure_sweep.csv")

STRUCTURES = {"I": dict(E_wale=12500.0, E_course=5000.0, nu=0.198, limit=3500.0),
              "II": dict(E_wale=8000.0, E_course=5100.0, nu=0.195, limit=4000.0)}   # limit: max stress, N/m
PRESTRAIN = {"nominal": (1.0, 1.0), "calibrated": (0.960, 1.018),     # (s_wale, s_course)
             "reference": (1.1, 1.1)}                               # Section 7.2 reference


def mesh_setup():
    L = open(MESH).read().split("\n")
    nv, nf, _ = map(int, L[1].split())
    F = [list(map(int, l.split()[1:4])) for l in L[2 + nv:2 + nv + nf]]
    ec = Counter(tuple(sorted((f[i], f[(i + 1) % 3]))) for f in F for i in range(3))
    rim = sorted({int(v) for e, c in ec.items() if c == 1 for v in e})
    mp = os.path.join(OUT, "map.json")
    # fem_batch_nregion angle convention: (cos, sin), so 0 deg = wale along x
    json.dump({"face_regions": [0] * nf, "face_knit_dirs_deg": [0.0] * nf}, open(mp, "w"))
    return rim, mp


def run_one(structure, prestrain, p, rim, mp):
    m = STRUCTURES[structure]
    sw, sc = PRESTRAIN[prestrain]
    tag = f"{structure}_{prestrain}_p{p:.0f}"
    pref = os.path.join(OUT, tag)
    params = {"pressure": float(p), "motif": 1, "E1": m["E_wale"], "E2": m["E_course"], "nu": m["nu"],
              "cable_paths": [], "fixed_vertices": rim,
              "regions": [{"sf_wale": sw, "sf_course": sc, "knit_dir_deg": 0.0}]}
    json.dump(params, open(pref + ".json", "w"))
    env = dict(os.environ)
    s = min(sw, sc)
    if s < 1.0:
        # slack rest shape: start on a dome taller than its no-stretch height
        env["FEM_INIT_DOME"] = str(1.15 * 0.6 * np.sqrt(1.0 / s ** 2 - 1.0) + 0.05)
    out = subprocess.run([BIN, MESH, mp, pref + ".json", pref], capture_output=True, text=True,
                         env=env, timeout=600)
    log = out.stdout + out.stderr
    open(pref + ".log", "w").write(log)
    statuses = [l.split()[1] for l in log.splitlines() if l.startswith("SOLVER_STATUS")]
    resid = [float(l.split("max=")[1].split()[0]) for l in log.splitlines() if l.startswith("SOLVER_RESIDUAL")]
    # a line search that stops on a flat energy at an equilibrium is converged:
    # judge on the final out-of-balance force, not only on the status
    ok = "OK" in log and bool(resid) and (statuses[-1] == "success" or resid[-1] < 1e-5)
    v = open(pref + "_scalars.csv").read().split("\n")[1].split(",")
    st = pd.read_csv(pref + "_stress.csv")
    return dict(structure=structure, prestrain=prestrain, stress_limit=m["limit"], s_wale=sw, s_course=sc, pressure=float(p),
                crown_mm=1000 * float(v[0]), max_vm=float(v[1]), mean_vm=float(v[2]),
                resid_max=resid[-1] if resid else np.nan,
                min_principal=float(st.principal_2.min()), frac_compressed=float((st.principal_2 < 0).mean()),
                converged=ok, at_limit=False)


def sweep(structure, prestrain, rim, mp, jobs, limit, step, p_max):
    """Steps up to the stress limit and on to p_max (rows past the limit kept,
    so the figure can draw them as out of range)."""
    rows, p0 = [], step
    with ThreadPoolExecutor(jobs) as ex:
        while True:
            batch = [p0 + k * step for k in range(jobs)]
            res = list(ex.map(lambda p: run_one(structure, prestrain, p, rim, mp), batch))
            rows += res
            if any(r["max_vm"] >= limit for r in res) and batch[-1] >= p_max:
                break
            p0 = batch[-1] + step
    rows.sort(key=lambda r: r["pressure"])
    first_over = next(r for r in rows if r["max_vm"] >= limit)
    below = [r for r in rows if r["pressure"] < first_over["pressure"]]
    past = [r for r in rows if r["pressure"] >= first_over["pressure"] and r["pressure"] <= p_max]
    lo, hi = below[-1]["pressure"] if below else 0.0, first_over["pressure"]
    while hi - lo > 1.0:                        # bisection on the stress limit
        mid = 0.5 * (lo + hi)
        if run_one(structure, prestrain, mid, rim, mp)["max_vm"] < limit: lo = mid
        else: hi = mid
    end = run_one(structure, prestrain, round(lo), rim, mp)
    end["at_limit"] = True
    return below + [end] + past


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None,
                    help="max von Mises stress, N/m (default: per structure, I 3500, II 4000)")
    ap.add_argument("--step", type=float, default=100.0, help="pressure step, Pa")
    ap.add_argument("--p-max", type=float, default=12000.0,
                    help="continue past the stress limit up to this pressure, Pa")
    ap.add_argument("--prestrain", default="reference",
                    help="comma list of " + ",".join(PRESTRAIN) + " (default: reference, s = 1.1)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rim, mp = mesh_setup()
    rows = []
    for s in STRUCTURES:
        for ps in a.prestrain.split(","):
            lim = a.limit if a.limit is not None else STRUCTURES[s]["limit"]
            r = sweep(s, ps, rim, mp, a.jobs, lim, a.step, a.p_max)
            end = next(x for x in r if x["at_limit"])
            print(f"structure {s:2s} {ps:10s}: {len(r)} points, stops at {end['pressure']:.0f} Pa, "
                  f"crown {end['crown_mm']:.1f} mm, max vM {end['max_vm']:.0f} N/m, "
                  f"all converged {all(x['converged'] for x in r)}", flush=True)
            rows += r
    df = pd.DataFrame(rows)
    df.to_csv(CSV, index=False)
    print(f"wrote {CSV}")


if __name__ == "__main__":
    main()
