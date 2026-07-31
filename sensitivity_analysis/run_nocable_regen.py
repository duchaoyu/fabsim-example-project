"""
Regenerate the no-cable FEA sample sets on the CURRENT mesh with the CORRECT
motif materials.

Why: validate_fem_runs.py established that the stored no-cable samples were run
on data/circular_flat.off scaled by 0.951149 (radius 0.5707 m instead of
0.600 m), and that motif2_nocable was run with motif 5 material (E1=12507,
E2=5000) instead of motif 2 (E1=5000, E2=8000).  Any sensitivity analysis built
on those samples describes the wrong geometry and, for motif 2, the wrong
material.  The cable groups are unaffected and are not touched here.

Writes results to a NEW file (data/results_nocable_v2.csv) and per-run output to
data/nocable_v2/, leaving the original dataset untouched.

Designs:
  lhs       Latin hypercube — for training the GP surrogates (default)
  saltelli  SALib Saltelli design — lets Sobol indices be computed directly from
            FEA, with no surrogate in the loop

Usage:
  python3 run_nocable_regen.py --n 300                 # LHS, 300 per motif
  python3 run_nocable_regen.py --design saltelli --n-base 256
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_NO_CABLE, RANDOM_SEED
from sampling import lhs
from run_uniform_sf_sweep import _section_metrics_from_files
from plot_section_sensitivity import ROUGHNESS_THRESHOLD, CROWN_MIN_M
from fea_interface import run_fea

KEYS = list(PARAMS_NO_CABLE.keys())

# Per-design output directory: the two designs reuse sample ids from 0, so a
# shared directory would have the second run overwrite the first run's
# per-sample vertex/stress files.
def _out_dir(design):
    return os.path.join(DATA_DIR,
                        "nocable_v2" if design == "lhs" else "nocable_v2_saltelli")


def _one(job):
    """Run a single sample; returns a result row. Safe to call in a subprocess."""
    idx, motif, params, out_dir = job
    prefix = os.path.join(out_dir, f"m{motif}_{idx:05d}")
    row = {"sample_id": idx, "motif": motif, "has_cable": False,
           "group": f"motif{motif}_nocable", **params}
    try:
        res = run_fea(params["sf_wale"], params["sf_course"],
                      params["knit_dir"], params["pressure"], motif,
                      prefix, timeout=600)
    except Exception as exc:
        row["sim_failed"] = True
        row["error"] = str(exc)[:200]
        return row

    row.update({"crown_height": res["crown_height"],
                "mean_stress":  res["mean_stress"],
                "max_stress":   res["max_stress"],
                "boundary_reaction_mean": res["boundary_reaction_mean"],
                "verts_path": res["verts_path"],
                "stress_path": res["stress_path"]})
    row.update(_section_metrics_from_files(res["verts_path"],
                                           res["stress_path"]))
    r_max = np.nanmax([row.get("r_x0", np.nan), row.get("r_y0", np.nan)])
    row["sim_failed"] = bool(
        not np.isfinite(row["crown_height"])
        or row["crown_height"] < CROWN_MIN_M
        or (np.isfinite(r_max) and r_max > ROUGHNESS_THRESHOLD))
    return row


def build_design(design, n, n_base, seed):
    if design == "lhs":
        df = lhs(n, PARAMS_NO_CABLE, seed)
        return df[KEYS].values, f"LHS n={n}"
    from SALib.sample import saltelli
    problem = {"num_vars": len(KEYS), "names": KEYS,
               "bounds": [list(PARAMS_NO_CABLE[k]) for k in KEYS]}
    X = saltelli.sample(problem, n_base, calc_second_order=False)
    return X, f"Saltelli n_base={n_base} -> {len(X)} points"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["lhs", "saltelli"], default="lhs")
    ap.add_argument("--n", type=int, default=300, help="samples per motif (lhs)")
    ap.add_argument("--n-base", type=int, default=256, help="SALib base N")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = _out_dir(args.design)
    os.makedirs(out_dir, exist_ok=True)
    X, desc = build_design(args.design, args.n, args.n_base, args.seed)
    out_csv = args.out or os.path.join(
        DATA_DIR, f"results_nocable_v2{'_saltelli' if args.design == 'saltelli' else ''}.csv")

    jobs = []
    sid = 0
    for motif in (1, 2):
        for row in X:
            jobs.append((sid, motif, dict(zip(KEYS, row)), out_dir))
            sid += 1

    print(f"{desc};  {len(jobs)} FEA runs over {args.workers} workers")
    # Each FEA is OpenMP-threaded; cap threads per worker so they do not fight.
    os.environ["OMP_NUM_THREADS"] = "2"

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, row in enumerate(ex.map(_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0:
                nb = sum(bool(r.get("sim_failed")) for r in rows)
                print(f"  {i+1}/{len(jobs)} done ({nb} flagged)")

    df = pd.DataFrame(rows).sort_values("sample_id")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
    for g, sub in df.groupby("group"):
        ok = sub[~sub["sim_failed"].astype(bool)]
        print(f"  {g}: {len(ok)}/{len(sub)} converged   "
              f"crown {ok.crown_height.min()*1000:.0f}-{ok.crown_height.max()*1000:.0f} mm   "
              f"H_x0 {ok.H_mean_x0.min():.2f}-{ok.H_mean_x0.max():.2f} m-1")
    return df


if __name__ == "__main__":
    main()
