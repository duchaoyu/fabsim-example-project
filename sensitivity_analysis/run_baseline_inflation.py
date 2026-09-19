"""
Baseline inflation of the flat disc for the two knitted stitch structures.

This is the reference run the single-variable sweeps sit on top of: no cable,
sf_wale = sf_course = 1.1, knit_dir = 0 deg, pressure stepped over the sampled
range (200-1200 Pa), motifs 1 and 2, mesh = config.MESH_PATH.

sf = 1.1 rather than 1.0 because run_e1r_grid.py identifies sf = 1.0 as an
unstable flat-membrane bifurcation point; every other study here avoids it.

The motif materials were corrected on 2026-09-19 to the measured values

    motif 1 (stitch structure 1):  E1 = 10300, E2 = 13400 N/m, nu12 = 0.58
    motif 2 (stitch structure 2):  E1 =  7700, E2 =  7600 N/m, nu12 = 0.65

from the earlier estimates 5000/12507/0.198 and 5000/8000/0.198.  Every result
in this directory predating that change was computed against the estimates, so
this script runs BOTH tables and reports them side by side: the "old" rows are
not there to be used, they are there to size the error in what is being
discarded.  The old material is reproduced through the binary's material
override (E1, r = E1/E2, nu), which bypasses the motif table.

Outputs:
    data/baseline_inflation.csv      one row per (motif, material, pressure)
    data/baseline_inflation/         per-run vertex and stress files

Usage:
    python3 run_baseline_inflation.py [--jobs 8]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH, MOTIF_PARAMS
from fea_interface import check_binary, run_fea

SF_DEFAULT = 1.1
KNIT_DIR   = 0.0
PRESSURES  = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0]

# The estimates the corrected values replace, kept only for the comparison.
OLD_MOTIF_PARAMS = {
    1: {"E1": 5000.0, "E2": 12507.0, "nu": 0.198},
    2: {"E1": 5000.0, "E2":  8000.0, "nu": 0.198},
}

OUT_DIR = os.path.join(DATA_DIR, "baseline_inflation")


def _one(job):
    motif, material, pressure, sf = job
    tag = f"m{motif}_{material}_sf{sf:g}_p{int(pressure)}"
    prefix = os.path.join(OUT_DIR, tag)

    kwargs = {}
    if material == "old":
        # The binary computes E2 = E1 / r, so r is E1/E2 here, not E2/E1.
        mp = OLD_MOTIF_PARAMS[motif]
        kwargs = {"E1": mp["E1"], "r": mp["E1"] / mp["E2"], "nu": mp["nu"]}
        E1, E2, nu = mp["E1"], mp["E2"], mp["nu"]
    else:
        mp = MOTIF_PARAMS[motif]
        E1, E2, nu = mp["E1"], mp["E2"], mp["nu"]

    res = run_fea(sf, sf, KNIT_DIR, pressure, motif, prefix, **kwargs)

    return {
        "motif":     motif,
        "material":  material,
        "E1":        E1,
        "E2":        E2,
        "E2_over_E1": E2 / E1,
        "nu":        nu,
        "pressure":  pressure,
        "sf_wale":   sf,
        "sf_course": sf,
        "knit_dir":  KNIT_DIR,
        "crown_height":          res["crown_height"],
        "max_stress":            res["max_stress"],
        "mean_stress":           res["mean_stress"],
        "boundary_reaction_mean": res["boundary_reaction_mean"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--motifs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sf", type=float, nargs="+", default=[SF_DEFAULT],
                    help="uniform stretch factor(s); 1.0 is the flat-membrane "
                         "bifurcation point and is expected to be ill-behaved")
    ap.add_argument("--out", default="baseline_inflation",
                    help="basename for the CSV and the per-run directory")
    args = ap.parse_args()

    global OUT_DIR
    OUT_DIR = os.path.join(DATA_DIR, args.out)

    check_binary()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"mesh   {MESH_PATH}")

    jobs = [(m, mat, p, sf)
            for m in args.motifs
            for mat in ("new", "old")
            for sf in args.sf
            for p in PRESSURES]

    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            job = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"  FAILED {job}: {exc}")
            print(f"  {i}/{len(jobs)}")

    df = pd.DataFrame(rows).sort_values(["motif", "material", "sf_wale", "pressure"])
    out = os.path.join(DATA_DIR, args.out + ".csv")
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} rows)")

    pd.set_option("display.width", 160)
    print(df[["motif", "material", "E2_over_E1", "nu", "sf_wale", "pressure",
              "crown_height", "mean_stress", "max_stress"]].to_string(index=False))


if __name__ == "__main__":
    main()
