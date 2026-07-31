"""
Direct FEA sweep over knitting direction (theta_knit) for figM.

figM previously read this slice off symmetry-augmented GPs fitted to the Sobol
samples — samples that were run on the wrong mesh and, for motif 2, with motif 5
material (see validate_fem_runs.py).  A run costs ~0.25 s, so the slice is
simulated directly instead.

Fixed: sf_wale = sf_course = 1.0, pressure = 1000 Pa (the same slice figM used).

Two exact identities are available as checks, both reported by --check:

  1. Rotational invariance.  theta_knit rotates the material frame and the
     stretch frame together, so on a circular domain the whole problem is merely
     rotated: crown height and global mean stress must be independent of theta.
     Any variation is pure discretisation noise and measures the noise floor.

  2. Mirror symmetry.  Reflecting about y = x maps direction (a,b) to (b,a), so
     wale(theta) -> wale(90-theta), and swaps the x=0 and y=0 cut planes:
         H_x0(theta) = H_y0(90 - theta)
     exactly, up to the mesh not being exactly mirror-symmetric itself.

Usage:
  python3 run_knit_dir_sweep.py               # 1 degree steps
  python3 run_knit_dir_sweep.py --step 0.5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from fea_interface import run_fea
from run_uniform_sf_sweep import _section_metrics_from_files
from plot_section_sensitivity import ROUGHNESS_THRESHOLD, CROWN_MIN_M

SWEEP_CSV = os.path.join(DATA_DIR, "knit_dir_sweep.csv")
SWEEP_DIR = os.path.join(DATA_DIR, "knit_dir_sweep")

THETA_RANGE = (0.0, 90.0)
SF          = 1.0
PRESSURE    = 1000.0
MOTIFS      = (1, 2)


def run_sweep(step=1.0, sf=SF, pressure=PRESSURE, motifs=MOTIFS):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    n = int(round((THETA_RANGE[1] - THETA_RANGE[0]) / step)) + 1
    thetas = np.linspace(*THETA_RANGE, n)
    print(f"Sweeping {n} angles x {len(motifs)} motifs "
          f"(sf_wale = sf_course = {sf}, p = {pressure} Pa)")

    rows = []
    for motif in motifs:
        for i, th in enumerate(thetas):
            prefix = os.path.join(SWEEP_DIR, f"m{motif}_th{th:06.2f}")
            try:
                res = run_fea(sf, sf, th, pressure, motif, prefix, timeout=600)
            except Exception as exc:
                print(f"  motif{motif} theta={th:5.1f}  FAILED: {exc}")
                rows.append({"motif": motif, "knit_dir": th, "sim_failed": True})
                continue
            row = {"motif": motif, "knit_dir": th, "sf_wale": sf,
                   "sf_course": sf, "pressure": pressure,
                   "crown_height": res["crown_height"],
                   "mean_stress":  res["mean_stress"],
                   "max_stress":   res["max_stress"],
                   "verts_path":   res["verts_path"],
                   "stress_path":  res["stress_path"]}
            row.update(_section_metrics_from_files(res["verts_path"],
                                                   res["stress_path"]))
            r_max = np.nanmax([row["r_x0"], row["r_y0"]])
            row["sim_failed"] = bool(
                not np.isfinite(row["crown_height"])
                or row["crown_height"] < CROWN_MIN_M
                or (np.isfinite(r_max) and r_max > ROUGHNESS_THRESHOLD))
            rows.append(row)
            if i % 15 == 0:
                print(f"  motif{motif} theta={th:5.1f}  "
                      f"crown={row['crown_height']*1000:6.1f} mm  "
                      f"H_x0={row['H_mean_x0']:.3f}  H_y0={row['H_mean_y0']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"\nSaved {SWEEP_CSV}  ({len(df)} runs, "
          f"{int(df['sim_failed'].sum())} flagged)")
    return df


def check(df=None):
    """Report the two exact identities (see module docstring)."""
    if df is None:
        df = pd.read_csv(SWEEP_CSV)
    print("\nConsistency checks")
    for motif, sub in df[~df["sim_failed"].astype(bool)].groupby("motif"):
        sub = sub.sort_values("knit_dir")
        th = sub["knit_dir"].values
        print(f"  motif {motif}:")
        for col, unit in (("crown_height", "mm"), ("mean_stress", "Pa")):
            v = sub[col].values * (1000 if col == "crown_height" else 1)
            print(f"    rotational invariance of {col:12s} "
                  f"spread {v.max()-v.min():.4g} {unit} "
                  f"({(v.max()-v.min())/np.mean(v)*100:.2f}% of mean)")
        # H_x0(theta) vs H_y0(90 - theta), on the common grid
        hx = sub["H_mean_x0"].values
        hy = sub["H_mean_y0"].values
        hy_mirror = np.interp(90.0 - th, th, hy)
        d = np.abs(hx - hy_mirror)
        scale = np.mean(np.abs(hx))
        print(f"    mirror symmetry  |H_x0(t) - H_y0(90-t)|  "
              f"median {np.median(d)/scale*100:.2f}%  max {d.max()/scale*100:.2f}% "
              f"of mean H")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--sf", type=float, default=SF)
    ap.add_argument("--pressure", type=float, default=PRESSURE)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if args.check_only:
        check()
    else:
        check(run_sweep(step=args.step, sf=args.sf, pressure=args.pressure))
