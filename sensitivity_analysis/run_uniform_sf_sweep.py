"""
Direct FEA sweep along the uniform stretch-factor diagonal (sf_wale = sf_course).

figK previously read this slice off a GP surrogate fitted to the Sobol samples.
That slice is barely covered by the samples — knit_dir = 0 is on the edge of the
sampled range and the corners of the (sf_wale, sf_course) square are sparse — so
the surrogate contributed most of the structure in the curvature panel.  A run
takes ~0.25 s, so we simply simulate the slice directly.

Outputs data/uniform_sf_sweep.csv with one row per (motif, sf):
  crown_height, mean_stress, max_stress, H_mean_x0, H_mean_y0, r_x0, r_y0
plus the quality flag used elsewhere in the pipeline.

Usage:
  python3 run_uniform_sf_sweep.py                 # default grid, both motifs
  python3 run_uniform_sf_sweep.py --step 0.005    # finer
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from curvature import compute_curvatures
from fea_interface import run_fea
from plot_section_profiles import _slice_plane
# Reuse the exact section metrics used to build results_with_sections.csv, so
# the sweep is directly comparable to the scatter of Sobol samples.
from plot_section_sensitivity import (
    _FACES, _SECTION_TOL, _profile_curvature, _profile_roughness,
    ROUGHNESS_THRESHOLD, CROWN_MIN_M,
)

SWEEP_CSV = os.path.join(DATA_DIR, "uniform_sf_sweep.csv")
SWEEP_DIR = os.path.join(DATA_DIR, "uniform_sf_sweep")

SF_LO, SF_HI = 0.8, 1.4
KNIT_DIR     = 0.0
PRESSURE     = 1000.0
MOTIFS       = (1, 2)


def _section_metrics_from_files(verts_path: str, stress_path: str) -> dict:
    """Same computation as plot_section_sensitivity._section_metrics, but for
    an explicit output prefix rather than a numbered sample id."""
    out = {k: np.nan for k in ["H_mean_x0", "H_mean_y0", "r_x0", "r_y0",
                               "von_mises_x0", "von_mises_y0",
                               "T_wale_x0", "T_course_y0"]}
    verts = pd.read_csv(verts_path).sort_values("vid")[["x", "y", "z"]].values
    H = compute_curvatures(verts, _FACES)["H"]

    for h_key, r_key, fixed_axis in [("H_mean_x0", "r_x0", 0),
                                     ("H_mean_y0", "r_y0", 1)]:
        pos, z_mm, _ = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
        if len(pos) < 5:
            continue
        out[h_key] = _profile_curvature(pos, z_mm)
        out[r_key] = _profile_roughness(z_mm)

    sdf = pd.read_csv(stress_path).sort_values("face")
    face_ids = sdf["face"].values.astype(int)
    centroids = verts[_FACES[face_ids]].mean(axis=1)
    mask_x0 = np.abs(centroids[:, 0]) < _SECTION_TOL
    mask_y0 = np.abs(centroids[:, 1]) < _SECTION_TOL
    if mask_x0.any():
        out["T_wale_x0"]    = float(np.mean(sdf["T_wale_Nm"].values[mask_x0]))
        out["von_mises_x0"] = float(np.mean(sdf["von_mises"].values[mask_x0]))
    if mask_y0.any():
        out["T_course_y0"]  = float(np.mean(sdf["T_course_Nm"].values[mask_y0]))
        out["von_mises_y0"] = float(np.mean(sdf["von_mises"].values[mask_y0]))
    return out


def run_sweep(step=0.01, knit_dir=KNIT_DIR, pressure=PRESSURE, motifs=MOTIFS):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    n = int(round((SF_HI - SF_LO) / step)) + 1
    sf_grid = np.linspace(SF_LO, SF_HI, n)
    print(f"Sweeping {n} sf values x {len(motifs)} motifs "
          f"(knit_dir={knit_dir}°, p={pressure} Pa)")

    rows = []
    for motif in motifs:
        for i, sf in enumerate(sf_grid):
            prefix = os.path.join(SWEEP_DIR, f"m{motif}_sf{sf:.3f}")
            try:
                res = run_fea(sf, sf, knit_dir, pressure, motif, prefix,
                              timeout=600)
            except Exception as exc:
                print(f"  motif{motif} sf={sf:.3f}  FAILED: {exc}")
                rows.append({"motif": motif, "sf": sf, "sim_failed": True})
                continue

            row = {"motif": motif, "sf": sf, "knit_dir": knit_dir,
                   "pressure": pressure,
                   "crown_height": res["crown_height"],
                   "mean_stress":  res["mean_stress"],
                   "max_stress":   res["max_stress"]}
            row.update(_section_metrics_from_files(res["verts_path"],
                                                   res["stress_path"]))
            # Same quality gate as the Sobol pipeline
            r_max = np.nanmax([row["r_x0"], row["r_y0"]])
            row["sim_failed"] = bool(
                not np.isfinite(row["crown_height"])
                or row["crown_height"] < CROWN_MIN_M
                or (np.isfinite(r_max) and r_max > ROUGHNESS_THRESHOLD)
            )
            rows.append(row)
            if i % 10 == 0 or row["sim_failed"]:
                flag = "  FLAGGED" if row["sim_failed"] else ""
                print(f"  motif{motif} sf={sf:.3f}  "
                      f"crown={row['crown_height']*1000:6.1f} mm  "
                      f"H_x0={row['H_mean_x0']:.3f}  H_y0={row['H_mean_y0']:.3f}"
                      f"{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    n_bad = int(df["sim_failed"].sum())
    print(f"\nSaved {SWEEP_CSV}  ({len(df)} runs, {n_bad} flagged)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--knit-dir", type=float, default=KNIT_DIR)
    ap.add_argument("--pressure", type=float, default=PRESSURE)
    args = ap.parse_args()
    run_sweep(step=args.step, knit_dir=args.knit_dir, pressure=args.pressure)
