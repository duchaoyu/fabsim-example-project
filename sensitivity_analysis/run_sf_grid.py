"""
Direct FEA grid over sf_wale x sf_course, for the figL response surfaces.

Why this replaces the surrogates
--------------------------------
figL used to be drawn from GP surrogates fitted to the `*_nocable` groups of
data/results_with_sections.csv.  validate_fem_runs.py established that those runs
are not what their labels say:

    motif1_nocable   ran on circular_flat.off scaled by 0.951149  (wrong mesh)
    motif2_nocable   wrong mesh AND motif 5 material (E1=12507, E2=5000),
                     i.e. wale-stiff -- the OPPOSITE anisotropy to motif 2

So the motif-1 vs motif-2 comparison the figure exists to make was comparing two
different meshes and two materials with opposed anisotropy, and any difference
between the columns was mostly that confound.  A run costs 0.04-0.6 s, so the
grid is simulated directly instead: no surrogate, no wrong mesh, no wrong
material.

The grid also records apex principal curvatures (apex_curvature.py) alongside the
section estimators, because the section estimators average |kappa| along a whole
diameter and are only weakly sensitive to direction.

Fixed: knit_dir = 0 deg, pressure = 1000 Pa.

Outputs:
    data/sf_grid.csv        one row per (motif, sf_wale, sf_course)
    data/sf_grid/           per-run vertex and stress files

Usage:
  python3 run_sf_grid.py                  # 26 x 26 over [0.9, 1.4], both motifs
  python3 run_sf_grid.py --n 16
  python3 run_sf_grid.py --lo 0.9 --hi 1.4
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off, compute_curvatures
from fea_interface import run_fea, check_binary
from plot_section_profiles import _slice_plane
from plot_section_sensitivity import (_profile_roughness, _SECTION_TOL,
                                     ROUGHNESS_THRESHOLD, CROWN_MIN_M)
from section_curvature import profile_curvature_fit, profile_curvature_binned
from apex_curvature import apex_curvature

GRID_CSV = os.path.join(DATA_DIR, "sf_grid.csv")
GRID_DIR = os.path.join(DATA_DIR, "sf_grid")

SF_LO, SF_HI = 0.9, 1.4
N_DEFAULT    = 26
KNIT_DIR     = 0.0
PRESSURE     = 1000.0
MOTIFS       = (1, 2)

_V0, _FACES = read_off(MESH_PATH)


def _metrics(verts_path, stress_path, knit_dir_deg):
    out = {}
    V = pd.read_csv(verts_path).sort_values("vid")[["x", "y", "z"]].values
    H = compute_curvatures(V, _FACES)["H"]

    for key, axis in (("x0", 0), ("y0", 1)):
        pos, z_mm, _ = _slice_plane(V, _FACES, H, fixed_axis=axis)
        if len(pos) < 6:
            out[f"H_fit_{key}"] = out[f"H_binned_{key}"] = np.nan
            out[f"r_{key}"] = np.nan
            continue
        out[f"H_fit_{key}"]    = profile_curvature_fit(pos, z_mm)
        out[f"H_binned_{key}"] = profile_curvature_binned(pos, z_mm)
        out[f"r_{key}"]        = _profile_roughness(z_mm)

    ap = apex_curvature(V, knit_dir_deg, ref_verts=_V0)
    if ap is not None:
        out.update({f"apex_{k}": v for k, v in ap.items()})

    sdf = pd.read_csv(stress_path).sort_values("face")
    cen = V[_FACES[sdf["face"].values.astype(int)]].mean(axis=1)
    for key, axis in (("x0", 0), ("y0", 1)):
        m = np.abs(cen[:, axis]) < _SECTION_TOL
        out[f"von_mises_{key}"] = (float(np.mean(sdf["von_mises"].values[m]))
                                   if m.any() else np.nan)
    m = np.abs(cen[:, 0]) < _SECTION_TOL
    out["T_wale_x0"] = float(np.mean(sdf["T_wale_Nm"].values[m])) if m.any() else np.nan
    m = np.abs(cen[:, 1]) < _SECTION_TOL
    out["T_course_y0"] = float(np.mean(sdf["T_course_Nm"].values[m])) if m.any() else np.nan
    return out


def run_grid(n=N_DEFAULT, lo=SF_LO, hi=SF_HI, motifs=MOTIFS):
    check_binary()
    os.makedirs(GRID_DIR, exist_ok=True)
    grid = np.linspace(lo, hi, n)
    total = len(motifs) * n * n
    print(f"sf grid: {n} x {n} over [{lo}, {hi}] x {len(motifs)} motifs "
          f"= {total} runs  (knit_dir={KNIT_DIR}, p={PRESSURE} Pa)")

    rows, t0, done = [], time.perf_counter(), 0
    for motif in motifs:
        for w in grid:
            for c in grid:
                prefix = os.path.join(GRID_DIR, f"m{motif}_w{w:.3f}_c{c:.3f}")
                done += 1
                try:
                    res = run_fea(w, c, KNIT_DIR, PRESSURE, motif, prefix,
                                  timeout=600)
                except Exception as exc:
                    print(f"  m{motif} w={w:.3f} c={c:.3f}  FAILED: {exc}")
                    rows.append({"motif": motif, "sf_wale": w, "sf_course": c,
                                 "sim_failed": True})
                    continue
                row = {"motif": motif, "sf_wale": w, "sf_course": c,
                       "knit_dir": KNIT_DIR, "pressure": PRESSURE,
                       "crown_height": res["crown_height"],
                       "mean_stress":  res["mean_stress"],
                       "max_stress":   res["max_stress"],
                       "verts_path":   res["verts_path"],
                       "stress_path":  res["stress_path"]}
                row.update(_metrics(res["verts_path"], res["stress_path"],
                                    KNIT_DIR))
                r_max = np.nanmax([row.get("r_x0", np.nan),
                                   row.get("r_y0", np.nan)])
                row["sim_failed"] = bool(
                    not np.isfinite(row["crown_height"])
                    or row["crown_height"] < CROWN_MIN_M
                    or (np.isfinite(r_max) and r_max > ROUGHNESS_THRESHOLD))
                rows.append(row)
                if done % 100 == 0:
                    el = time.perf_counter() - t0
                    print(f"  {done}/{total}  {el:5.0f}s elapsed, "
                          f"~{el/done*(total-done):4.0f}s left")

    df = pd.DataFrame(rows)
    df.to_csv(GRID_CSV, index=False)
    n_fail = int(df["sim_failed"].sum())
    print(f"\nSaved {GRID_CSV}  ({len(df)} rows, {n_fail} flagged "
          f"({n_fail/len(df)*100:.1f}%))")

    ok = df[~df["sim_failed"].astype(bool)]
    for motif, sub in ok.groupby("motif"):
        dH = ((sub["H_fit_x0"] - sub["H_fit_y0"]) /
              (sub["H_fit_x0"].abs() + sub["H_fit_y0"].abs()))
        ad = ((sub["apex_k_y"] - sub["apex_k_x"]) /
              (sub["apex_k_y"].abs() + sub["apex_k_x"].abs()))
        print(f"  motif{motif}: crown {sub.crown_height.min()*1000:6.1f}-"
              f"{sub.crown_height.max()*1000:6.1f} mm   "
              f"section dH {dH.min():+.3f}..{dH.max():+.3f}   "
              f"apex dH {ad.min():+.3f}..{ad.max():+.3f}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--lo", type=float, default=SF_LO)
    ap.add_argument("--hi", type=float, default=SF_HI)
    args = ap.parse_args()
    run_grid(n=args.n, lo=args.lo, hi=args.hi)
