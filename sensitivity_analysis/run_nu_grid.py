"""
Run FEA grids for Poisson's ratio (nu12) analysis.

Grid A (figS): E1 (10 values) × nu12 (8 values) at 3 fixed E2/E1 slices
               E2/E1 = 0.25, 0.50, 1.00  →  240 runs  (IDs 7000-7239)

Grid B (figT): E2/E1 (8 values) × nu12 (8 values) at fixed E1 = 7000 N/m
               64 runs  (IDs 7240-7303)

Fixed: sf=1.1, knit_dir=0°, pressure=1000 Pa, motif=1.

Usage:
  python sensitivity_analysis/run_nu_grid.py [--jobs 8]
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR
from fea_interface import run_fea, check_binary
from plot_material_section_sobol import _section_metrics

RESULTS_A_CSV  = os.path.join(DATA_DIR, "nu_grid_A_results.csv")
SECTIONS_A_CSV = os.path.join(DATA_DIR, "nu_grid_A_sections.csv")
RESULTS_B_CSV  = os.path.join(DATA_DIR, "nu_grid_B_results.csv")
SECTIONS_B_CSV = os.path.join(DATA_DIR, "nu_grid_B_sections.csv")

# Grid A
E1_VALUES   = np.array([1000, 2000, 3500, 5000, 7000, 10000, 13000, 16000, 18000, 20000], dtype=float)
E2_OVER_E1_SLICES = np.array([0.25, 0.50, 1.00])   # r = 4.0, 2.0, 1.0

# Grid B
E1_FIXED    = 7000.0
R_VALUES_B  = np.array([1.0, 1.25, 1.67, 2.0, 2.5, 3.0, 4.0, 5.0], dtype=float)

# Shared
NU12_VALUES = np.linspace(0.05, 0.40, 8)   # [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SF    = 1.1
KNIT  = 0.0
PRES  = 1000.0
MOTIF = 1

START_ID_A = 7000   # 240 runs → 7000-7239
START_ID_B = 7240   # 64 runs  → 7240-7303


def _build_samples_A():
    samples, sid = [], START_ID_A
    for e2r in E2_OVER_E1_SLICES:
        r = 1.0 / e2r
        for E1 in E1_VALUES:
            for nu in NU12_VALUES:
                samples.append(dict(
                    sample_id=sid, E1=float(E1), r=float(r),
                    E2=float(E1 * e2r), nu=float(nu),
                    sf_wale=SF, sf_course=SF, knit_dir=KNIT,
                    pressure=PRES, motif=MOTIF,
                    grid="A", e2_over_e1=float(e2r),
                ))
                sid += 1
    return samples


def _build_samples_B():
    samples, sid = [], START_ID_B
    for r in R_VALUES_B:
        for nu in NU12_VALUES:
            samples.append(dict(
                sample_id=sid, E1=E1_FIXED, r=float(r),
                E2=float(E1_FIXED / r), nu=float(nu),
                sf_wale=SF, sf_course=SF, knit_dir=KNIT,
                pressure=PRES, motif=MOTIF,
                grid="B", e2_over_e1=round(1.0 / r, 6),
            ))
            sid += 1
    return samples


def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")
    scalars_path = prefix + "_scalars.csv"
    verts_path   = prefix + "_verts.csv"
    if os.path.exists(scalars_path) and os.path.exists(verts_path):
        row = {"sample_id": sid}
        with open(scalars_path) as f:
            for k, v in next(csv.DictReader(f)).items():
                try:    row[k] = float(v)
                except: row[k] = v
        return {**sample, **row}, True, "cached"
    try:
        result = run_fea(
            sf_wale=sample["sf_wale"], sf_course=sample["sf_course"],
            knit_dir_deg=sample["knit_dir"], pressure=sample["pressure"],
            motif=sample["motif"], output_prefix=prefix,
            E1=sample["E1"], r=sample["r"], nu=sample["nu"],
        )
        return {**sample, **result}, True, "ok"
    except Exception as e:
        return {**sample, "crown_height": 0.0, "mean_stress": 0.0,
                "max_stress": 0.0, "boundary_reaction_mean": 0.0}, False, str(e)


def _run_grid(samples, jobs, label):
    rows = []
    done = failed = cached = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, msg = fut.result()
            rows.append(row)
            if msg == "cached": cached += 1
            elif ok:            done   += 1
            else:
                failed += 1
                print(f"  FAIL {label} id={row['sample_id']}  E1={row['E1']:.0f}  nu={row['nu']:.2f}: {msg}")
            total = done + failed + cached
            if total % 20 == 0:
                print(f"  {label}: {total}/{len(samples)}  done={done}  cached={cached}  failed={failed}")
    return pd.DataFrame(rows)


def _compute_sections(df, csv_path):
    valid = df[df["crown_height"] > 0.01]
    rows = []
    for sid in valid["sample_id"]:
        m = _section_metrics(int(sid))
        m["sample_id"] = int(sid)
        rows.append(m)
    sec = pd.DataFrame(rows)
    merged = df[["sample_id", "E1", "r", "E2", "nu", "e2_over_e1", "grid"]].merge(
        sec, on="sample_id", how="left"
    )
    merged.to_csv(csv_path, index=False)
    print(f"Saved sections → {csv_path}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",  type=int, default=4)
    parser.add_argument("--grids", default="A,B", help="which grids to run (A,B)")
    parser.add_argument("--steps", default="run,sections")
    args = parser.parse_args()
    grids = [g.strip().upper() for g in args.grids.split(",")]
    steps = [s.strip() for s in args.steps.split(",")]

    check_binary()

    if "A" in grids:
        sA = _build_samples_A()
        print(f"Grid A: {len(sA)} runs ({len(E2_OVER_E1_SLICES)} E2/E1 × {len(E1_VALUES)} E1 × {len(NU12_VALUES)} nu)")
        if "run" in steps:
            dfA = _run_grid(sA, args.jobs, "A")
            dfA.sort_values(["e2_over_e1", "E1", "nu"]).to_csv(RESULTS_A_CSV, index=False)
            print(f"Saved → {RESULTS_A_CSV}  (n={len(dfA)})")
        else:
            dfA = pd.read_csv(RESULTS_A_CSV) if os.path.exists(RESULTS_A_CSV) else None
        if "sections" in steps and dfA is not None:
            _compute_sections(dfA, SECTIONS_A_CSV)

    if "B" in grids:
        sB = _build_samples_B()
        print(f"Grid B: {len(sB)} runs ({len(R_VALUES_B)} E2/E1 × {len(NU12_VALUES)} nu, E1={E1_FIXED:.0f})")
        if "run" in steps:
            dfB = _run_grid(sB, args.jobs, "B")
            dfB.sort_values(["e2_over_e1", "nu"]).to_csv(RESULTS_B_CSV, index=False)
            print(f"Saved → {RESULTS_B_CSV}  (n={len(dfB)})")
        else:
            dfB = pd.read_csv(RESULTS_B_CSV) if os.path.exists(RESULTS_B_CSV) else None
        if "sections" in steps and dfB is not None:
            _compute_sections(dfB, SECTIONS_B_CSV)

    print("Done.")
