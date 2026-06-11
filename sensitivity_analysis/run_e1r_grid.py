"""
Run FEA grid: E1 (1000–20000 N/m) × r = E1/E2 (1–5) at fixed conditions.

Fixed: sf_wale=sf_course=1.0, knit_dir=0°, pressure=1000 Pa, nu=0.195, motif=1.
Grid: 10 E1 × 8 r values = 80 FEA runs.

Outputs:
  data/e1r_grid_results.csv   — scalar outputs (crown_height, mean_stress, …)
  data/e1r_grid_sections.csv  — section metrics (H_mean_x0/y0, vm_x0/y0)

Usage (on server):
  python sensitivity_analysis/run_e1r_grid.py [--jobs 8]
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

RESULTS_CSV  = os.path.join(DATA_DIR, "e1r_grid_results.csv")
SECTIONS_CSV = os.path.join(DATA_DIR, "e1r_grid_sections.csv")

# Grid parameters
E1_VALUES = np.array([1000, 2000, 3500, 5000, 7000, 10000, 13000, 16000, 18000, 20000],
                     dtype=float)
R_VALUES  = np.array([1.0, 1.25, 1.67, 2.0, 2.5, 3.0, 4.0, 5.0],
                     dtype=float)   # r = E1/E2

SF    = 1.0
KNIT  = 0.0
PRES  = 1000.0
NU    = 0.195
MOTIF = 1

START_ID = 5000   # offset to avoid conflicts with LHS sample IDs


def _build_samples():
    samples = []
    sid = START_ID
    for E1 in E1_VALUES:
        for r in R_VALUES:
            samples.append({
                "sample_id": sid,
                "E1":        float(E1),
                "r":         float(r),
                "E2":        float(E1 / r),
                "nu":        NU,
                "sf_wale":   SF,
                "sf_course": SF,
                "knit_dir":  KNIT,
                "pressure":  PRES,
                "motif":     MOTIF,
            })
            sid += 1
    return samples


def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")

    # Re-use cached result if both files exist
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
            sf_wale      = sample["sf_wale"],
            sf_course    = sample["sf_course"],
            knit_dir_deg = sample["knit_dir"],
            pressure     = sample["pressure"],
            motif        = sample["motif"],
            output_prefix= prefix,
            E1           = sample["E1"],
            r            = sample["r"],
            nu           = sample["nu"],
        )
        return {**sample, **result}, True, "ok"
    except Exception as e:
        return {**sample, "crown_height": 0.0, "mean_stress": 0.0,
                "max_stress": 0.0, "boundary_reaction_mean": 0.0}, False, str(e)


def run_grid(jobs: int = 4):
    check_binary()
    samples = _build_samples()
    print(f"Grid: {len(samples)} runs  ({len(E1_VALUES)} E1 × {len(R_VALUES)} r)")

    rows = []
    done = failed = cached = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, msg = fut.result()
            rows.append(row)
            if msg == "cached":
                cached += 1
            elif ok:
                done += 1
            else:
                failed += 1
                print(f"  FAIL  id={row['sample_id']}  E1={row['E1']:.0f}  r={row['r']:.2f}: {msg}")
            total = done + failed + cached
            if total % 10 == 0:
                print(f"  {total}/{len(samples)}  done={done}  cached={cached}  failed={failed}")

    df = pd.DataFrame(rows).sort_values(["E1", "r"])
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved scalar results → {RESULTS_CSV}  (n={len(df)}, failed={failed})")
    return df


def compute_sections(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(RESULTS_CSV)
    valid = df[df["crown_height"] > 0.01]
    rows = []
    for sid in valid["sample_id"]:
        m = _section_metrics(int(sid))
        m["sample_id"] = int(sid)
        rows.append(m)
    sec = pd.DataFrame(rows)
    merged = df[["sample_id", "E1", "r", "E2"]].merge(sec, on="sample_id", how="left")
    merged.to_csv(SECTIONS_CSV, index=False)
    print(f"Saved section metrics → {SECTIONS_CSV}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--steps", default="run,sections")
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    df = None
    if "run" in steps:
        df = run_grid(jobs=args.jobs)
    if "sections" in steps:
        compute_sections(df)
    print("Done.")
