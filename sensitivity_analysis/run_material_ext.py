"""
Extended material study — no cable.

Generates FEA data over E1 × E2 space (1–20 kN/m each) plus nu 0.09–0.3,
covering wale-stiffer, course-stiffer, and isotropic regimes.

Steps:
  1. generate   — LHS sampling + parallel FEA (saves _scalars/_verts/_stress)
  2. sections   — compute section metrics from verts/stress files
  3. train      — train scalar + section GP surrogates
  4. plot       — generate figP_material_e1e2_surface

Usage:
  python run_material_ext.py [--jobs N] [--steps generate,sections,train,plot]
"""

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, PARAMS_MATERIAL_EXT_NO_CABLE,
    QUALITY_CROWN_MAX, QUALITY_STRESS_RATIO_MAX,
)
from fea_interface import run_fea, check_binary
from sampling import generate_material_ext_samples
from surrogate import ScalarSurrogate
from plot_material_section_sobol import _section_metrics, SECTION_OUTPUTS

EXT_RESULTS_CSV      = os.path.join(DATA_DIR, "material_ext_results.csv")
EXT_SECTION_CSV      = os.path.join(DATA_DIR, "material_ext_section_metrics.csv")
EXT_SCALAR_SUR_PATH  = os.path.join(DATA_DIR, "material_ext_nocable_scalar_surrogate.pkl")
EXT_SECTION_SUR_PATH = os.path.join(DATA_DIR, "material_ext_nocable_section_surrogate.pkl")

SCALAR_OUTPUTS_EXT = ["crown_height", "max_stress", "mean_stress", "boundary_reaction_mean"]


# ── quality filter ────────────────────────────────────────────────────────────

def _check_quality(result: dict):
    h = result.get("crown_height", 0)
    if h <= 0 or h > QUALITY_CROWN_MAX:
        return False, f"crown_height={h:.3f}"
    ms = result.get("max_stress", 0)
    mn = result.get("mean_stress", 1e-9)
    if mn > 0 and ms / mn > QUALITY_STRESS_RATIO_MAX:
        return False, f"stress_ratio={ms/mn:.1f}"
    return True, "ok"


# ── single FEA run (top-level for pickling) ───────────────────────────────────

def _run_one(sample: dict) -> tuple:
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")

    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        row = {"sample_id": sid}
        with open(prefix + "_scalars.csv") as f:
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
            E1           = sample["E1"],
            r            = sample["r"],   # E1/E2, derived in sampling
            nu           = sample["nu"],
            output_prefix= prefix,
        )
    except Exception as e:
        return sample, False, f"FEA failed: {e}\n{traceback.format_exc()}"

    ok, reason = _check_quality(result)
    if not ok:
        for ext in ("_scalars.csv", "_verts.csv", "_stress.csv"):
            p = prefix + ext
            if os.path.exists(p):
                os.unlink(p)
        return sample, False, f"quality rejected: {reason}"

    return {**sample, **result}, True, "ok"


# ── step 1: generate ─────────────────────────────────────────────────────────

def step_generate(jobs: int = 4) -> pd.DataFrame:
    check_binary()
    samples = generate_material_ext_samples()
    print(f"\n[Step 1] Generating {len(samples)} FEA runs ({jobs} workers)...")

    rows = []
    n_ok = n_fail = n_quality = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                rows.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] OK  "
                          f"h={row.get('crown_height', 0):.4f} m  "
                          f"E1={row.get('E1',0):.0f}  E2={row.get('E2',0):.0f}",
                          flush=True)
            else:
                if "quality" in info:
                    n_quality += 1
                else:
                    n_fail += 1
                print(f"  [{row['sample_id']:05d}] SKIP: {info}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(EXT_RESULTS_CSV, index=False)
    print(f"\n  Done: {n_ok} OK, {n_quality} quality-rejected, {n_fail} errors")
    print(f"  Results → {EXT_RESULTS_CSV}")
    return df


# ── step 2: section metrics ───────────────────────────────────────────────────

def step_sections(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(EXT_RESULTS_CSV)

    print(f"\n[Step 2] Computing section metrics for {len(df)} samples...")
    rows = []
    for i, row in df.iterrows():
        m = _section_metrics(int(row["sample_id"]))
        rows.append(m)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(df)} done")

    metrics_df = pd.DataFrame(rows)
    Hx = metrics_df["H_mean_x0"]
    Hy = metrics_df["H_mean_y0"]
    Hm = 0.5 * (Hx + Hy)
    metrics_df["H_anisotropy"] = np.where(Hm > 1e-6, (Hx - Hy) / Hm, np.nan)

    r_max  = metrics_df[["r_x0", "r_y0"]].max(axis=1)
    failed = (df["crown_height"].values < 0.02) | (r_max.values > 0.10)
    metrics_df.loc[failed, SECTION_OUTPUTS] = np.nan

    enriched = pd.concat([df.reset_index(drop=True),
                           metrics_df.reset_index(drop=True)], axis=1)
    enriched.to_csv(EXT_SECTION_CSV, index=False)
    n_ok = (~metrics_df["H_mean_x0"].isna()).sum()
    print(f"  Saved → {EXT_SECTION_CSV}  ({n_ok}/{len(df)} valid)")
    return enriched


# ── step 3: train surrogates ──────────────────────────────────────────────────

def step_train(df: pd.DataFrame = None):
    if df is None:
        df = pd.read_csv(EXT_SECTION_CSV)

    print(f"\n[Step 3a] Training scalar surrogate on {len(df)} samples...")
    valid_sc = df.dropna(subset=["crown_height"]).copy()
    sur_sc = ScalarSurrogate(has_cable=False, bounds=PARAMS_MATERIAL_EXT_NO_CABLE)
    for col, m in sur_sc.fit(valid_sc, output_cols=SCALAR_OUTPUTS_EXT).items():
        print(f"  {col}: R²={m['r2']:.3f}  RMSE={m['rmse']:.4g}")
    sur_sc.save(EXT_SCALAR_SUR_PATH)
    print(f"  Saved → {EXT_SCALAR_SUR_PATH}")

    print(f"\n[Step 3b] Training section surrogate...")
    valid_sec = df.dropna(subset=SECTION_OUTPUTS).copy()
    sur_sec = ScalarSurrogate(has_cable=False, bounds=PARAMS_MATERIAL_EXT_NO_CABLE)
    for col, m in sur_sec.fit(valid_sec, output_cols=SECTION_OUTPUTS).items():
        print(f"  {col}: R²={m['r2']:.3f}  RMSE={m['rmse']:.4g}")
    sur_sec.save(EXT_SECTION_SUR_PATH)
    print(f"  Saved → {EXT_SECTION_SUR_PATH}")

    return sur_sc, sur_sec


# ── step 4: plot ──────────────────────────────────────────────────────────────

def step_plot():
    print(f"\n[Step 4] Generating E1×E2 surface figure...")
    import plot_material_e1e2_surface as fig_mod
    fig_mod.plot_surface(save=True)
    print("  Done.")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs",  type=int, default=4)
    parser.add_argument("--steps", type=str,
                        default="generate,sections,train,plot")
    args = parser.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    df = None
    if "generate" in steps:
        df = step_generate(jobs=args.jobs)
    if "sections" in steps:
        df = step_sections(df)
    if "train" in steps:
        step_train(df)
    if "plot" in steps:
        step_plot()
    print("\nAll done.")
