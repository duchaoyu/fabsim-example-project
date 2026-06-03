"""
Main pipeline runner for sensitivity analysis.

Steps:
  1. generate  — run all FEA simulations (LHS sampling)
  2. train     — train GP surrogates (scalar + field)
  3. sobol     — Sobol sensitivity analysis via surrogate
  4. visualize — produce all figures

Usage:
  python3 run_pipeline.py [--steps generate train sobol visualize]
                          [--jobs 4]
                          [--no-cache]
"""

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MOTIFS, HAS_CABLE, SCALAR_OUTPUTS, MESH_PATH,
    PARAMS_MATERIAL_NO_CABLE, PARAMS_MATERIAL_CABLE,
    QUALITY_CROWN_MAX, QUALITY_STRESS_RATIO_MAX,
)
from sampling import generate_all_samples, generate_material_samples
from fea_interface import run_fea, check_binary
from surrogate import ScalarSurrogate
from sobol_analysis import run_all_sobol, run_material_sobol
import visualization as viz

os.makedirs(DATA_DIR, exist_ok=True)

# Load fixed cable path (same 23-vertex path used by run_lrest_sweep.py)
_CABLE_IDX_FILE = os.path.join(os.path.dirname(os.path.dirname(DATA_DIR)), "data", "cable_indices.txt")
_CABLE_INDICES = np.loadtxt(_CABLE_IDX_FILE, dtype=int).tolist() if os.path.exists(_CABLE_IDX_FILE) else None


# ── Step 1: Data generation ───────────────────────────────────────────────────

def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")
    # Skip if already done
    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        row = {"sample_id": sid}
        with open(prefix + "_scalars.csv") as f:
            for k, v in next(csv.DictReader(f)).items():
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = v
        return {**sample, **row}, True, "cached"

    cable_indices = None
    if sample["has_cable"]:
        if _CABLE_INDICES is None:
            return sample, False, f"cable_path failed: {_CABLE_IDX_FILE} not found"
        cable_indices = _CABLE_INDICES

    try:
        result = run_fea(
            sf_wale      = sample["sf_wale"],
            sf_course    = sample["sf_course"],
            knit_dir_deg = sample["knit_dir"],
            pressure     = sample["pressure"],
            motif        = sample["motif"],
            cable_indices= cable_indices,
            L_rest       = sample.get("L_rest"),
            output_prefix= prefix,
        )
        return {**sample, **result}, True, "ok"
    except Exception as e:
        return sample, False, f"FEA failed: {e}\n{traceback.format_exc()}"


def step_generate(jobs: int = 4):
    check_binary()
    samples = generate_all_samples()
    print(f"\n[Step 1] Generating {len(samples)} FEA simulations ({jobs} workers)...")

    results = []
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                results.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] OK  "
                          f"h={row.get('crown_height', 0):.4f}m", flush=True)
            else:
                n_fail += 1
                print(f"  [{row['sample_id']:05d}] FAILED: {info}", flush=True)

    df = pd.DataFrame(results)
    path = os.path.join(DATA_DIR, "results.csv")
    df.to_csv(path, index=False)
    print(f"\n  Done: {n_ok} OK, {n_fail} failed.  Results → {path}")
    return df


# ── Step 2: Train surrogates ──────────────────────────────────────────────────

CROWN_MIN_M = 0.02  # collapsed dome threshold — matches plot_section_sensitivity.py


def _check_quality(row: dict) -> tuple:
    """
    Returns (ok: bool, reason: str).

    Rejects:
      - NaN/Inf in any scalar output
      - crown_height < CROWN_MIN_M  → flat / dome didn't inflate
      - crown_height > QUALITY_CROWN_MAX → exploded simulation
      - max_stress / mean_stress > QUALITY_STRESS_RATIO_MAX → stress
        concentration, likely wrinkled or buckled state
    """
    import math
    for k in ("crown_height", "max_stress", "mean_stress"):
        v = row.get(k, float("nan"))
        if math.isnan(v) or math.isinf(v):
            return False, f"NaN/Inf in {k}"

    ch = row.get("crown_height", 0.0)
    if ch < CROWN_MIN_M:
        return False, f"flat (crown={ch:.4f} m)"
    if ch > QUALITY_CROWN_MAX:
        return False, f"exploded (crown={ch:.3f} m)"

    mx = row.get("max_stress", 0.0)
    mn = row.get("mean_stress", 1.0)
    if mn > 0 and mx / mn > QUALITY_STRESS_RATIO_MAX:
        return False, f"unsmooth — stress ratio {mx/mn:.1f} > {QUALITY_STRESS_RATIO_MAX}"

    return True, "ok"

def step_train(df= None):
    if df is None:
        enriched = os.path.join(DATA_DIR, "results_with_curvature.csv")
        path = enriched if os.path.exists(enriched) else os.path.join(DATA_DIR, "results.csv")
        if not os.path.exists(path):
            print("[Step 2] No results.csv found. Run generate first.")
            return {}
        df = pd.read_csv(path)

    # Merge section metrics (H_mean, sim_failed) from sections file if available
    sections_path = os.path.join(DATA_DIR, "results_with_sections.csv")
    if os.path.exists(sections_path):
        sec_cols = ["sample_id", "sim_failed"]
        df_sec_all = pd.read_csv(sections_path)
        for c in ["H_mean_x0", "H_mean_y0"]:
            if c in df_sec_all.columns:
                sec_cols.append(c)
        df_sec = df_sec_all[sec_cols]
        df = df.merge(df_sec, on="sample_id", how="left")
        df["sim_failed"] = df["sim_failed"].fillna(False)

    print(f"\n[Step 2] Training GP surrogates...")
    surrogates = {}
    for motif in MOTIFS:
        for has_cable in HAS_CABLE:
            group  = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
            subset = df[(df["motif"] == motif) &
                        (df["has_cable"] == has_cable)]
            # Filter failed simulations
            if "sim_failed" in subset.columns:
                n_before = len(subset)
                subset = subset[~subset["sim_failed"]]
                n_removed = n_before - len(subset)
                if n_removed:
                    print(f"  {group}: removed {n_removed} failed samples")
            subset = subset[subset["crown_height"] >= CROWN_MIN_M]
            # Only require non-null on columns that actually have data for this group
            available = [c for c in SCALAR_OUTPUTS
                         if c in subset.columns and subset[c].notna().any()]
            subset = subset.dropna(subset=available)
            if len(subset) < 10:
                print(f"  {group}: only {len(subset)} samples — skipping")
                continue
            print(f"  {group}: {len(subset)} samples")
            surrogate = ScalarSurrogate(has_cable=has_cable)
            metrics = surrogate.fit(subset)
            for col, m in metrics.items():
                print(f"    {col:30s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")
            save_path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
            surrogate.save(save_path)
            surrogates[group] = surrogate
            print(f"  Saved → {save_path}")
    return surrogates


# ── Step 3: Sobol analysis ────────────────────────────────────────────────────

def step_sobol(surrogates= None):
    if surrogates is None:
        surrogates = {}
        for motif in MOTIFS:
            for has_cable in HAS_CABLE:
                group = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
                path  = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
                if os.path.exists(path):
                    surrogates[group] = ScalarSurrogate.load(path)

    print(f"\n[Step 3] Sobol sensitivity analysis...")
    sobol_results = run_all_sobol(surrogates)

    # Save to CSV
    for group, group_results in sobol_results.items():
        for col, df in group_results.items():
            path = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
            df.to_csv(path)
    return sobol_results


# ── Step 4: Visualize ─────────────────────────────────────────────────────────

def step_visualize(surrogates= None,
                    sobol_results= None):
    print(f"\n[Step 4] Generating figures → {viz.FIG_DIR}")

    if surrogates is None:
        surrogates = {}
        for motif in MOTIFS:
            for has_cable in HAS_CABLE:
                group = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
                path  = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
                if os.path.exists(path):
                    surrogates[group] = ScalarSurrogate.load(path)

    if sobol_results is None:
        sobol_results = {}
        for motif in MOTIFS:
            for has_cable in HAS_CABLE:
                group = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
                group_results = {}
                for col in SCALAR_OUTPUTS:
                    path = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
                    if os.path.exists(path):
                        group_results[col] = pd.read_csv(path, index_col=0)
                if group_results:
                    sobol_results[group] = group_results

    # Fig 1+2: need motif1 no-cable surrogate
    key_m1 = "motif1_nocable"
    key_m2 = "motif2_nocable"
    if key_m1 in surrogates:
        viz.plot_response_surface(surrogates[key_m1])
        if key_m2 in surrogates:
            viz.plot_pressure_curve(surrogates[key_m1], surrogates[key_m2])
        else:
            print(f"  Skipping fig 2: {key_m2} surrogate not available")
    else:
        print(f"  Skipping fig 1+2: {key_m1} surrogate not available")

    # Fig 3: Sobol bar charts
    if sobol_results:
        viz.plot_sobol(sobol_results)
    else:
        print("  Skipping fig 3: no Sobol results")

    print("  Figures 4+5 (field comparisons) require field surrogate data.")
    print("  Run field surrogate training separately if needed.")


# ── Material sensitivity study ────────────────────────────────────────────────

def _run_material_one(sample: dict) -> tuple:
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

    cable_indices = None
    if sample["has_cable"]:
        if _CABLE_INDICES is None:
            return sample, False, f"cable_path failed: {_CABLE_IDX_FILE} not found"
        cable_indices = _CABLE_INDICES

    try:
        result = run_fea(
            sf_wale           = sample["sf_wale"],
            sf_course         = sample["sf_course"],
            knit_dir_deg      = sample["knit_dir"],
            pressure          = sample["pressure"],
            motif             = sample["motif"],
            cable_wale_lrest  = sample.get("cable_wale_lrest", -1.0),
            cable_course_lrest= sample.get("cable_course_lrest", -1.0),
            E1                = sample["E1"],
            r                 = sample["r"],
            nu                = sample["nu"],
            output_prefix     = prefix,
        )
    except Exception as e:
        return sample, False, f"FEA failed: {e}\n{traceback.format_exc()}"

    ok, reason = _check_quality(result)
    if not ok:
        # Remove output files for rejected samples to save disk space
        for ext in ("_scalars.csv", "_verts.csv", "_stress.csv"):
            p = prefix + ext
            if os.path.exists(p):
                os.unlink(p)
        return sample, False, f"quality rejected: {reason}"

    return {**sample, **result}, True, "ok"


def step_generate_material(jobs: int = 4) -> pd.DataFrame:
    check_binary()
    samples = generate_material_samples()
    print(f"\n[Material Step 1] Generating {len(samples)} FEA simulations "
          f"({jobs} workers)...")

    results = []
    n_ok = n_fail = n_quality = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_material_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                results.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] OK  "
                          f"h={row.get('crown_height', 0):.4f}m", flush=True)
            else:
                if "quality" in info:
                    n_quality += 1
                else:
                    n_fail += 1
                print(f"  [{row['sample_id']:05d}] SKIP: {info}", flush=True)

    df = pd.DataFrame(results)
    path = os.path.join(DATA_DIR, "material_results.csv")
    df.to_csv(path, index=False)
    print(f"\n  Done: {n_ok} OK, {n_quality} quality-rejected, "
          f"{n_fail} errors.  Results → {path}")
    return df


def step_train_material(df: pd.DataFrame = None) -> dict:
    if df is None:
        path = os.path.join(DATA_DIR, "material_results.csv")
        if not os.path.exists(path):
            print("[Material Step 2] No material_results.csv found.")
            return {}
        df = pd.read_csv(path)

    print(f"\n[Material Step 2] Training GP surrogates for material study...")
    surrogates = {}
    for has_cable in HAS_CABLE:
        group  = f"material_{'cable' if has_cable else 'nocable'}"
        bounds = PARAMS_MATERIAL_CABLE if has_cable else PARAMS_MATERIAL_NO_CABLE
        subset = df[df["group"] == group].copy()
        subset = subset[subset["crown_height"] >= CROWN_MIN_M]
        available = [c for c in SCALAR_OUTPUTS
                     if c in subset.columns and subset[c].notna().any()]
        subset = subset.dropna(subset=available)
        # Remove snap-through outliers in cable tension (discontinuous → GP can't fit)
        for tcol in ("cable_wale_tension", "cable_course_tension"):
            if tcol in subset.columns:
                p99 = subset[tcol].quantile(0.99)
                n_snap = (subset[tcol] > p99 * 2).sum()
                if n_snap:
                    subset = subset[subset[tcol] <= p99 * 2]
                    print(f"  {group}: removed {n_snap} snap-through outliers in {tcol} "
                          f"(threshold={p99*2:.1f} N)")
        if len(subset) < 20:
            print(f"  {group}: only {len(subset)} samples — skipping")
            continue
        print(f"  {group}: {len(subset)} samples, {len(bounds)}D input")
        surrogate = ScalarSurrogate(has_cable=has_cable, bounds=bounds)
        metrics = surrogate.fit(subset)
        for col, m in metrics.items():
            print(f"    {col:30s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")
        save_path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        surrogate.save(save_path)
        surrogates[group] = surrogate
        print(f"  Saved → {save_path}")
    return surrogates


def step_sobol_material(surrogates: dict = None) -> dict:
    if surrogates is None:
        surrogates = {}
        for has_cable in HAS_CABLE:
            group = f"material_{'cable' if has_cable else 'nocable'}"
            path  = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
            if os.path.exists(path):
                surrogates[group] = ScalarSurrogate.load(path)

    print(f"\n[Material Step 3] Sobol sensitivity analysis (material study)...")
    sobol_results = run_material_sobol(surrogates)
    for group, group_results in sobol_results.items():
        for col, df in group_results.items():
            path = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
            df.to_csv(path)
    return sobol_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis pipeline")
    parser.add_argument("--steps", nargs="+",
                        default=["generate", "train", "sobol", "visualize"],
                        choices=["generate", "train", "sobol", "visualize"])
    parser.add_argument("--material", action="store_true",
                        help="Run material sensitivity study instead of motif study")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Parallel FEA workers (step 1)")
    args = parser.parse_args()

    t0 = time.time()
    df          = None
    surrogates  = None
    sobol_res   = None

    if args.material:
        if "generate" in args.steps:
            df = step_generate_material(jobs=args.jobs)
        if "train" in args.steps:
            surrogates = step_train_material(df)
        if "sobol" in args.steps:
            sobol_res = step_sobol_material(surrogates)
    else:
        if "generate" in args.steps:
            df = step_generate(jobs=args.jobs)
        if "train" in args.steps:
            surrogates = step_train(df)
        if "sobol" in args.steps:
            sobol_res = step_sobol(surrogates)
        if "visualize" in args.steps:
            step_visualize(surrogates, sobol_res)

    print(f"\nPipeline completed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
