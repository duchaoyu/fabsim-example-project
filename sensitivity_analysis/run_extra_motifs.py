"""
Extend sensitivity analysis to motifs 3, 4, 5 (no-cable only).

Motif 3: E1=E2=5000      isotropic         E2/E1 = 1.00
Motif 4: E1=8000, E2=5000 wale-stiff        E1/E2 = 1.60
Motif 5: E1=12507, E2=5000 strongly wale-stiff  E1/E2 = 2.50

Steps:
  1. Generate LHS samples  (IDs starting from 2100)
  2. Run FEA in parallel
  3. Append scalar results to results.csv
  4. Compute section metrics and append to results_with_sections.csv
  5. Train GP surrogates
  6. Run Sobol analysis and save CSVs
"""

import csv
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, PARAMS_NO_CABLE, RANDOM_SEED, N_SAMPLES, SCALAR_OUTPUTS
from sampling import lhs
from fea_interface import run_fea, check_binary
from surrogate import ScalarSurrogate
from sobol_analysis import run_sobol_for_group

# Import section-metric helpers (no top-level GUI imports)
from plot_section_sensitivity import _section_metrics, CROWN_MIN_M, ROUGHNESS_THRESHOLD

os.makedirs(DATA_DIR, exist_ok=True)

NEW_MOTIFS   = [3, 4, 5]
NEXT_ID      = 2100   # first ID for new samples
JOBS         = 4


# ── 1. Sample generation ──────────────────────────────────────────────────────

def generate_new_samples():
    """LHS samples for motifs 3-5, nocable only. IDs start at NEXT_ID."""
    samples    = []
    current_id = NEXT_ID
    for i, motif in enumerate(NEW_MOTIFS):
        seed  = RANDOM_SEED + (4 + i) * 1000   # beyond existing 4 groups
        df    = lhs(N_SAMPLES, PARAMS_NO_CABLE, seed=seed)
        group = f"motif{motif}_nocable"
        for _, row in df.iterrows():
            samples.append({
                "sample_id":   current_id,
                "group":       group,
                "motif":       motif,
                "has_cable":   False,
                "sf_wale":     float(row["sf_wale"]),
                "sf_course":   float(row["sf_course"]),
                "knit_dir":    float(row["knit_dir"]),
                "pressure":    float(row["pressure"]),
                "cable_angle": 0.0,
            })
            current_id += 1
    return samples


# ── 2. FEA ────────────────────────────────────────────────────────────────────

def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")
    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        row = {"sample_id": sid}
        with open(prefix + "_scalars.csv") as f:
            row.update(next(csv.DictReader(f)))
        return {**sample, **row}, True, "cached"

    try:
        result = run_fea(
            sf_wale      = sample["sf_wale"],
            sf_course    = sample["sf_course"],
            knit_dir_deg = sample["knit_dir"],
            pressure     = sample["pressure"],
            motif        = sample["motif"],
            cable_indices= None,
            output_prefix= prefix,
        )
        return {**sample, **result}, True, "ok"
    except Exception as e:
        return sample, False, f"FEA failed: {e}\n{traceback.format_exc()}"


def step_generate(samples):
    check_binary()
    print(f"\n[1] Running {len(samples)} FEA simulations ({JOBS} workers)...")
    results, n_ok, n_fail = [], 0, 0
    with ProcessPoolExecutor(max_workers=JOBS) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                results.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] {row.get('group','')}  "
                          f"h={row.get('crown_height',0):.4f}m", flush=True)
            else:
                n_fail += 1
                print(f"  [{row['sample_id']:05d}] FAILED: {info}", flush=True)
    print(f"  Done: {n_ok} OK, {n_fail} failed.")
    return results


# ── 3. Append to results.csv ──────────────────────────────────────────────────

def step_append_results(new_rows):
    path       = os.path.join(DATA_DIR, "results.csv")
    df_exist   = pd.read_csv(path)
    existing   = set(df_exist["sample_id"])
    df_new     = pd.DataFrame(new_rows)
    df_new     = df_new[~df_new["sample_id"].isin(existing)]
    if df_new.empty:
        print("[3] No new rows — results.csv already up to date.")
        return df_exist
    df_out = pd.concat([df_exist, df_new], ignore_index=True)
    df_out.to_csv(path, index=False)
    print(f"[3] Appended {len(df_new)} rows to results.csv  (total {len(df_out)})")
    return df_out


# ── 4. Section metrics ────────────────────────────────────────────────────────

def step_section_metrics(samples):
    sec_path   = os.path.join(DATA_DIR, "results_with_sections.csv")
    df_sec     = pd.read_csv(sec_path) if os.path.exists(sec_path) else pd.DataFrame()
    done_ids   = set(df_sec["sample_id"].astype(int)) if len(df_sec) else set()
    results_df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))

    pending = [s for s in samples if s["sample_id"] not in done_ids]
    print(f"\n[4] Section metrics for {len(pending)} new samples...")

    new_rows = []
    for i, s in enumerate(pending):
        sid = s["sample_id"]
        m   = _section_metrics(sid)
        row = results_df[results_df["sample_id"] == sid]
        if row.empty:
            continue
        d = row.iloc[0].to_dict()
        d.update(m)
        r_max = max(m.get("r_x0") or 0.0, m.get("r_y0") or 0.0)
        d["sim_failed"] = (
            (d.get("crown_height", 0) < CROWN_MIN_M) or
            (not np.isnan(r_max) and r_max > ROUGHNESS_THRESHOLD)
        )
        new_rows.append(d)
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(pending)}...", flush=True)

    if new_rows:
        df_combined = pd.concat([df_sec, pd.DataFrame(new_rows)], ignore_index=True)
        df_combined.to_csv(sec_path, index=False)
        n_bad = sum(r.get("sim_failed", False) for r in new_rows)
        print(f"  Appended {len(new_rows)} rows  ({n_bad} flagged failed) → {sec_path}")
        return df_combined
    print("  Already up to date.")
    return df_sec


# ── 5. Train surrogates ───────────────────────────────────────────────────────

def step_train_new(df_sec):
    surrogates = {}
    for motif in NEW_MOTIFS:
        group  = f"motif{motif}_nocable"
        subset = df_sec[df_sec["group"] == group].copy()

        if "sim_failed" in subset.columns:
            n_before = len(subset)
            subset = subset[~subset["sim_failed"]]
            removed = n_before - len(subset)
            if removed:
                print(f"  {group}: removed {removed} failed samples")

        subset = subset[subset["crown_height"] >= CROWN_MIN_M]

        available = [c for c in SCALAR_OUTPUTS if c in subset.columns]
        subset = subset.dropna(subset=available)

        if len(subset) < 10:
            print(f"  {group}: only {len(subset)} valid samples — skipping")
            continue

        print(f"\n[5] {group}: training on {len(subset)} samples...")
        surr    = ScalarSurrogate(has_cable=False)
        metrics = surr.fit(subset)
        for col, m in metrics.items():
            print(f"    {col:30s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")
        path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        surr.save(path)
        surrogates[group] = surr
        print(f"  Saved → {path}")

    return surrogates


# ── 6. Sobol ─────────────────────────────────────────────────────────────────

def step_sobol_new(surrogates):
    for motif in NEW_MOTIFS:
        group = f"motif{motif}_nocable"
        if group not in surrogates:
            continue
        print(f"\n[6] Sobol for {group}...")
        results = run_sobol_for_group(surrogates[group], motif, False)
        for col, df_s in results.items():
            path = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
            df_s.to_csv(path)
            st = df_s["ST"].dropna()
            if len(st):
                top = st.idxmax()
                print(f"    {col}: top={top} (ST={st[top]:.3f})")
            else:
                print(f"    {col}: all-NaN ST (zero-variance output)")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0      = time.time()
    samples = generate_new_samples()
    print(f"New sample specs: {len(samples)}  "
          f"(motifs {NEW_MOTIFS}, IDs {samples[0]['sample_id']}–{samples[-1]['sample_id']})")

    fea_rows = step_generate(samples)
    step_append_results(fea_rows)
    df_sec = step_section_metrics(samples)
    surrogates = step_train_new(df_sec)
    step_sobol_new(surrogates)

    print(f"\nExtra-motifs pipeline complete in {time.time()-t0:.1f}s")
    print("Now run:  python3 plot_motif_comparison.py")
