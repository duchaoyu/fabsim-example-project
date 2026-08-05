"""
Material Sobol study on the r = E2/E1 box, with curvature outputs.

Parameter box (config.PARAMS_MATERIAL_R_{NO_CABLE,CABLE}):

    sf_wale, sf_course   0.8 – 1.4
    knit_dir             0 – 90 deg
    pressure             200 – 2000 Pa
    E1                   1000 – 20000 N/m
    r = E2/E1            1 – 5            (course-stiffer; binary gets 1/r)
    nu12                 0.1 – 0.5
    cable_*_lrest        0.90 – 1.0       (cable group only)

Why new FEA rather than reusing the existing material data — checked, not
assumed:

  material_results.csv      0/929 samples in the box.  Its r is E1/E2 in 3-5,
                            so E2/E1 = 0.20-0.33, entirely outside r >= 1.
  material_ext_results.csv  223/499 in the box, but those sit at nu <= 0.3 and
                            p <= 1200, leaving half the nu range and 40% of the
                            p range unsampled; and E1/E2 are sampled
                            independently there, so r is not a box input.

Sobol needs a box design in the variables it apportions, hence a fresh LHS.

Outputs, in the paper's column order:
    crown_height, H_mean_x0, H_mean_y0, H_anisotropy,
    max_stress, mean_stress, boundary_reaction_mean,
    cable_wale_tension, cable_course_tension   (cable group)

with  delta H = (H_x0 - H_y0) / (H_x0 + H_y0).

Products:
    data/material_r_results.csv                   raw FEA scalars
    data/material_r_{group}_section_metrics.csv   + curvature
    data/sobol_material_r_{group}_{output}.csv
    figures/fig3_sobol_material_r_{group}{,_S1}.{pdf,png}

Usage:
    python3 run_material_r_sobol.py --jobs 16
    python3 run_material_r_sobol.py --steps sobol,plot     # reuse cached FEA
"""

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, SOBOL_N_BASE,
    PARAMS_MATERIAL_R_NO_CABLE, PARAMS_MATERIAL_R_CABLE,
    QUALITY_CROWN_MIN, QUALITY_STRESS_RATIO_MAX,
)
from fea_interface import run_fea, check_binary
from sampling import generate_material_r_samples
from surrogate import ScalarSurrogate
from plot_material_section_sobol import _section_metrics
import visualization

RESULTS_CSV = os.path.join(DATA_DIR, "material_r_results.csv")

GROUPS = {
    "material_r_nocable": PARAMS_MATERIAL_R_NO_CABLE,
    "material_r_cable":   PARAMS_MATERIAL_R_CABLE,
}

CURV_OUTPUTS  = ["H_mean_x0", "H_mean_y0", "H_anisotropy"]
BASE_OUTPUTS  = ["crown_height"] + CURV_OUTPUTS + [
    "max_stress", "mean_stress", "boundary_reaction_mean"]
CABLE_OUTPUTS = ["cable_wale_tension", "cable_course_tension"]

# Study-local crown ceiling: h/R <= 1 on the R = 0.6 m dome.  The global
# QUALITY_CROWN_MAX of 2.0 m is h/R = 3.3 — loose enough to admit balloon states
# that the membrane model does not describe.  Existing runs already reach
# h/R = 1.21, and p up to 2000 Pa makes those more likely, so cap at the
# hemisphere.
CROWN_MAX = 0.6
_R_DOME   = 0.6

# Curvature quality filter (thresholds from plot_material_section_sobol)
_ROUGHNESS_MAX = 0.10


def _metrics_path(group):
    return os.path.join(DATA_DIR, f"{group}_section_metrics.csv")


def _outputs_for(group):
    return BASE_OUTPUTS + (CABLE_OUTPUTS if group.endswith("_cable") else [])


# ── Step 1: FEA ───────────────────────────────────────────────────────────────

def _check_quality(res):
    h = res.get("crown_height", 0.0)
    if not np.isfinite(h) or h < QUALITY_CROWN_MIN:
        return False, f"crown_height={h:.5f} (uninflated / diverged)"
    if h > CROWN_MAX:
        return False, f"crown_height={h:.3f} (h/R={h/_R_DOME:.2f} > 1)"
    ms, mn = res.get("max_stress", 0.0), res.get("mean_stress", 1e-9)
    if mn > 0 and ms / mn > QUALITY_STRESS_RATIO_MAX:
        return False, f"stress_ratio={ms/mn:.1f}"
    return True, "ok"


def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")

    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        row = {"sample_id": sid}
        with open(prefix + "_scalars.csv") as f:
            for k, v in next(csv.DictReader(f)).items():
                try:    row[k] = float(v)
                except (TypeError, ValueError): row[k] = v
        return {**sample, **row}, True, "cached"

    try:
        res = run_fea(
            sf_wale            = sample["sf_wale"],
            sf_course          = sample["sf_course"],
            knit_dir_deg       = sample["knit_dir"],
            pressure           = sample["pressure"],
            motif              = sample["motif"],
            cable_wale_frac    = sample.get("cable_wale_frac"),
            cable_course_frac  = sample.get("cable_course_frac"),
            E1                 = sample["E1"],
            r                  = sample["r_binary"],   # binary wants E1/E2
            nu                 = sample["nu"],
            output_prefix      = prefix,
            timeout            = 600,
        )
    except Exception as exc:
        return sample, False, f"FEA failed: {exc}"

    ok, why = _check_quality(res)
    if not ok:
        for ext in ("_scalars.csv", "_verts.csv", "_stress.csv"):
            p = prefix + ext
            if os.path.exists(p):
                os.unlink(p)
        return sample, False, f"quality rejected: {why}"
    return {**sample, **res}, True, "ok"


def step_generate(jobs, n=None, start_id=3000, seed=None, append=False):
    check_binary()
    from config import RANDOM_SEED
    samples = generate_material_r_samples(
        start_id=start_id, seed=seed if seed is not None else RANDOM_SEED, n=n)
    print(f"[1] {len(samples)} FEA runs on {jobs} workers "
          f"(ids {samples[0]['sample_id']}-{samples[-1]['sample_id']})")

    rows, n_ok, n_fail, n_rej = [], 0, 0, 0
    reasons = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futs = [pool.submit(_run_one, s) for s in samples]
        for i, fut in enumerate(as_completed(futs), 1):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                rows.append(row)
            elif info.startswith("quality"):
                n_rej += 1
                reasons[info.split("=")[0]] = reasons.get(info.split("=")[0], 0) + 1
            else:
                n_fail += 1
                reasons["FEA failed"] = reasons.get("FEA failed", 0) + 1
            if i % 100 == 0:
                print(f"    {i}/{len(samples)}  ok={n_ok} rejected={n_rej} "
                      f"failed={n_fail}", flush=True)

    df = pd.DataFrame(rows).sort_values("sample_id")
    if append and os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        df = (pd.concat([prev, df], ignore_index=True)
                .drop_duplicates(subset="sample_id", keep="last")
                .sort_values("sample_id"))
        print(f"[1] appended to {len(prev)} existing rows -> {len(df)} total")
    df.to_csv(RESULTS_CSV, index=False)
    print(f"[1] converged {n_ok}/{len(samples)}  "
          f"(quality-rejected {n_rej}, solver-failed {n_fail})")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {v:4d}  {k}")
    print(f"[1] saved {RESULTS_CSV}")
    for g, sub in df.groupby("group"):
        print(f"      {g}: {len(sub)}")
    return df


# ── Step 2: curvature ─────────────────────────────────────────────────────────

def step_sections(force=False):
    src = pd.read_csv(RESULTS_CSV)
    out = {}
    for group in GROUPS:
        path = _metrics_path(group)
        if not force and os.path.exists(path):
            out[group] = pd.read_csv(path)
            print(f"[2] {group}: cached ({out[group]['H_mean_x0'].notna().sum()} valid)")
            continue

        sub = src[src["group"] == group].copy().reset_index(drop=True)
        print(f"[2] {group}: curvature for {len(sub)} samples")
        met = pd.DataFrame([_section_metrics(int(s)) for s in sub["sample_id"]])

        Hs = met["H_mean_x0"] + met["H_mean_y0"]
        met["H_anisotropy"] = np.where(
            Hs > 1e-6, (met["H_mean_x0"] - met["H_mean_y0"]) / Hs, np.nan)

        rough  = met[["r_x0", "r_y0"]].max(axis=1)
        failed = (sub["crown_height"] < 0.02) | (rough > _ROUGHNESS_MAX)
        met.loc[failed.values, CURV_OUTPUTS] = np.nan

        df = pd.concat([sub, met], axis=1)
        df.to_csv(path, index=False)
        print(f"[2] {group}: {met['H_mean_x0'].notna().sum()}/{len(sub)} valid "
              f"({int(failed.sum())} filtered) -> {path}")
        out[group] = df
    return out


# ── Step 3: surrogate + Sobol ─────────────────────────────────────────────────

def step_sobol(section_dfs):
    results = {}
    for group, bounds in GROUPS.items():
        df      = section_dfs[group]
        keys    = list(bounds.keys())
        outputs = [c for c in _outputs_for(group) if c in df.columns]
        valid   = df.dropna(subset=keys + outputs)
        print(f"\n[3] {group}: GPs on {len(valid)}/{len(df)} samples, "
              f"{len(keys)}-D input")

        sur = ScalarSurrogate(has_cable=group.endswith("_cable"), bounds=bounds)
        metrics = sur.fit(valid, output_cols=outputs)
        for col in outputs:
            m = metrics.get(col)
            if m:
                flag = "  <- LOW" if m["r2"] < 0.6 else ""
                print(f"      {col:24s} R2={m['r2']:6.3f}  RMSE={m['rmse']:.4g}{flag}")
        sur.save(os.path.join(DATA_DIR, f"{group}_surrogate.pkl"))

        problem = {"num_vars": len(bounds), "names": keys,
                   "bounds": [list(v) for v in bounds.values()]}
        X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=False)
        preds = sur.predict(X)

        gr = {}
        for col in outputs:
            if col not in preds or np.std(preds[col]) < 1e-10:
                continue
            si = sobol_analyze.analyze(problem, preds[col],
                                       calc_second_order=False,
                                       print_to_console=False)
            idx = pd.DataFrame({"S1": si["S1"], "ST": si["ST"],
                                "S1_conf": si["S1_conf"],
                                "ST_conf": si["ST_conf"]}, index=keys)
            gr[col] = idx
            idx.to_csv(os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv"))
            st = idx["ST"].clip(0)
            print(f"      {col:24s} top={st.idxmax():18s} ST={st.max():.3f}")
        results[group] = gr
    return results


# ── Step 4: figures ───────────────────────────────────────────────────────────

def step_plot(results):
    visualization.OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")
    visualization.GROUP_TITLES.setdefault(
        "material_r_nocable",
        "Material study ($r = E_2/E_1$) — no cable")
    visualization.GROUP_TITLES.setdefault(
        "material_r_cable",
        "Material study ($r = E_2/E_1$) — cable")
    visualization.plot_sobol(results, save=True, index="ST")
    visualization.plot_sobol(results, save=True, index="S1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--steps", default="generate,sections,sobol,plot")
    ap.add_argument("--force", action="store_true",
                    help="recompute cached curvature metrics")
    ap.add_argument("--extend", type=int, default=0,
                    help="run an extra LHS block of N samples per group and "
                         "append to material_r_results.csv")
    ap.add_argument("--extend-start-id", type=int, default=9000)
    ap.add_argument("--extend-seed", type=int, default=777)
    args = ap.parse_args()
    steps = [s.strip() for s in args.steps.split(",")]

    if "generate" in steps:
        step_generate(args.jobs)
    if args.extend:
        step_generate(args.jobs, n=args.extend, start_id=args.extend_start_id,
                      seed=args.extend_seed, append=True)
    section_dfs = step_sections(force=args.force) if "sections" in steps else None
    if "sobol" in steps:
        results = step_sobol(section_dfs)
        if "plot" in steps:
            step_plot(results)
        print("\nCurvature ST:")
        for g, gr in results.items():
            for col in CURV_OUTPUTS:
                if col in gr:
                    st = gr[col]["ST"].clip(0).sort_values(ascending=False)
                    print(f"  {g:20s} {col:14s} " +
                          "  ".join(f"{p}={v:.2f}" for p, v in st.head(3).items()))


if __name__ == "__main__":
    main()
