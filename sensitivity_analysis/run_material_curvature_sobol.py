"""
Add curvature outputs to the material Sobol study.

The material Sobol analysis (data/sobol_material_*.csv, figures/fig3_sobol_material_*)
was run on material_results.csv, which never carried curvature columns — so
crown height, stresses, boundary reaction and cable tensions were apportioned,
but shape itself was not.  The curvature that *was* analysed lived in a separate
no-cable-only script (plot_material_section_sobol.py, fig_material_section_sobol_*).

This script closes that gap for both material groups, on the same parameter
spaces the existing material figures use — nothing about the input design changes:

  material_nocable  7-D  sf_wale, sf_course, knit_dir, pressure, E1, r, nu
  material_cable    9-D  the above + cable_wale_lrest, cable_course_lrest

The existing surrogates (data/material_{group}_scalar_surrogate.pkl) are reused
untouched for the outputs they already carry — only the missing curvature
outputs get newly fitted GPs.  So crown height, stresses, boundary reaction and
cable tensions keep exactly the indices they had; the analysis gains columns
rather than being redone.

Curvature outputs added:
  H_mean_x0     mean curvature along the x=0 (wale) section      (m⁻¹)
  H_mean_y0     mean curvature along the y=0 (course) section    (m⁻¹)
  H_anisotropy  ΔH = (H_x0 - H_y0) / (H_x0 + H_y0)  signed, dimensionless

Curvature is recomputed from the per-sample verts/stress files already on disk
(474 no-cable + 455 cable samples) — no new FEA is required.  The estimator and
quality filter are the ones from plot_material_section_sobol.py, reused directly
so the numbers stay comparable to fig_material_section_sobol_ST.

Outputs:
  data/material_{nocable,cable}_section_metrics.csv    (enriched sample tables)
  data/sobol_material_{group}_{output}.csv             (all outputs, incl. curvature)
  figures/fig3_sobol_material_{nocable,cable}.{pdf,png}

Usage:
  python3 run_material_curvature_sobol.py            # cached metrics if present
  python3 run_material_curvature_sobol.py --force    # recompute curvature
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, SOBOL_N_BASE,
    PARAMS_MATERIAL_NO_CABLE, PARAMS_MATERIAL_CABLE,
)
from surrogate import ScalarSurrogate
# _section_metrics carries the curvature estimator (per-vertex cotangent H,
# interpolated at section crossings) and _profile_roughness for the quality
# filter.  Importing rather than copying keeps this consistent with
# fig_material_section_sobol_ST.
from plot_material_section_sobol import _section_metrics
import visualization

CURV_OUTPUTS = ["H_mean_x0", "H_mean_y0", "H_anisotropy"]

# Column order for fig3: crown height, the three curvature measures, then the
# stress / reaction outputs, then cable tension (cable group only).
BASE_OUTPUTS = ["crown_height"] + CURV_OUTPUTS + [
    "max_stress", "mean_stress", "boundary_reaction_mean",
]
CABLE_OUTPUTS = ["cable_wale_tension", "cable_course_tension"]

GROUPS = {
    "material_nocable": PARAMS_MATERIAL_NO_CABLE,
    "material_cable":   PARAMS_MATERIAL_CABLE,
}

# Quality filter (same thresholds as plot_material_section_sobol.build_section_df)
_CROWN_MIN     = 0.02   # m   — below this the dome did not inflate
_ROUGHNESS_MAX = 0.10   # normalised d²z/ds² RMS — above this the profile is noise


def _metrics_path(group):
    return os.path.join(DATA_DIR, f"{group}_section_metrics.csv")


def _outputs_for(group):
    return BASE_OUTPUTS + (CABLE_OUTPUTS if group == "material_cable" else [])


# ── Step 1: curvature metrics per sample ──────────────────────────────────────

def build_section_df(group, force=False):
    path = _metrics_path(group)
    if not force and os.path.exists(path):
        df = pd.read_csv(path)
        if all(c in df.columns for c in CURV_OUTPUTS):
            print(f"{group}: loaded cached curvature metrics from {path} "
                  f"({df['H_mean_x0'].notna().sum()}/{len(df)} valid)")
            return df
        print(f"{group}: cached file lacks curvature columns — recomputing")

    src = pd.read_csv(os.path.join(DATA_DIR, "material_results.csv"))
    sub = src[src["group"] == group].copy().reset_index(drop=True)
    print(f"{group}: computing curvature for {len(sub)} samples...")

    rows = []
    for i, row in sub.iterrows():
        rows.append(_section_metrics(int(row["sample_id"])))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sub)}")
    met = pd.DataFrame(rows)

    Hx, Hy = met["H_mean_x0"], met["H_mean_y0"]
    Hs = Hx + Hy
    met["H_anisotropy"] = np.where(Hs > 1e-6, (Hx - Hy) / Hs, np.nan)

    roughness = met[["r_x0", "r_y0"]].max(axis=1)
    failed = (sub["crown_height"] < _CROWN_MIN) | (roughness > _ROUGHNESS_MAX)
    met.loc[failed.values, CURV_OUTPUTS] = np.nan

    out = pd.concat([sub, met], axis=1)
    out.to_csv(path, index=False)
    n_ok = int(met["H_mean_x0"].notna().sum())
    print(f"{group}: saved {path}  ({n_ok}/{len(sub)} valid, "
          f"{int(failed.sum())} rejected by the quality filter)")
    return out


# ── Step 2: surrogates ────────────────────────────────────────────────────────

def load_base_surrogate(group, bounds):
    """The study's existing surrogate, reused as-is for the outputs it has."""
    path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
    sur = ScalarSurrogate.load(path)
    if sur.input_keys != list(bounds.keys()):
        raise RuntimeError(
            f"{group}: cached surrogate input keys {sur.input_keys} do not match "
            f"the configured parameter space {list(bounds.keys())} — refusing to "
            f"mix designs")
    print(f"\n{group}: reusing {os.path.basename(path)} for "
          f"{sorted(sur.gps)} ({len(sur.input_keys)}-D input)")
    return sur


def train_curvature_surrogate(group, df, bounds, force=False):
    """New GPs for the curvature outputs only — the base model is left alone."""
    path = os.path.join(DATA_DIR, f"{group}_curvature_surrogate.pkl")
    if not force and os.path.exists(path):
        print(f"{group}: loaded cached curvature surrogate")
        return ScalarSurrogate.load(path)

    keys = list(bounds.keys())
    valid = df.dropna(subset=keys + CURV_OUTPUTS)
    print(f"{group}: fitting curvature GPs on {len(valid)}/{len(df)} samples")

    sur = ScalarSurrogate(has_cable=(group == "material_cable"), bounds=bounds)
    metrics = sur.fit(valid, output_cols=CURV_OUTPUTS)
    for col in CURV_OUTPUTS:
        m = metrics.get(col)
        if m is not None:
            print(f"    {col:24s} R²={m['r2']:6.3f}  RMSE={m['rmse']:.4g}")
    sur.save(path)
    return sur


# ── Step 3: Sobol ─────────────────────────────────────────────────────────────

def run_sobol(group, base_sur, curv_sur, outputs, bounds):
    """One Saltelli design, both surrogates evaluated on it."""
    problem = {"num_vars": len(bounds),
               "names":    list(bounds.keys()),
               "bounds":   [list(v) for v in bounds.values()]}
    X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=False)

    preds = base_sur.predict(X)
    preds.update(curv_sur.predict(X))   # curvature keys are disjoint from base

    results = {}
    for col in outputs:
        if col not in preds:
            continue
        Y = preds[col]
        if np.std(Y) < 1e-10:
            continue    # e.g. cable tension in the no-cable group
        si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   print_to_console=False)
        df = pd.DataFrame({"S1": si["S1"], "ST": si["ST"],
                           "S1_conf": si["S1_conf"], "ST_conf": si["ST_conf"]},
                          index=problem["names"])
        results[col] = df
        df.to_csv(os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv"))
        st = df["ST"].clip(0)
        print(f"    {col:24s} top={st.idxmax():18s} ST={st.max():.3f}"
              f"{'   <- curvature (new)' if col in CURV_OUTPUTS else ''}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="recompute curvature metrics and refit the curvature GPs")
    args = ap.parse_args()

    # H_anisotropy is new to the shared label table used by fig3.
    visualization.OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")

    all_results = {}
    for group, bounds in GROUPS.items():
        df = build_section_df(group, force=args.force)
        base_sur = load_base_surrogate(group, bounds)
        curv_sur = train_curvature_surrogate(group, df, bounds, force=args.force)
        outputs = [c for c in _outputs_for(group)
                   if c in base_sur.gps or c in curv_sur.gps]
        print(f"  Sobol ({len(bounds)}-D, N={SOBOL_N_BASE}):")
        all_results[group] = run_sobol(group, base_sur, curv_sur, outputs, bounds)

    visualization.plot_sobol(all_results, save=True)

    print("\nCurvature ST by group:")
    for group, res in all_results.items():
        for col in CURV_OUTPUTS:
            if col not in res:
                continue
            st = res[col]["ST"].clip(0).sort_values(ascending=False)
            top = "  ".join(f"{p}={v:.2f}" for p, v in st.head(3).items())
            print(f"  {group:18s} {col:14s} {top}")
    return all_results


if __name__ == "__main__":
    main()
