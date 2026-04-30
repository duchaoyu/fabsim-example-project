"""
Extend the FEA dataset with additional LHS samples per group.

New samples get IDs starting from ID_OFFSET (default 600), using a
different LHS seed so they fill different regions of parameter space.
After running FEA, curvature is computed and results are merged into
results_with_curvature.csv, then surrogates are retrained.

Usage:
    python3 extend_samples.py [--n-extra 150] [--jobs 4]
"""
import argparse, csv, json, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    DATA_DIR, MESH_PATH, MOTIFS, HAS_CABLE,
    PARAMS_NO_CABLE, PARAMS_CABLE, RANDOM_SEED, SCALAR_OUTPUTS,
)
from sampling import lhs
from fea_interface import run_fea, check_binary
from curvature import (read_off, compute_curvatures,
                       boundary_vertices, curvature_stats)
from surrogate import ScalarSurrogate
from sobol_analysis import run_all_sobol

# ── Curvature section setup ───────────────────────────────────────────────────
_rest_verts, _faces = read_off(MESH_PATH)
_bdry = boundary_vertices(_faces, len(_rest_verts))
_interior = np.array([i not in _bdry for i in range(len(_rest_verts))])
_R    = np.sqrt(_rest_verts[:,0]**2 + _rest_verts[:,1]**2).max()
_BAND = 0.08 * _R


def _section_curvature(verts_path):
    """Return (H_mean_x0, H_mean_y0) for a deformed mesh."""
    vdf   = pd.read_csv(verts_path).sort_values("vid")
    verts = vdf[["x","y","z"]].values
    curv  = compute_curvatures(verts, _faces)
    H     = np.abs(curv["H"])
    mask_x0 = (np.abs(verts[:,0]) < _BAND) & _interior & np.isfinite(H)
    mask_y0 = (np.abs(verts[:,1]) < _BAND) & _interior & np.isfinite(H)
    hx = float(np.mean(H[mask_x0])) if mask_x0.sum() > 0 else float("nan")
    hy = float(np.mean(H[mask_y0])) if mask_y0.sum() > 0 else float("nan")
    return hx, hy


def _run_one(sample):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")

    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        with open(prefix + "_scalars.csv") as f:
            row = next(csv.DictReader(f))
        return {**sample, **{k: float(v) for k, v in row.items()}}, True, "cached"

    try:
        result = run_fea(
            sf_wale            = sample["sf_wale"],
            sf_course          = sample["sf_course"],
            knit_dir_deg       = sample["knit_dir"],
            pressure           = sample["pressure"],
            motif              = sample["motif"],
            output_prefix      = prefix,
            cable_wale_lrest   = sample.get("cable_wale_lrest",   -1.0),
            cable_course_lrest = sample.get("cable_course_lrest", -1.0),
        )
        return {**sample, **result}, True, "ok"
    except Exception as e:
        return sample, False, f"FEA failed: {e}\n{traceback.format_exc()}"


def generate_extra_samples(n_extra, id_offset, seed_extra):
    samples = []
    sid = id_offset
    for gi, (motif, has_cable) in enumerate(
        [(m, c) for m in MOTIFS for c in HAS_CABLE]
    ):
        bounds = PARAMS_CABLE if has_cable else PARAMS_NO_CABLE
        df = lhs(n_extra, bounds, seed=seed_extra + gi * 1000)
        for _, row in df.iterrows():
            samples.append({
                "sample_id":   sid,
                "group":       f"motif{motif}_{'cable' if has_cable else 'nocable'}",
                "motif":       motif,
                "has_cable":   has_cable,
                "sf_wale":     float(row["sf_wale"]),
                "sf_course":   float(row["sf_course"]),
                "knit_dir":    float(row["knit_dir"]),
                "pressure":    float(row["pressure"]),
                "cable_wale_lrest":   float(row.get("cable_wale_lrest",   -1.0)),
                "cable_course_lrest": float(row.get("cable_course_lrest", -1.0)),
            })
            sid += 1
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-extra", type=int, default=150)
    parser.add_argument("--jobs",    type=int, default=4)
    args = parser.parse_args()

    check_binary()

    # Determine ID offset from existing data
    existing = os.path.join(DATA_DIR, "results_with_curvature.csv")
    fallback  = os.path.join(DATA_DIR, "results.csv")
    if os.path.exists(existing):
        df_exist = pd.read_csv(existing)
    elif os.path.exists(fallback):
        df_exist = pd.read_csv(fallback)
    else:
        df_exist = pd.DataFrame()
    id_offset = int(df_exist["sample_id"].max()) + 1 if len(df_exist) else 600
    seed_extra = RANDOM_SEED + 77777

    print(f"Generating {args.n_extra} extra samples per group "
          f"(IDs {id_offset}–{id_offset + 4*args.n_extra - 1}), "
          f"seed={seed_extra}")

    samples = generate_extra_samples(args.n_extra, id_offset, seed_extra)

    # ── Run FEA ──────────────────────────────────────────────────────────────
    print(f"\nRunning {len(samples)} FEA simulations ({args.jobs} workers)...")
    results, n_ok, n_fail = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_run_one, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                results.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] h={row.get('crown_height',0):.4f}m",
                          flush=True)
            else:
                n_fail += 1
                print(f"  [{row['sample_id']:05d}] FAILED: {info[:120]}", flush=True)

    print(f"\nFEA done: {n_ok} OK, {n_fail} failed.")
    df_new = pd.DataFrame(results)

    # ── Compute curvature ────────────────────────────────────────────────────
    print("\nComputing section curvature for new samples...")
    hx0, hy0 = [], []
    for _, row in df_new.iterrows():
        vp = os.path.join(DATA_DIR, f"{int(row['sample_id']):05d}_verts.csv")
        if os.path.exists(vp):
            hx, hy = _section_curvature(vp)
        else:
            hx, hy = float("nan"), float("nan")
        hx0.append(hx); hy0.append(hy)
    df_new["H_mean_x0"] = hx0
    df_new["H_mean_y0"] = hy0

    # ── Merge with existing ──────────────────────────────────────────────────
    df_all = pd.concat([df_exist, df_new], ignore_index=True)
    df_all.to_csv(existing, index=False)
    print(f"Merged → {existing}  ({len(df_all)} total rows)")

    # ── Retrain surrogates ───────────────────────────────────────────────────
    print("\nRetraining GP surrogates...")
    surrogates = {}
    for motif in MOTIFS:
        for has_cable in HAS_CABLE:
            group  = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
            subset = df_all[(df_all["motif"] == motif) &
                            (df_all["has_cable"] == has_cable)].dropna(
                                subset=[o for o in SCALAR_OUTPUTS if o in df_all.columns])
            print(f"  {group}: {len(subset)} samples")
            sur = ScalarSurrogate(has_cable=has_cable)
            metrics = sur.fit(subset)
            for col, m in metrics.items():
                print(f"    {col:30s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")
            path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
            sur.save(path)
            surrogates[group] = sur

    # ── Re-run Sobol ─────────────────────────────────────────────────────────
    print("\nRunning Sobol analysis...")
    sobol_results = run_all_sobol(surrogates)
    for group, group_results in sobol_results.items():
        for col, df_s in group_results.items():
            df_s.to_csv(os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv"))

    print("\nDone. Run  python3 plot_paper_figures.py  to regenerate figures.")


if __name__ == "__main__":
    main()
