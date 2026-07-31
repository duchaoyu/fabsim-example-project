"""
Recompute section curvature for an existing sweep from its saved vertex files,
with both the binned estimator and the polynomial-fit estimator.

No FEA is re-run — this only re-derives the metric, so it is cheap and lets the
two estimators be compared on identical solutions.

Adds columns H_fit_x0 / H_fit_y0 (and H_binned_x0 / H_binned_y0 as a check that
the reproduction matches the stored H_mean_x0 / H_mean_y0).

Usage:
  python3 recompute_section_curvature.py data/knit_dir_sweep.csv
  python3 recompute_section_curvature.py data/uniform_sf_sweep.csv --x-col sf
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off, compute_curvatures
from plot_section_profiles import _slice_plane
from section_curvature import profile_curvature_fit, profile_curvature_binned

_V0, _FACES = read_off(MESH_PATH)


def _verts_path(row, sweep_dir):
    """Sweep CSVs may or may not carry verts_path; fall back to the naming
    convention used by the sweep runners."""
    p = row.get("verts_path")
    if isinstance(p, str) and os.path.exists(p):
        return p
    if "knit_dir" in row and sweep_dir and "knit" in sweep_dir:
        return os.path.join(sweep_dir,
                            f"m{int(row['motif'])}_th{row['knit_dir']:06.2f}_verts.csv")
    if "sf" in row and sweep_dir:
        return os.path.join(sweep_dir,
                            f"m{int(row['motif'])}_sf{row['sf']:.3f}_verts.csv")
    return None


def recompute(csv_path, sweep_dir=None, degree=6):
    df = pd.read_csv(csv_path)
    if sweep_dir is None:
        sweep_dir = os.path.splitext(csv_path)[0]

    out = {k: [] for k in ["H_fit_x0", "H_fit_y0",
                           "H_binned_x0", "H_binned_y0",
                           "fit_rms_resid_frac"]}
    for _, row in df.iterrows():
        vp = _verts_path(row, sweep_dir)
        if not (vp and os.path.exists(vp)):
            for k in out:
                out[k].append(np.nan)
            continue
        V = pd.read_csv(vp).sort_values("vid")[["x", "y", "z"]].values
        H = compute_curvatures(V, _FACES)["H"]
        vals, resid = {}, []
        for key, axis in (("x0", 0), ("y0", 1)):
            pos, z_mm, _ = _slice_plane(V, _FACES, H, fixed_axis=axis)
            if len(pos) < 6:
                vals[f"H_fit_{key}"] = vals[f"H_binned_{key}"] = np.nan
                continue
            k_fit, info = profile_curvature_fit(pos, z_mm, degree=degree,
                                                return_fit=True)
            vals[f"H_fit_{key}"] = k_fit
            vals[f"H_binned_{key}"] = profile_curvature_binned(pos, z_mm)
            if info and info["z_range_mm"] > 0:
                resid.append(info["rms_resid_mm"] / info["z_range_mm"])
        for k in ["H_fit_x0", "H_fit_y0", "H_binned_x0", "H_binned_y0"]:
            out[k].append(vals.get(k, np.nan))
        out["fit_rms_resid_frac"].append(np.mean(resid) if resid else np.nan)

    for k, v in out.items():
        df[k] = v
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")

    # ── report ────────────────────────────────────────────────────────────────
    ok = df[~df["sim_failed"].astype(bool)] if "sim_failed" in df else df
    if "H_mean_x0" in ok:
        d = (ok["H_binned_x0"] - ok["H_mean_x0"]).abs()
        print(f"  binned estimator reproduces stored H_mean_x0 to "
              f"{np.nanmax(d):.2e} m^-1 (should be ~0)")
    print(f"  polynomial fit: rms residual {np.nanmedian(ok['fit_rms_resid_frac'])*100:.2f}% "
          f"of profile height (median over runs)")

    xcol = "knit_dir" if "knit_dir" in ok else "sf"
    print(f"\n  stability along {xcol} (max step between adjacent runs, "
          f"as % of the mean):")
    for motif, sub in ok.groupby("motif"):
        sub = sub.sort_values(xcol)
        for label, cols in (("binned", ("H_binned_x0", "H_binned_y0")),
                            ("fit   ", ("H_fit_x0", "H_fit_y0"))):
            steps = []
            for c in cols:
                v = sub[c].values
                v = v[np.isfinite(v)]
                if len(v) > 2:
                    steps.append(np.max(np.abs(np.diff(v))) / np.mean(v) * 100)
            print(f"    motif{motif}  {label}  {max(steps):5.1f}%")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sweep-dir", default=None)
    ap.add_argument("--degree", type=int, default=6)
    args = ap.parse_args()
    recompute(args.csv, args.sweep_dir, args.degree)
