"""
Validate the reconstructed src/stress_analysis.h against archived output.

The original header was never committed and was lost, so both batch drivers had
become unbuildable.  The replacement must reproduce the cached CSVs bit-for-bit
in every column, because max_stress and mean_stress are study outputs and every
surrogate and Sobol table on disk depends on them.

Re-runs no-cable material-r samples (untouched by the cable-geometry fix, so any
difference is down to the stress code alone) and diffs:
  - every column of <id>_stress.csv, per element
  - max_stress / mean_stress in <id>_scalars.csv
  - crown_height, as a check that the run itself reproduced

Usage:
    python3 check_stress_reconstruction.py [--n 6]
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR
from fea_interface import run_fea

STRESS_COLS = ["S11", "S22", "S12", "von_mises", "principal_1", "principal_2",
               "T_wale_Nm", "T_course_Nm"]
# Both files are written at 4 decimal places, so two independently rounded
# values straddling a .00005 boundary differ by a full unit in the last place.
# Anything at or below that is rounding, not a difference in the stress code;
# a real discrepancy shows up orders of magnitude above it (stresses are ~1e4).
TOL = 1.5e-4


def check(row, tmpdir):
    sid = int(row.sample_id)
    ref_stress = os.path.join(DATA_DIR, f"{sid:05d}_stress.csv")
    if not os.path.exists(ref_stress):
        return None

    prefix = os.path.join(tmpdir, f"{sid:05d}")
    got = run_fea(
        sf_wale=row.sf_wale, sf_course=row.sf_course,
        knit_dir_deg=row.knit_dir, pressure=row.pressure,
        motif=int(row.motif), output_prefix=prefix,
        cable_wale_lrest=-1.0, cable_course_lrest=-1.0,
        # the binary parameterises anisotropy as E1/E2, the study as E2/E1
        E1=row.E1, r=row.r_binary, nu=row.nu, timeout=600,
    )

    ref = pd.read_csv(ref_stress).sort_values("face").reset_index(drop=True)
    new = pd.read_csv(prefix + "_stress.csv").sort_values("face").reset_index(drop=True)
    if len(ref) != len(new):
        return dict(sid=sid, ok=False, note=f"{len(ref)} vs {len(new)} elements")

    worst = {c: float(np.nanmax(np.abs(ref[c].values - new[c].values)))
             for c in STRESS_COLS}
    res = dict(sid=sid, ok=max(worst.values()) <= TOL, **worst)
    res["d_max_stress"]  = abs(row.max_stress  - got["max_stress"])
    res["d_mean_stress"] = abs(row.mean_stress - got["mean_stress"])
    res["d_crown"]       = abs(row.crown_height - got["crown_height"])
    return res


def main(n):
    df = pd.read_csv(os.path.join(DATA_DIR,
                                  "material_r_nocable_section_metrics.csv"))
    df = df.dropna(subset=["max_stress", "crown_height"])
    # spread over the box rather than taking the first n rows
    df = df.sort_values("E1").iloc[np.linspace(0, len(df) - 1, n).astype(int)]

    tmpdir = tempfile.mkdtemp(prefix="stress_check_")
    try:
        rows = [r for r in (check(t, tmpdir) for t in df.itertuples())
                if r is not None]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    worst = out[STRESS_COLS].to_numpy().max()
    print(f"\nlargest absolute difference over {len(out)} samples x "
          f"{len(STRESS_COLS)} columns: {worst:.3e}   (tolerance {TOL:g})")
    print(f"largest max_stress difference:  {out.d_max_stress.max():.3e}")
    print(f"largest mean_stress difference: {out.d_mean_stress.max():.3e}")
    if out.ok.all() and worst <= TOL:
        print("\nPASS — the reconstructed header reproduces the archived stresses.")
        return 0
    print("\nFAIL — reconstruction does not match the cached output.")
    return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    sys.exit(main(ap.parse_args().n))
