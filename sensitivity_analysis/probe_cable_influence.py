"""
Probe: does the corrected cable actually influence the dome shape?

The zigzag cable path (cable_path.py, fixed) gave L_rest a Sobol index of
~0.00-0.03 on every shape metric while its own tension sat at ~0.50 — a cable
that could absorb any rest-length change by wiggling sideways.  This sweeps the
new absolute rest length over its range at otherwise fixed parameters and
reports the shape response, alongside a no-cable reference.

Usage:
    python3 probe_cable_influence.py [--n 9] [--jobs 8]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DATA_DIR, MESH_PATH
from cable_path import (generate_cable_path, cable_path_length, load_off,
                        WALE_CABLE_ANGLE, COURSE_CABLE_ANGLE)
from fea_interface import run_fea
from plot_material_section_sobol import _section_metrics

# Box centre of the validity box, so the probe sits where the study lives
BASE = {k: 0.5 * (lo + hi)
        for k, (lo, hi) in config.PARAMS_MATERIAL_R_VALID_CABLE.items()}

PROBE_ID0 = 90000     # far from any study's sample ids


def one(sid, lrest_wale, lrest_course):
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")
    out = run_fea(
        sf_wale=BASE["sf_wale"], sf_course=BASE["sf_course"],
        knit_dir_deg=BASE["knit_dir"], pressure=BASE["pressure"],
        motif=1, output_prefix=prefix,
        cable_wale_lrest=lrest_wale, cable_course_lrest=lrest_course,
        E1=BASE["E1"], r=BASE["r"], nu=BASE["nu"], timeout=600,
    )
    out.update(_section_metrics(sid))
    out["H_anisotropy"] = out.get("H_mean_x0", np.nan) - out.get("H_mean_y0", np.nan)
    return out


def main(n, jobs):
    V, _ = load_off(MESH_PATH)
    for nm, ang in (("wale", WALE_CABLE_ANGLE), ("course", COURSE_CABLE_ANGLE)):
        idx = generate_cable_path(ang, MESH_PATH)
        print(f"{nm:7s} cable: {len(idx)} nodes, flat arc length "
              f"{cable_path_length(idx, V):.4f} m")
    lo, hi = config.PARAMS_MATERIAL_R_VALID_CABLE["cable_wale_lrest"]
    print(f"sweeping L_rest over ({lo}, {hi}) m at the validity-box centre:")
    print({k: round(v, 4) for k, v in BASE.items() if "lrest" not in k})

    jobs_list = [(PROBE_ID0, -1.0, -1.0)]          # no-cable reference
    grid = np.linspace(lo, hi, n)
    jobs_list += [(PROBE_ID0 + 1 + i, L, L) for i, L in enumerate(grid)]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(one, sid, a, b): (sid, a) for sid, a, b in jobs_list}
        for f in as_completed(fut):
            sid, a = fut[f]
            try:
                rows[sid] = dict(f.result(), L_rest=a)
            except Exception as e:
                print(f"  sid {sid} failed: {e}")

    COLS = ["L_rest", "crown_height", "H_mean_x0", "H_mean_y0", "H_anisotropy",
            "max_stress", "mean_stress", "cable_wale_tension",
            "cable_course_tension"]
    df = pd.DataFrame([rows[k] for k in sorted(rows)])[COLS]
    df.to_csv(os.path.join(DATA_DIR, "probe_cable_influence.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    ref = df[df.L_rest < 0]
    swept = df[df.L_rest > 0]
    print("\nresponse to L_rest across its range (cable rows only):")
    for c in COLS[1:6]:
        v = swept[c].values
        rng = np.nanmax(v) - np.nanmin(v)
        mid = np.nanmedian(v)
        base = ref[c].values[0] if len(ref) else np.nan
        print(f"  {c:16s} range {rng:10.4g}  = {100*rng/abs(mid):6.1f}% of its "
              f"median   |  no-cable {base:10.4g} -> cable {mid:10.4g} "
              f"({100*(mid-base)/abs(base):+.1f}%)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="L_rest grid points")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    main(args.n, args.jobs)
