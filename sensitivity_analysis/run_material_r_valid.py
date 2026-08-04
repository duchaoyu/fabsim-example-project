"""
Re-run the material-r Sobol study on the model-validity box.

Why: a membrane element carries no compression, so a converged run with
compressive principal stress is outside the formulation's validity, not merely
noisy.  Sampling 250 retained runs, the fraction of faces with sigma_2 < 0 is

    s_course 0.80-0.90   39%   (94% of runs affected)
    s_course 0.90-0.95   14%   (60%)
    s_course >= 1.00      3%   (8%)

with the same pattern in s_wale.  The lower quarter of the stretch-factor range
therefore contributed outputs the model cannot produce.  Restricting to
s_wale, s_course >= 0.95 removes that region.

Pressure is deliberately NOT restricted.  The model is valid at 200 Pa; what
degrades there is the section-curvature estimator on a barely-inflated dome.
Truncating an operating variable to compensate for a post-processing weakness
would discard valid runs (a soft fabric at 200 Pa has pR/E1 = 0.12).

No new FEA: 1095 (no cable) and 1024 (cable) curvature-valid runs already lie in
the restricted box, so the surrogates are simply refitted there.

Products:
    data/{group}_valid_surrogate.pkl
    data/sobol_{group}_valid_{output}.csv
and a printed before/after comparison of R2 and of the total-order indices.

Usage:
    python3 run_material_r_valid.py
"""

import os
import sys

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, SOBOL_N_BASE,
    PARAMS_MATERIAL_R_VALID_NO_CABLE, PARAMS_MATERIAL_R_VALID_CABLE,
)
from surrogate import ScalarSurrogate
from run_material_r_sobol import _outputs_for

GROUPS = {
    "material_r_nocable": PARAMS_MATERIAL_R_VALID_NO_CABLE,
    "material_r_cable":   PARAMS_MATERIAL_R_VALID_CABLE,
}


def in_box(df, bounds):
    m = np.ones(len(df), dtype=bool)
    for k, (lo, hi) in bounds.items():
        m &= (df[k] >= lo) & (df[k] <= hi)
    return m


def main():
    summary = []
    for group, bounds in GROUPS.items():
        df   = pd.read_csv(os.path.join(DATA_DIR, f"{group}_section_metrics.csv"))
        keys = list(bounds)
        outs = [c for c in _outputs_for(group) if c in df.columns]

        sub   = df[in_box(df, bounds)]
        valid = sub.dropna(subset=keys + outs)
        print(f"\n=== {group}: {len(valid)} runs in the validity box "
              f"(of {df.dropna(subset=keys+outs).shape[0]} in the full box)")

        sur = ScalarSurrogate(has_cable=group.endswith("_cable"), bounds=bounds)
        met = sur.fit(valid, output_cols=outs)
        sur.save(os.path.join(DATA_DIR, f"{group}_valid_surrogate.pkl"))

        problem = {"num_vars": len(keys), "names": keys,
                   "bounds": [list(v) for v in bounds.values()]}
        X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=False)
        preds = sur.predict(X)

        for col in outs:
            if col not in preds or np.std(preds[col]) < 1e-10:
                continue
            si = sobol_analyze.analyze(problem, preds[col],
                                       calc_second_order=False,
                                       print_to_console=False)
            new = pd.DataFrame({"S1": si["S1"], "ST": si["ST"],
                                "S1_conf": si["S1_conf"],
                                "ST_conf": si["ST_conf"]}, index=keys)
            new.to_csv(os.path.join(DATA_DIR, f"sobol_{group}_valid_{col}.csv"))

            old_path = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
            old = pd.read_csv(old_path, index_col=0) if os.path.exists(old_path) else None
            row = {"group": group, "output": col,
                   "R2_new": met.get(col, {}).get("r2", np.nan)}
            if old is not None:
                shared = [k for k in keys if k in old.index]
                d = (new.loc[shared, "ST"].clip(0) - old.loc[shared, "ST"].clip(0))
                row["ST_maxshift"] = d.abs().max()
                row["ST_shift_on"] = d.abs().idxmax()
                row["top_old"] = old["ST"].clip(0).idxmax()
                row["top_new"] = new["ST"].clip(0).idxmax()
            summary.append(row)

    s = pd.DataFrame(summary)
    # Held-out R2 of the full-box surrogates, for the comparison column
    old_r2 = {}
    for group in GROUPS:
        import pickle
        p = os.path.join(DATA_DIR, f"{group}_surrogate.pkl")
        if os.path.exists(p):
            sur = pickle.load(open(p, "rb"))
            for c, m in sur.metrics.items():
                old_r2[(group, c)] = m["r2"]
    s["R2_old"] = [old_r2.get((r.group, r.output), np.nan) for r in s.itertuples()]
    s["dR2"]    = s.R2_new - s.R2_old

    print("\n=== full box vs validity box ===")
    cols = ["group", "output", "R2_old", "R2_new", "dR2",
            "ST_maxshift", "ST_shift_on", "top_old", "top_new"]
    print(s[cols].to_string(index=False,
          formatters={"R2_old": "{:.3f}".format, "R2_new": "{:.3f}".format,
                      "dR2": "{:+.3f}".format, "ST_maxshift": "{:.3f}".format}))
    print(f"\nmean dR2 = {s.dR2.mean():+.3f}   "
          f"curvature outputs only = "
          f"{s[s.output.str.startswith(('H_',))].dR2.mean():+.3f}")
    print(f"top-ranked parameter changed for "
          f"{(s.top_old != s.top_new).sum()}/{len(s)} outputs")


if __name__ == "__main__":
    main()
