"""
Latin Hypercube Sampling for the four experiment groups.

Groups:
  (motif=1, cable=False) — 4D LHS: sf_wale, sf_course, knit_dir, pressure
  (motif=1, cable=True)  — 5D LHS: + cable_angle
  (motif=2, cable=False) — 4D LHS
  (motif=2, cable=True)  — 5D LHS

Returns a list of parameter dicts ready for fea_interface.run_fea().
"""

import numpy as np
import pandas as pd
from config import (
    PARAMS_NO_CABLE, PARAMS_CABLE, PARAMS_CABLE_ORIENT,
    MOTIFS, HAS_CABLE, CABLE_AXES, N_SAMPLES, N_SAMPLES_ORIENT, RANDOM_SEED,
)


def lhs(n: int, bounds: dict, seed: int) -> pd.DataFrame:
    """Manual LHS: returns DataFrame with one row per sample."""
    rng   = np.random.default_rng(seed)
    d     = len(bounds)
    keys  = list(bounds.keys())
    cuts  = np.linspace(0, 1, n + 1)
    unit  = np.zeros((n, d))
    for j in range(d):
        pts = rng.uniform(cuts[:-1], cuts[1:])
        unit[:, j] = rng.permutation(pts)

    rows = {}
    for j, k in enumerate(keys):
        lo, hi = bounds[k]
        rows[k] = lo + unit[:, j] * (hi - lo)
    return pd.DataFrame(rows)


def generate_all_samples(seed=RANDOM_SEED):
    """
    Generate all samples for all 4 groups.
    Each sample dict has keys matching fea_interface.run_fea() arguments
    plus 'motif', 'has_cable', 'group'.
    """
    all_samples = []
    sample_id   = 0
    for gi, (motif, has_cable) in enumerate(
        [(m, c) for m in MOTIFS for c in HAS_CABLE]
    ):
        bounds = PARAMS_CABLE if has_cable else PARAMS_NO_CABLE
        # Use a deterministic but distinct seed per group
        df = lhs(N_SAMPLES, bounds, seed=seed + gi * 1000)

        for _, row in df.iterrows():
            s = {
                "sample_id": sample_id,
                "group":     f"motif{motif}_{'cable' if has_cable else 'nocable'}",
                "motif":     motif,
                "has_cable": has_cable,
                "sf_wale":   float(row["sf_wale"]),
                "sf_course": float(row["sf_course"]),
                "knit_dir":  float(row["knit_dir"]),
                "pressure":  float(row["pressure"]),
                "L_rest":    float(row.get("L_rest", 1.325)),
            }
            all_samples.append(s)
            sample_id += 1

    return all_samples


def samples_for_group(motif, has_cable, seed=RANDOM_SEED):
    return [s for s in generate_all_samples(seed)
            if s["motif"] == motif and s["has_cable"] == has_cable]


def generate_orient_samples(start_id: int = 600, seed: int = RANDOM_SEED) -> list:
    """
    Generate LHS samples for the cable-orientation study.

    Groups:  motif{1,2}_cable_{wale,course}  — 4 groups × N_SAMPLES_ORIENT
    L_rest is a dimensionless ratio; the actual cable rest length is computed
    at run time as  L_rest_actual = L_rest_ratio × L_flat(knit_dir, axis).

    Sample IDs start at start_id (default 600) to avoid collision with the
    original 0–599 sample set.
    """
    all_samples = []
    sid = start_id
    groups = [(m, ax) for m in MOTIFS for ax in CABLE_AXES]
    for gi, (motif, cable_axis) in enumerate(groups):
        df = lhs(N_SAMPLES_ORIENT, PARAMS_CABLE_ORIENT,
                 seed=seed + (gi + 20) * 1000)
        for _, row in df.iterrows():
            s = {
                "sample_id":  sid,
                "group":      f"motif{motif}_cable_{cable_axis}",
                "motif":      motif,
                "has_cable":  True,
                "cable_axis": cable_axis,
                "sf_wale":    float(row["sf_wale"]),
                "sf_course":  float(row["sf_course"]),
                "knit_dir":   float(row["knit_dir"]),
                "pressure":   float(row["pressure"]),
                "L_rest":     float(row["L_rest"]),   # ratio, not metres
            }
            all_samples.append(s)
            sid += 1
    return all_samples


if __name__ == "__main__":
    samples = generate_all_samples()
    print(f"Total samples: {len(samples)}")
    for group in set(s["group"] for s in samples):
        n = sum(1 for s in samples if s["group"] == group)
        print(f"  {group}: {n}")

    print()
    orient = generate_orient_samples()
    print(f"Orient samples: {len(orient)}")
    for group in set(s["group"] for s in orient):
        n = sum(1 for s in orient if s["group"] == group)
        print(f"  {group}: {n}")
