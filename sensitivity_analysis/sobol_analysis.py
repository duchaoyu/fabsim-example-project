"""
Sobol sensitivity analysis using SALib + GP surrogates.

For each group and each scalar output, computes S1 (first-order) and
ST (total-order) Sobol indices via Saltelli sampling through the GP.
"""

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol

from config import (
    PARAMS_NO_CABLE, PARAMS_CABLE, SCALAR_OUTPUTS,
    PARAMS_MATERIAL_NO_CABLE, PARAMS_MATERIAL_CABLE,
    MOTIFS, HAS_CABLE, SOBOL_N_BASE, RANDOM_SEED,
)
from surrogate import ScalarSurrogate


def _salib_problem(has_cable: bool, bounds: dict = None) -> dict:
    if bounds is None:
        bounds = PARAMS_CABLE if has_cable else PARAMS_NO_CABLE
    return {
        "num_vars": len(bounds),
        "names":    list(bounds.keys()),
        "bounds":   [list(v) for v in bounds.values()],
    }


def run_sobol_for_group(
    surrogate: ScalarSurrogate,
    motif: int,
    has_cable: bool,
    n_base: int = SOBOL_N_BASE,
    bounds: dict = None,
):
    """
    Returns {output_name: DataFrame(index=param_names, columns=[S1, ST, S1_conf, ST_conf])}.
    """
    problem = _salib_problem(has_cable, bounds)
    X_sobol = saltelli.sample(problem, n_base, calc_second_order=False)

    # Predict with GP surrogate
    preds = surrogate.predict(X_sobol)

    results = {}
    for col in SCALAR_OUTPUTS:
        if col not in preds:
            continue
        Y = preds[col]
        if np.std(Y) < 1e-10:
            continue  # constant output (e.g. cable tension for no-cable group)
        si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)
        df = pd.DataFrame(
            {
                "S1":      si["S1"],
                "ST":      si["ST"],
                "S1_conf": si["S1_conf"],
                "ST_conf": si["ST_conf"],
            },
            index=problem["names"],
        )
        results[col] = df

    return results


def run_all_sobol(
    surrogates: dict,
    n_base: int = SOBOL_N_BASE,
):
    """
    surrogates: {group_key: ScalarSurrogate}
    Returns nested dict: {group_key: {output: DataFrame}}.
    """
    all_results = {}
    for motif in MOTIFS:
        for has_cable in HAS_CABLE:
            group = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
            if group not in surrogates:
                print(f"  Skipping {group}: no surrogate found")
                continue
            print(f"  Sobol analysis for {group}...")
            results = run_sobol_for_group(
                surrogates[group], motif, has_cable, n_base
            )
            all_results[group] = results
            for col, df in results.items():
                st = df['ST'].dropna()
                top = st.idxmax() if len(st) else "nan"
                val = st.max()    if len(st) else float("nan")
                print(f"    {col}: top param = {top} (ST={val:.3f})")
    return all_results


def run_material_sobol(surrogates: dict, n_base: int = SOBOL_N_BASE) -> dict:
    """
    Run Sobol analysis for the material sensitivity study groups
    (material_nocable, material_cable).
    """
    all_results = {}
    for has_cable in HAS_CABLE:
        group = f"material_{'cable' if has_cable else 'nocable'}"
        if group not in surrogates:
            print(f"  Skipping {group}: no surrogate found")
            continue
        bounds = PARAMS_MATERIAL_CABLE if has_cable else PARAMS_MATERIAL_NO_CABLE
        print(f"  Sobol analysis for {group} ({len(bounds)}D)...")
        results = run_sobol_for_group(
            surrogates[group], motif=0, has_cable=has_cable,
            n_base=n_base, bounds=bounds,
        )
        all_results[group] = results
        for col, df in results.items():
            st = df["ST"].dropna()
            top = st.idxmax() if len(st) else "nan"
            val = st.max()    if len(st) else float("nan")
            print(f"    {col}: top param = {top} (ST={val:.3f})")
    return all_results


if __name__ == "__main__":
    import os
    from config import DATA_DIR
    print("Loading surrogates and running Sobol analysis...")
    surrogates = {}
    for motif in MOTIFS:
        for has_cable in HAS_CABLE:
            group = f"motif{motif}_{'cable' if has_cable else 'nocable'}"
            path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
            if os.path.exists(path):
                surrogates[group] = ScalarSurrogate.load(path)
    results = run_all_sobol(surrogates)
    print("Done.")
