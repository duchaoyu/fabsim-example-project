"""
Sobol sensitivity analysis for the section-curvature outputs, on regenerated
no-cable data (current mesh, correct motif materials).

Two independent estimates per output:

  surrogate  the pipeline's method — fit a GP to the LHS samples, then evaluate
             a Saltelli design through the GP (sobol_analysis.run_sobol_for_group)
  direct     Saltelli design evaluated by FEA itself, no surrogate in the loop
             (requires data/results_nocable_v2_saltelli.csv)

Reporting both matters here: the curvature estimator carries 8-12% discretisation
noise, which a GP absorbs into its WhiteKernel (and therefore excludes from the
variance it apportions) but which direct FEA leaves in the output variance.  If
the two estimates agree, the ranking is not an artifact of either choice.

Outputs:
  data/curvature_sobol_indices.csv
  figures/figL_curvature_sobol.{pdf,png}

Usage:  python3 run_curvature_sobol.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_NO_CABLE, SOBOL_N_BASE
from surrogate import ScalarSurrogate
from sobol_analysis import _salib_problem

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LHS_CSV      = os.path.join(DATA_DIR, "results_nocable_v2.csv")
SALTELLI_CSV = os.path.join(DATA_DIR, "results_nocable_v2_saltelli.csv")

CURV_OUTPUTS = ["H_mean_x0", "H_mean_y0"]
REF_OUTPUTS  = ["crown_height"]          # for context in the figure
KEYS  = list(PARAMS_NO_CABLE.keys())
GROUPS = ["motif1_nocable", "motif2_nocable"]

PARAM_LABELS = {"sf_wale": r"$s_{wale}$", "sf_course": r"$s_{course}$",
                "knit_dir": r"$\theta_{knit}$", "pressure": r"$p$"}
OUT_LABELS = {"H_mean_x0": r"$\bar{H}_{x=0}$  (wale cut)",
              "H_mean_y0": r"$\bar{H}_{y=0}$  (course cut)",
              "crown_height": "crown height"}
COLORS = {"motif1_nocable": "#2E8B57", "motif2_nocable": "#20B2AA"}
LABELS = {"motif1_nocable": "Motif 1", "motif2_nocable": "Motif 2"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})


def _load(path, group):
    df = pd.read_csv(path)
    df = df[(df["group"] == group) & (~df["sim_failed"].astype(bool))]
    return df.dropna(subset=KEYS)


def surrogate_indices(group, outputs):
    """Fit a GP on the LHS samples and run Sobol through it."""
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    df = _load(LHS_CSV, group).dropna(subset=outputs)
    sur = ScalarSurrogate(has_cable=False)
    metrics = sur.fit(df, output_cols=outputs)

    problem = _salib_problem(has_cable=False)
    X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=False)
    preds = sur.predict(X)

    res = {}
    for col in outputs:
        Y = preds[col]
        si = sobol.analyze(problem, Y, calc_second_order=False,
                           print_to_console=False)
        res[col] = pd.DataFrame({"S1": si["S1"], "ST": si["ST"],
                                 "S1_conf": si["S1_conf"],
                                 "ST_conf": si["ST_conf"]},
                                index=problem["names"])
    return res, metrics, len(df)


def direct_indices(group, outputs):
    """Sobol straight from the FEA-evaluated Saltelli design."""
    from SALib.analyze import sobol
    if not os.path.exists(SALTELLI_CSV):
        return None, 0
    df = pd.read_csv(SALTELLI_CSV)
    df = df[df["group"] == group]
    problem = _salib_problem(has_cable=False)
    res = {}
    n_bad = int(df["sim_failed"].astype(bool).sum())
    for col in outputs:
        Y = df[col].values.astype(float)
        # Saltelli's estimator needs the full design in order; substitute the
        # output median for non-converged / missing entries rather than dropping
        # rows, which would break the A/B matrix pairing.
        bad = ~np.isfinite(Y) | df["sim_failed"].astype(bool).values
        if bad.all():
            continue
        Y = Y.copy()
        Y[bad] = np.median(Y[~bad])
        si = sobol.analyze(problem, Y, calc_second_order=False,
                           print_to_console=False)
        res[col] = pd.DataFrame({"S1": si["S1"], "ST": si["ST"],
                                 "S1_conf": si["S1_conf"],
                                 "ST_conf": si["ST_conf"]},
                                index=problem["names"])
    return res, n_bad


def main():
    outputs = CURV_OUTPUTS + REF_OUTPUTS
    rows = []
    sur_res, dir_res = {}, {}

    for group in GROUPS:
        s, metrics, n = surrogate_indices(group, outputs)
        sur_res[group] = s
        print(f"\n{group}: GP fitted on {n} converged LHS samples")
        for col in outputs:
            m = metrics.get(col, {})
            print(f"    {col:14s} hold-out R2={m.get('r2', float('nan')):.3f}  "
                  f"RMSE={m.get('rmse', float('nan')):.4g}")

        d, n_bad = direct_indices(group, outputs)
        if d:
            dir_res[group] = d
            print(f"  direct-FEA Saltelli design available "
                  f"({n_bad} non-converged points replaced by the median)")

        for col in outputs:
            for est, res in (("surrogate", s), ("direct", d or {})):
                if col not in res:
                    continue
                for p in KEYS:
                    rows.append({"group": group, "output": col,
                                 "estimator": est, "param": p,
                                 "S1": res[col].loc[p, "S1"],
                                 "ST": res[col].loc[p, "ST"],
                                 "S1_conf": res[col].loc[p, "S1_conf"],
                                 "ST_conf": res[col].loc[p, "ST_conf"]})

    idx = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, "curvature_sobol_indices.csv")
    idx.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # ── figure: ST bars per parameter, motifs side by side, one row per output ─
    have_direct = bool(dir_res)
    fig, axes = plt.subplots(len(outputs), 1, figsize=(5.6, 2.3 * len(outputs)),
                             constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(KEYS))
    w = 0.36

    for ax, col in zip(axes, outputs):
        for i, group in enumerate(GROUPS):
            if col not in sur_res[group]:
                continue
            st = sur_res[group][col]["ST"].reindex(KEYS).values
            s1 = sur_res[group][col]["S1"].reindex(KEYS).values
            off = (i - 0.5) * w
            ax.bar(x + off, st, w * 0.92, color=COLORS[group], alpha=0.35,
                   label=f"{LABELS[group]}  $S_T$" if col == outputs[0] else None)
            ax.bar(x + off, s1, w * 0.92, color=COLORS[group],
                   label=f"{LABELS[group]}  $S_1$" if col == outputs[0] else None)
            if group in dir_res and col in dir_res[group]:
                dst = dir_res[group][col]["ST"].reindex(KEYS).values
                ax.plot(x + off, dst, ls="none", marker="_", ms=11, mew=1.8,
                        color="0.15",
                        label="direct FEA  $S_T$" if (col == outputs[0] and i == 0)
                        else None)
        ax.set_ylabel("Sobol index")
        ax.set_title(OUT_LABELS.get(col, col))
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([PARAM_LABELS[k] for k in KEYS])
    axes[0].legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    axes[-1].set_xlabel("Input parameter")

    fig.suptitle("Sobol sensitivity of section curvature\n"
                 "(no-cable groups, regenerated on the current mesh with the "
                 "correct motif materials)", fontsize=9)

    path = os.path.join(FIG_DIR, "figL_curvature_sobol.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"Saved: {path}")

    # ── console summary ───────────────────────────────────────────────────────
    print("\nTotal-order indices (ST):")
    for col in outputs:
        print(f"  {OUT_LABELS.get(col, col)}")
        for group in GROUPS:
            for est in (["surrogate", "direct"] if have_direct else ["surrogate"]):
                sub = idx[(idx.group == group) & (idx.output == col) &
                          (idx.estimator == est)]
                if sub.empty:
                    continue
                s = "  ".join(f"{PARAM_LABELS[p]:>12s}={sub[sub.param==p].ST.iloc[0]:5.2f}"
                              for p in KEYS)
                print(f"    {LABELS[group]:8s} {est:9s} {s}")
    return idx


if __name__ == "__main__":
    main()
