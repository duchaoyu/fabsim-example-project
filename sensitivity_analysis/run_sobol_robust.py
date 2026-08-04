"""
Robust indices for the heavy-tailed outputs of the material-r study.

Why: the convergence sweep (figW) shows the cable tension and cable Delta-H
columns still drifting at N = 2048, while crown height and the stress outputs
settle by N ~ 256.  The cause is not the estimator but the output distribution —
Sobol indices divide by Var(Y), and for those columns the variance is dominated
by a handful of extreme states:

    output                 kurtosis   variance share of top 1% of runs
    H_anisotropy (cable)      12.9                 30%
    cable_course_tension      11.6                 27%
    cable_wale_tension         4.7                 20%
    mean_stress / crown        2.8                 15-16%

Two remedies, both applied here:

  1. The tensions are strictly positive, so analysing log T removes the tail
     (kurtosis 11.6 -> 0.3, top-1% share 27% -> 9%).  The GP already fits these
     outputs in log space and only exponentiates on prediction, so the
     well-behaved scale costs nothing.  Indices then describe the sensitivity of
     the relative variation of tension.
  2. Delta-H changes sign and cannot be logged.  For it, PAWN [Pianosi & Wagener
     2015] provides a cross-check: it compares conditional CDFs instead of
     variances, so a heavy tail does not slow it down.  If PAWN agrees with the
     Sobol ranking, the Sobol values can be reported with a caveat rather than
     dropped.

Products:
    data/sobol_material_r_cable_valid_log_{cable_wale,cable_course}_tension.csv
    data/pawn_material_r_{group}_valid_{output}.csv
    figures/figX_robust_indices.{pdf,png}

Usage:
    python3 run_sobol_robust.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import pawn as pawn_analyze

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, SOBOL_N_BASE, RANDOM_SEED
from surrogate import ScalarSurrogate
from visualization import OUTPUT_LABELS, FIG_DIR
from plot_material_r_sobol_combined import GROUPS, PARAM_LABELS, SUR_PREFIX

OUTPUT_LABELS = dict(OUTPUT_LABELS)
OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")
SHORT = {p: lab.split(" (")[0].replace("\n", "") for p, lab in PARAM_LABELS.items()}

LOG_OUTPUTS  = ["cable_wale_tension", "cable_course_tension"]   # cable group only
PAWN_OUTPUTS = {"material_r_cable":   ["H_anisotropy", "cable_course_tension"],
                "material_r_nocable": ["H_anisotropy"]}

N_SWEEP  = [128, 256, 512, 1024, 2048, 4096]
N_PAWN   = 20000     # given-data estimator: one large sample, S=10 conditioning slices
_COLORS  = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#AA4499",
            "#EE3377", "#33BBEE", "#555555", "#BBBB44"]


def _problem(bounds):
    return {"num_vars": len(bounds), "names": list(bounds),
            "bounds": [list(v) for v in bounds.values()]}


def _surrogate(group):
    return ScalarSurrogate.load(os.path.join(
        DATA_DIR, f"{group}_{SUR_PREFIX}surrogate.pkl"))


# ── 1. Sobol on log-transformed tensions, with a convergence sweep ────────────

def log_tension_indices(group="material_r_cable"):
    bounds, prob = GROUPS[group][1], _problem(GROUPS[group][1])
    keys, sur = list(bounds), _surrogate(group)
    sweep = {c: {"raw": {}, "log": {}} for c in LOG_OUTPUTS}

    for N in N_SWEEP:
        X = saltelli.sample(prob, N, calc_second_order=False)
        preds = sur.predict(X)
        for col in LOG_OUTPUTS:
            y = preds[col]
            for lab, Y in [("raw", y), ("log", np.log(np.clip(y, 1e-9, None)))]:
                si = sobol_analyze.analyze(prob, Y, calc_second_order=False,
                                           print_to_console=False)
                sweep[col][lab][N] = pd.DataFrame(
                    {"S1": si["S1"], "ST": si["ST"],
                     "S1_conf": si["S1_conf"], "ST_conf": si["ST_conf"]},
                    index=keys)
        print(f"  N={N:5d} ({len(X)} evaluations)")

    for col in LOG_OUTPUTS:
        t = sweep[col]["log"][SOBOL_N_BASE]
        path = os.path.join(DATA_DIR,
                            f"sobol_{group}_valid_log_{col}.csv")
        t.to_csv(path)
        drift_raw = max(abs(sweep[col]["raw"][4096].loc[p, "ST"]
                            - sweep[col]["raw"][1024].loc[p, "ST"]) for p in keys)
        drift_log = max(abs(sweep[col]["log"][4096].loc[p, "ST"]
                            - sweep[col]["log"][1024].loc[p, "ST"]) for p in keys)
        print(f"  {col}: max |ST(4096) - ST(1024)|  raw {drift_raw:.3f} "
              f"-> log {drift_log:.3f}   (saved {os.path.basename(path)})")
    return sweep


# ── 2. PAWN cross-check ──────────────────────────────────────────────────────

def pawn_indices():
    out = {}
    for group, cols in PAWN_OUTPUTS.items():
        bounds, prob = GROUPS[group][1], _problem(GROUPS[group][1])
        keys, sur = list(bounds), _surrogate(group)
        rng = np.random.default_rng(RANDOM_SEED)
        X = np.column_stack([rng.uniform(lo, hi, N_PAWN)
                             for lo, hi in bounds.values()])
        preds = sur.predict(X)
        for col in cols:
            if col not in preds:
                continue
            res = pawn_analyze.analyze(prob, X, preds[col], S=10,
                                       print_to_console=False, seed=RANDOM_SEED)
            df = pd.DataFrame({k: res[k] for k in ("minimum", "mean", "median",
                                                    "maximum", "CV")},
                              index=keys)
            df.to_csv(os.path.join(DATA_DIR, f"pawn_{group}_valid_{col}.csv"))
            out[(group, col)] = df
            top = df["median"].idxmax()
            print(f"  {group} {col}: PAWN top = {top} "
                  f"(median KS {df['median'].max():.3f})")
    return out


# ── 3. Figure ────────────────────────────────────────────────────────────────

def plot_robust(sweep, pawn, save=True):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.19, wspace=0.33)

    keys = list(GROUPS["material_r_cable"][1])

    # (a), (b): raw vs log convergence for the two tensions
    for ax, col, tag in zip(axes[:2], LOG_OUTPUTS, "ab"):
        final = sweep[col]["log"][N_SWEEP[-1]]["ST"].clip(0)
        top3  = list(final.sort_values(ascending=False).index[:3])
        for p in top3:
            ci = _COLORS[keys.index(p) % len(_COLORS)]
            for lab, style, mk in [("raw", "--", "x"), ("log", "-", "o")]:
                y = [max(0, sweep[col][lab][N].loc[p, "ST"]) for N in N_SWEEP]
                ax.plot(N_SWEEP, y, style, marker=mk, color=ci, linewidth=1.3,
                        markersize=3.2, alpha=0.95 if lab == "log" else 0.45,
                        label=f"{SHORT.get(p, p)} ({lab})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_SWEEP); ax.set_xticklabels([str(n) for n in N_SWEEP],
                                                    fontsize=7)
        ax.axvline(SOBOL_N_BASE, color="#333333", linestyle=":", linewidth=0.7)
        ax.set_ylim(-0.03, 1.02)
        ax.set_xlabel("$N$", fontsize=8.5)
        ax.set_ylabel("$S_T$", fontsize=8.5)
        ax.set_title(f"({tag}) {OUTPUT_LABELS.get(col, col)}: raw vs $\\log$",
                     fontsize=9.5, pad=4)
        ax.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper left",
                  handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
        ax.tick_params(labelsize=7)

    # (c): PAWN vs Sobol ranking for cable Delta-H
    ax = axes[2]
    g, col = "material_r_cable", "H_anisotropy"
    pw = pawn[(g, col)]["median"]
    so = pd.read_csv(os.path.join(DATA_DIR, f"sobol_{g}_valid_{col}.csv"),
                     index_col=0)["ST"].clip(0)
    order = so.sort_values(ascending=False).index
    x = np.arange(len(order))
    ax.bar(x - 0.2, so[order].values, 0.4, label="Sobol $S_T$", color="#EE7733")
    ax.bar(x + 0.2, (pw[order] / pw.max()).values, 0.4,
           label="PAWN median KS (scaled)", color="#0077BB")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(p, p) for p in order], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("index", fontsize=8.5)
    rho = pd.Series(so[order].values).corr(pd.Series(pw[order].values),
                                           method="spearman")
    ax.set_title(f"(c) cable $\\Delta H$: variance-based vs moment-independent\n"
                 f"Spearman rank correlation {rho:.2f}", fontsize=9.5, pad=4)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    fig.suptitle("Robustness of the indices for the heavy-tailed outputs",
                 fontsize=11, y=0.965)

    if save:
        base = os.path.join(FIG_DIR, "figX_robust_indices")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    print("[1] Sobol on log tensions, with convergence sweep")
    sweep = log_tension_indices()
    print("[2] PAWN cross-check")
    pawn = pawn_indices()
    print("[3] figure")
    plot_robust(sweep, pawn)
