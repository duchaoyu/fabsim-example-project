"""
Robust indices for the awkwardly-distributed outputs of the material-r study.

Why: the convergence sweep (figW) shows the cable tension and cable Delta-H
columns drifting for longer than crown height and the stress outputs, which
settle by N ~ 256.  The cause is not the estimator but the output distribution —
Sobol indices divide by Var(Y).  On the corrected cable geometry the tensions
are not so much heavy-tailed as ZERO-INFLATED: L_rest runs over (1.2, 1.4) m
against a 1.29 m flat arc, so the upper part of the range leaves the cable
slack and it carries no load at all.  In the 660-sample cable batch:

    L_rest bin (m)      1.20-1.225  ...  1.275-1.30  1.30-1.325  1.375-1.40
    runs with T = 0          0%              15%          56%         81%

38% of cable runs have T identically 0 (wale; 37% course) over the full box, and
40% (wale; 41% course) inside the model-validity box these figures use.

Note the surrogate reproduces only ~24% slack against that sampled 40%: a GP is
continuous, so it cannot represent a point mass and interpolates across the
slack/taut switch.  The forthcoming L_rest = f * L_nocable parameterisation
removes the plateau by construction, at which point this whole correction becomes
unnecessary.

That has three consequences, all handled here:

  1. log T is NOT defined on this output.  The previous version of this script
     analysed np.log(np.clip(y, 1e-9, None)), which mapped the whole slack
     plateau onto a single spike at log(1e-9) = -20.72.  That spike carried 98%
     of the variance (Var = 114.5, against 2.3 among the genuinely taut
     samples), so the resulting "log tension" indices largely measured which
     parameters trip the clip, not the sensitivity of tension.  Replaced by
     log1p(T / T_REF), which is defined at T = 0, needs no clipping, and still
     compresses the taut range: kurtosis 1.9 -> -0.9, top-1% variance share
     13% -> 3%.
  2. The tension GP is fitted on RAW tension, not log — surrogate.py disables
     its log transform when the sample contains zeros.  An unconstrained GP
     overshoots the point mass at zero and predicted tension down to -271 N
     across ~21% of the box; surrogate._NONNEG_OUTPUTS now clamps that at 0 on
     prediction.  The docstring numbers above are post-clamp.
  3. Whether the cable is engaged at all is a separate response from how hard it
     pulls, and the tension indices conflate the two.  Indices for the engagement
     indicator 1[T > 0] are estimated and written to
     sobol_*_valid_engaged_*.csv, but deliberately NOT plotted here: this figure
     is about whether the indices can be trusted, and engagement is a substantive
     result, not a robustness check.  It is also the one quantity the GP gets
     materially wrong (see the 24%-vs-40% note above), because the indicator is
     exactly the discontinuity a continuous surrogate interpolates across.  If
     the result matters it belongs in the regime bars, on a classifier rather
     than a GP.

Delta-H changes sign and cannot be logged either.  For it, PAWN [Pianosi &
Wagener 2015] provides a cross-check: it compares conditional CDFs instead of
variances, so a skewed output does not slow it down.  If PAWN agrees with the
Sobol ranking, the Sobol values can be reported with a caveat rather than
dropped.

Products:
    data/sobol_material_r_cable_valid_log1p_{cable_wale,cable_course}_tension.csv
    data/sobol_material_r_cable_valid_engaged_{cable_wale,cable_course}_tension.csv
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
from plot_material_r_sobol_combined import (
    GROUPS, PARAM_LABELS, SUR_PREFIX, T_REF, tension_scale,
)

OUTPUT_LABELS = dict(OUTPUT_LABELS)
OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")
SHORT = {p: lab.split(" (")[0].replace("\n", "") for p, lab in PARAM_LABELS.items()}

LOG_OUTPUTS  = ["cable_wale_tension", "cable_course_tension"]   # cable group only
# T_REF and tension_scale are imported from plot_material_r_sobol_combined so the
# reported scale is defined once: log1p(T / T_REF) with T_REF = 1 N, which is
# ~log in the taut range (T ~ 1e2-1e3 N) and smoothly zero at the slack plateau,
# so no clipping is needed and the point mass at T = 0 stays a point mass.
# Tension below this counts as slack for the engagement indicator.  The FEM
# returns exact zeros, so any small positive threshold picks out the same set;
# it exists only to absorb surrogate round-off near the clamp.
T_ENGAGED    = 1e-6
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


# ── 1. Sobol on the tensions: raw, log1p, and slack/taut engagement ───────────

# Labels tracked in the convergence sweep.  "engaged" is a different response
# (a 0/1 indicator), not another scale for the same one — it is swept alongside
# so its convergence can be read off the same figure.
SWEEP_LABELS = ["raw", "log1p", "engaged"]


def tension_indices(group="material_r_cable"):
    bounds, prob = GROUPS[group][1], _problem(GROUPS[group][1])
    keys, sur = list(bounds), _surrogate(group)
    sweep = {c: {lab: {} for lab in SWEEP_LABELS} for c in LOG_OUTPUTS}

    for N in N_SWEEP:
        X = saltelli.sample(prob, N, calc_second_order=False)
        preds = sur.predict(X)
        slack = {}
        for col in LOG_OUTPUTS:
            y = preds[col]
            if (y < 0).any():                     # clamp must already have run
                raise RuntimeError(
                    f"{col}: {(y < 0).sum()} negative predictions — expected "
                    "surrogate._NONNEG_OUTPUTS to clamp tension at 0")
            engaged = (y > T_ENGAGED).astype(float)
            slack[col] = 1.0 - engaged.mean()
            for lab, Y in [("raw", y),
                           ("log1p", tension_scale(y)),
                           ("engaged", engaged)]:
                si = sobol_analyze.analyze(prob, Y, calc_second_order=False,
                                           print_to_console=False)
                sweep[col][lab][N] = pd.DataFrame(
                    {"S1": si["S1"], "ST": si["ST"],
                     "S1_conf": si["S1_conf"], "ST_conf": si["ST_conf"]},
                    index=keys)
        print(f"  N={N:5d} ({len(X)} evaluations)  slack fraction: "
              + ", ".join(f"{c.replace('cable_','').replace('_tension','')} "
                          f"{slack[c]:.1%}" for c in LOG_OUTPUTS))

    for col in LOG_OUTPUTS:
        drift = {}
        for lab in SWEEP_LABELS:
            sweep[col][lab][SOBOL_N_BASE].to_csv(os.path.join(
                DATA_DIR, f"sobol_{group}_valid_{lab}_{col}.csv"))
            drift[lab] = max(abs(sweep[col][lab][4096].loc[p, "ST"]
                                 - sweep[col][lab][1024].loc[p, "ST"])
                             for p in keys)
        print(f"  {col}: max |ST(4096) - ST(1024)|  "
              + "  ".join(f"{lab} {drift[lab]:.3f}" for lab in SWEEP_LABELS))
        top = sweep[col]["engaged"][SOBOL_N_BASE]["ST"].clip(0).idxmax()
        print(f"    engagement 1[T>0] is driven most by {top}")
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

    # (a), (b): raw vs log1p convergence for the two tensions
    for ax, col, tag in zip(axes[:2], LOG_OUTPUTS, "ab"):
        final = sweep[col]["log1p"][N_SWEEP[-1]]["ST"].clip(0)
        top3  = list(final.sort_values(ascending=False).index[:3])
        for p in top3:
            ci = _COLORS[keys.index(p) % len(_COLORS)]
            for lab, style, mk in [("raw", "--", "x"), ("log1p", "-", "o")]:
                y = [max(0, sweep[col][lab][N].loc[p, "ST"]) for N in N_SWEEP]
                ax.plot(N_SWEEP, y, style, marker=mk, color=ci, linewidth=1.3,
                        markersize=3.2, alpha=0.95 if lab == "log1p" else 0.45,
                        label=f"{SHORT.get(p, p)} ({lab})")
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_SWEEP); ax.set_xticklabels([str(n) for n in N_SWEEP],
                                                    fontsize=7)
        ax.axvline(SOBOL_N_BASE, color="#333333", linestyle=":", linewidth=0.7)
        ax.set_ylim(-0.03, 1.02)
        ax.set_xlabel("$N$", fontsize=8.5)
        ax.set_ylabel("$S_T$", fontsize=8.5)
        ax.set_title(f"({tag}) {OUTPUT_LABELS.get(col, col)}: "
                     f"raw vs $\\log(1+T)$", fontsize=9.5, pad=4)
        ax.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper left",
                  handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
        ax.tick_params(labelsize=7)

    # (c): PAWN vs Sobol ranking for cable Delta-H.  The engagement indices are
    # estimated in tension_indices() and written to CSV, but not plotted — see
    # the module docstring for why they do not belong on a robustness figure.
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

    fig.suptitle("Robustness of the indices for the zero-inflated cable "
                 "outputs ($L_\\mathrm{rest}$ = 1.2–1.4 m, 40% of runs slack)",
                 fontsize=11, y=0.965)

    if save:
        base = os.path.join(FIG_DIR, "figX_robust_indices")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    print("[1] Sobol on the tensions (raw / log1p / engagement), "
          "with convergence sweep")
    sweep = tension_indices()
    print("[2] PAWN cross-check")
    pawn = pawn_indices()
    print("[3] figure")
    plot_robust(sweep, pawn)
