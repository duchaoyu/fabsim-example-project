"""
Surrogate validation figures for the material-r study.

figV_surrogate_parity      predicted vs FEA on the held-out 20%, one panel per
                           output, with 95% predictive intervals.  Shows *where*
                           a surrogate fails, which Table 6.x cannot.
figD_design_coverage_{g}   pairwise projections of the training design, with the
                           per-parameter histogram on the diagonal.  Documents
                           that LHS filled the box and that the quality filter
                           did not carve out a region.

Both reconstruct the same 80/20 split the surrogates were fitted with
(train_test_split, random_state=RANDOM_SEED) so the held-out points here are the
ones the GPs never saw.

Usage:
    python3 plot_surrogate_validation.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DATA_DIR, TRAIN_VAL_SPLIT, RANDOM_SEED
from run_material_r_sobol import _outputs_for
from sampling import generate_material_r_samples
from visualization import OUTPUT_LABELS, FIG_DIR
from plot_material_r_sobol_combined import (
    PARAM_LABELS, VALID, SUR_PREFIX, FIG_SUFFIX,
)

# The LHS blocks that were actually run, as (start_id, seed, n) — needed to count
# how many planned runs fall inside a box, and so to report retention honestly
# for whichever box is being plotted.
_LHS_BLOCKS = [(3000, 42, 800), (9000, 777, 1600)]
_BOX_LABEL  = ("validity box, $s \\geq 0.95$" if VALID else "full box")

OUTPUT_LABELS = dict(OUTPUT_LABELS)
OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")

_B = ((config.PARAMS_MATERIAL_R_VALID_NO_CABLE,
       config.PARAMS_MATERIAL_R_VALID_CABLE) if VALID else
      (config.PARAMS_MATERIAL_R_NO_CABLE, config.PARAMS_MATERIAL_R_CABLE))

GROUPS = {
    "material_r_nocable": ("no cable", _B[0]),
    "material_r_cable":   ("cable",    _B[1]),
}


def in_box(df, bounds):
    m = np.ones(len(df), dtype=bool)
    for k, (lo, hi) in bounds.items():
        m &= (df[k] >= lo) & (df[k] <= hi)
    return m


_PLANNED = None


def n_planned(group, bounds):
    """Planned runs of this group that fall inside `bounds`."""
    global _PLANNED
    if _PLANNED is None:
        rows = []
        for sid, seed, n in _LHS_BLOCKS:
            rows += generate_material_r_samples(start_id=sid, seed=seed, n=n)
        _PLANNED = pd.DataFrame(rows)
    P = _PLANNED[_PLANNED.group == group]
    return int(in_box(P, bounds).sum())

_C_OK   = "#0077BB"
_C_EDGE = "#333333"


def load_split(group):
    """Return (valid_df, keys, outputs, surrogate, train_idx, val_idx)."""
    sur   = pickle.load(open(os.path.join(
        DATA_DIR, f"{group}_{SUR_PREFIX}surrogate.pkl"), "rb"))
    df    = pd.read_csv(os.path.join(DATA_DIR, f"{group}_section_metrics.csv"))
    keys  = list(GROUPS[group][1])
    outs  = [c for c in _outputs_for(group) if c in df.columns]
    df    = df[in_box(df, GROUPS[group][1])]
    valid = df.dropna(subset=keys + outs).reset_index(drop=True)
    tr, va = train_test_split(np.arange(len(valid)), test_size=TRAIN_VAL_SPLIT,
                              random_state=RANDOM_SEED)
    n_fit = sur.gps[outs[0]].X_train_.shape[0]
    if n_fit != len(tr):
        raise SystemExit(f"{group}: split mismatch ({n_fit} fitted vs {len(tr)} "
                         f"reconstructed) — the cached surrogate is stale")
    return valid, keys, outs, sur, tr, va


def _predict_interval(sur, group_keys, valid, idx, col):
    """Mean and 95% interval in physical units for one output."""
    Xs   = sur.scaler_X.transform(valid[group_keys].values)[idx]
    gp   = sur.gps[col]
    sc   = sur.scalers_y[col]
    m, s = gp.predict(Xs, return_std=True)
    inv  = lambda z: sc.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    pred, lo, hi = inv(m), inv(m - 1.96 * s), inv(m + 1.96 * s)
    if col in getattr(sur, "_log_cols", set()):
        pred, lo, hi = np.exp(pred), np.exp(lo), np.exp(hi)
    return pred, lo, hi


# ── Figure V: parity ─────────────────────────────────────────────────────────

def plot_parity(save=True):
    data = {g: load_split(g) for g in GROUPS}
    n_col = max(len(d[2]) for d in data.values())

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(GROUPS), n_col,
                             figsize=(1.65 * n_col + 0.9, 4.4),
                             squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.845, bottom=0.135,
                        wspace=0.42, hspace=0.62)

    for r, (group, (label, _)) in enumerate(GROUPS.items()):
        valid, keys, outs, sur, tr, va = data[group]
        for c in range(n_col):
            ax = axes[r][c]
            if c >= len(outs):
                ax.set_visible(False)
                continue
            col = outs[c]
            true = valid.loc[va, col].values
            pred, lo, hi = _predict_interval(sur, keys, valid, va, col)

            ax.errorbar(true, pred, yerr=[pred - lo, hi - pred], fmt="none",
                        ecolor=_C_OK, elinewidth=0.4, alpha=0.28, zorder=1)
            ax.scatter(true, pred, s=3.0, color=_C_OK, alpha=0.75,
                       linewidths=0, zorder=2)
            lim = [min(true.min(), pred.min()), max(true.max(), pred.max())]
            pad = 0.04 * (lim[1] - lim[0])
            lim = [lim[0] - pad, lim[1] + pad]
            ax.plot(lim, lim, "--", color=_C_EDGE, linewidth=0.7, zorder=3)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal", adjustable="box")

            r2   = r2_score(true, pred)
            rmse = np.sqrt(np.mean((true - pred) ** 2))
            cov  = np.mean((true >= lo) & (true <= hi)) * 100
            ax.set_title(OUTPUT_LABELS.get(col, col), fontsize=9, pad=3)
            ax.text(0.05, 0.95,
                    f"$R^2$={r2:.3f}\nnRMSE={100*rmse/(true.max()-true.min()):.1f}%"
                    f"\ncov={cov:.0f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
                    linespacing=1.35)
            ax.tick_params(labelsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{label}\nsurrogate", fontsize=8.5)

    fig.supxlabel("FEA (held-out 20%)", fontsize=9, y=0.035)
    fig.suptitle("Surrogate accuracy on held-out runs: prediction vs FEA, "
                 f"with 95% predictive intervals ({_BOX_LABEL})",
                 fontsize=11, y=0.965)

    if save:
        base = os.path.join(FIG_DIR, f"figV_surrogate_parity{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


# ── Figure D: design coverage ────────────────────────────────────────────────

def plot_coverage(group, save=True):
    label, bounds = GROUPS[group]
    planned = n_planned(group, bounds)
    valid, keys, outs, sur, tr, va = load_split(group)
    X = valid[keys].values
    d = len(keys)

    corr = np.corrcoef(X, rowvar=False)
    off  = np.abs(corr - np.eye(d))
    i, j = np.unravel_index(off.argmax(), off.shape)
    max_r = off[i, j]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.linewidth": 0.7, "figure.dpi": 150,
                         "axes.spines.top": True, "axes.spines.right": True})
    side = 1.05 * d + 1.0
    fig, axes = plt.subplots(d, d, figsize=(side, side), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.905, bottom=0.075,
                        wspace=0.12, hspace=0.12)

    for a in range(d):
        for b in range(d):
            ax = axes[a][b]
            if b > a:
                ax.set_visible(False)
                continue
            if a == b:
                ax.hist(X[:, a], bins=25, color="#BBBBBB",
                        edgecolor="white", linewidth=0.3)
                ax.set_yticks([])
            else:
                ax.scatter(X[:, b], X[:, a], s=1.2, color=_C_OK,
                           alpha=0.30, linewidths=0)
                ax.set_ylim(*bounds[keys[a]])
                if (a, b) in [(i, j), (j, i)]:
                    for s in ax.spines.values():
                        s.set_color("#CC3311"); s.set_linewidth(1.4)
            ax.set_xlim(*bounds[keys[b]])
            ax.tick_params(labelsize=6, length=2)
            if a != d - 1:
                ax.set_xticklabels([])
            if b != 0 or a == 0:
                ax.set_yticklabels([])
            if a == d - 1:
                ax.set_xlabel(PARAM_LABELS.get(keys[b], keys[b]).replace("\n", " "),
                              fontsize=8)
            if b == 0 and a > 0:
                ax.set_ylabel(PARAM_LABELS.get(keys[a], keys[a]).replace("\n", " "),
                              fontsize=8)

    fig.suptitle(
        f"Training design coverage — {label}, {_BOX_LABEL} ({d}-D, "
        f"{len(valid)} runs of {planned} planned, "
        f"{100*len(valid)/planned:.0f}% retained)\n"
        f"largest pairwise $|r|$ = {max_r:.3f} "
        f"({keys[i]}, {keys[j]}), outlined in red",
        fontsize=10, y=0.975)

    if save:
        base = os.path.join(FIG_DIR,
                            f"figD_design_coverage_{label.replace(' ', '')}{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf   max|r|={max_r:.3f}")
    return fig


if __name__ == "__main__":
    plot_parity()
    for g in GROUPS:
        plot_coverage(g)
