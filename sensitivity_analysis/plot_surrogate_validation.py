"""
Surrogate validation figures for the material-r study.

table_design_reduction     Table 6.3 — planned design -> converged -> in the
                           validity box -> retained -> training / held-out.
table_surrogate_accuracy   Table 6.4 — R2, RMSE, nRMSE and interval coverage on
                           the held-out 20%, per group and output.
figV_surrogate_parity      predicted vs FEA on the held-out 20%, one panel per
                           output, with 95% predictive intervals.  Shows *where*
                           a surrogate fails, which the table cannot.

The table and the figure are computed from the same predictions, so they cannot
disagree — see write_metrics_table for why that mattered.
figD_design_coverage_{g}   pairwise projections of the training design, with the
                           per-parameter histogram on the diagonal.  Documents
                           that LHS filled the box and that the quality filter
                           did not carve out a region.

Both reconstruct the same 80/20 split the surrogates were fitted with
(train_test_split, random_state=RANDOM_SEED) so the held-out points here are the
ones the GPs never saw.

Usage:
    python3 plot_surrogate_validation.py [--full-box]
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
from surrogate import _NONNEG_OUTPUTS, _LOG1P_OUTPUTS, _DERIVED_OUTPUTS  # noqa: F401
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
    # First output that actually has a GP — outs may lead with a derived output,
    # which has none (surrogate._DERIVED_OUTPUTS).
    n_fit = sur.gps[next(c for c in outs if c in sur.gps)].X_train_.shape[0]
    if n_fit != len(tr):
        raise SystemExit(f"{group}: split mismatch ({n_fit} fitted vs {len(tr)} "
                         f"reconstructed) — the cached surrogate is stale")
    return valid, keys, outs, sur, tr, va


def _untransform(sur, col, z):
    """Undo the fitting transform for one output, on any array shape."""
    flat = sur.scalers_y[col].inverse_transform(
        np.asarray(z).reshape(-1, 1)).ravel().reshape(np.shape(z))
    if col in getattr(sur, "_log_cols", set()):
        flat = np.exp(flat)
    if col in getattr(sur, "_log1p_cols", set()):
        flat = np.expm1(flat)
    if col in _NONNEG_OUTPUTS:
        flat = np.maximum(flat, 0.0)
    return flat


def _derived_interval(sur, group_keys, valid, idx, col, n_draw=512):
    """Mean and 95% interval for an output derived from other outputs.

    There is no GP for a derived output, so the interval is propagated by
    sampling each component's predictive normal and pushing the draws through
    the same function predict() uses.  The point estimate is the function of the
    component MEANS, not the mean of the draws, so it matches predict() exactly
    and Table 6.4 cannot disagree with this figure.

    The components are drawn independently, which is conservative: Hx and Hy come
    from the same inputs and are positively correlated, so the true interval on
    their normalised difference is narrower than this.
    """
    deps, fn = _DERIVED_OUTPUTS[col]
    Xs  = sur.scaler_X.transform(valid[group_keys].values)[idx]
    rng = np.random.default_rng(RANDOM_SEED)
    means, draws = [], []
    for d in deps:
        m, s = sur.gps[d].predict(Xs, return_std=True)
        means.append(_untransform(sur, d, m))
        draws.append(_untransform(sur, d,
                                  m + s * rng.normal(size=(n_draw, len(m)))))
    vals = fn(*draws)
    with np.errstate(invalid="ignore"):
        lo = np.nanpercentile(vals, 2.5, axis=0)
        hi = np.nanpercentile(vals, 97.5, axis=0)
    return fn(*means), lo, hi


def _predict_interval(sur, group_keys, valid, idx, col):
    """Mean and 95% interval in physical units for one output."""
    if col in _DERIVED_OUTPUTS:
        return _derived_interval(sur, group_keys, valid, idx, col)
    Xs   = sur.scaler_X.transform(valid[group_keys].values)[idx]
    gp   = sur.gps[col]
    sc   = sur.scalers_y[col]
    m, s = gp.predict(Xs, return_std=True)
    inv  = lambda z: sc.inverse_transform(np.asarray(z).reshape(-1, 1)).ravel()
    pred, lo, hi = inv(m), inv(m - 1.96 * s), inv(m + 1.96 * s)
    if col in getattr(sur, "_log_cols", set()):
        pred, lo, hi = np.exp(pred), np.exp(lo), np.exp(hi)
    if col in getattr(sur, "_log1p_cols", set()):
        pred, lo, hi = np.expm1(pred), np.expm1(lo), np.expm1(hi)
    if col in _NONNEG_OUTPUTS:
        # This path reaches into gp.predict rather than going through
        # sur.predict(), so it has to apply the same physical floor: a slack
        # cable carries no load, and the tension GP is fitted on raw tension
        # over a sample that is ~40% zeros, so it overshoots below zero.  Without
        # this, figV plots negative tension and its R2 disagrees with the
        # surrogate's own metrics.  The interval is clamped too — a 95% interval
        # on a non-negative quantity must not extend below 0.
        pred, lo, hi = (np.maximum(v, 0.0) for v in (pred, lo, hi))
    return pred, lo, hi


# ── Table: reduction of the planned design to the training set ───────────────

def write_reduction_table(save=True):
    """Table 6.3 — how the planned design reduces to the surrogate training set.

    Every row is an ACHIEVED count on the same population, so the chain reads as
    a reduction.  The published version mixed one planned count into that chain:
    its "within the validity range" row carried 1345 / 1356, which is how many of
    the 2400 PLANNED samples fall inside the box, not how many converged runs do
    (1336 / 451 at the time).  Those are different quantities and only one of them
    belongs in a reduction.
    """
    rows = {}
    for group, (label, bounds) in GROUPS.items():
        df    = pd.read_csv(os.path.join(
            DATA_DIR, f"{group}_section_metrics.csv"))
        keys  = list(bounds)
        outs  = [c for c in _outputs_for(group) if c in df.columns]
        inbox = df[in_box(df, bounds)]
        kept  = inbox.dropna(subset=keys + outs)
        n_tr  = len(kept) - int(round(TRAIN_VAL_SPLIT * len(kept)))
        rows[label] = {
            "planned":              sum(n for _, _, n in _LHS_BLOCKS),
            "planned_in_box":       n_planned(group, bounds),
            "converged":            len(df),
            "converged_in_box":     len(inbox),
            "retained_all_filters": len(kept),
            "training":             n_tr,
            "held_out":             len(kept) - n_tr,
        }
    t = pd.DataFrame(rows)
    t.index.name = "stage"

    print(f"\nTable 6.3 — reduction of the planned design ({_BOX_LABEL})")
    print(t.to_string())
    if save:
        path = os.path.join(DATA_DIR, f"table_design_reduction{FIG_SUFFIX}.csv")
        t.to_csv(path)
        print(f"Saved: {path}")
    return t


# ── Table: surrogate accuracy on the held-out 20% ────────────────────────────

def write_metrics_table(save=True):
    """Table 6.4 — held-out accuracy, from the same numbers figV plots.

    The table used to be transcribed from ScalarSurrogate.metrics, which is
    computed at fit time, while figV recomputed its own R2 from the reconstructed
    split.  Two sources for one quantity is one too many: they agree only while
    nothing downstream of the GP changes, and the non-negative clamp on the cable
    tensions broke exactly that assumption.  Both now come from here.
    """
    rows = []
    for group, (label, _) in GROUPS.items():
        valid, keys, outs, sur, tr, va = load_split(group)
        for col in outs:
            true = valid.loc[va, col].values
            pred, lo, hi = _predict_interval(sur, keys, valid, va, col)
            rng  = true.max() - true.min()
            rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
            rows.append({
                "group":      label,
                "output":     col,
                "n_heldout":  len(va),
                "R2":         round(r2_score(true, pred), 3),
                "RMSE":       float(f"{rmse:.4g}"),
                "nRMSE_pct":  round(100 * rmse / rng, 1) if rng > 0 else np.nan,
                "coverage_pct": round(float(np.mean((true >= lo) &
                                                    (true <= hi)) * 100), 0),
            })
    t = pd.DataFrame(rows)

    print(f"\nTable 6.4 — surrogate accuracy on the held-out 20% ({_BOX_LABEL})")
    print(t.to_string(index=False))
    if save:
        path = os.path.join(DATA_DIR,
                            f"table_surrogate_accuracy{FIG_SUFFIX}.csv")
        t.to_csv(path, index=False)
        print(f"Saved: {path}")
    return t


# ── Figure V: parity ─────────────────────────────────────────────────────────

def plot_parity(save=True):
    data = {g: load_split(g) for g in GROUPS}

    # Columns are set by the *smaller* group, so the grid carries no blank slots
    # inside a row: the cable-only outputs (the two cable tensions) wrap onto a
    # row of their own.  Fewer columns for the same page width means every panel
    # is drawn larger.
    n_col = min(len(d[2]) for d in data.values())

    place, row_label, r = [], {}, 0
    for group, (label, _) in GROUPS.items():
        outs = data[group][2]
        # Name the held-out count on the row.  The two groups are far from
        # symmetric — the cable validity box retains 370 runs against 1095, so
        # its panels rest on ~74 points against ~219 — and a reader comparing
        # R2 down a column should not have to infer that from the scatter.
        n_held = len(data[group][5])
        for k in range(0, len(outs), n_col):
            for c, col in enumerate(outs[k:k + n_col]):
                place.append((r, c, group, col))
            row_label[r] = (f"{label}\nsurrogate\n($n$={n_held})" if k == 0 else
                            f"{label}\nsurrogate (cont.)\n($n$={n_held})")
            r += 1
    n_row = r

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})

    # Per-cell inches, plus fixed room for the row labels, suptitle and supxlabel
    PW, PH = 2.05, 2.30
    pad_l, pad_t, pad_b = 1.05, 0.62, 0.60
    fig_w, fig_h = PW * n_col + pad_l, PH * n_row + pad_t + pad_b
    fig, axes = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h),
                             squeeze=False)
    fig.subplots_adjust(left=pad_l / fig_w, right=0.995,
                        top=1 - pad_t / fig_h, bottom=pad_b / fig_h,
                        wspace=0.42, hspace=0.55)

    for ax in axes.ravel():
        ax.set_visible(False)

    for r, c, group, col in place:
        valid, keys, outs, sur, tr, va = data[group]
        ax = axes[r][c]
        ax.set_visible(True)
        true = valid.loc[va, col].values
        pred, lo, hi = _predict_interval(sur, keys, valid, va, col)

        ax.errorbar(true, pred, yerr=[pred - lo, hi - pred], fmt="none",
                    ecolor=_C_OK, elinewidth=0.4, alpha=0.28, zorder=1)
        ax.scatter(true, pred, s=4.0, color=_C_OK, alpha=0.75,
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
        ax.set_title(OUTPUT_LABELS.get(col, col), fontsize=10, pad=4)
        ax.text(0.05, 0.95,
                f"$R^2$={r2:.3f}\nnRMSE={100*rmse/(true.max()-true.min()):.1f}%"
                f"\ncov={cov:.0f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                linespacing=1.35)
        ax.tick_params(labelsize=7.5)
        if c == 0:
            ax.set_ylabel(row_label[r], fontsize=9.5)

    fig.supxlabel("FEA (held-out 20%)", fontsize=10, y=0.012)
    fig.suptitle("Surrogate accuracy on held-out runs: prediction vs FEA, "
                 f"with 95% predictive intervals ({_BOX_LABEL})",
                 fontsize=12, y=1 - 0.16 / fig_h)

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

    # The L_rest axes are labelled with the bare symbol, but carry the sampled
    # FRACTION of the cable-free section length (0.93-0.99), not metres — say so
    # once here rather than crowding two pair-plot axis labels with the ratio.
    frac_note = ("\n$L_\\mathrm{rest}$ axes are fractions of the cable-free "
                 "section length" if any(k.endswith("_frac") for k in keys)
                 else "")
    fig.suptitle(
        f"Training design coverage — {label}, {_BOX_LABEL} ({d}-D, "
        f"{len(valid)} runs of {planned} planned, "
        f"{100*len(valid)/planned:.0f}% retained)\n"
        f"largest pairwise $|r|$ = {max_r:.3f} "
        f"({keys[i]}, {keys[j]}), outlined in red"
        f"{frac_note}",
        fontsize=10, y=0.975)

    if save:
        base = os.path.join(FIG_DIR,
                            f"figD_design_coverage_{label.replace(' ', '')}{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf   max|r|={max_r:.3f}")
    return fig


if __name__ == "__main__":
    write_reduction_table()
    write_metrics_table()
    plot_parity()
    for g in GROUPS:
        plot_coverage(g)
