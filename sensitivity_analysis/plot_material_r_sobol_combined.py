"""
Combine the four material-r Sobol heatmaps into one 2x2 figure.

Panels:  (a) no cable, S_T    (b) cable, S_T
         (c) no cable, S_1    (d) cable, S_1

Reads the per-output index tables written by run_material_r_sobol.py
(data/sobol_material_r_{group}_{output}.csv) — no FEA, surrogate or Sobol
re-run needed.  Panels are laid out with a fixed cell size so all four share
the same visual scale even though the cable group has two extra parameter rows
and two extra output columns; one shared colorbar at the bottom.

Usage:
    python3 plot_material_r_sobol_combined.py
Products:
    figures/fig3_sobol_material_r_combined.{pdf,png}
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DATA_DIR
from visualization import PARAM_LABELS, OUTPUT_LABELS, FIG_DIR

# The model-validity box (stretch factors from 0.95; config.PARAMS_MATERIAL_R_
# VALID_*) is the default, because the full box includes slack states where the
# membrane carries compression it physically cannot — see run_material_r_valid.py.
# --full-box reproduces the original figures under a _fullbox suffix.  Read from
# argv at import so the regime-map and validation scripts inherit it.
VALID      = "--full-box" not in sys.argv
CSV_PREFIX = "valid_" if VALID else ""
SUR_PREFIX = "valid_" if VALID else ""
FIG_SUFFIX = "" if VALID else "_fullbox"
_BOUNDS = ((config.PARAMS_MATERIAL_R_VALID_NO_CABLE,
            config.PARAMS_MATERIAL_R_VALID_CABLE) if VALID else
           (config.PARAMS_MATERIAL_R_NO_CABLE, config.PARAMS_MATERIAL_R_CABLE))
PARAMS_MATERIAL_R_NO_CABLE, PARAMS_MATERIAL_R_CABLE = _BOUNDS

OUTPUT_LABELS = dict(OUTPUT_LABELS)
OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")

# The cable tensions are zero-inflated, not merely heavy-tailed: L_rest spans
# (1.2, 1.4) m against a 1.29 m flat arc, so 38% of cable runs leave the cable
# slack at T = 0 exactly (see run_sobol_robust.py).  log T is therefore not
# defined on this output — analysing log(clip(T, 1e-9, inf)) collapses the whole
# slack plateau onto a spike at -20.72 that carries 98% of the variance, so the
# indices then measure what trips the clip.  Use log1p(T / T_REF), which is
# defined at T = 0: kurtosis 1.9 -> -0.9, top-1% variance share 13% -> 3%, and
# drift max|ST(4096) - ST(1024)| 0.027 -> 0.015.  --raw-tensions opts out.
LOG_TENSIONS = "--raw-tensions" not in sys.argv
T_REF        = 1.0     # newtons; keep in step with run_sobol_robust.T_REF
if LOG_TENSIONS:
    OUTPUT_LABELS["cable_wale_tension"]   = r"$\log(1{+}T_\mathrm{wale})$"
    OUTPUT_LABELS["cable_course_tension"] = r"$\log(1{+}T_\mathrm{course})$"


def tension_scale(Y):
    """The scale the study reports the cable tensions on.

    Defined at T = 0, so the slack runs stay a point mass at 0 instead of
    becoming an artificial spike at log(1e-9).  Tension is clamped at 0 by
    surrogate._NONNEG_OUTPUTS; the maximum here only guards round-off.
    """
    return np.log1p(np.maximum(Y, 0.0) / T_REF)

# E1 is the modulus along face_dirs, which anisotropic_rest_shape.h and
# fem_batch_sensitivity.cpp both take as the wale direction — name it on the row.
PARAM_LABELS = dict(PARAM_LABELS)
PARAM_LABELS["E1"] = PARAM_LABELS["E1"] + "\n(wale)"

GROUPS = {
    "material_r_nocable": ("no cable", PARAMS_MATERIAL_R_NO_CABLE),
    "material_r_cable":   ("cable",    PARAMS_MATERIAL_R_CABLE),
}

BASE_OUTPUTS = ["crown_height", "H_mean_x0", "H_mean_y0", "H_anisotropy",
                "max_stress", "mean_stress", "boundary_reaction_mean"]
CABLE_OUTPUTS = ["cable_wale_tension", "cable_course_tension"]

_ST_MIN = 0.02          # drop columns that nothing moves (same rule as plot_sobol)

# Cell / margin geometry, inches
CELL_W, CELL_H = 0.68, 0.58
PAD_LEFT, PAD_RIGHT = 1.05, 0.25
COL_GAP = 1.05          # clears the right panel's y labels
PAD_TOP, ROW_GAP = 0.85, 1.05   # ROW_GAP clears the rotated x labels
PAD_BOT = 1.95          # rotated x labels of the bottom row + colorbar


def _is_masked(p, o):
    """A cable's rest length cannot act on the other cable's tension."""
    # Both rest-length conventions: metres (legacy) and fraction of the
    # cable-free section (current).  A name missing here silently stops masking
    # rather than erroring, so match the pair by suffix.
    return ((p in ("cable_wale_lrest", "cable_wale_frac")
             and o == "cable_course_tension") or
            (p in ("cable_course_lrest", "cable_course_frac")
             and o == "cable_wale_tension"))


def load_group(group):
    outputs = BASE_OUTPUTS + (CABLE_OUTPUTS if group.endswith("_cable") else [])
    bounds_keys = list(GROUPS[group][1])
    tables = {}
    for col in outputs:
        path = os.path.join(DATA_DIR, f"sobol_{group}_{CSV_PREFIX}{col}.csv")
        if LOG_TENSIONS and col in CABLE_OUTPUTS:
            log_path = os.path.join(DATA_DIR,
                                    f"sobol_{group}_{CSV_PREFIX}log1p_{col}.csv")
            if os.path.exists(log_path):
                path = log_path
            else:
                print(f"  {col}: no log1p table, using raw indices")
        if not os.path.exists(path):
            print(f"  missing, skipped: {os.path.basename(path)}")
            continue
        df = pd.read_csv(path, index_col=0)
        # A cached table written under a different parameter naming (the cable
        # rest length was cable_*_lrest in metres before it became cable_*_frac)
        # indexes rows the current box does not contain.  That used to surface as
        # a KeyError deep in _matrices; refuse the stale table here instead, and
        # say which one, so the fix is obvious.
        missing = [p for p in bounds_keys if p not in df.index]
        if missing:
            print(f"  {col}: {os.path.basename(path)} is indexed by a different "
                  f"parameter set (missing {missing}) — stale, skipping. "
                  f"Re-run run_sobol_robust.py / run_material_r_valid.py.")
            continue
        if df["ST"].clip(0).max() <= _ST_MIN:
            print(f"  {col}: ST_max <= {_ST_MIN}, column dropped")
            continue
        tables[col] = df
    return tables


def _matrices(tables, param_names, out_names, index):
    conf_key = index + "_conf"
    n_p, n_o = len(param_names), len(out_names)
    mat  = np.zeros((n_p, n_o))
    conf = np.zeros((n_p, n_o))
    mask = np.zeros((n_p, n_o), dtype=bool)
    for j, col in enumerate(out_names):
        df = tables[col]
        for i, p in enumerate(param_names):
            if _is_masked(p, col):
                mask[i, j] = True
                continue
            mat[i, j]  = max(0.0, df.loc[p, index])
            conf[i, j] = df.loc[p, conf_key]
    return np.where(mask, np.nan, mat), conf, mask


def _draw_panel(ax, mat, conf, mask, param_names, out_names, title):
    im = ax.imshow(mat, cmap=plt.cm.YlOrRd, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mask[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="lightgrey", hatch="////",
                                       edgecolor="grey", linewidth=0.0, zorder=2))
                continue
            v, c = mat[i, j], conf[i, j]
            color = "white" if v > 0.6 else "black"
            ax.text(j, i - 0.20, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold")
            ax.text(j, i + 0.20, f"±{c:.2f}", ha="center", va="center",
                    fontsize=6.0, color=color)

    ax.set_xticks(range(len(out_names)))
    ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names], fontsize=9)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(out_names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(param_names)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, fontsize=10.5, pad=5)
    return im


def plot_combined(save=True):
    panels = {}
    for group in GROUPS:
        print(f"[{group}]")
        tables = load_group(group)
        if not tables:
            raise SystemExit(f"no Sobol tables found for {group}")
        panels[group] = tables

    groups   = list(GROUPS)
    param_of = {g: list(GROUPS[g][1].keys()) for g in groups}
    outs_of  = {g: [c for c in (BASE_OUTPUTS +
                                (CABLE_OUTPUTS if g.endswith("_cable") else []))
                    if c in panels[g]] for g in groups}

    # ── layout, in inches ─────────────────────────────────────────────────────
    col_w = [len(outs_of[g])  * CELL_W for g in groups]
    col_h = [len(param_of[g]) * CELL_H for g in groups]
    row_h = max(col_h)
    fig_w = PAD_LEFT + col_w[0] + COL_GAP + col_w[1] + PAD_RIGHT
    fig_h = PAD_TOP + row_h + ROW_GAP + row_h + PAD_BOT

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8, "figure.dpi": 150,
        "axes.spines.top": True, "axes.spines.right": True,
    })
    fig = plt.figure(figsize=(fig_w, fig_h))

    col_left = [PAD_LEFT, PAD_LEFT + col_w[0] + COL_GAP]
    row_top  = [fig_h - PAD_TOP, fig_h - PAD_TOP - row_h - ROW_GAP]

    tags = ["a", "b", "c", "d"]
    im = None
    for ri, index in enumerate(["ST", "S1"]):
        for ci, group in enumerate(groups):
            gname = GROUPS[group][0]
            params, outs = param_of[group], outs_of[group]
            mat, conf, mask = _matrices(panels[group], params, outs, index)

            h = col_h[ci]
            ax = fig.add_axes([col_left[ci] / fig_w,
                               (row_top[ri] - h) / fig_h,
                               col_w[ci] / fig_w,
                               h / fig_h])
            idx_lab = r"$S_T$" if index == "ST" else r"$S_1$"
            tag = tags[ri * 2 + ci]
            im = _draw_panel(ax, mat, conf, mask, params, outs,
                             f"({tag}) {gname} — {idx_lab}")

    # ── shared colorbar ───────────────────────────────────────────────────────
    cbar_w = 0.42 * fig_w
    cbar_ax = fig.add_axes([(fig_w - cbar_w) / 2 / fig_w, 0.40 / fig_h,
                            cbar_w / fig_w, 0.16 / fig_h])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.set_label("Sobol sensitivity indices", fontsize=10)

    fig.suptitle("Sobol sensitivity indices",
                 fontsize=13, y=1 - 0.18 / fig_h)

    if save:
        base = os.path.join(FIG_DIR,
                            f"fig3_sobol_material_r_combined{FIG_SUFFIX}")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    plot_combined()
