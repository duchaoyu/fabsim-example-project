"""
Sobol sensitivity regime map for the material-r study (r = E2/E1).

Same panels, parameters and outputs as fig3_sobol_material_r_combined, but each
cell is coloured by *regime* rather than by one index:

    colour = S1 * blue + (ST - S1) * orange + (1 - ST) * grey

so a blue cell is direct sensitivity, an orange cell is sensitivity that lives
almost entirely in interactions, and a grey cell is inert.  Italic text names
the dominant second-order partner where the interaction share is non-trivial.

S1/ST come from the cached data/sobol_material_r_{group}_{output}.csv tables, so
the numbers printed here are the same ones in fig3_sobol_material_r_combined.
S2 is not in those tables (they were produced with calc_second_order=False), so
the partner is taken from a separate, smaller second-order Saltelli run on the
same surrogates — it only has to rank partners, not report a value.

Usage:
    python3 plot_material_r_regime.py [--n-s2 256]
Products:
    figures/fig3_sobol_material_r_regime.{pdf,png}
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, PARAMS_MATERIAL_R_NO_CABLE, PARAMS_MATERIAL_R_CABLE
from surrogate import ScalarSurrogate
from visualization import OUTPUT_LABELS, FIG_DIR
from plot_material_r_sobol_combined import (
    GROUPS, BASE_OUTPUTS, CABLE_OUTPUTS, PARAM_LABELS, load_group, _is_masked,
)
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

OUTPUT_LABELS = dict(OUTPUT_LABELS)
OUTPUT_LABELS.setdefault("H_anisotropy", r"$\Delta H$")

# Row labels carry units ("$E_1$ (N/m)"); the in-cell partner tags should not.
PARTNER_LABELS = {p: lab.split(" (")[0] for p, lab in PARAM_LABELS.items()}

BLUE   = np.array([0.18, 0.45, 0.73])
ORANGE = np.array([0.96, 0.50, 0.06])
GREY   = np.array([0.88, 0.88, 0.88])

# The blend  S1*blue + (ST-S1)*orange + (1-ST)*grey  is identical to mixing the
# hue  (S1/ST)*blue + (1-S1/ST)*orange  onto grey with weight ST.  Keeping that
# weight linear in ST left the whole figure grey: half the cells sit below
# ST = 0.2, and a 0.4 cell only got 40% colour, so nothing stood out.  Raising
# the weight to ST**GAMMA with GAMMA < 1 saturates the informative cells
# (0.4 -> 0.58, 0.7 -> 0.81) while ST < 0.05 stays grey.  Hue still means
# direct-vs-interaction and saturation still means "how sensitive"; only the
# saturation ramp is non-linear.
GAMMA = 0.6

# An interaction share below this is not worth naming a partner for
_INTERACTION_MIN = 0.05
_S2_MIN          = 0.02

# Cell / margin geometry, inches.  Cells are taller than in the combined figure
# because each one carries three lines of text.
CELL_W, CELL_H = 0.95, 0.80
PAD_LEFT, PAD_RIGHT = 1.05, 0.25
COL_GAP = 1.10
PAD_TOP, PAD_BOT = 0.80, 2.45   # bottom holds x labels + legend + key + caption


def s2_partners(group, bounds, outputs, n_base):
    """{output: {param: partner_param}} from a second-order Saltelli run."""
    path = os.path.join(DATA_DIR, f"{group}_surrogate.pkl")
    if not os.path.exists(path):
        print(f"  no surrogate at {path} — partners omitted")
        return {}

    names   = list(bounds.keys())
    problem = {"num_vars": len(names), "names": names,
               "bounds": [list(v) for v in bounds.values()]}
    X = saltelli.sample(problem, n_base, calc_second_order=True)
    print(f"  {group}: {len(X)} surrogate evaluations for S2")
    preds = ScalarSurrogate.load(path).predict(X)

    out = {}
    for col in outputs:
        if col not in preds or np.std(preds[col]) < 1e-10:
            continue
        si = sobol_analyze.analyze(problem, preds[col], calc_second_order=True,
                                   print_to_console=False)
        S2 = np.array(si["S2"], dtype=float)
        # S2 is upper-triangular with NaN elsewhere; symmetrise for lookup
        S2 = np.where(np.isnan(S2), 0.0, S2)
        S2 = S2 + S2.T
        col_partners = {}
        for i, p in enumerate(names):
            row = S2[i].copy()
            row[i] = -1.0
            j = int(np.argmax(row))
            if row[j] > _S2_MIN:
                col_partners[p] = names[j]
        out[col] = col_partners
    return out


def _cell_rgb(s1, st, gamma=GAMMA):
    """Hue from the direct share S1/ST, saturation from ST**gamma."""
    if st <= 0:
        return GREY.copy()
    frac = np.clip(s1 / st, 0.0, 1.0)
    hue  = frac * BLUE + (1.0 - frac) * ORANGE
    w    = st ** gamma
    return np.clip(w * hue + (1.0 - w) * GREY, 0, 1)


def _draw_panel(ax, tables, params, outs, partners, title, gamma=GAMMA):
    n_p, n_o = len(params), len(outs)
    rgb = np.zeros((n_p, n_o, 3))
    s1m = np.zeros((n_p, n_o))
    stm = np.zeros((n_p, n_o))
    mask = np.zeros((n_p, n_o), dtype=bool)

    for i, p in enumerate(params):
        for j, o in enumerate(outs):
            if _is_masked(p, o):
                mask[i, j] = True
                rgb[i, j] = GREY * 0.7
                continue
            s1 = float(np.clip(tables[o].loc[p, "S1"], 0, 1))
            st = float(np.clip(tables[o].loc[p, "ST"], 0, 1))
            st = max(st, s1)
            s1m[i, j], stm[i, j] = s1, st
            rgb[i, j] = _cell_rgb(s1, st, gamma)

    ax.imshow(rgb, aspect="auto", interpolation="nearest",
              extent=(-0.5, n_o - 0.5, n_p - 0.5, -0.5))

    for i, p in enumerate(params):
        for j, o in enumerate(outs):
            if mask[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="lightgrey", hatch="////",
                                       edgecolor="grey", linewidth=0.0, zorder=2))
                continue
            s1, st = s1m[i, j], stm[i, j]
            lum = float(rgb[i, j] @ np.array([0.299, 0.587, 0.114]))
            fc  = "white" if lum < 0.55 else "black"
            ax.text(j, i - 0.22, r"$S_T$={:.2f}".format(st), ha="center",
                    va="center", fontsize=8.5, color=fc, fontweight="bold")
            ax.text(j, i + 0.04, r"$S_1$={:.2f}".format(s1), ha="center",
                    va="center", fontsize=8, color=fc)
            partner = partners.get(o, {}).get(p) if st - s1 > _INTERACTION_MIN else None
            if partner:
                ax.text(j, i + 0.30,
                        "w. " + PARTNER_LABELS.get(partner, partner),
                        ha="center", va="center", fontsize=7, color=fc,
                        style="italic")

    ax.set_xticks(range(n_o))
    ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in outs],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in params], fontsize=9)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_o), minor=True)
    ax.set_yticks(np.arange(-0.5, n_p), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, fontsize=10.5, pad=5)


def plot_regime(n_s2=256, gamma=GAMMA, save=True):
    groups   = list(GROUPS)
    tables   = {}
    partners = {}
    for g in groups:
        print(f"[{g}]")
        tables[g] = load_group(g)
        if not tables[g]:
            raise SystemExit(f"no Sobol tables found for {g}")
        partners[g] = s2_partners(g, GROUPS[g][1], list(tables[g]), n_s2)

    param_of = {g: list(GROUPS[g][1].keys()) for g in groups}
    outs_of  = {g: [c for c in (BASE_OUTPUTS +
                                (CABLE_OUTPUTS if g.endswith("_cable") else []))
                    if c in tables[g]] for g in groups}

    col_w = [len(outs_of[g])  * CELL_W for g in groups]
    col_h = [len(param_of[g]) * CELL_H for g in groups]
    fig_w = PAD_LEFT + col_w[0] + COL_GAP + col_w[1] + PAD_RIGHT
    fig_h = PAD_TOP + max(col_h) + PAD_BOT

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.linewidth": 0.8, "figure.dpi": 150,
        "axes.spines.top": True, "axes.spines.right": True,
    })
    fig = plt.figure(figsize=(fig_w, fig_h))
    col_left = [PAD_LEFT, PAD_LEFT + col_w[0] + COL_GAP]
    top      = fig_h - PAD_TOP

    for ci, g in enumerate(groups):
        ax = fig.add_axes([col_left[ci] / fig_w, (top - col_h[ci]) / fig_h,
                           col_w[ci] / fig_w, col_h[ci] / fig_h])
        tag = "ab"[ci]
        _draw_panel(ax, tables[g], param_of[g], outs_of[g], partners[g],
                    f"({tag}) {GROUPS[g][0]}", gamma)

    legend = [
        Patch(facecolor=BLUE,   label=r"Direct ($S_1/S_T$ large)"),
        Patch(facecolor=ORANGE, label=r"Interaction ($S_1/S_T$ small)"),
        Patch(facecolor=GREY,   label=r"Negligible ($S_T \to 0$)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 1.45 / fig_h))

    # Saturation key: the same ramp the cells use, so a reader can invert the
    # non-linear mapping by eye instead of trusting a formula.
    ticks = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    strip = np.array([[_cell_rgb(0.0, t, gamma) for t in ticks],
                      [_cell_rgb(t,   t, gamma) for t in ticks]])
    key_w = 0.30 * fig_w
    key_ax = fig.add_axes([(fig_w - key_w) / 2 / fig_w, 0.90 / fig_h,
                           key_w / fig_w, 0.28 / fig_h])
    key_ax.imshow(strip, aspect="auto", interpolation="nearest",
                  extent=(-0.5, len(ticks) - 0.5, 1.5, -0.5))
    key_ax.set_xticks(range(len(ticks)))
    key_ax.set_xticklabels([f"{t:.1f}" for t in ticks], fontsize=8)
    key_ax.set_yticks([0, 1])
    key_ax.set_yticklabels(["interaction", "direct"], fontsize=8)
    key_ax.tick_params(length=0)
    key_ax.set_xlabel(r"$S_T$  (saturation $\propto S_T^{%.1f}$)" % gamma,
                      fontsize=9, labelpad=2)
    for s in key_ax.spines.values():
        s.set_linewidth(0.6)

    fig.text(0.5, 0.14 / fig_h,
             r"Hue = direct share $S_1/S_T$ (blue $\to$ orange), "
             r"saturation = $S_T^{%.1f}$ over grey;  "
             r"italic = dominant $S_2$ partner" % gamma,
             ha="center", va="bottom", fontsize=10)

    fig.suptitle("Sobol sensitivity regime map", fontsize=13,
                 y=1 - 0.18 / fig_h)

    if save:
        base = os.path.join(FIG_DIR, "fig3_sobol_material_r_regime")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
        print(f"Saved: {base}.png / .pdf")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s2", type=int, default=256,
                    help="Saltelli base N for the second-order partner run")
    ap.add_argument("--gamma", type=float, default=GAMMA,
                    help="saturation exponent; 1.0 is the old linear ramp, "
                         "smaller means more colour at mid/high ST")
    args = ap.parse_args()
    plot_regime(n_s2=args.n_s2, gamma=args.gamma)
