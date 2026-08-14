"""Deviation against parameter error in percent of nominal.

The companion figure sweeps each factor in multiples of its own assumed
tolerance, which answers "what if the estimate is wrong" but hides how the
factors compare, because the assumed tolerances differ by a factor of thirty.
This one puts them on a common footing: the same relative error applied to every
parameter, so the curves may be read against each other.

That comparison reorders them. R is the steepest factor per percent on both
geometries, ahead of s_course, and looks harmless in the other figure only
because it is anchored to 0.33% where the stretch factor is assumed to 4.5%.

    python3 figure_tolerance_percent.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imperfection_config as cfg
from tolerances import TOLERANCES

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e3

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110,
})

COLOUR = {"s_course": "#CC3311", "s_wale": "#EE7733", "E1": "#0077BB",
          "r": "#33BBEE", "pressure": "#009988", "R": "#000000"}
LABEL = {"s_course": "$s_\\mathrm{course}$", "s_wale": "$s_\\mathrm{wale}$",
         "E1": "$E_1$", "r": "$E_2/E_1$", "pressure": "$p$", "R": "$R$"}
ORDER = ["R", "s_course", "s_wale", "E1", "pressure", "r"]


def load(geom):
    path = os.path.join(cfg.DATA_DIR, f"tolerance_sweep_{geom}_percent.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def curve(rows, factor, column, signed_from_base):
    """Signed percent error on x; deviation on y, through the baseline."""
    base = float(next(r for r in rows if r["factor"] == "baseline")[column])
    pts = [(0.0, 0.0 if signed_from_base else base)]
    for r in rows:
        if r["factor"] != factor:
            continue
        v = float(r[column])
        pts.append((float(r["percent"]),
                    abs(v - base) if signed_from_base else v))
    pts.sort()
    return [p[0] for p in pts], [MM * p[1] for p in pts], base


def panel(ax, rows, column, signed_from_base, title, ylabel):
    for factor in ORDER:
        xs, ys, _ = curve(rows, factor, column, signed_from_base)
        ax.plot(xs, ys, "-o", ms=2.6, lw=1.4, color=COLOUR[factor],
                label=LABEL[factor], zorder=3 if factor == "R" else 2)
    ax.axvline(0, color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("parameter error (% of nominal)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(-2.15, 2.15)
    ax.set_xticks([-2, -1, 0, 1, 2])
    return ax


def main():
    disc, twopart = load("disc"), load("2part")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    panel(axes[0], disc, "L_pos", True,
          "(a) circular dome: how far the surface moves",
          "surface deviation from baseline (mm)")
    panel(axes[1], twopart, "L_pos", True,
          "(b) creased shell: how far the surface moves",
          "surface deviation from baseline (mm)")
    ax = panel(axes[2], twopart, "L_target", False,
               "(c) creased shell: deviation from the design target",
               "deviation from design target (mm)")

    base_t = MM * float(next(r for r in twopart
                             if r["factor"] == "baseline")["L_target"])
    ax.axhline(base_t, color="#000000", lw=1.0, ls="--", zorder=0)
    ax.annotate(f"standing mismatch, {base_t:.1f} mm",
                xy=(-2.05, base_t), xytext=(0, 5), textcoords="offset points",
                fontsize=7.5)
    ax.set_ylim(0, 1.35 * base_t)

    # Where each factor's assumed tolerance actually sits, for the two that fall
    # inside the frame; the rest are named in the caption.
    for a in axes[:2]:
        a.axvspan(-1, 1, color="#F2F2F2", zorder=0)
    axes[0].annotate("±1%", xy=(1.0, axes[0].get_ylim()[1]), xytext=(3, -10),
                     textcoords="offset points", fontsize=7.5, color="#666666")

    axes[0].legend(ncol=2, frameon=False, loc="upper center",
                   handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()

    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    out = os.path.join(cfg.FIG_DIR, "tolerance_percent.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # The headline numbers, so the text can quote them without re-deriving.
    print("response per 1% of nominal, surface deviation (mm):")
    for name, rows in (("disc", disc), ("2part", twopart)):
        print(f"  {name}")
        for factor in ORDER:
            xs, ys, _ = curve(rows, factor, "L_pos", True)
            at1 = [y for x, y in zip(xs, ys) if abs(x - 1.0) < 1e-9]
            print(f"    {factor:9s} {at1[0]:5.2f} mm at +1%")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
