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
          "r": "#33BBEE", "pressure": "#009988", "R": "#000000",
          "nu": "#AA4499", "cable_L": "#999933"}
LABEL = {"s_course": "$s_\\mathrm{course}$", "s_wale": "$s_\\mathrm{wale}$",
         "E1": "$E_1$", "r": "$E_2/E_1$", "pressure": "$p$", "R": "$R$",
         "nu": "$\\nu$", "cable_L": "cable $L_\\mathrm{rest}$"}
ORDER = ["R", "cable_L", "s_course", "s_wale", "E1", "pressure", "r", "nu"]


def load(geom):
    path = os.path.join(cfg.DATA_DIR, f"tolerance_sweep_{geom}_percent.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def curve(rows, factor, column, signed_from_base):
    """Magnitude of the error on x, 0 to 2%; deviation on y.

    Both signs are run; the curve takes the worse of the two at each magnitude,
    since a tolerance budget should quote the worse case and the responses are
    not perfectly symmetric.
    """
    base = float(next(r for r in rows if r["factor"] == "baseline")[column])
    worst = {0.0: 0.0 if signed_from_base else base}
    for r in rows:
        if r["factor"] != factor:
            continue
        v = float(r[column])
        d = abs(v - base) if signed_from_base else v
        pct = abs(float(r["percent"]))
        worst[pct] = max(worst.get(pct, 0.0), d)
    xs = sorted(worst)
    return xs, [MM * worst[x] for x in xs], base


def panel(ax, rows, column, signed_from_base, title, ylabel):
    present = {r["factor"] for r in rows}
    for factor in ORDER:
        if factor not in present:
            continue
        xs, ys, _ = curve(rows, factor, column, signed_from_base)
        ax.plot(xs, ys, "-o", ms=2.8, lw=1.5, color=COLOUR[factor],
                label=LABEL[factor],
                zorder=3 if factor in ("R", "cable_L") else 2)
    ax.set_xlabel("parameter error (% of nominal)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 2.05)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    return ax


def main():
    disc, cable = load("disc"), load("2part_cable")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))

    panel(axes[0], disc, "L_pos", True,
          "(a) circular dome",
          "deviation from the designed equilibrium (mm)")
    panel(axes[1], cable, "L_pos", True,
          "(b) creased shell as designed: optimised factors + crease cable",
          "deviation from the designed equilibrium (mm)")

    for a in axes:
        a.axvline(1.0, color="#BBBBBB", lw=0.8, ls=":", zorder=0)
    axes[0].annotate("1%", xy=(1.0, 0), xytext=(3, 6),
                     textcoords="offset points", fontsize=7.5, color="#888888")

    for a in axes:
        a.legend(ncol=2, frameon=False, loc="upper left",
                 handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()

    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    out = os.path.join(cfg.FIG_DIR, "tolerance_percent.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    # The headline numbers, so the text can quote them without re-deriving.
    print("response per 1% of nominal, surface deviation (mm):")
    for name, rows in (("disc", disc), ("2part_cable", cable)):
        print(f"  {name}")
        present = {r["factor"] for r in rows}
        for factor in ORDER:
            if factor not in present:
                continue
            xs, ys, _ = curve(rows, factor, "L_pos", True)
            at1 = [y for x, y in zip(xs, ys) if abs(x - 1.0) < 1e-9]
            print(f"    {factor:9s} {at1[0]:5.2f} mm at 1%")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
