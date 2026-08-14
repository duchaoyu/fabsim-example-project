"""Deviation against tolerance magnitude, rather than at one assumed tolerance.

The Block A table reports each factor's response at plus and minus one assumed
delta. Five of the six deltas are estimates, so the question a reader actually
has is what the answer becomes if an estimate is wrong. These curves answer it
directly: read up from whatever tolerance turns out to be right.

The creased shell panel carries the argument of the section. Its deviation from
the design target starts at the standing mismatch and only rises: no tolerance,
at any magnitude in range, brings the built shape closer to what was designed.

    python3 figure_tolerance_sweep.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imperfection_config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
MM = 1e3

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110,
})

# One colour per factor, stable across panels.
COLOUR = {"s_course": "#CC3311", "s_wale": "#EE7733", "E1": "#0077BB",
          "r": "#33BBEE", "pressure": "#009988", "R": "#BBBBBB"}
LABEL = {"s_course": "$s_\\mathrm{course}$", "s_wale": "$s_\\mathrm{wale}$",
         "E1": "$E_1$", "r": "$E_2/E_1$", "pressure": "$p$", "R": "$R$"}
ORDER = ["s_course", "s_wale", "E1", "r", "pressure", "R"]


def load(geom):
    path = os.path.join(cfg.DATA_DIR, f"tolerance_sweep_{geom}.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def series(rows, factor, column):
    """(multiple, value) for one factor, worse sign at each magnitude.

    The two signs differ by up to 17% on some factors, and a tolerance budget
    should quote the worse one, so the curve is the envelope rather than a mean.
    """
    by_mult = {}
    for r in rows:
        if r["factor"] != factor:
            continue
        m = abs(float(r["multiple"]))
        v = abs(float(r[column]) - by_mult.get("__base__", 0.0))
        by_mult[m] = max(by_mult.get(m, 0.0), v)
    base = next(r for r in rows if r["factor"] == "baseline")
    out = [(0.0, 0.0)]
    for m in sorted(by_mult):
        out.append((m, by_mult[m]))
    return [p[0] for p in out], [p[1] for p in out], float(base[column])


def deviation_curve(rows, factor, column, relative_to_base):
    xs, ys = [0.0], [0.0]
    base = float(next(r for r in rows if r["factor"] == "baseline")[column])
    by_mult = {}
    for r in rows:
        if r["factor"] != factor:
            continue
        m = abs(float(r["multiple"]))
        v = float(r[column])
        d = abs(v - base) if relative_to_base else v
        by_mult[m] = max(by_mult.get(m, 0.0), d)
    for m in sorted(by_mult):
        xs.append(m)
        ys.append(by_mult[m])
    if not relative_to_base:
        ys[0] = base
    return xs, ys, base


def panel(ax, rows, column, relative_to_base, title, ylabel):
    for factor in ORDER:
        xs, ys, base = deviation_curve(rows, factor, column, relative_to_base)
        ax.plot(xs, [MM * y for y in ys], "-o", ms=3, lw=1.4,
                color=COLOUR[factor], label=LABEL[factor])
    ax.axvline(1.0, color="#666666", lw=0.8, ls=":", zorder=0)
    ax.annotate("assumed\ntolerance", xy=(1.0, ax.get_ylim()[1]),
                xytext=(2, -2), textcoords="offset points",
                fontsize=7, color="#666666", va="top")
    ax.set_xlabel("tolerance, as a multiple of the assumed value")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, 2.05)
    return ax


def main():
    disc, twopart = load("disc"), load("2part")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

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
    ax.annotate(f"standing mismatch at the nominal, {base_t:.1f} mm",
                xy=(0.05, base_t), xytext=(0, 6), textcoords="offset points",
                fontsize=7.5)
    ax.set_ylim(0, 1.25 * max(base_t, ax.get_ylim()[1]))

    axes[0].legend(ncol=2, frameon=False, loc="upper left",
                   handlelength=1.4, columnspacing=1.0)

    fig.tight_layout()
    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    out = os.path.join(cfg.FIG_DIR, "tolerance_sweep.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved: {out}")
    print(f"  saved: {out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()
