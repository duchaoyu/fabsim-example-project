"""Figure for Section 6.5.1 — the same convergence, against solves and against time.

Panel (a) puts the loss on an evaluation-count axis, which is how an optimiser
reports itself. Panel (b) puts it on a wall-clock axis, which is what the work
actually costs. The two disagree, and the disagreement is the point: the cheap
evaluations are the ones that make progress, so counting evaluations flatters the
long tail and hides how little the last hours buy.

Reads the trajectory rebuilt by reconstruct_loss_history.py.

    python3 reconstruct_loss_history.py
    python3 figure_convergence_cost.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY = os.path.join(HERE, "data", "D5", "loss_history.json")

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
# Paul Tol vibrant, extending the house pair of analyse_block_A.py with the teal
# from the same scheme; the scheme is designed for CVD separation.
C_A, C_B, C_C = "#0077BB", "#EE7733", "#009988"
C_GREY = "#666666"
PENALTY = 1e3

RUNS = [
    ("d5_4ra_v3",   "4 adaptive regions",       C_A),
    ("d5_10lap_v4", "10 field-aligned regions", C_B),
    ("d5_sym_v1",   "10 symmetric regions",     C_C),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures",
                                                  "convergence_cost"))
    args = ap.parse_args()
    hist = json.load(open(HISTORY))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    rows = []

    for key, label, colour in RUNS:
        if key not in hist:
            print(f"  {key}: not in history, skipped")
            continue
        h = hist[key]
        loss = np.array(h["loss"]) * 1e3          # mm
        calls = np.array(h["calls"])
        mins = np.array(h["elapsed"]) / 60.0
        ok = loss < PENALTY * 1e3 * 0.5           # rejected evaluations
        best = np.minimum.accumulate(np.where(ok, loss, np.inf))

        # 99% of the total improvement, on each axis.
        span = best[0] - best[-1]
        k99 = int(np.argmax(best <= best[0] - 0.99 * span)) if span > 0 else 0

        for ax, x, xmax in ((axA, calls, calls[-1]), (axB, mins, mins[-1])):
            ax.plot(x[ok], loss[ok], linestyle="none", marker="o", markersize=1.5,
                    color=colour, alpha=0.25)
            ax.plot(x, best, linewidth=2.0, color=colour, label=label)
            ax.plot([x[k99]], [best[k99]], marker="o", markersize=6.5,
                    color=colour, markeredgecolor="white", markeredgewidth=1.8,
                    zorder=5)

        rows.append(dict(label=label, colour=colour, n=len(calls),
                         total_calls=int(h["n_evaluations"]),
                         call99=int(calls[k99]), min99=float(mins[k99]),
                         wall=float(mins[-1]),
                         frac_calls=100 * calls[k99] / h["n_evaluations"],
                         frac_time=100 * mins[k99] / mins[-1],
                         start=best[0], end=best[-1],
                         rejected=int((~ok).sum())))

    axA.set_xlabel("forward solves")
    axA.set_ylabel("loss, RMS deviation (mm)")
    axA.set_title("(a) Against evaluations", loc="left")
    axB.set_xlabel("wall clock (min)")
    axB.set_title("(b) Against wall clock", loc="left")
    for ax in (axA, axB):
        ax.set_ylim(0, 22)
        ax.grid(color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
    axA.legend(loc="upper right", frameon=False)

    # The run where the evaluation axis flatters the tail most, and the
    # warm-started run, which starts where the others finish.
    by = {r["label"]: r for r in rows}
    r = by.get("10 field-aligned regions")
    if r:
        axB.annotate(f"99% of the improvement,\nafter {r['frac_time']:.0f}% of "
                     f"the wall clock\n(vs {r['frac_calls']:.0f}% of the "
                     f"evaluations)",
                     xy=(r["min99"], r["end"]), xytext=(0.30, 0.50),
                     textcoords="axes fraction", fontsize=7.5, color=r["colour"],
                     arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                     color=r["colour"], shrinkB=6))
    r = by.get("10 symmetric regions")
    if r:
        axB.annotate("warm-started from an earlier\noptimum: begins where the\n"
                     "others finish",
                     xy=(r["wall"], r["end"]), xytext=(0.24, 0.86),
                     textcoords="axes fraction", fontsize=7.5, color=r["colour"],
                     arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                     color=r["colour"], shrinkB=6))
    r = by.get("4 adaptive regions")
    if r:
        axB.annotate("a real late gain: the last\nregion sweep still moved it",
                     xy=(r["wall"], r["end"]), xytext=(0.46, 0.09),
                     textcoords="axes fraction", fontsize=7.5, color=r["colour"],
                     ha="left",
                     arrowprops=dict(arrowstyle="->", linewidth=0.7,
                                     color=r["colour"], shrinkB=6))

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"  saved: {args.out}.{ext}")

    print(f"\n  {'run':26s} {'solves':>7s} {'wall/min':>9s} {'loss mm':>16s} "
          f"{'99% by solve':>13s} {'= % solves':>11s} {'= % time':>9s} {'rejected':>9s}")
    for r in rows:
        print(f"  {r['label']:26s} {r['total_calls']:7d} {r['wall']:9.1f} "
              f"{r['start']:7.2f} -> {r['end']:5.2f} {r['call99']:13d} "
              f"{r['frac_calls']:10.0f}% {r['frac_time']:8.0f}% {r['rejected']:9d}")


if __name__ == "__main__":
    main()
