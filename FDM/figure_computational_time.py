"""Figure for Section 6.5.1 — where the wall-clock time of the inverse problem goes.

The optimisation drivers never recorded timings, but every objective evaluation
wrote <prefix>_NNNNN_verts.csv, so two things are recoverable from the run
directory alone:

  * the cost of a converged evaluation — the interval between two consecutively
    numbered output files;
  * which evaluations failed — a missing index. A non-converged inner solve is
    abandoned at the driver's wall-clock limit and writes nothing, so the gaps in
    the numbering are exactly the timeouts.

Only the D5 runs kept their original mtimes; the B5 and C5 output files were
rewritten on 2026-05-12 and their wall clocks are lost.

    python3 figure_computational_time.py [--out figures/computational_time]
"""

import argparse
import collections
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIM_DIR = os.path.join(HERE, "optimisation")

# House style, matching imperfection_study/analyse_block_A.py so the figure sits
# with the rest of the chapter.
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
C_PLUS, C_MINUS = "#0077BB", "#EE7733"
C_GREY = "#666666"

# The runs that kept their timestamps, with the wall-clock limit the driver used.
RUNS = [
    ("d5_4ra_v3",   "4 adaptive regions",     40.0, C_PLUS),
    ("d5_10lap_v4", "10 field-aligned regions", 40.0, C_MINUS),
]


def read_run(prefix):
    """Return (index array, mtime array, failed index array) for one run."""
    pat = re.compile(rf"{re.escape(prefix)}_(\d{{5}})_verts\.csv$")
    stamps = {}
    for f in os.listdir(OPTIM_DIR):
        m = pat.match(f)
        if m:
            stamps[int(m.group(1))] = os.path.getmtime(os.path.join(OPTIM_DIR, f))
    if not stamps:
        return None
    idx = np.array(sorted(stamps))
    t = np.array([stamps[i] for i in idx])
    present = set(idx.tolist())
    failed = np.array([i for i in range(1, int(idx.max()) + 1) if i not in present])
    return idx, t, failed


def converged_costs(idx, t):
    """Cost of each evaluation whose predecessor is the immediately previous index.

    Restricting to steps of one keeps a timeout's wall time out of the following
    evaluation's cost, which is what makes these numbers the cost of a *converged*
    solve rather than a blend of the two populations.
    """
    step = np.diff(idx) == 1
    dt = np.diff(t)[step]
    at = idx[1:][step]
    keep = dt > 0
    return at[keep], dt[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "computational_time"))
    args = ap.parse_args()

    data = []
    for prefix, label, limit, colour in RUNS:
        r = read_run(prefix)
        if r is None:
            print(f"  skipped {prefix}: no output files found")
            continue
        idx, t, failed = r
        at, dt = converged_costs(idx, t)
        total = t[-1] - t[0]
        n_eval = int(idx.max())
        data.append(dict(prefix=prefix, label=label, limit=limit, colour=colour,
                         idx=idx, at=at, dt=dt, failed=failed,
                         total=total, n_eval=n_eval, n_ok=len(idx)))

    if not data:
        sys.exit("no runs with usable timestamps")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # ── Left: the cost of every evaluation over the course of one run ──────────
    d = data[0]
    axL.plot(d["at"], d["dt"], linestyle="none", marker="o", markersize=1.6,
             color=d["colour"], alpha=0.55, label="converged solve")
    axL.axhline(d["limit"], color=C_GREY, linewidth=0.8, linestyle="--")
    axL.text(d["n_eval"] * 0.99, d["limit"] * 1.12,
             f"driver limit, {d['limit']:.0f} s", color=C_GREY, fontsize=7,
             ha="right", va="bottom")
    # Failed evaluations: a rug at the ceiling, since each one spent the full limit.
    axL.plot(d["failed"], np.full(len(d["failed"]), d["limit"]), linestyle="none",
             marker="|", markersize=5, markeredgewidth=0.7, color=C_GREY,
             label=f"abandoned at the limit (n={len(d['failed'])})")
    med = float(np.median(d["dt"]))
    axL.axhline(med, color=d["colour"], linewidth=0.8)
    axL.text(d["n_eval"] * 0.01, med * 0.72, f"median {med:.2f} s",
             color=d["colour"], fontsize=7, ha="left", va="top")
    axL.set_yscale("log")
    axL.set_xlabel("forward solve")
    axL.set_ylabel("cost of one evaluation (s)")
    axL.set_title(f"(a) {d['label']}: {d['n_eval']} evaluations, "
                  f"{d['total']/60:.0f} min", loc="left")
    axL.legend(loc="lower right", frameon=False, handletextpad=0.4,
               borderaxespad=0.2)
    axL.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axL.set_axisbelow(True)

    # ── Right: how concentrated that wall clock is ─────────────────────────────
    for k, d in enumerate(data):
        # Every evaluation, costed: converged ones at their measured dt, failed
        # ones at the driver limit. Sorted cheapest-first, then accumulated.
        costs = np.concatenate([d["dt"], np.full(len(d["failed"]), d["limit"])])
        costs = np.sort(costs)
        frac_eval = np.arange(1, len(costs) + 1) / len(costs)
        frac_time = np.cumsum(costs) / costs.sum()
        axR.plot(100 * frac_eval, 100 * frac_time, linewidth=2.0,
                 color=d["colour"], label=d["label"])
        # Mark the share of wall time carried by the evaluations over 15 s.
        slow = costs > 15.0
        if slow.any():
            x = 100 * (1 - slow.sum() / len(costs))
            y = 100 * (1 - costs[slow].sum() / costs.sum())
            axR.plot([x], [y], marker="o", markersize=6, color=d["colour"],
                     markeredgecolor="white", markeredgewidth=2.0, zorder=5)
            axR.annotate(f"{100*slow.sum()/len(costs):.0f}% of evaluations "
                         f"carry {100*costs[slow].sum()/costs.sum():.0f}% "
                         f"of the wall clock",
                         xy=(x, y), xytext=(-14, 30 + 34 * k),
                         textcoords="offset points",
                         fontsize=7, color=d["colour"], ha="right",
                         arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                         color=d["colour"],
                                         shrinkA=2, shrinkB=3))
    axR.plot([0, 100], [0, 100], color=C_GREY, linewidth=0.8, linestyle=":")
    axR.text(52, 46, "uniform cost", color=C_GREY, fontsize=7, rotation=38,
             ha="center", va="top")
    axR.set_xlim(0, 100)
    axR.set_ylim(0, 100)
    axR.set_xlabel("evaluations, cheapest first (%)")
    axR.set_ylabel("cumulative wall clock (%)")
    axR.set_title("(b) The tail is the cost", loc="left")
    axR.legend(loc="upper left", frameon=False)
    axR.grid(color="#DDDDDD", linewidth=0.5)
    axR.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
        print(f"  saved: {args.out}.{ext}")

    print("\nNumbers behind the figure")
    print(f"  {'run':14s} {'evals':>6s} {'converged':>10s} {'failed':>7s} "
          f"{'wall/min':>9s} {'median':>7s} {'p90':>7s} {'>15s share':>11s}")
    for d in data:
        costs = np.concatenate([d["dt"], np.full(len(d["failed"]), d["limit"])])
        slow = costs > 15.0
        print(f"  {d['prefix']:14s} {d['n_eval']:6d} {d['n_ok']:10d} "
              f"{len(d['failed']):7d} {d['total']/60:9.1f} "
              f"{np.median(d['dt']):7.2f} {np.percentile(d['dt'], 90):7.2f} "
              f"{100*costs[slow].sum()/costs.sum():10.0f}%")


if __name__ == "__main__":
    main()
