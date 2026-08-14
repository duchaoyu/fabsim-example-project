"""How the fit degrades as the same design is built at a larger span.

The free-form shell of Section 6.4.2 was optimised at four spans against a target
scaled with it, so the question is not whether a bigger shell is harder to fit -
everything is bigger - but whether it is *proportionally* harder. It is.

Reads:
    optimisation/B5_multiscale_summary.json   per-span optimum, cost, tensions
    optimisation/check_<tag>_verts.csv        verification solve at that optimum
    ../data/B5_remeshed_shared[_<tag>].off    the target at that span
    ../data/B5_remeshed_interior_idx.npy      the 434 interior vertices of 497

Writes figures/scalability.{pdf,png} and data/scalability.json, the latter so the
section text and this figure quote one set of numbers.

    python3 figure_scalability.py
"""

import csv
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MM = 1e3

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})

C_MEAN, C_MAX, C_COST = "#0077BB", "#CC3311", "#009988"

SPANS = [("1p2m", 1.2, "B5_remeshed_shared.off"),
         ("1p5m", 1.5, "B5_remeshed_shared_1p5m.off"),
         ("2p0m", 2.0, "B5_remeshed_shared_2p0m.off"),
         ("3p0m", 3.0, "B5_remeshed_shared_3p0m.off")]


def load_off(path):
    t = open(path).read().split()
    nv, nf = int(t[1]), int(t[2])
    V = np.array([[float(t[4 + 3 * k]), float(t[5 + 3 * k]),
                   float(t[6 + 3 * k])] for k in range(nv)])
    o = 4 + 3 * nv
    F = np.array([[int(t[o + 4 * k + 1]), int(t[o + 4 * k + 2]),
                   int(t[o + 4 * k + 3])] for k in range(nf)])
    return V, F


def load_verts(path):
    rows = sorted(((int(r["vid"]), (float(r["x"]), float(r["y"]), float(r["z"])))
                   for r in csv.DictReader(open(path))), key=lambda t: t[0])
    return np.array([xyz for _, xyz in rows])


def collect():
    summary = {d["tag"]: d for d in json.load(
        open(os.path.join(HERE, "optimisation", "B5_multiscale_summary.json")))}
    interior = np.load(os.path.join(ROOT, "data",
                                    "B5_remeshed_interior_idx.npy"))
    out = []
    for tag, D, target in SPANS:
        V = load_verts(os.path.join(HERE, "optimisation",
                                    f"check_{tag}_verts.csv"))
        T, F = load_off(os.path.join(ROOT, "data", target))
        d = np.linalg.norm(V[interior] - T[interior], axis=1)
        s = summary[tag]
        out.append(dict(
            tag=tag, D=D, mean_mm=MM * d.mean(), max_mm=MM * d.max(),
            mean_pct=100 * d.mean() / D, max_pct=100 * d.max() / D,
            n_calls=s["n_calls"], converged=s["converged"],
            T_course=s["tension_global"]["T_course_mean_Npm"],
            T_wale=s["tension_global"]["T_wale_mean_Npm"],
            crown_ratio=s["final_crown_m"] / D,
            target_crown_ratio=s["target_crown_m"] / D,
            sf_wale_max=max(r["sf_wale"] for r in s["regions"]),
            sf_course_max=max(r["sf_course"] for r in s["regions"]),
            dev=d, V=V, T=T, F=F, interior=interior))
    return out


def exponent(x, y):
    """Slope of a log-log fit: y ~ x^k."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    runs = collect()
    D = np.array([r["D"] for r in runs])
    mean = np.array([r["mean_mm"] for r in runs])
    mx = np.array([r["max_mm"] for r in runs])
    calls = np.array([r["n_calls"] for r in runs], dtype=float)

    k_mean, k_max, k_cost = (exponent(D, mean), exponent(D, mx),
                             exponent(D, calls))

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.95], hspace=0.42,
                          wspace=0.42)

    # (a) absolute deviation, log-log with the fitted power law
    ax = fig.add_subplot(gs[0, 0])
    fit = np.linspace(D.min(), D.max(), 50)
    for y, c, lab, k in ((mean, C_MEAN, "mean", k_mean),
                         (mx, C_MAX, "max", k_max)):
        ax.loglog(D, y, "o-", color=c, ms=4, lw=1.4, label=f"{lab}")
        ax.loglog(fit, y[0] * (fit / D[0]) ** k, ":", color=c, lw=1.0)
    ax.loglog(fit, mean[0] * (fit / D[0]), "--", color="#888888", lw=1.0,
              label="proportional, $D^{1}$")
    ax.set_xlabel("span $D$ (m)")
    ax.set_ylabel("deviation from target (mm)")
    ax.set_title("(a) deviation vs span", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(which="both", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.annotate(f"mean $\\propto D^{{{k_mean:.2f}}}$\nmax $\\propto D^{{{k_max:.2f}}}$",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=8)

    # (b) the same, normalised by span: the question a client asks
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(D, [r["mean_pct"] for r in runs], "o-", color=C_MEAN, ms=4,
            lw=1.4, label="mean / $D$")
    ax.plot(D, [r["max_pct"] for r in runs], "o-", color=C_MAX, ms=4,
            lw=1.4, label="max / $D$")
    ax.axhline(runs[0]["mean_pct"], color=C_MEAN, ls=":", lw=0.9)
    ax.axhline(runs[0]["max_pct"], color=C_MAX, ls=":", lw=0.9)
    ax.set_xlabel("span $D$ (m)")
    ax.set_ylabel("deviation as % of span")
    ax.set_title("(b) relative to the span", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.15 * max(r["max_pct"] for r in runs))

    # (c) what it costs to get even that
    ax = fig.add_subplot(gs[0, 2])
    conv = [r["converged"] for r in runs]
    ax.loglog(D, calls, "-", color=C_COST, lw=1.4, zorder=1)
    ax.scatter(D[np.array(conv)], calls[np.array(conv)], s=28, color=C_COST,
               zorder=3, label="converged")
    ax.scatter(D[~np.array(conv)], calls[~np.array(conv)], s=34,
               facecolor="white", edgecolor=C_COST, zorder=3,
               label="stopped at the iteration cap")
    ax.loglog(fit, calls[0] * (fit / D[0]) ** k_cost, ":", color=C_COST, lw=1.0)
    ax.set_xlabel("span $D$ (m)")
    ax.set_ylabel("FEM solves to optimise")
    ax.set_title("(c) cost of optimising", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(which="both", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.annotate(f"$\\propto D^{{{k_cost:.2f}}}$", xy=(0.97, 0.06),
                xycoords="axes fraction", ha="right", fontsize=8)

    # (d) where the accuracy went: crown ratio against surface error
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(D, [100 * r["crown_ratio"] for r in runs], "o-", color="#AA4499",
            ms=4, lw=1.4, label="achieved")
    ax.axhline(100 * runs[0]["target_crown_ratio"], color="#000000", ls="--",
               lw=1.0, label="target, self-similar")
    ax.set_xlabel("span $D$ (m)")
    ax.set_ylabel("crown height as % of span")
    ax.set_title("(d) crown height", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    # Bottom row: where on the surface the deviation is, at each span, each
    # normalised by its own span so the patterns are comparable.
    vmax = max(r["max_pct"] for r in runs)
    for i, r in enumerate(runs):
        ax = fig.add_subplot(gs[1, i])
        V, F, interior = r["V"], r["F"], r["interior"]
        full = np.zeros(len(V))
        full[interior] = 100 * r["dev"] / r["D"]
        tri = Triangulation(r["T"][:, 0] / r["D"], r["T"][:, 1] / r["D"], F)
        tp = ax.tripcolor(tri, full, cmap="magma_r", vmin=0, vmax=vmax,
                          shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"D = {r['D']:.1f} m", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if i == 3:
            cb = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("deviation, % of span", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.455, "(e) where the deviation is, each normalised by its "
             "own span: the same pattern throughout, growing",
             fontsize=10, ha="center")

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    out = os.path.join(HERE, "figures", "scalability.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    rec = dict(
        exponent_mean=k_mean, exponent_max=k_max, exponent_cost=k_cost,
        n_interior=int(len(runs[0]["interior"])),
        runs=[{k: v for k, v in r.items()
               if k not in ("dev", "V", "T", "F", "interior")} for r in runs])
    with open(os.path.join(HERE, "data", "scalability.json"), "w") as f:
        json.dump(rec, f, indent=1)

    print(f"  mean deviation ~ D^{k_mean:.2f}, max ~ D^{k_max:.2f}, "
          f"cost ~ D^{k_cost:.2f}")
    for r in runs:
        print(f"  D={r['D']:.1f}  mean {r['mean_mm']:6.2f} mm "
              f"({r['mean_pct']:.3f}%)  max {r['max_mm']:6.2f} mm "
              f"({r['max_pct']:.3f}%)  {r['n_calls']:5d} solves"
              f"{'' if r['converged'] else '  [capped]'}")
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
