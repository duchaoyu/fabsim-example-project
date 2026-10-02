"""Figure for Section 7.5.2 — the middle-crease shell (7.4.1) at nine spans.

Strategies D (anisotropic pre-strain + crease cable, one region) and E (three
adaptive regions + cable) are re-optimised from scratch at each span by
run_2part_span.sh.  Geometry and cable EA scale with the span; pressure and
material do not, so the shell is not a scaled copy of itself: membrane tension
grows with the span and so does the pre-strain needed to hold the shape.

Deviation is the Section 7.4 measure (deviation_tools.closest_dist: each
simulated vertex to the nearest point on the scaled target), over all vertices.

    bash FDM/run_2part_span.sh
    python3 FDM/figure_2part_span.py
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines
from matplotlib.tri import Triangulation

from deviation_tools import read_off, closest_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_DIR = os.path.join(HERE, "optimisation", "2part_span")
TARGET = os.path.join(ROOT, "data", "2part", "2part_opt_simu_m.off")
BASE_SPAN = 1.2
MM = 1e3

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
})

STRATEGIES = [("D", "D  one region + cable", "#0077BB"),
              ("E", "E  three adaptive regions + cable", "#EE7733")]
MAP_SPANS = [0.6, 1.2, 3.0, 6.0]


def read_faces(path, n_faces):
    reg = np.full(n_faces, -1)
    cur = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("region"):
            cur = int(line.split()[0].split("_")[1])   # "REGION_<k> <count>"
            continue
        for tok in line.replace(",", " ").split():
            if tok.lstrip("-").isdigit() and cur is not None:
                reg[int(tok)] = cur
    return reg


def collect():
    T0, F = read_off(TARGET, faces=True)
    runs = {}
    for key, _, _ in STRATEGIES:
        rows = []
        for js in glob.glob(os.path.join(RUN_DIR, f"{key}_*_result.json")):
            r = json.load(open(js))
            prefix = js[:-len("_result.json")]
            T = T0 * r["scale"]
            V = read_off(prefix + "_result.off")
            d = closest_dist(V, T, F)
            r.update(D=r["span_m"], dev=d, V=V, T=T, F=F,
                     mean_mm=MM * d.mean(), max_mm=MM * d.max(),
                     mean_pct=100 * d.mean() / r["span_m"],
                     max_pct=100 * d.max() / r["span_m"])
            faces = prefix + "_faces.txt"
            r["regions"] = read_faces(faces, len(F)) if os.path.exists(faces) else None
            rows.append(r)
        runs[key] = sorted(rows, key=lambda r: r["D"])
    return runs


def exponent(x, y):
    """Slope of a log-log fit: y ~ x^k."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    runs = collect()
    fig = plt.figure(figsize=(14.0, 7.6))
    gs = fig.add_gridspec(2, 16, height_ratios=[1.0, 0.95], hspace=0.45, wspace=3.0)
    axA = fig.add_subplot(gs[0, 0:4])
    axB = fig.add_subplot(gs[0, 4:8])
    axC = fig.add_subplot(gs[0, 8:12])
    axD = fig.add_subplot(gs[0, 12:16])
    allD = sorted({r["D"] for rs in runs.values() for r in rs})
    fits = {}

    for key, label, col in STRATEGIES:
        rs = runs[key]
        if not rs:
            continue
        D = np.array([r["D"] for r in rs])
        mean = np.array([r["mean_mm"] for r in rs])
        mx = np.array([r["max_mm"] for r in rs])
        conv = np.array([r["converged"] for r in rs])
        fits[key] = (exponent(D, mean), exponent(D, mx)) if len(rs) > 1 else (np.nan, np.nan)

        # (a) absolute deviation, log-log
        axA.plot(D, mean, "-o", color=col, ms=4.5, lw=1.6, label=f"{key} mean")
        axA.plot(D, mx, "--o", color=col, ms=4.5, lw=1.2, mfc="white", label=f"{key} max")
        # (b) relative to the span
        axB.plot(D, [r["mean_pct"] for r in rs], "-o", color=col, ms=4.5, lw=1.6)
        axB.plot(D, [r["max_pct"] for r in rs], "--o", color=col, ms=4.5, lw=1.2, mfc="white")
        # (c) the pre-strain needed: largest wale stretch factor over the regions
        sfw = [np.max(r["sf_wale"]) for r in rs]
        sfc = [np.max(r["sf_course"]) for r in rs]
        axC.plot(D, sfw, "-o", color=col, ms=4.5, lw=1.6)
        axC.plot(D, sfc, ":s", color=col, ms=3.5, lw=1.2)
        # (d) cost; hollow = stopped at a cap
        axD.plot(D, [r["n_solves"] for r in rs], "-", color=col, lw=1.4)
        for r, c in zip(rs, conv):
            axD.plot(r["D"], r["n_solves"], "o", ms=6, color=col,
                     mfc=col if c else "white", mew=1.4)

    if allD:
        d0 = np.array([allD[0], allD[-1]])
        ref = runs["D"][0]["mean_mm"] if runs["D"] else 1.0
        axA.plot(d0, ref * d0 / d0[0], color="#888888", lw=1.0, ls=(0, (4, 3)),
                 label="proportional, $D^1$", zorder=0)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_ylabel("deviation from target (mm)")
    axA.set_title("(a) deviation vs span", loc="left")
    axA.legend(frameon=False, loc="upper left", fontsize=7.5)
    txt = "\n".join(f"{k}: mean $\\propto D^{{{fits[k][0]:.2f}}}$, max $\\propto D^{{{fits[k][1]:.2f}}}$"
                    for k, _, _ in STRATEGIES if k in fits)
    axA.text(0.97, 0.04, txt, transform=axA.transAxes, ha="right", va="bottom", fontsize=7.5)

    axB.set_ylim(bottom=0)
    axB.set_ylabel("deviation as % of span")
    axB.set_title("(b) relative to the span", loc="left")
    axB.plot([], [], "-", color="#444444", label="mean / D")
    axB.plot([], [], "--", color="#444444", label="max / D")
    axB.legend(frameon=False, loc="upper left")

    axC.set_ylabel("stretch factor (max over regions)")
    axC.set_title("(c) pre-strain needed", loc="left")
    axC.legend(handles=[matplotlib.lines.Line2D([], [], color="#444444", ls="-", label="wale"),
                        matplotlib.lines.Line2D([], [], color="#444444", ls=":", label="course")],
               frameon=False, loc="upper left")

    axD.set_yscale("log")
    axD.set_ylabel("FEM solves to optimise")
    axD.set_title("(d) cost of optimising", loc="left")
    axD.plot([], [], "o", color="#444444", label="converged")
    axD.plot([], [], "o", color="#444444", mfc="white", label="stopped at the cap")
    axD.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.2))

    for ax in (axA, axB, axC, axD):
        ax.set_xscale("log")
        ax.set_xticks(allD)
        # 1.5 and 1.8 sit too close on a log axis to label both
        ax.set_xticklabels(["" if d == 1.5 else f"{d:g}" for d in allD])
        ax.set_xlabel("span $D$ (m)")
        ax.grid(color="#E6E6E6", lw=0.5); ax.set_axisbelow(True)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # proxy legend entries for the strategy colours
    # strategy colours, keyed once for the whole top row
    fig.legend(handles=[matplotlib.lines.Line2D([], [], color=col, marker="o", lw=1.6, label=label)
                        for _, label, col in STRATEGIES],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))

    # Bottom row: strategy E deviation field, each normalised by its own span,
    # with the region boundaries drawn over it.
    E = {round(r["D"], 1): r for r in runs["E"]}
    maps = [E[s] for s in MAP_SPANS if s in E]
    vmax = max([r["max_pct"] for r in maps], default=1.0)
    for i, r in enumerate(maps):
        ax = fig.add_subplot(gs[1, 4 * i:4 * i + 4])
        tri = Triangulation(r["T"][:, 0] / r["D"], r["T"][:, 1] / r["D"], r["F"])
        tp = ax.tripcolor(tri, 100 * r["dev"] / r["D"], cmap="magma_r", vmin=0,
                          vmax=vmax, shading="gouraud")
        if r["regions"] is not None:
            reg = r["regions"]
            Fm = r["F"]
            edges = {}
            for f, tri_f in enumerate(Fm):
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    e = tuple(sorted((tri_f[a], tri_f[b])))
                    edges.setdefault(e, []).append(reg[f])
            P = r["T"][:, :2] / r["D"]
            for (a, b), rr in edges.items():
                if len(rr) == 2 and rr[0] != rr[1]:
                    ax.plot(P[[a, b], 0], P[[a, b], 1], color="#1A1A1A", lw=0.9)
        ax.set_aspect("equal")
        tag = "" if r["converged"] else "  (cap)"
        ax.set_title(f"D = {r['D']:.1f} m{tag}", loc="left", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if i == len(maps) - 1:
            cb = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("deviation, % of span", fontsize=8)
            cb.ax.tick_params(labelsize=7)
    fig.text(0.5, 0.455, "(e) strategy E: where the deviation is, each normalised "
             "by its own span; black lines are the region boundaries",
             fontsize=10, ha="center")

    out = os.path.join(HERE, "figures", "2part_span.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    rec = {"exponents": {k: {"mean": v[0], "max": v[1]} for k, v in fits.items()},
           "runs": {k: [{q: v for q, v in r.items()
                         if q not in ("dev", "V", "T", "F", "regions")} for r in rs]
                    for k, rs in runs.items()}}
    with open(os.path.join(HERE, "data", "2part_span.json"), "w") as f:
        json.dump(rec, f, indent=1)
    for k, rs in runs.items():
        for r in rs:
            print(f"  {k} D={r['D']:.1f}  mean {r['mean_mm']:6.2f} mm ({r['mean_pct']:.3f}%)"
                  f"  max {r['max_mm']:6.2f} mm ({r['max_pct']:.3f}%)  {r['n_solves']:4d} solves"
                  f"  {'conv' if r['converged'] else 'CAP'}")
    print("  exponents:", {k: tuple(round(x, 2) for x in v) for k, v in fits.items()})
    print("  saved:", out)


if __name__ == "__main__":
    main()
