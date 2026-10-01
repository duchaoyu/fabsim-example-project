"""
Figure 7.6: crown height against inflation pressure, from run_pressure_sweep.py.

(a) crown height of structures I and II at the stated pre-strain, solid from
    500 Pa up to the structure's stress limit (I 3.5, II 4.0 kN/m) and dashed
    outside it, the limit point marked; rise-to-span labelled every 1000 Pa.
(b) section profiles of structure I through the crown (plane y = 0, along the
    wale) at every 1000 Pa and at the limit.

Usage:
    python3 plot_pressure_sweep.py [--prestrain reference]
"""
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
MESH = os.path.join(ROOT, "data", "circular_flat.off")
SPAN_MM = 1200.0

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"I": "#2E8B57", "II": "#20B2AA"}          # seagreen / lightseagreen, as the other 7.2 figures
P_LO = 500.0                                     # below this pressure the curve is dashed, Pa


def faces():
    L = open(MESH).read().split("\n")
    nv, nf, _ = map(int, L[1].split())
    return np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])


def section(verts_csv, F, axis=1, value=0.0):
    """Polyline where the deformed surface cuts the plane x[axis] = value."""
    X = pd.read_csv(verts_csv).sort_values("vid")[["x", "y", "z"]].to_numpy()
    pts = []
    for tri in F:
        P = X[tri]
        d = P[:, axis] - value
        hit = []
        for i in range(3):
            a, b = i, (i + 1) % 3
            if d[a] * d[b] < 0:
                t = d[a] / (d[a] - d[b])
                hit.append(P[a] + t * (P[b] - P[a]))
        if len(hit) == 2:
            pts += hit
    pts = np.array(pts)
    other = 0 if axis == 1 else 1
    pts = pts[np.argsort(pts[:, other])]
    return pts[:, other] * 1000, pts[:, 2] * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prestrain", default="reference")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(DATA, "pressure_sweep.csv"))
    df = df[df.prestrain == a.prestrain]
    s_w, s_c = df.s_wale.iloc[0], df.s_course.iloc[0]
    F = faces()

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "sans-serif"})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})

    # ── (a) crown height vs pressure ────────────────────────────────────────
    for s, g in df.groupby("structure"):
        g = g.sort_values("pressure")
        p = np.r_[0.0, g.pressure.to_numpy()]
        h = np.r_[0.0, g.crown_mm.to_numpy()]
        lim = g.stress_limit.iloc[0]
        vm = np.r_[g.max_vm.iloc[0], g.max_vm.to_numpy()]     # p -> 0 carries the pre-strain stress
        # solid for p >= 500 Pa up to the structure's stress limit (I 3.5, II 4.0 kN/m)
        inside = (p >= P_LO - 1e-6) & (vm <= lim + 1e-6)
        # solid inside the stress range, dashed outside.  Each run of equal status is
        # one polyline (a 100 Pa piece is shorter than a dash), sharing its end points.
        seg_ok = inside[:-1] & inside[1:]
        k = 0
        while k < len(seg_ok):
            j = k
            while j + 1 < len(seg_ok) and seg_ok[j + 1] == seg_ok[k]:
                j += 1
            ax.plot(p[k:j + 2], h[k:j + 2], ls="-" if seg_ok[k] else (0, (5, 3)), color=COL[s], lw=1.6,
                    solid_capstyle="butt", zorder=3)
            k = j + 1
        ax.plot([], [], "-", color=COL[s], lw=1.6, label=f"structure {s}")
        end = g[g.at_limit].iloc[0]
        ax.plot(end.pressure, end.crown_mm, "o", ms=8, mfc="white", mec=COL[s], mew=2, zorder=4)
        below = s == "I"          # structure I's end sits under the structure II curve
        ax.annotate(f"{end.pressure:.0f} Pa\n{end.crown_mm:.0f} mm", (end.pressure, end.crown_mm),
                    xytext=(8, -10) if below else (-6, 8), textcoords="offset points",
                    ha="left" if below else "right", va="top" if below else "bottom",
                    fontsize=7.5, color=INK)
        marks = g[(g.pressure % 1000 == 0) & (g.max_vm <= lim)]
        ax.plot(marks.pressure, marks.crown_mm, "o", ms=8, color=COL[s], mec="white", mew=1.5, zorder=4)
        if s == "I":
            for _, r in marks.iterrows():
                ax.annotate(f"{r.crown_mm / SPAN_MM:.2f}", (r.pressure, r.crown_mm), xytext=(4, -12),
                            textcoords="offset points", fontsize=7.5, color=INK2)
        last = g.iloc[-1]          # direct label: the two greens are close
        ax.annotate(f"structure {s}", (last.pressure, last.crown_mm), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8, color=INK)
    ax.plot([], [], ls=(0, (5, 3)), color=INK2, lw=1.6,
            label="p < 500 Pa, or max stress above\nthe limit (I 3.5, II 4.0 kN/m)")
    ax.set_xlabel("inflation pressure  p  (Pa)")
    ax.set_ylabel("crown height  (mm)")
    ax.set_xlim(0, df.pressure.max() * 1.13)
    ax.set_ylim(0, df.crown_mm.max() * 1.15)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title(f"(a)  crown height against pressure,  "
                 f"$s_{{wale}}$ = {s_w:g}, $s_{{course}}$ = {s_c:g}",
                 loc="left", fontsize=9, color=INK)
    ax.text(0.02, 0.97, "labels: rise-to-span (structure I)   ○ max von Mises at the limit",
            transform=ax.transAxes, fontsize=7.5, color=INK2, va="top")

    # ── (b) section profiles, structure I ──────────────────────────────────
    gI = df[(df.structure == "I") & (df.max_vm <= df.stress_limit + 1e-6)].sort_values("pressure")
    show = list(gI[gI.pressure % 1000 == 0].itertuples()) + list(gI[gI.at_limit].itertuples())
    greys = np.linspace(0.78, 0.25, len(show))
    for (r, gv) in zip(show, greys):
        tag = f"I_{a.prestrain}_p{r.pressure:.0f}"
        x, z = section(os.path.join(DATA, "pressure_sweep", tag + "_verts.csv"), F)
        c = COL["I"] if r.at_limit else str(gv)
        bx.plot(x, z, color=c, lw=2 if r.at_limit else 1.2)
        if r.at_limit or r.pressure in (1000, 3000, 5000):
            k = np.argmax(z)                     # four labelled crowns sit >= 40 mm apart
            bx.annotate(f"{r.pressure:.0f} Pa", (x[k], z[k]), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK if r.at_limit else INK2,
                        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    bx.axhline(0, color=INK2, lw=0.8)
    bx.set_aspect("equal")
    bx.set_xlim(-640, 640)
    bx.set_ylim(-10, max(z) * 1.18)
    bx.set_xlabel("x  (mm), section y = 0 along the wale")
    bx.set_ylabel("z  (mm)")
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    bx.set_title("(b)  structure I, section through the crown", loc="left", fontsize=9, color=INK)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    stem = os.path.join(FIG, f"fig_pressure_sweep_{a.prestrain}")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
    print("wrote", stem + ".{pdf,png}")


if __name__ == "__main__":
    main()
