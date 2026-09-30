"""Divide the pattern_smooth rm2 mesh into N vertical strips, west to east.

An alternative to the 4 cable-bounded regions of field_regions_pattern_smooth.py:
each face goes to the strip its plan centroid falls in.  The strip boundaries
are x-quantiles of the face centroids, so every strip has the same number of
faces (--equal-width gives equal widths instead: the end strips then shrink to
~50 faces and the west one splits into two disconnected corner tips).  The strips ignore the cables; the
Laplacian penalty in optimise_pattern_smooth.py (--lambda-smooth) then keeps
neighbouring strips' stretch factors from jumping.

The knit direction is NOT re-derived: face_knit_dirs_deg is copied from the rm2
4-region map, i.e. the field held to the three hand-drawn cables.

    python3 FDM/strip_regions_pattern_smooth.py [--n 10]
writes optimisation/pattern_smooth_rm2_<N>strip_map.json and
       data/pattern/remesh2/pattern_smooth_rm2_<N>strip.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

import optimise_2part as o2

HERE = os.path.dirname(os.path.abspath(__file__))
RM2 = os.path.join(HERE, "data", "pattern", "remesh2")
MESH = os.path.join(RM2, "pattern_smooth_rm2_tri_m.off")
CABLES = os.path.join(RM2, "cable_paths_pattern_smooth_rm2.json")
KNIT_FROM = os.path.join(HERE, "optimisation", "pattern_smooth_rm2_4region_map.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--equal-width", action="store_true")
    args = ap.parse_args()
    n = args.n
    V, F = o2.load_off(MESH)
    cen = V[F].mean(1)
    x0, x1 = V[:, 0].min(), V[:, 0].max()
    edges = (np.linspace(x0, x1, n + 1) if args.equal_width else
             np.r_[x0, np.quantile(cen[:, 0], np.linspace(0, 1, n + 1)[1:-1]), x1])
    region = np.clip(np.digitize(cen[:, 0], edges) - 1, 0, n - 1)
    counts = np.bincount(region, minlength=n)
    if (counts == 0).any():
        raise SystemExit(f"empty strip(s): {counts.tolist()}")
    knit = json.load(open(KNIT_FROM))["face_knit_dirs_deg"]
    out = os.path.join(HERE, "optimisation", f"pattern_smooth_rm2_{n}strip_map.json")
    json.dump({"face_regions": region.tolist(), "face_knit_dirs_deg": knit,
               "x_edges": edges.tolist()}, open(out, "w"))
    print(f"{n} strips, widths {np.round(np.diff(edges) * 1e3).astype(int).tolist()} mm, "
          f"faces {counts.tolist()}")
    print(f"wrote {os.path.relpath(out, HERE)}")

    cab = json.load(open(CABLES))
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    ax.add_collection(PolyCollection([V[f][:, :2] for f in F],
                                     facecolors=[cmap(r % cmap.N) for r in region],
                                     edgecolors="white", linewidths=0.15, alpha=0.8))
    for k, p in cab.items():
        ax.plot(*V[p][:, :2].T, "-", color="#b3261e" if k.startswith("C") else "#5a1a14",
                lw=2.0 if k.startswith("C") else 1.2)
    for r in range(n):
        m = region == r
        ax.text(cen[m, 0].mean(), cen[m, 1].mean(), f"R{r}\n{m.sum()}", ha="center",
                va="center", fontsize=8)
    ax.set_aspect("equal"); ax.autoscale(); ax.set_axis_off()
    ax.set_title(f"{n} vertical strips (west -> east) with the drawn cables", fontsize=11)
    png = os.path.join(RM2, f"pattern_smooth_rm2_{n}strip.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(png, HERE)}")


if __name__ == "__main__":
    main()
