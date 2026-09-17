"""
Deviation maps for the 2-part smooth fits, all on ONE shared colour scale so the
three runs can actually be compared by eye.

The per-vertex CSVs hold 581 values, one per mesh vertex (interior AND boundary);
the reported RMSE is over the 521 interior vertices only, so the boundary ring is
masked out here rather than drawn as a spurious zero.

    .venv/bin/python FDM/figure_2part_deviation_maps.py
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")
OPT  = os.path.join(HERE, "optimisation")
OUT  = os.path.join(DATA, "2part_deviation_maps.png")

L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
tri = Triangulation(V[:, 0], V[:, 1], F)

cnt = collections.Counter()
for t in F:
    for k in range(3):
        cnt[tuple(sorted((t[k], t[(k + 1) % 3])))] += 1
bdry = {v for e, c in cnt.items() if c == 1 for v in e}
interior = np.array([i for i in range(nv) if i not in bdry])

RUNS = [("2part_p2",    "24 cables, mirror-tied (15 par)"),
        ("2part_p3",    "24 cables, all free (48 par)"),
        ("2part_Y_p2",  "middle Y only (14 par)")]

devs, metas = [], []
for tag, label in RUNS:
    d = np.loadtxt(os.path.join(OPT, f"{tag}_deviation_mm.csv"), skiprows=1)
    assert len(d) == nv, f"{tag}: {len(d)} values for {nv} vertices"
    r = json.load(open(os.path.join(OPT, f"{tag}_result.json")))
    devs.append(d); metas.append((label, r))

vmax = max(d[interior].max() for d in devs)
print(f"shared colour scale 0 .. {vmax:.1f} mm")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, d, (label, r) in zip(axes, devs, metas):
    dm = d.copy(); dm[list(bdry)] = np.nan
    tp = ax.tripcolor(tri, dm, cmap="magma_r", vmin=0, vmax=vmax, shading="gouraud")
    ax.tricontour(tri, np.nan_to_num(dm), levels=[10, 20, 30],
                  colors="k", linewidths=0.4, alpha=0.35)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    di = d[interior]
    ax.set_title(f"{label}\nRMSE {r['rmse_interior_mm']:.2f} mm   "
                 f"median {np.median(di):.1f}   p90 {np.percentile(di, 90):.1f}   "
                 f"max {di.max():.1f}", fontsize=9)
cb = fig.colorbar(tp, ax=axes, fraction=0.022, pad=0.02)
cb.set_label("|FEM - target|, mm")
cb.add_lines(plt.matplotlib.contour.ContourSet, [], [])  if False else None
fig.suptitle("2-part smooth — deviation from target, shared scale, "
             "boundary ring masked (RMSE is over the 521 interior vertices)",
             fontsize=12)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved", OUT)
for d, (label, r) in zip(devs, metas):
    di = d[interior]
    print(f"  {label:34s} RMSE {r['rmse_interior_mm']:6.2f}  median {np.median(di):5.1f}  "
          f"p90 {np.percentile(di,90):5.1f}  max {di.max():5.1f}")
