"""
Pick the middle Y-shaped cable out of the 24 extracted 2-part cables.

C14 is the stem: it sits exactly on x = 0, 1.364 m long, and is the longest
single cable in the set.  What forks off it at the bottom is ambiguous, so this
draws the candidate arm pairs side by side.

    .venv/bin/python FDM/figure_2part_middle_Y.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")
OUT  = os.path.join(DATA, "2part_middle_Y.png")

L  = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
cab = json.load(open(os.path.join(DATA, "cable_paths_2part.json")))

STEM = "C14"
CANDIDATES = [
    ("A   C14 + C01/C06", ["C14", "C01", "C06"]),
    ("B   C14 + C02/C07", ["C14", "C02", "C07"]),
    ("C   C14 + C00/C05", ["C14", "C00", "C05"]),
]


def mesh(ax):
    ax.triplot(V[:, 0], V[:, 1], F, color="0.88", lw=0.3, zorder=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def total_len(keys):
    return sum(float(np.linalg.norm(np.diff(V[cab[k]], axis=0), axis=1).sum()) for k in keys)


fig, axes = plt.subplots(1, 4, figsize=(19, 5.6))

ax = axes[0]; mesh(ax)
for k, p in cab.items():
    P = V[p]
    hot = k == STEM
    ax.plot(P[:, 0], P[:, 1], color="#e34948" if hot else "#b9bcc4",
            lw=2.4 if hot else 1.2, zorder=3 if hot else 2)
for k in ["C00", "C01", "C02", "C05", "C06", "C07", "C14"]:
    P = V[cab[k]].mean(axis=0)
    ax.annotate(k, (P[0], P[1]), fontsize=7, color="#2a3550",
                ha="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
ax.set_title(f"all 24 cables — stem {STEM} in red\n(x = 0 exactly, "
             f"{total_len([STEM]):.3f} m, the longest single cable)", fontsize=9)

for ax, (title, keys) in zip(axes[1:], CANDIDATES):
    mesh(ax)
    for k, p in cab.items():
        if k in keys:
            continue
        P = V[p]
        ax.plot(P[:, 0], P[:, 1], color="#e2e4e8", lw=1.0, zorder=1)
    for k in keys:
        P = V[cab[k]]
        ax.plot(P[:, 0], P[:, 1], color="#e34948", lw=2.6, zorder=3)
        ax.scatter(P[[0, -1], 0], P[[0, -1], 1], s=16, color="#2a78d6", zorder=4)
    ax.set_title(f"{title}\n{len(keys)} cables, {total_len(keys):.3f} m total", fontsize=9)

fig.suptitle("2-part smooth: which cables make the middle Y?", fontsize=12)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved", OUT)
for t, k in CANDIDATES:
    print(f"{t:22s} {k}  total {total_len(k):.3f} m")
