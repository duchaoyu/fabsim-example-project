"""
The 4-part knit angle per face, before and after the x/y mirror lines were added
as guides to the directional field (FDM/directional_field_4part.py).

knit_dir_deg is pi-periodic, so it is drawn with a cyclic colormap over 0..180;
equal colours mean equal knit direction, and the wrap at 0/180 is not a jump.

    .venv/bin/python FDM/figure_4part_knit_angles.py [old_field.json]
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "4part")
NEW  = os.path.join(DATA, "directional_field_4part.json")
OLD  = sys.argv[1] if len(sys.argv) > 1 else None
OUT  = os.path.join(DATA, "4part_knit_angles.png")


def load_off(path):
    L = open(path).read().split("\n")
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


V, F = load_off(os.path.join(DATA, "4part_tri_m.off"))
tri  = Triangulation(V[:, 0], V[:, 1], F)
new  = json.load(open(NEW))
old  = json.load(open(OLD)) if OLD and os.path.exists(OLD) else None
cen  = np.array(new["centroid"])
cables = list(json.load(open(os.path.join(DATA, "cable_paths_4part.json"))).values())
R = np.linalg.norm(V[:, :2], axis=1).max()


def overlay(ax, ticks=None):
    for c in cables:
        ax.plot(V[c][:, 0], V[c][:, 1], color="#e34948", lw=1.6, zorder=3)
    ax.plot([-R, R], [0, 0], color="#f2a93b", lw=1.6, zorder=3)
    ax.plot([0, 0], [-R, R], color="#f2a93b", lw=1.6, zorder=3)
    if ticks is not None:
        s = 0.016
        d = np.stack([np.cos(np.radians(ticks)), np.sin(np.radians(ticks))], 1)
        ax.add_collection(LineCollection(
            [[c[:2] - s * v, c[:2] + s * v] for c, v in zip(cen, d)],
            colors="k", linewidths=0.5, zorder=4))
    ax.set_aspect("equal"); ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05)
    ax.set_xticks([]); ax.set_yticks([])


panels = [("after  (cables + x/y guides)", np.array(new["knit_dir_deg_face"]))]
if old is not None:
    panels.insert(0, ("before  (cables only)", np.array(old["knit_dir_deg_face"])))

ncol = len(panels) + (1 if old is not None else 0)
fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.6))
axes = np.atleast_1d(axes)

for ax, (title, a) in zip(axes, panels):
    # cyclic over 0..180: double the angle so twilight's 0 and 2pi meet at 0/180
    tp = ax.tripcolor(tri, facecolors=a, cmap="twilight", vmin=0, vmax=180,
                      shading="flat", zorder=1)
    overlay(ax, a)
    ax.set_title(title, fontsize=10)
    cb = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.03)
    cb.set_ticks([0, 45, 90, 135, 180]); cb.set_label("knit_dir_deg", fontsize=8)

if old is not None:
    a_o = np.array(old["knit_dir_deg_face"]); a_n = np.array(new["knit_dir_deg_face"])
    dd  = np.abs((a_n - a_o + 90) % 180 - 90)
    ax  = axes[-1]
    tp  = ax.tripcolor(tri, facecolors=dd, cmap="magma", vmin=0, shading="flat", zorder=1)
    overlay(ax)
    ax.set_title(f"|change|  mean {dd.mean():.1f}, max {dd.max():.1f} deg", fontsize=10)
    cb = fig.colorbar(tp, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("deg", fontsize=8)

fig.suptitle("4-part knit angle per face — red = cables, orange = x/y mirror lines",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved", OUT)
