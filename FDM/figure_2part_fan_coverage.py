"""
What "the straightened cables no longer follow the fan" means, in pictures.

The FDM force concentration has two parts: a vertical crease and, below it, a
FAN that splays out to the bottom boundary.  The crossed routing followed both.
The straightened cables keep the crease and leave the fan uncovered.

    .venv/bin/python FDM/figure_2part_fan_coverage.py
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")
OUT  = os.path.join(DATA, "2part_fan_coverage.png")

L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])

fdm = json.load(open(os.path.join(DATA, "mesh_out_2part_smooth_latest.json")))
g = fdm.get("data", fdm)
qe = {}
for k, d in g["edgedata"].items():
    u, v = eval(k)
    qv = d.get("qpre", 1.0)
    qe[tuple(sorted((int(u), int(v))))] = float(qv[0] if isinstance(qv, (list, tuple)) else qv)
qmax = max(qe.values())

old = json.load(open(os.path.join(DATA, "cable_paths_2part_continuous.json")))
Lm = open(os.path.join(DATA, "2part_cablemesh.off")).read().split("\n")
nv2, nf2 = int(Lm[1].split()[0]), int(Lm[1].split()[1])
V2 = np.array([[float(x) for x in Lm[2 + i].split()] for i in range(nv2)])
new = json.load(open(os.path.join(DATA, "cable_paths_2part_cablemesh.json")))

# the high-q edges, and which of them sit in the fan
HI = 1.22          # the p95 threshold the extraction used
FAN_Y = -0.33      # below this is the fan


def hi_segments(fan_only=None):
    segs, cs = [], []
    for (u, v), q in qe.items():
        if q < HI:
            continue
        ym = 0.5 * (V[u, 1] + V[v, 1])
        if fan_only is True and ym > FAN_Y:
            continue
        if fan_only is False and ym <= FAN_Y:
            continue
        segs.append([V[u, :2], V[v, :2]]); cs.append(q)
    return segs, cs


def dist_to_cables(pts, cable_pts):
    d = np.full(len(pts), np.inf)
    for c in cable_pts:
        for i in range(len(c) - 1):
            a, b = c[i], c[i + 1]
            ab = b - a; t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            proj = a + t[:, None] * ab
            d = np.minimum(d, np.linalg.norm(pts - proj, axis=1))
    return d


old_pts = [V[p][:, :2] for p in old.values()]
new_pts = [V2[p][:, :2] for p in new.values()]

# how far is each high-q EDGE MIDPOINT from the nearest cable?
mids, qs = [], []
for (u, v), q in qe.items():
    if q >= HI:
        mids.append(0.5 * (V[u, :2] + V[v, :2])); qs.append(q)
mids = np.array(mids); qs = np.array(qs)
d_old = dist_to_cables(mids, old_pts)
d_new = dist_to_cables(mids, new_pts)
fan = mids[:, 1] <= FAN_Y

print(f"high-q edges (q >= {HI}): {len(mids)}   of which in the fan (y <= {FAN_Y}): {fan.sum()}")
for nm, d in [("crossed routing", d_old), ("straightened", d_new)]:
    print(f"  {nm:16s} mean distance to nearest cable: "
          f"crease {d[~fan].mean()*1000:6.1f} mm   fan {d[fan].mean()*1000:6.1f} mm")

fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))

for ax, (title, pts, col) in zip(
        axes[:2],
        [("crossed routing — follows the crease AND the fan", old_pts, "#2a78d6"),
         ("straightened — follows the crease, leaves the fan", new_pts, "#3fa46a")]):
    ax.triplot(V[:, 0], V[:, 1], F, color="0.93", lw=0.25, zorder=0)
    segs, cs = hi_segments()
    ax.add_collection(LineCollection(segs, linewidths=2.6, array=np.array(cs),
                                     cmap="inferno_r",
                                     norm=plt.Normalize(HI, qmax), zorder=2))
    for c in pts:
        ax.plot(c[:, 0], c[:, 1], color=col, lw=2.4, zorder=4)
    ax.axhline(FAN_Y, color="0.55", ls="--", lw=1, zorder=1)
    ax.text(-0.57, FAN_Y - 0.03, "the fan", fontsize=9, color="0.35")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

ax = axes[2]
ax.triplot(V[:, 0], V[:, 1], F, color="0.93", lw=0.25, zorder=0)
sc = ax.scatter(mids[:, 0], mids[:, 1], c=(d_new - d_old) * 1000, cmap="coolwarm",
                vmin=-150, vmax=150, s=26, zorder=3)
for c in new_pts:
    ax.plot(c[:, 0], c[:, 1], color="#3fa46a", lw=2.0, zorder=4)
ax.axhline(FAN_Y, color="0.55", ls="--", lw=1)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("change in distance from each high-q edge\n"
             "to the nearest cable (red = now further away)", fontsize=10)
cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03); cb.set_label("mm")

fig.suptitle("2-part smooth — the straightened cables abandon the fan", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved", OUT)
