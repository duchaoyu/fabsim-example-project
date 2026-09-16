"""
Figures for the 4-part pipeline: the D4 region map with its fixed knit
directions, and the deviation of the best FEM fit from the target.

    .venv/bin/python FDM/visualise_4part.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "4part")
OPT  = os.path.join(HERE, "optimisation")


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


V, F = load_off(os.path.join(DATA, "4part_tri_m.off"))
rmap = json.load(open(os.path.join(OPT, "4part_region_map.json")))["face_regions"]
res  = json.load(open(os.path.join(OPT, "4part_result.json")))
rmap = np.array(rmap)
cen  = V[F].mean(axis=1)

# ── region map ───────────────────────────────────────────────────────────────
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7))
n_wedge = res["n_wedge_slots"]
slot = rmap // 4
cmap = plt.get_cmap("tab10")
ax.add_collection(PolyCollection([V[f][:, :2] for f in F],
                                 facecolors=[cmap(s % 10) for s in slot],
                                 edgecolors="white", linewidths=0.15))
knit = {r["region_id"]: r["knit_dir_deg"] for r in res["regions"]}
s = 0.02
segs = [[c[:2] - s*np.array([np.cos(np.deg2rad(knit[r])), np.sin(np.deg2rad(knit[r]))]),
         c[:2] + s*np.array([np.cos(np.deg2rad(knit[r])), np.sin(np.deg2rad(knit[r]))])]
        for c, r in zip(cen, rmap)]
ax.add_collection(LineCollection(segs, colors="k", linewidths=0.5, alpha=0.6))
if os.path.exists(os.path.join(DATA, "cable_paths_4part.json")):
    for p in json.load(open(os.path.join(DATA, "cable_paths_4part.json"))).values():
        ax.plot(V[p][:, 0], V[p][:, 1], color="#e34948", lw=2.0)
ax.set_aspect("equal"); ax.autoscale()
ax.set_title(f"D4 region map: {res['n_regions']} regions = {n_wedge} shared slots "
             f"x 4 quadrants\ncolour = slot (shared sf), black = fixed knit dir, "
             f"red = cables", fontsize=9)

# ── deviation ────────────────────────────────────────────────────────────────
npy = os.path.join(DATA, "4part_fem_best_verts.npy")
if os.path.exists(npy):
    W = np.load(npy)
    d = np.linalg.norm(W - V, axis=1) * 1000.0
    sc = ax2.tripcolor(V[:, 0], V[:, 1], F, d, shading="gouraud", cmap="viridis")
    fig.colorbar(sc, ax=ax2, label="mm", shrink=0.8)
    ax2.set_title(f"FEM best fit vs target — interior RMSE {res['rmse_mm']:.2f} mm, "
                  f"max {res['max_dev_mm']:.2f} mm", fontsize=9)
else:
    ax2.text(0.5, 0.5, "no FEM result yet", ha="center")
ax2.set_aspect("equal"); ax2.autoscale()

fig.tight_layout()
out = os.path.join(DATA, "4part_regions_and_deviation.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("Saved:", out)
