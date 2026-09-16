"""
C5 pipeline in three stages: original remesh -> FDM form-finding -> FEM optimum.

All three at the project's 1.2 m diameter convention:
  1. FDM/data/C5/C5_remeshed_fem.off              target / FEM rest mesh
  2. FDM/data/mesh_out_C5_dense_20260506084323.json   FDM form-found net
  3. FDM/data/C5/C5_optim_inflated.obj            FEM result at the optimum
Panel 4 is the optimum's deviation from the target.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off")
FDMJS  = os.path.join(HERE, "data", "mesh_out_C5_dense_20260506084323.json")
OPTOBJ = os.path.join(HERE, "data", "C5", "C5_optim_inflated.obj")
OUT    = os.path.join(HERE, "data", "C5", "C5_pipeline.png")
INK, INK2 = "#0b0b0b", "#52514e"


def load_off(p):
    L = open(p).readlines()
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in l.split()[:3]] for l in L[2:2 + nv]])
    F = [[int(x) for x in l.split()[1:]] for l in L[2 + nv:2 + nv + nf]]
    return V, F


def load_obj(p):
    V, F = [], []
    for l in open(p):
        if l.startswith("v "):   V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    return np.array(V), F


def load_fdm(p):
    d = json.load(open(p))["data"]
    keys = sorted(d["vertex"], key=int)
    idx  = {k: i for i, k in enumerate(keys)}
    V = np.array([[d["vertex"][k]["x"], d["vertex"][k]["y"], d["vertex"][k]["z"]] for k in keys])
    F = [[idx[str(v)] for v in f] for f in d["face"].values()]
    return V, F


Vt, Ft = load_off(TARGET)
Vf, Ff = load_fdm(FDMJS)
Vo, Fo = load_obj(OPTOBJ)
dev = np.linalg.norm(Vo - Vt, axis=1) * 1000.0

fig = plt.figure(figsize=(20, 5.6))
panels = [
    (Vt, Ft, "#7a7a75", "1. original remesh\nC5_remeshed_fem.off"),
    (Vf, Ff, "#2a78d6", "2. FDM form-finding\nmesh_out_C5_dense_*.json"),
    (Vo, Fo, "#eb6834", "3. FEM optimum\nC5_optim_inflated.obj"),
]
for k, (V, F, colr, name) in enumerate(panels):
    ax = fig.add_subplot(1, 4, k + 1, projection="3d")
    ax.add_collection3d(Poly3DCollection([V[f] for f in F], alpha=0.92,
                        facecolor=colr, edgecolor="#2c2c2a", linewidths=0.12))
    ax.set_xlim(Vt[:, 0].min(), Vt[:, 0].max())
    ax.set_ylim(Vt[:, 1].min(), Vt[:, 1].max())
    ax.set_zlim(0, Vt[:, 2].max() * 1.05)
    ax.set_box_aspect([1, 1, 0.52]); ax.view_init(elev=26, azim=-60)
    ax.set_xlabel("x (m)", fontsize=8, color=INK2)
    ax.set_ylabel("y (m)", fontsize=8, color=INK2)
    ax.set_zlabel("z (m)", fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.set_title(f"{name}\n{len(V)}v / {len(F)}f,  crown {V[:,2].max():.4f} m",
                 fontsize=9, color=INK)

ax = fig.add_subplot(1, 4, 4)
sc = ax.tripcolor(Vt[:, 0], Vt[:, 1], [f for f in Ft if len(f) == 3], dev,
                  shading="gouraud", cmap="magma")
ax.set_aspect("equal")
ax.set_xlabel("x (m)", fontsize=8, color=INK2); ax.set_ylabel("y (m)", fontsize=8, color=INK2)
ax.tick_params(labelsize=7, colors=INK2)
ax.set_title("4. deviation: optimum vs target\n"
             f"RMSE {np.sqrt((dev**2).mean()):.3f} mm,  max {dev.max():.3f} mm",
             fontsize=9, color=INK)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
cb.set_label("mm", fontsize=8, color=INK2)
cb.ax.tick_params(labelsize=7, colors=INK2); cb.outline.set_visible(False)

fig.suptitle("C5 (D8 dome, Ø1.2 m) — original mesh, FDM result, optimised geometry",
             fontsize=12, color=INK, y=1.01)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", OUT)
print("target crown %.5f | fdm crown %.5f | optimum crown %.5f"
      % (Vt[:,2].max(), Vf[:,2].max(), Vo[:,2].max()))
print("dev mm: rmse %.4f median %.4f p95 %.4f max %.4f"
      % (np.sqrt((dev**2).mean()), np.median(dev), np.percentile(dev,95), dev.max()))
