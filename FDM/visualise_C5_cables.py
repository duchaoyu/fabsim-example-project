"""
Visualise C5 mesh with spoke + circular cables.
Saves FDM/data/C5/C5_cables_full.png
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ  = os.path.join(HERE, "data", "C5", "C5_remeshed_smooth.obj")
SPOKE_J    = os.path.join(HERE, "data", "C5", "cable_paths_C5.json")
CIRCULAR_J = os.path.join(HERE, "data", "C5", "cable_paths_C5_circular.json")
OUT  = os.path.join(HERE, "data", "C5", "C5_cables_full.png")


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                faces.append(idx)
    return np.array(verts), faces


V, faces = load_obj(OBJ)
with open(SPOKE_J)    as f: spokes   = json.load(f)
with open(CIRCULAR_J) as f: circs    = json.load(f)

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")

# ── Mesh surface (transparent) ───────────────────────────────────────────────
tris = [V[face] for face in faces if len(face) == 3]
mesh_col = Poly3DCollection(tris, alpha=0.08, linewidth=0,
                             facecolor="steelblue", edgecolor="none")
ax.add_collection3d(mesh_col)

# ── Spoke cables ──────────────────────────────────────────────────────────────
spoke_cmap = plt.get_cmap("Set1")
for i, (name, vids) in enumerate(spokes.items()):
    pts = V[vids]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color=spoke_cmap(i / len(spokes)), linewidth=1.8,
            label=name, zorder=5)

# ── Circular cables ───────────────────────────────────────────────────────────
# Colour by radius (inner→outer: cool→warm)
n_circ = len(circs)
circ_cmap = plt.get_cmap("plasma")
for i, (name, vids) in enumerate(circs.items()):
    pts = V[vids]
    color = circ_cmap(i / max(n_circ - 1, 1))
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            color=color, linewidth=1.4, alpha=0.85, zorder=4)

# Dummy entries for legend
ax.plot([], [], [], color="grey",   linewidth=1.8, label="spokes (×8)")
ax.plot([], [], [], color="purple", linewidth=1.4, label="circular K<0 (×6)")

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("C5 remeshed smooth (Ø1.2 m) — spokes + circular cables (K<0)")
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.set_box_aspect([1, 1, 0.55])
ax.view_init(elev=30, azim=-60)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
plt.close(fig)
