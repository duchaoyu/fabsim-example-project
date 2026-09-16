"""Plot the FDM edge forces (f = q * L) as line thickness."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from compas.datastructures import Mesh

m = Mesh.from_json('FDM/data/2part/mesh_out_2part_smooth_latest.json')
E = list(m.edges())
q = np.array([m.edge_attribute(e, 'qpre') for e in E])
L = np.array([m.edge_length(e) for e in E])
f = q * L                                    # axial force per edge [N]

P = np.array([[m.vertex_coordinates(e[0]), m.vertex_coordinates(e[1])] for e in E])

# Force spans ~500x, so clip the thickness scale at p98 to keep the bulk visible.
f_ref = np.percentile(f, 98)
LW_MAX, LW_MIN = 4.5, 0.12
lw = LW_MIN + (LW_MAX - LW_MIN) * np.clip(f / f_ref, 0, 1)

fig = plt.figure(figsize=(16, 6.5))

# Top view
ax1 = fig.add_subplot(121)
ax1.add_collection(LineCollection(P[:, :, :2], linewidths=lw, colors="#1b2a4a", alpha=0.85))
ax1.set_xlim(-0.65, 0.65); ax1.set_ylim(-0.65, 0.65); ax1.set_aspect("equal")
ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
ax1.set_title("Top view — line thickness $\\propto$ axial force $f = qL$", fontsize=10)

# Perspective
ax2 = fig.add_subplot(122, projection="3d")
ax2.add_collection3d(Line3DCollection(P, linewidths=lw, colors="#1b2a4a", alpha=0.85))
V = np.array([m.vertex_coordinates(v) for v in m.vertices()])
ax2.set_xlim(V[:,0].min(), V[:,0].max()); ax2.set_ylim(V[:,1].min(), V[:,1].max())
ax2.set_zlim(0, V[:,2].max()*1.05)
ax2.set_box_aspect([1, 1, 0.45]); ax2.view_init(elev=28, azim=-60)
ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_zlabel("z (m)")
ax2.set_title("Perspective", fontsize=10)

# Thickness key
keys = [np.percentile(f, 25), np.percentile(f, 50), np.percentile(f, 75), f_ref, f.max()]
labels = ["p25", "median", "p75", "p98", "max"]
handles = [plt.Line2D([0], [0], color="#1b2a4a",
                      linewidth=LW_MIN + (LW_MAX - LW_MIN) * min(v / f_ref, 1.0),
                      label=f"{lab}: {v*1000:.1f} mN") for v, lab in zip(keys, labels)]
ax1.legend(handles=handles, loc="upper right", fontsize=8, frameon=True,
           title="axial force", title_fontsize=8)

fig.suptitle("2parts_smooth — FDM axial forces  (pressure 1.0, "
             f"{len(E)} edges, {f.min()*1000:.2f}–{f.max()*1000:.1f} mN)",
             fontsize=11, y=0.98)
fig.tight_layout()
out = "FDM/data/2part/2part_smooth_fdm_forces.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
