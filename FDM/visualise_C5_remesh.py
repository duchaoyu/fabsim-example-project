"""Render the C5 D8-symmetric remesh with cable paths highlighted."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE     = os.path.dirname(os.path.abspath(__file__))
OFF_PATH = os.path.join(HERE, "..", "data", "C5_remeshed.off")
CABLES   = os.path.join(HERE, "..", "data", "cable_paths_C5.json")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_remeshed.png")
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

# ── Read OFF ──────────────────────────────────────────────────────────────────
with open(OFF_PATH) as f:
    lines = f.read().strip().splitlines()
nv, nf, _ = map(int, lines[1].split())
V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
with open(CABLES) as f:
    cables = json.load(f)
print(f"Loaded {OFF_PATH}: {nv} verts, {nf} faces, {len(cables)} cables")

face_verts = [[V[F[i, j]] for j in range(3)] for i in range(nf)]

# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

# 3D
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
poly = Poly3DCollection(face_verts, alpha=0.6,
                        facecolor=cm.viridis((V[F].mean(axis=1)[:, 2]) / V[:, 2].max()),
                        edgecolor="white", linewidth=0.15)
ax1.add_collection3d(poly)

# Cable paths in red
for name, path in cables.items():
    pts = V[path]
    ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
             color="red", linewidth=2.0, alpha=0.95, label=name)

ax1.set_title(f"C5 D8 remesh — {nv} verts / {nf} faces  (red = 8 cables)",
              color="white", fontsize=11)
ax1.set_xlabel("x (m)", color="white", fontsize=8)
ax1.set_ylabel("y (m)", color="white", fontsize=8)
ax1.set_zlabel("z (m)", color="white", fontsize=8)
ax1.tick_params(colors="white", labelsize=7)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
ax1.set_xlim(-11, 11); ax1.set_ylim(-11, 11); ax1.set_zlim(0, 5.5)
ax1.view_init(elev=25, azim=-50)

# Top-down with cables overlaid
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

# Draw mesh edges
for face in F:
    p = V[face]
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        ax2.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                 color="white", linewidth=0.2, alpha=0.3)

# Cable paths
for name, path in cables.items():
    pts = V[path]
    ax2.plot(pts[:, 0], pts[:, 1], color="red", linewidth=2.5, alpha=0.95)

# Boundary anchors (z=0)
bdry_idx = np.where(np.abs(V[:, 2]) < 1e-6)[0]
ax2.scatter(V[bdry_idx, 0], V[bdry_idx, 1], c="cyan", s=22, zorder=5,
            label=f"boundary ({len(bdry_idx)} anchors)")

# Mark spoke endpoints on boundary
for name, path in cables.items():
    p = V[path[-1]]
    ax2.scatter(p[0], p[1], c="orange", s=80, zorder=6, marker="*",
                edgecolors="red", linewidths=1)

# Annotate cable angles
for name, path in cables.items():
    p = V[path[-1]]
    angle = float(name.split("_")[1].rstrip("deg"))
    ax2.annotate(f"{angle:.1f}°", xy=(p[0], p[1]),
                 xytext=(p[0]*1.13, p[1]*1.13), color="red", fontsize=8,
                 ha="center", va="center")

ax2.set_title("Top-down — cables along D8 ridges (every 4th anchor)",
              color="white", fontsize=11)
ax2.set_xlabel("x (m)", color="white", fontsize=8)
ax2.set_ylabel("y (m)", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
ax2.set_xlim(-12, 12); ax2.set_ylim(-12, 12)

fig.suptitle("C5 D8-projected remesh with stiffener spokes",
             color="white", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PNG}")
