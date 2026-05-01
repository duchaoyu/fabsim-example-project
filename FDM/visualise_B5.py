"""
Visualise FDM force flow in B5 result mesh.

Shows:
  - 3D perspective: edges coloured + scaled by axial force f = q * l
  - Top-down view: force density pattern on the flat projection
  - Face-coloured surface: mean q per face (shows load-path zones)
  - Cross-sections: x=0 and y=0 slices with force magnitudes

Saves figures/B5_force_flow.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves PNG
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from compas.datastructures import Mesh

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
TARGET_OBJ  = os.path.join(HERE, "input", "B5.obj")
OUT_DIR     = os.path.join(HERE, "..")   # repo root figures? use FDM dir
OUT_PNG     = os.path.join(HERE, "B5_force_flow.png")

# ── Load meshes ───────────────────────────────────────────────────────────────
mesh   = Mesh.from_json(RESULT_JSON)
target = Mesh.from_obj(TARGET_OBJ)

fixed = list(mesh.vertices_on_boundary())
free  = [v for v in mesh.vertices() if v not in fixed]

# ── Per-edge quantities ───────────────────────────────────────────────────────
edges = list(mesh.edges())
q_arr = np.array([mesh.edge_attribute(e, "qpre") for e in edges], dtype=float)

# Axial force f = q * l
pts   = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
mid   = np.array([(pts[u] + pts[v]) / 2 for u, v in edges])
l_arr = np.array([np.linalg.norm(pts[u] - pts[v]) for u, v in edges])
f_arr = q_arr * l_arr

# Normalised colour value [0,1]
def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-12)

cmap_f = cm.plasma
cmap_q = cm.viridis

# ── Per-face mean q ───────────────────────────────────────────────────────────
faces       = list(mesh.faces())
face_verts  = [[mesh.vertex_coordinates(v) for v in mesh.face_vertices(f)] for f in faces]
face_q_mean = []
for f in faces:
    face_edges = mesh.face_halfedges(f)
    qs = []
    for u, v in face_edges:
        q = mesh.edge_attribute((u, v), "qpre") or mesh.edge_attribute((v, u), "qpre")
        if q is not None:
            qs.append(q)
    face_q_mean.append(np.mean(qs) if qs else 0.0)
face_q_mean = np.array(face_q_mean)

# ── Target vertices for comparison ───────────────────────────────────────────
tpts = np.array([target.vertex_coordinates(v) for v in target.vertices()])

# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#1a1a2e")

# ── 1. 3-D perspective: edges coloured by axial force ─────────────────────────
ax1 = fig.add_subplot(2, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")

# Draw faces (transparent grey)
poly = Poly3DCollection(face_verts, alpha=0.08, facecolor="white", edgecolor="none")
ax1.add_collection3d(poly)

# Draw target wireframe (green, thin)
for u, v in target.edges():
    p0, p1 = target.vertex_coordinates(u), target.vertex_coordinates(v)
    ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
             color="lime", alpha=0.15, linewidth=0.4)

# Draw FDM edges coloured and scaled by axial force f
f_norm = norm01(f_arr)
for i, (u, v) in enumerate(edges):
    p0, p1 = pts[u], pts[v]
    c   = cmap_f(f_norm[i])
    lw  = 0.4 + 3.5 * f_norm[i]
    ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
             color=c, linewidth=lw, alpha=0.85)

sm1 = cm.ScalarMappable(cmap=cmap_f,
                        norm=mcolors.Normalize(f_arr.min(), f_arr.max()))
sm1.set_array([])
cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.55, pad=0.05)
cb1.set_label("Axial force  f = q·l  (N/m equiv.)", color="white", fontsize=8)
cb1.ax.yaxis.set_tick_params(color="white")
plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

ax1.set_title("3D: axial force  f = q·l\n(green = target B5)", color="white", fontsize=10)
ax1.set_xlabel("x (m)", color="white", fontsize=7)
ax1.set_ylabel("y (m)", color="white", fontsize=7)
ax1.set_zlabel("z (m)", color="white", fontsize=7)
ax1.tick_params(colors="white", labelsize=6)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
ax1.view_init(elev=25, azim=-50)

# ── 2. Top-down: force density q on projected edges ──────────────────────────
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

q_norm = norm01(q_arr)
for i, (u, v) in enumerate(edges):
    p0, p1 = pts[u], pts[v]
    c  = cmap_q(q_norm[i])
    lw = 0.5 + 3.0 * q_norm[i]
    ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], color=c, linewidth=lw, alpha=0.9)

# Mark fixed boundary
bx = [mesh.vertex_coordinates(v)[0] for v in fixed]
by = [mesh.vertex_coordinates(v)[1] for v in fixed]
ax2.scatter(bx, by, c="cyan", s=18, zorder=5, label="fixed (boundary)")

sm2 = cm.ScalarMappable(cmap=cmap_q,
                        norm=mcolors.Normalize(q_arr.min(), q_arr.max()))
sm2.set_array([])
cb2 = fig.colorbar(sm2, ax=ax2, shrink=0.75)
cb2.set_label("Force density  q", color="white", fontsize=8)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

ax2.set_title("Top-down: force density q\n(thicker = higher q)", color="white", fontsize=10)
ax2.set_xlabel("x (m)", color="white", fontsize=8)
ax2.set_ylabel("y (m)", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

# ── 3. Face surface coloured by mean q ───────────────────────────────────────
ax3 = fig.add_subplot(2, 2, 3, projection="3d")
ax3.set_facecolor("#1a1a2e")

fq_norm = norm01(face_q_mean)
face_colors = [cmap_f(fq_norm[i]) for i in range(len(faces))]
poly3 = Poly3DCollection(face_verts, alpha=0.75,
                         facecolor=face_colors, edgecolor="none")
ax3.add_collection3d(poly3)

sm3 = cm.ScalarMappable(cmap=cmap_f,
                        norm=mcolors.Normalize(face_q_mean.min(), face_q_mean.max()))
sm3.set_array([])
cb3 = fig.colorbar(sm3, ax=ax3, shrink=0.55, pad=0.05)
cb3.set_label("Mean face q", color="white", fontsize=8)
cb3.ax.yaxis.set_tick_params(color="white")
plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

ax3.set_title("Surface: mean force density per face\n(load-path zones)", color="white", fontsize=10)
ax3.set_xlabel("x (m)", color="white", fontsize=7)
ax3.set_ylabel("y (m)", color="white", fontsize=7)
ax3.set_zlabel("z (m)", color="white", fontsize=7)
ax3.tick_params(colors="white", labelsize=6)
for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
    pane.fill = False
ax3.set_xlim(-11, 11); ax3.set_ylim(-11, 11); ax3.set_zlim(0, 5)
ax3.view_init(elev=28, azim=-50)

# ── 4. Cross-sections x=0 (y-z) and y=0 (x-z) ───────────────────────────────
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor("#1a1a2e")

BAND = 2.5   # include edges whose midpoint is within ±BAND of the slice plane

# x=0 slice → project to y-z
for i, (u, v) in enumerate(edges):
    if abs(mid[i, 0]) < BAND:
        p0, p1 = pts[u], pts[v]
        c  = cmap_f(f_norm[i])
        lw = 0.5 + 3.0 * f_norm[i]
        ax4.plot([p0[1], p1[1]], [p0[2], p1[2]], color=c, linewidth=lw, alpha=0.9)

# y=0 slice → project to x-z (offset to right half of plot)
x_off = 12.0
for i, (u, v) in enumerate(edges):
    if abs(mid[i, 1]) < BAND:
        p0, p1 = pts[u], pts[v]
        c  = cmap_f(f_norm[i])
        lw = 0.5 + 3.0 * f_norm[i]
        ax4.plot([p0[0] + x_off, p1[0] + x_off], [p0[2], p1[2]],
                 color=c, linewidth=lw, alpha=0.9)

ax4.axvline(0,        color="white", alpha=0.15, linewidth=0.5, linestyle="--")
ax4.axvline(x_off,    color="white", alpha=0.15, linewidth=0.5, linestyle="--")
ax4.text(-9, -0.3, "slice x≈0  (y-z)", color="cyan", fontsize=8)
ax4.text( 3, -0.3, "slice y≈0  (x-z)", color="orange", fontsize=8)

ax4.set_title("Cross-sections: axial force  f = q·l", color="white", fontsize=10)
ax4.set_xlabel("projected axis (m)", color="white", fontsize=8)
ax4.set_ylabel("z (m)", color="white", fontsize=8)
ax4.tick_params(colors="white", labelsize=7)
for sp in ax4.spines.values(): sp.set_color("white")
ax4.set_aspect("equal")

# ── Stats annotation ──────────────────────────────────────────────────────────
stats = (
    f"Edges: {len(edges)}   Free nodes: {len(free)}\n"
    f"q  min={q_arr.min():.2f}  max={q_arr.max():.2f}  mean={q_arr.mean():.2f}\n"
    f"f  min={f_arr.min():.2f}  max={f_arr.max():.2f}  mean={f_arr.mean():.2f}\n"
    f"Span 20 m  |  Height {pts[:,2].max():.2f} m"
)
fig.text(0.5, 0.01, stats, ha="center", va="bottom", color="white",
         fontsize=8, fontfamily="monospace",
         bbox=dict(facecolor="#0d0d1a", edgecolor="white", alpha=0.6, pad=4))

fig.suptitle("B5 FDM — Force Flow Visualisation", color="white", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.06, 1, 0.97])

plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PNG}")
