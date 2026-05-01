"""
Visualise candidate cable paths and region layout from B5 FDM result.

Shows the top-force edges at three thresholds (90/95/99 percentile),
top-down and 3D, so we can pick the right cable locations and
confirm the 9-region layout before implementing the multi-region FEM.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from compas.datastructures import Mesh

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
OUT_PNG     = os.path.join(HERE, "B5_regions.png")

# ── Load ──────────────────────────────────────────────────────────────────────
mesh  = Mesh.from_json(RESULT_JSON)
verts = list(mesh.vertices())
pts   = np.array([mesh.vertex_coordinates(v) for v in verts])
edges = list(mesh.edges())
faces = list(mesh.faces())
face_verts_xyz = [[mesh.vertex_coordinates(v) for v in mesh.face_vertices(f)]
                  for f in faces]

q_arr = np.array([mesh.edge_attribute(e, "qpre") for e in edges])
l_arr = np.array([np.linalg.norm(pts[u] - pts[v]) for u, v in edges])
f_arr = q_arr * l_arr
mids  = np.array([(pts[u] + pts[v]) / 2 for u, v in edges])

# Edge direction angle (0–180°)
dirs  = np.array([pts[v] - pts[u] for u, v in edges])
dirs  = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 180

THRESHOLDS = [90, 95, 99]
COLORS_THRESH = ["#ffaa00", "#ff5500", "#ff0055"]   # orange → red → pink
BG = "#1a1a2e"

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor(BG)

for col, (pct, col_high) in enumerate(zip(THRESHOLDS, COLORS_THRESH)):
    thresh = np.percentile(f_arr, pct)
    is_cable = f_arr >= thresh
    is_low   = ~is_cable

    ax_top = axes[0, col]   # top-down
    ax_3d  = axes[1, col]   # 3D — can't use projection="3d" in subplots easily
    # Use regular axes for 3D approximation (x vs z cross-section)
    ax_3d  = axes[1, col]

    for ax in [ax_top, ax_3d]:
        ax.set_facecolor(BG)

    # ── Top-down ──────────────────────────────────────────────────────────────
    # Low-force edges: grey
    for i in np.where(is_low)[0]:
        u, v = edges[i]
        ax_top.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]],
                    color="white", alpha=0.10, linewidth=0.4)

    # High-force (cable) edges
    f_max = f_arr.max()
    for i in np.where(is_cable)[0]:
        u, v = edges[i]
        t  = (f_arr[i] - thresh) / (f_max - thresh + 1e-12)
        lw = 1.5 + 3.5 * t
        ax_top.plot([pts[u, 0], pts[v, 0]], [pts[u, 1], pts[v, 1]],
                    color=col_high, linewidth=lw, alpha=0.9, zorder=3)

    # Identify region boundaries: connected components of low-force faces
    # Simple approach: colour faces by which side of cable lines they're on
    # Detect dominant cable directions
    h_mask  = (angles < 20) | (angles > 160)
    v_mask  = (angles > 70) & (angles < 110)

    cable_h = mids[is_cable & h_mask]
    cable_v = mids[is_cable & v_mask]

    # Shade regions: 9 regions from cable_x and cable_y boundaries
    if len(cable_h) and len(cable_v):
        # approximate cable y-positions (horizontal cables) and x-positions (vertical cables)
        y_split = np.median(cable_h[:, 1]) if len(cable_h) else 0
        x_split = np.median(cable_v[:, 0]) if len(cable_v) else 0

        # For each face, assign region colour
        REGION_COLORS = [
            "#3a3a6a", "#3a5a3a", "#5a3a3a",
            "#4a4a2a", "#2a4a5a", "#5a2a4a",
            "#4a2a2a", "#2a5a4a", "#3a3a5a",
        ]

        face_cx = np.array([np.mean([mesh.vertex_coordinates(v)[0]
                                     for v in mesh.face_vertices(f)])
                            for f in faces])
        face_cy = np.array([np.mean([mesh.vertex_coordinates(v)[1]
                                     for v in mesh.face_vertices(f)])
                            for f in faces])

        for fi, f in enumerate(faces):
            fv = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(f)]
            px = [v[0] for v in fv]
            py = [v[1] for v in fv]
            rx = 0 if face_cx[fi] < -x_split/2 else (2 if face_cx[fi] > x_split/2 else 1)
            ry = 0 if face_cy[fi] < -y_split/2 else (2 if face_cy[fi] > y_split/2 else 1)
            rc = rx * 3 + ry
            ax_top.fill(px, py, color=REGION_COLORS[rc], alpha=0.25, zorder=1)

    # Stats
    n_cable = is_cable.sum()
    cable_h_cnt = h_mask[is_cable].sum()
    cable_v_cnt = v_mask[is_cable].sum()
    ax_top.set_title(
        f"Top-down  |  threshold: top {100-pct}%  ({n_cable} edges)\n"
        f"horiz: {cable_h_cnt}  vert: {cable_v_cnt}  diag: {n_cable-cable_h_cnt-cable_v_cnt}",
        color="white", fontsize=9)
    ax_top.set_aspect("equal")
    ax_top.tick_params(colors="white", labelsize=7)
    ax_top.set_xlabel("x (m)", color="white", fontsize=7)
    ax_top.set_ylabel("y (m)", color="white", fontsize=7)
    for sp in ax_top.spines.values(): sp.set_color("white")

    # ── x=0 Cross-section ────────────────────────────────────────────────────
    ax_3d.set_facecolor(BG)
    BAND = 2.0
    for i, (u, v) in enumerate(edges):
        if abs(mids[i, 0]) < BAND:
            c  = col_high if is_cable[i] else "white"
            lw = (1.5 + 3.5 * (f_arr[i] - thresh) / (f_max - thresh + 1e-12)
                  if is_cable[i] else 0.3)
            a  = 0.9 if is_cable[i] else 0.12
            ax_3d.plot([pts[u, 1], pts[v, 1]], [pts[u, 2], pts[v, 2]],
                       color=c, linewidth=lw, alpha=a)

    ax_3d.set_title(f"Cross-section x≈0 (y-z)", color="white", fontsize=9)
    ax_3d.set_aspect("equal")
    ax_3d.tick_params(colors="white", labelsize=7)
    ax_3d.set_xlabel("y (m)", color="white", fontsize=7)
    ax_3d.set_ylabel("z (m)", color="white", fontsize=7)
    for sp in ax_3d.spines.values(): sp.set_color("white")

fig.suptitle("B5 FDM — Cable paths & region layout (top-force threshold sweep)",
             color="white", fontsize=13, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
