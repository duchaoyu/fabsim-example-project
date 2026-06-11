"""
Plot the D5 directional field (d1 = wale direction) as quiver arrows
on top of the FEM mesh, shown from above (xy-plane projection) and in 3D.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE   = os.path.dirname(os.path.abspath(__file__))
FIELD  = os.path.join(HERE, "directional_field_D5.json")
MESH   = os.path.join(HERE, "D5_remeshed_fem.off")
OUT    = os.path.join(HERE, "D5_directional_field_plot.png")

ARROW_SCALE  = 0.045   # half-length of each arrow (m)
SUBSAMPLE    = 3        # plot every Nth face arrow to reduce clutter


def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def main():
    V, F = load_off(MESH)
    with open(FIELD) as f:
        field = json.load(f)

    n_f = len(F)
    centroids = np.array([field[str(i)]["centroid"] for i in range(n_f)
                          if str(i) in field])
    d1 = np.array([field[str(i)]["d1"] for i in range(n_f)
                   if str(i) in field])
    face_ids = [i for i in range(n_f) if str(i) in field]

    # Angle of d1 in the xy-plane (for colouring by knit direction)
    angle_deg = np.degrees(np.arctan2(d1[:, 1], d1[:, 0])) % 180

    # ── Build triangulation for mesh surface ─────────────────────────────────
    triang = mtri.Triangulation(V[:, 0], V[:, 1], F)

    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.90, wspace=0.08)

    cmap  = plt.get_cmap("hsv")
    norm  = Normalize(vmin=0, vmax=180)
    sm    = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ── LEFT: top-down view ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect("equal")

    # mesh wireframe (faint)
    ax1.triplot(triang, color="#cccccc", lw=0.3, alpha=0.6)

    # quiver: show both +d1 and −d1 (headless line field)
    idx = slice(None, None, SUBSAMPLE)
    cx, cy = centroids[idx, 0], centroids[idx, 1]
    dx, dy = d1[idx, 0], d1[idx, 1]
    col    = cmap(norm(angle_deg[idx]))

    scale = ARROW_SCALE
    # draw as line segments (headless) — both directions
    for i in range(len(cx)):
        c = col[i]
        ax1.plot([cx[i] - scale * dx[i], cx[i] + scale * dx[i]],
                 [cy[i] - scale * dy[i], cy[i] + scale * dy[i]],
                 color=c, lw=0.8, solid_capstyle="round")

    ax1.set_xlabel("x  (m)")
    ax1.set_ylabel("y  (m)")
    ax1.set_title("Top view  (d₁ = wale direction)")

    # ── RIGHT: 3-D view ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # mesh surface (light grey)
    ax2.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                     color="#dddddd", edgecolor="none", alpha=0.5, zorder=1)

    cx3, cy3, cz3 = centroids[idx, 0], centroids[idx, 1], centroids[idx, 2]
    dx3, dy3, dz3 = d1[idx, 0], d1[idx, 1], d1[idx, 2]

    for i in range(len(cx3)):
        c = col[i]
        ax2.plot([cx3[i] - scale * dx3[i], cx3[i] + scale * dx3[i]],
                 [cy3[i] - scale * dy3[i], cy3[i] + scale * dy3[i]],
                 [cz3[i] - scale * dz3[i], cz3[i] + scale * dz3[i]],
                 color=c, lw=0.8, solid_capstyle="round", zorder=5)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("3D view  (d₁ = wale direction)")
    ax2.view_init(elev=35, azim=-60)

    # ── Colourbar ─────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="vertical",
                        fraction=0.02, pad=0.02, shrink=0.7)
    cbar.set_label("d₁ angle in xy-plane  (°)")
    cbar.set_ticks([0, 45, 90, 135, 180])

    fig.suptitle("D5 — directional field  (d₁ wale direction per face)",
                 fontsize=12)

    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
