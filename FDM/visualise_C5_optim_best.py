"""
Visualise the best C5 16-region optimisation result (lowest max_stress).

Panels:
  A – 3D isometric view of the dome coloured by elevation
  B – side elevation coloured by elevation, with crown marker
  C – convergence history: max_stress (left) and crown_height (right) vs FEM call
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import csv

HERE    = os.path.dirname(os.path.abspath(__file__))
OFF     = os.path.join(HERE, "data", "C5_remeshed.off")
OPT_DIR = os.path.join(HERE, "optimisation")
OUT_PNG = os.path.join(HERE, "data", "C5", "C5_optim_best.png")
BG      = "#0f0f1a"


# ── Load base mesh topology ────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F

V0, F = load_off(OFF)


# ── Collect convergence history from all scalar files ─────────────────────────
scalars_files = sorted(glob.glob(os.path.join(OPT_DIR, "c5_16r_*_scalars.csv")))
history = []
for sf in scalars_files:
    it = int(os.path.basename(sf).split("_")[-2])
    with open(sf) as fp:
        rows = list(csv.DictReader(fp))
        if rows:
            r = rows[0]
            history.append((it, float(r["crown_height"]),
                            float(r["max_stress"]), float(r["mean_stress"])))
history.sort()
iters      = np.array([h[0] for h in history])
crowns     = np.array([h[1] for h in history])
max_stress = np.array([h[2] for h in history])

best_idx = int(np.argmin(max_stress))
best_it  = iters[best_idx]
print(f"Best iteration: {best_it}  max_stress={max_stress[best_idx]:.1f}  "
      f"crown={crowns[best_idx]:.3f} m")


# ── Load best-iteration verts and stress ──────────────────────────────────────
verts_path  = os.path.join(OPT_DIR, f"c5_16r_{best_it:05d}_verts.csv")
stress_path = os.path.join(OPT_DIR, f"c5_16r_{best_it:05d}_stress.csv")

V = np.loadtxt(verts_path, delimiter=",", skiprows=1)[:, 1:]   # drop vid column
stress_data = np.loadtxt(stress_path, delimiter=",", skiprows=1)
vm_stress   = stress_data[:, 4]   # von_mises column

# Per-face elevation (mean z of vertices)
face_z = V[F].mean(axis=1)[:, 2]

# Face centroids for region overlay
cx = V[F].mean(axis=1)[:, 0]
cy = V[F].mean(axis=1)[:, 1]
r_face = np.hypot(cx, cy)


# ── Shared elevation colormap ─────────────────────────────────────────────────
z_all  = V[:, 2]
norm_z = Normalize(vmin=z_all.min(), vmax=z_all.max())
cmap_z = plt.cm.viridis


def fill_faces(ax, xi, yi, triangles, values, norm, cmap):
    colors = cmap(norm(values))
    for fi, tri in enumerate(triangles):
        ax.fill(xi[tri], yi[tri], color=colors[fi], linewidth=0, zorder=1)
    for tri in triangles:
        for a, b in [(0,1),(1,2),(2,0)]:
            ax.plot([xi[tri[a]], xi[tri[b]]], [yi[tri[a]], yi[tri[b]]],
                    color="white", alpha=0.04, linewidth=0.2, zorder=2)


def style_ax(ax, title_str, xlabel, ylabel, equal=True):
    ax.set_facecolor(BG)
    ax.set_title(title_str, color="white", fontsize=10, pad=5)
    ax.set_xlabel(xlabel, color="white", fontsize=8)
    ax.set_ylabel(ylabel, color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#555")
    if equal:
        ax.set_aspect("equal")


def add_colorbar(ax, cmap, norm, label):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cb.set_label(label, color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    cb.outline.set_edgecolor("white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    return cb


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.32, left=0.04, right=0.98,
                        top=0.92, bottom=0.08)
ax_3d   = fig.add_subplot(gs[0, 0], projection="3d")
ax_side = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[0, 2])

title = (f"C5 16-region FEM optimisation  |  best call #{best_it} of {iters[-1]}"
         f"  |  max σ_vm = {max_stress[best_idx]:.0f} Pa  (uniform)"
         f"  |  crown = {crowns[best_idx]:.3f} m")
fig.suptitle(title, color="white", fontsize=11, y=0.98)


# ── Panel A: 3D isometric view ─────────────────────────────────────────────────
ax_3d.set_facecolor(BG)
ax_3d.set_title("3D dome shape (coloured by elevation)", color="white",
                fontsize=10, pad=5)

face_colors = cmap_z(norm_z(face_z))
poly = ax_3d.plot_trisurf(V[:, 0], V[:, 1], V[:, 2],
                           triangles=F, antialiased=False, zorder=1)
poly.set_facecolors(face_colors)
poly.set_edgecolor("none")

ax_3d.set_xlabel("x (m)", color="white", fontsize=7, labelpad=2)
ax_3d.set_ylabel("y (m)", color="white", fontsize=7, labelpad=2)
ax_3d.set_zlabel("z (m)", color="white", fontsize=7, labelpad=2)
ax_3d.tick_params(colors="white", labelsize=6)
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor("#333")
ax_3d.yaxis.pane.set_edgecolor("#333")
ax_3d.zaxis.pane.set_edgecolor("#333")
ax_3d.view_init(elev=30, azim=45)


# ── Panel B: side elevation (x-z plane) ───────────────────────────────────────
style_ax(ax_side, "Side elevation (x–z)", "x (m)", "z (m)", equal=False)
fill_faces(ax_side, V[:, 0], V[:, 2], F, face_z, norm_z, cmap_z)

z_max = V[:, 2].max()
ax_side.axhline(z_max, color="#b0e87c", linewidth=1.0, linestyle=":",
                alpha=0.8, zorder=3)
ax_side.text(0, z_max + 0.08, f"crown = {z_max:.3f} m",
             color="#b0e87c", fontsize=7, va="bottom", ha="center")
ax_side.set_xlim(V[:, 0].min() - 0.5, V[:, 0].max() + 0.5)

add_colorbar(ax_side, cmap_z, norm_z, "Elevation z (m)")


# ── Panel C: convergence history ──────────────────────────────────────────────
ax_hist.set_facecolor(BG)
ax_hist.set_title("Convergence history", color="white", fontsize=10, pad=5)
ax_hist.set_xlabel("FEM call #", color="white", fontsize=8)
ax_hist.set_ylabel("Max von Mises stress (Pa)", color="#ff6b35", fontsize=8)
ax_hist.tick_params(colors="#ff6b35", labelsize=7)
ax_hist.spines["left"].set_color("#ff6b35")
for sp in ["top", "bottom", "right"]: ax_hist.spines[sp].set_color("#555")

ax_hist.semilogy(iters, max_stress, color="#ff6b35", linewidth=1.2, zorder=3,
                 label="max σ_vm")
ax_hist.scatter([best_it], [max_stress[best_idx]], color="#ffdd00", s=80,
                zorder=5, label=f"Best #{best_it} — {max_stress[best_idx]:.0f} Pa")

# Crown on twin axis
ax2 = ax_hist.twinx()
ax2.plot(iters, crowns, color="#b0e87c", linewidth=1.0, linestyle="--",
         alpha=0.8, label="crown h")
ax2.set_ylabel("Crown height (m)", color="#b0e87c", fontsize=8)
ax2.tick_params(colors="#b0e87c", labelsize=7)
ax2.spines["right"].set_color("#b0e87c")
ax2.spines["left"].set_color("#ff6b35")

lines1, lab1 = ax_hist.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax_hist.legend(lines1 + lines2, lab1 + lab2,
               loc="upper right", facecolor=BG, labelcolor="white", fontsize=7)
ax_hist.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

# Annotation at best point
ax_hist.annotate(f"#{best_it}", xy=(best_it, max_stress[best_idx]),
                 xytext=(best_it + 15, max_stress[best_idx] * 3),
                 color="#ffdd00", fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="#ffdd00", lw=0.8))


# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
