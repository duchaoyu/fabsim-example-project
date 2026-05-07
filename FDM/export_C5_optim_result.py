"""
Export C5 optimised FEM result as OBJ and visualise:
  - 3D side-by-side: target vs inflated shape (coloured by deviation)
  - Profile (side view) overlay
  - Deviation histogram

Best call is determined from the c5_1042v3 run (call #55, RMSE=0.22 mm).
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE    = os.path.dirname(os.path.abspath(__file__))
OFF     = os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off")
OPT_DIR = os.path.join(HERE, "optimisation")
OBJ_OUT = os.path.join(HERE, "data", "C5", "C5_optim_inflated.obj")
PNG_OUT = os.path.join(HERE, "data", "C5", "C5_optim_result_3d.png")
BG      = "#0f0f1a"

BEST_CALL = 24
RUN_PREFIX = "c5_v5"


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def load_verts_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 1:]   # drop vid column


# ── Load data ─────────────────────────────────────────────────────────────────
V_target, F = load_off(OFF)
n_verts = len(V_target)

verts_path = os.path.join(OPT_DIR, f"{RUN_PREFIX}_{BEST_CALL:05d}_verts.csv")
V_inflated = load_verts_csv(verts_path)
assert V_inflated.shape == V_target.shape, "Vertex count mismatch"

dev = np.linalg.norm(V_inflated - V_target, axis=1)   # per-vertex deviation (m)
face_dev = dev[F].mean(axis=1)

bdry_mask    = np.hypot(V_target[:, 0], V_target[:, 1]) > \
               np.hypot(V_target[:, 0], V_target[:, 1]).max() * 0.98
interior_idx = np.where(~bdry_mask)[0]
rmse = float(np.sqrt(np.mean(dev[interior_idx] ** 2)))

print(f"Mesh:         {n_verts} verts, {len(F)} faces")
print(f"Target crown: {V_target[:, 2].max():.4f} m")
print(f"Inflated crown: {V_inflated[:, 2].max():.4f} m")
print(f"RMSE (interior): {rmse*1000:.3f} mm")
print(f"Max deviation:   {dev.max()*1000:.3f} mm")
print(f"Mean deviation:  {dev.mean()*1000:.3f} mm")


# ── Export OBJ ────────────────────────────────────────────────────────────────
with open(OBJ_OUT, "w") as f:
    f.write(f"# C5 optimal inflated shape (FEM result, call #{BEST_CALL})\n")
    f.write(f"# RMSE={rmse*1000:.3f} mm  max_dev={dev.max()*1000:.3f} mm\n")
    f.write(f"# sf_wale_in=1.0315 sf_course_in=1.0417 sf_wale_out=1.0345 sf_course_out=1.0459\n")
    f.write(f"# scale_Si=1.0098 scale_So=0.9907 scale_Ha=0.9966 (D8-symmetric, 7 params)\n")
    f.write(f"# pressure=1000 Pa, motif=1\n")
    f.write("g C5_inflated\n")
    for v in V_inflated:
        f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for tri in F:
        f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
print(f"Saved OBJ: {OBJ_OUT}")


# ── Figure: 3-panel ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 8), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.05, left=0.02, right=0.98,
                        top=0.90, bottom=0.05)

ax3d_t = fig.add_subplot(gs[0, 0], projection="3d")
ax3d_i = fig.add_subplot(gs[0, 1], projection="3d")
ax_prof = fig.add_subplot(gs[0, 2])

fig.suptitle(
    f"C5 FEM optimal inflated shape  |  call #{BEST_CALL}  "
    f"|  RMSE={rmse*1000:.2f} mm  |  max_dev={dev.max()*1000:.2f} mm  "
    f"|  crown={V_inflated[:,2].max():.4f} m (target {V_target[:,2].max():.4f} m)",
    color="white", fontsize=11, y=0.97)


def plot_mesh_3d(ax, V, F, face_colors, title_str, elev=30, azim=-60):
    ax.set_facecolor(BG)
    tris = [V[tri] for tri in F]
    coll = Poly3DCollection(tris, zsort="min", linewidth=0, antialiased=False)
    coll.set_facecolor(face_colors)
    coll.set_edgecolor("none")
    ax.add_collection3d(coll)
    ax.set_xlim(V[:, 0].min(), V[:, 0].max())
    ax.set_ylim(V[:, 1].min(), V[:, 1].max())
    ax.set_zlim(0, V[:, 2].max() * 1.1)
    ax.set_xlabel("x", color="white", fontsize=7, labelpad=1)
    ax.set_ylabel("y", color="white", fontsize=7, labelpad=1)
    ax.set_zlabel("z", color="white", fontsize=7, labelpad=1)
    ax.tick_params(colors="white", labelsize=6, pad=1)
    ax.set_title(title_str, color="white", fontsize=9, pad=4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 0.6])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")
    ax.grid(color="#333", linewidth=0.4)


# Normalise deviation for colour mapping
norm_dev = Normalize(vmin=0, vmax=dev.max())
cmap_dev = plt.cm.plasma
face_colors_dev = cmap_dev(norm_dev(face_dev))

# Target — flat steel-blue coloring
target_grey = np.full((len(F), 4), [0.25, 0.45, 0.75, 0.85])
plot_mesh_3d(ax3d_t, V_target, F, target_grey, "Target (C5_remeshed_fem.off)")

# Inflated — coloured by deviation
plot_mesh_3d(ax3d_i, V_inflated, F, face_colors_dev,
             f"Inflated (FEM call #{BEST_CALL}) — coloured by deviation")

# Colourbar for deviation panel
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(cmap=cmap_dev, norm=norm_dev)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax3d_i, fraction=0.03, pad=0.08, shrink=0.6)
cb.set_label("Deviation (m)", color="white", fontsize=7)
cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
cb.outline.set_edgecolor("white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")


# ── Profile view (side cross-section near x≈0) ────────────────────────────────
ax_prof.set_facecolor(BG)
ax_prof.set_title("Side profile: target vs inflated (vertices near y=0 slice)",
                  color="white", fontsize=9, pad=4)
ax_prof.set_xlabel("x (m)", color="white", fontsize=8)
ax_prof.set_ylabel("z (m)", color="white", fontsize=8)
ax_prof.tick_params(colors="white", labelsize=7)
for sp in ax_prof.spines.values():
    sp.set_color("#555")

# All vertices sorted by x for profile
r_max = float(np.hypot(V_target[:, 0], V_target[:, 1]).max())
y_slice = np.abs(V_target[:, 1]) < r_max * 0.08
idx_s = np.where(y_slice)[0]
xs = V_target[idx_s, 0]
order = np.argsort(xs)
idx_sorted = idx_s[order]

ax_prof.plot(V_target[idx_sorted, 0], V_target[idx_sorted, 2],
             "o-", color="#4a9eff", markersize=2, linewidth=1.0,
             alpha=0.8, label="Target")
ax_prof.plot(V_inflated[idx_sorted, 0], V_inflated[idx_sorted, 2],
             "o--", color="#ff6b35", markersize=2, linewidth=1.0,
             alpha=0.8, label="Inflated")

# Shade deviation band
ax_prof.fill_between(V_target[idx_sorted, 0],
                     V_target[idx_sorted, 2],
                     V_inflated[idx_sorted, 2],
                     color="#ffdd00", alpha=0.25, label="Deviation band")

ax_prof.legend(facecolor=BG, labelcolor="white", fontsize=8, loc="upper center")
ax_prof.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.5)
ax_prof.set_aspect("equal")

# Stats box
ax_prof.text(0.02, 0.02,
             f"Interior RMSE:  {rmse*1000:.3f} mm\n"
             f"Max deviation:  {dev.max()*1000:.3f} mm\n"
             f"Mean deviation: {dev.mean()*1000:.3f} mm\n"
             f"Crown target:   {V_target[:,2].max()*1000:.1f} mm\n"
             f"Crown inflated: {V_inflated[:,2].max()*1000:.1f} mm\n"
             f"\nsf_w_in=1.0315  sf_c_in=1.0417\n"
             f"sf_w_out=1.0345  sf_c_out=1.0459\n"
             f"Si=1.010  So=0.991  Ha=0.997",
             transform=ax_prof.transAxes, va="bottom", ha="left",
             color="white", fontsize=8, fontfamily="monospace",
             bbox=dict(facecolor="#0d0d1a", edgecolor="#444", alpha=0.9, pad=4))

plt.savefig(PNG_OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved PNG: {PNG_OUT}")
