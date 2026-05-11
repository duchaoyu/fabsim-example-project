"""
Visualise per-vertex deviation between B5 FEM simulation result and target geometry.
Uses the most recent run_*_verts.csv (the final FEM evaluation at the optimum).
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE    = os.path.dirname(os.path.abspath(__file__))
OPTIM   = os.path.join(HERE, "optimisation")
DATA    = os.path.join(HERE, "..", "data")
PARAMS  = os.path.join(OPTIM, "B5_optimised_params.json")
OUT_PNG = os.path.join(HERE, "B5_deviation.png")
BG      = "#0f0f1a"

# ── Load mesh + target ────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

V_target, F = load_off(os.path.join(DATA, "B5_remeshed_shared.off"))
interior_idx = np.load(os.path.join(DATA, "B5_remeshed_interior_idx.npy"))

# ── Find latest FEM run (= final eval at optimum) ─────────────────────────────
runs = sorted([f for f in os.listdir(OPTIM)
               if f.startswith("run_") and f.endswith("_verts.csv")])
if not runs:
    raise FileNotFoundError("No run_*_verts.csv in optimisation/")
latest = runs[-1]
verts_path = os.path.join(OPTIM, latest)
print(f"Final FEM result: {latest}")

# Load simulated verts
V_sim = np.zeros_like(V_target)
with open(verts_path) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        i = int(row["vid"])
        V_sim[i] = [float(row["x"]), float(row["y"]), float(row["z"])]

# ── Compute deviations ────────────────────────────────────────────────────────
dev_3d = np.linalg.norm(V_sim - V_target, axis=1)        # 3D Euclidean
dev_z  = V_sim[:, 2] - V_target[:, 2]                    # signed z-error

dev_int   = dev_3d[interior_idx]
rmse_int  = float(np.sqrt(np.mean(dev_int ** 2)))
mean_int  = float(dev_int.mean())
max_int   = float(dev_int.max())
p95_int   = float(np.percentile(dev_int, 95))

print(f"Deviation (interior, n={len(interior_idx)}):")
print(f"  RMSE = {rmse_int*1000:.2f} mm   mean = {mean_int*1000:.2f} mm")
print(f"  max  = {max_int*1000:.2f} mm    p95  = {p95_int*1000:.2f} mm")
print(f"Crown sim={V_sim[:,2].max():.4f}  target={V_target[:,2].max():.4f} m")

# ── Load optim summary ────────────────────────────────────────────────────────
with open(PARAMS) as f:
    P = json.load(f)

# ── Figure: 1 row × 3 cols ────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.18)
ax0 = fig.add_subplot(gs[0, 0])                                # top-down |dev|
ax1 = fig.add_subplot(gs[0, 1], projection="3d")               # 3D shape comparison
ax2 = fig.add_subplot(gs[0, 2])                                # histogram

fig.suptitle(
    f"B5 FEM Deviation from Target  |  mean={mean_int*1000:.2f} mm  |  "
    f"max={max_int*1000:.2f} mm  |  RMSE={rmse_int*1000:.2f} mm  |  "
    f"crown sim={V_sim[:,2].max():.4f} m  target={V_target[:,2].max():.4f} m",
    color="white", fontsize=11, y=1.01)

# Common colour scale
vmax_dev = max_int
norm = Normalize(vmin=0, vmax=vmax_dev)
cmap = plt.cm.magma

# ── Panel 0: top-down filled by per-vertex |deviation| ────────────────────────
ax0.set_facecolor(BG); ax0.set_aspect("equal")
ax0.set_title(f"Top-down |deviation| (mm)", color="white", fontsize=10, pad=6)
ax0.set_xlabel("x (m)", color="white", fontsize=8)
ax0.set_ylabel("y (m)", color="white", fontsize=8)
ax0.tick_params(colors="white", labelsize=7)
for sp in ax0.spines.values(): sp.set_color("#555")

# Shade each face with its mean vertex deviation
for tri in F:
    fv = V_target[tri, :2]
    fd = dev_3d[tri].mean()
    color = cmap(norm(fd))
    ax0.fill(fv[:, 0], fv[:, 1], color=color, alpha=0.95, zorder=1, edgecolor="none")

# Boundary outline
boundary_mask = np.ones(len(V_target), dtype=bool)
boundary_mask[interior_idx] = False
bdry_pts = V_target[boundary_mask]
# order by angle for clean outline
ang = np.arctan2(bdry_pts[:,1], bdry_pts[:,0])
order = np.argsort(ang)
bp = bdry_pts[order]
ax0.plot(np.append(bp[:,0], bp[0,0]), np.append(bp[:,1], bp[0,1]),
         color="white", linewidth=1.0, alpha=0.6, zorder=4)

sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cb = plt.colorbar(sm, ax=ax0, fraction=0.038, pad=0.04,
                  format=lambda x, _: f"{x*1000:.1f}")
cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
cb.outline.set_edgecolor("white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
cb.set_label("|dev| (mm)", color="white", fontsize=8)

# ── Panel 1: 3D — target wireframe + sim coloured ─────────────────────────────
ax1.set_facecolor(BG)
ax1.set_title("3D: simulated vs target (sim coloured)", color="white",
              fontsize=10, pad=6)

# Target as wireframe
tri_xyz_t = V_target[F]
poly_t = Poly3DCollection(tri_xyz_t, alpha=0.0, edgecolor="#888888", linewidth=0.3)
ax1.add_collection3d(poly_t)

# Sim mesh coloured by deviation
tri_xyz_s = V_sim[F]
face_dev  = dev_3d[F].mean(axis=1)
face_col  = cmap(norm(face_dev))
poly_s = Poly3DCollection(tri_xyz_s, alpha=0.85, facecolor=face_col,
                          edgecolor="none")
ax1.add_collection3d(poly_s)

ax1.set_xlim(-0.65, 0.65); ax1.set_ylim(-0.65, 0.65); ax1.set_zlim(0, 0.35)
ax1.set_xlabel("x (m)", color="white", fontsize=7)
ax1.set_ylabel("y (m)", color="white", fontsize=7)
ax1.set_zlabel("z (m)", color="white", fontsize=7)
ax1.tick_params(colors="white", labelsize=6)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False; pane.set_edgecolor("#444")
ax1.view_init(elev=22, azim=-55)

# ── Panel 2: histogram of deviations ──────────────────────────────────────────
ax2.set_facecolor(BG)
ax2.set_title("Distribution of vertex deviation (interior only)",
              color="white", fontsize=10, pad=6)
ax2.set_xlabel("deviation (mm)", color="white", fontsize=8)
ax2.set_ylabel("count", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#555")

vals_mm = dev_int * 1000
ax2.hist(vals_mm, bins=30, color="#00d4ff", edgecolor="#003344", alpha=0.85)
ax2.axvline(mean_int*1000, color="#ffd866", linewidth=2,
            label=f"mean = {mean_int*1000:.2f} mm")
ax2.axvline(rmse_int*1000, color="#ff6b35", linewidth=2,
            label=f"RMSE = {rmse_int*1000:.2f} mm")
ax2.axvline(p95_int*1000, color="#b0e87c", linewidth=1.5, linestyle="--",
            label=f"p95  = {p95_int*1000:.2f} mm")
ax2.axvline(max_int*1000, color="#ff3366", linewidth=1.5, linestyle=":",
            label=f"max  = {max_int*1000:.2f} mm")
ax2.legend(loc="upper right", facecolor=BG, labelcolor="white", fontsize=7)
ax2.grid(color="#333", linestyle="--", linewidth=0.4, alpha=0.6)

# Stats box
stats = (
    f"Optimisation:\n"
    f"  pressure = {P['pressure']:.0f} Pa\n"
    f"  loss_rmse = {P['loss_rmse_m']*1000:.2f} mm\n"
    f"  calls = {P['n_calls']}\n"
    f"  converged = {P['converged']}\n"
    f"  knit_dir = {P['regions'][0]['knit_dir_deg']}° (uniform)"
)
fig.text(0.99, 0.01, stats, ha="right", va="bottom", color="white",
         fontsize=8, fontfamily="monospace",
         bbox=dict(facecolor="#0d0d1a", edgecolor="#444", alpha=0.8, pad=5))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
