"""
Visualise C5 16-region FEM optimisation result (symmetric D8 run).

Panels:
  A – sf_wale per region (top-down ring map)
  B – sf_course per region (top-down ring map)
  C – deviation |V_sim − V_target| for the best call
  D – convergence history (RMSE vs FEM call)

Parameters loaded from C5_16region_optimised_sym.json (7-param D8 run).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import csv

HERE    = os.path.dirname(os.path.abspath(__file__))
OFF     = os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off")
OPT_DIR = os.path.join(HERE, "optimisation")
OUT_PNG = os.path.join(HERE, "data", "C5", "C5_optim_best.png")
BG      = "#0f0f1a"

N_REGIONS = 16
RUN_PREFIX = "c5_p2"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def style_ax(ax, title_str, xlabel="x (m)", ylabel="y (m)", equal=True):
    ax.set_facecolor(BG)
    ax.set_title(title_str, color="white", fontsize=10, pad=5)
    ax.set_xlabel(xlabel, color="white", fontsize=8)
    ax.set_ylabel(ylabel, color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#555")
    if equal:
        ax.set_aspect("equal")


def add_colorbar(ax, cmap, norm, label):
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label(label, color="white", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
    cb.outline.set_edgecolor("white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    return cb


def fill_faces(ax, xi, yi, triangles, values, norm, cmap, edge_alpha=0.04):
    colors = cmap(norm(values))
    for fi, tri in enumerate(triangles):
        ax.fill(xi[tri], yi[tri], color=colors[fi], linewidth=0, zorder=1)
    if edge_alpha > 0:
        for tri in triangles:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                ax.plot([xi[tri[a]], xi[tri[b]]], [yi[tri[a]], yi[tri[b]]],
                        color="white", alpha=edge_alpha, linewidth=0.2, zorder=2)


# ── Load mesh topology + target ───────────────────────────────────────────────
V_target, F = load_off(OFF)
n_verts = len(V_target)

r_max = float(np.hypot(V_target[:, 0], V_target[:, 1]).max())
R_HOOP = 0.53 * r_max   # inner/outer split radius (matches optimiser)

# Face centroids and region assignment
cx_t  = V_target[F].mean(axis=1)[:, 0]
cy_t  = V_target[F].mean(axis=1)[:, 1]
r_t   = np.hypot(cx_t, cy_t)
az_t  = np.degrees(np.arctan2(cy_t, cx_t)) % 360
sect  = (az_t / 45.0).astype(int) % 8
freg  = np.where(r_t < R_HOOP, sect, sect + 8)   # 0-7 inner, 8-15 outer

# Vertex-level region
vx, vy = V_target[:, 0], V_target[:, 1]
vr     = np.hypot(vx, vy)
vaz    = np.degrees(np.arctan2(vy, vx)) % 360
vsect  = (vaz / 45.0).astype(int) % 8
vreg   = np.where(vr < R_HOOP, vsect, vsect + 8)


# ── Load stretch-factor params from symmetric optimisation result ─────────────
SF_FILE = os.path.join(OPT_DIR, "C5_16region_optimised_sym.json")
with open(SF_FILE) as f:
    sf_params = json.load(f)

sf_wale   = np.array([r["sf_wale"]       for r in sf_params["regions"]])
sf_course = np.array([r["sf_course"]     for r in sf_params["regions"]])
knit_dirs = np.array([r["knit_dir_deg"]  for r in sf_params["regions"]])

# Per-face sf values
face_sf_w = sf_wale[freg]
face_sf_c = sf_course[freg]


# ── Region centroids in 2D (for annotations) ──────────────────────────────────
reg_cx = np.array([cx_t[freg == r].mean() if (freg == r).any() else 0.0
                   for r in range(N_REGIONS)])
reg_cy = np.array([cy_t[freg == r].mean() if (freg == r).any() else 0.0
                   for r in range(N_REGIONS)])


# ── Convergence history ───────────────────────────────────────────────────────
bdry_mask    = np.hypot(V_target[:, 0], V_target[:, 1]) > r_max * 0.98
interior_idx = np.where(~bdry_mask)[0]

scalars_files = sorted(glob.glob(os.path.join(OPT_DIR, f"{RUN_PREFIX}_*_scalars.csv")))
history = []
for sf_path in scalars_files:
    vf = sf_path.replace("_scalars.csv", "_verts.csv")
    if not os.path.exists(vf):
        continue
    call_n = int(os.path.basename(sf_path).replace(f"{RUN_PREFIX}_", "").split("_")[0])
    with open(sf_path) as fp:
        rows = list(csv.DictReader(fp))
        if not rows:
            continue
        crown = float(rows[0]["crown_height"])
    V_sim = np.loadtxt(vf, delimiter=",", skiprows=1)[:, 1:]
    if V_sim.shape[0] != len(V_target):
        continue
    # Reject unconverged runs that returned the undeformed rest shape
    if float(np.max(np.linalg.norm(V_sim - V_target, axis=1))) < 1e-5:
        continue
    diff = V_sim[interior_idx] - V_target[interior_idx]
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    history.append((call_n, crown, rmse, V_sim))

history.sort(key=lambda h: h[0])
calls  = np.array([h[0] for h in history])
crowns = np.array([h[1] for h in history])
rmses  = np.array([h[2] for h in history])

best_idx  = int(np.argmin(rmses))
best_call = calls[best_idx]

V_best  = history[best_idx][3]
V_final = history[-1][3]

dev_best  = np.linalg.norm(V_best  - V_target, axis=1)
dev_final = np.linalg.norm(V_final - V_target, axis=1)

face_dev_best  = dev_best[F].mean(axis=1)
face_dev_final = dev_final[F].mean(axis=1)

print(f"Best   (call #{best_call}):  RMSE={rmses[best_idx]:.4f} m  "
      f"max_dev={dev_best.max()*1000:.2f} mm  crown={V_best[:,2].max():.4f} m")
print(f"Final  (call #{calls[-1]}):  RMSE={rmses[-1]:.4f} m  "
      f"max_dev={dev_final.max()*1000:.2f} mm  crown={V_final[:,2].max():.4f} m")
print(f"Target crown: {V_target[:,2].max():.4f} m")


# ── Figure: 2×2 grid ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 12), facecolor=BG)
gs  = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.35,
                        left=0.04, right=0.97, top=0.93, bottom=0.07)
ax_sfw  = fig.add_subplot(gs[0, 0])
ax_sfc  = fig.add_subplot(gs[0, 1])
ax_dev  = fig.add_subplot(gs[1, 0])
ax_hist = fig.add_subplot(gs[1, 1])

fig.suptitle(
    f"C5 16-region FEM optimisation (D8-symmetric, 7 params)  |  "
    f"best call #{best_call}: RMSE={rmses[best_idx]*1000:.2f} mm  |  "
    f"final call #{calls[-1]}: RMSE={rmses[-1]*1000:.2f} mm",
    color="white", fontsize=11, y=0.98)


# ── sf_wale and sf_course region maps ─────────────────────────────────────────
L_arrow = r_max * 0.08   # annotation arrow half-length

for ax, face_sf, sf_vals, cmap_name, label, title_str in [
    (ax_sfw, face_sf_w,  sf_wale,   "Blues",   "sf_wale",   "sf_wale per region (radial stretch)"),
    (ax_sfc, face_sf_c,  sf_course, "Oranges", "sf_course", "sf_course per region (hoop stretch)"),
]:
    style_ax(ax, title_str)

    all_v = np.concatenate([sf_wale, sf_course])
    vmin, vmax = all_v.min() - 0.002, all_v.max() + 0.002
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fill_faces(ax, V_target[:, 0], V_target[:, 1], F, face_sf, norm, cmap, edge_alpha=0.03)

    # Hoop ring boundary
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R_HOOP * np.cos(theta), R_HOOP * np.sin(theta),
            color="white", linewidth=1.2, alpha=0.7, zorder=4)

    # Spoke lines
    for k in range(8):
        ang = np.radians(k * 45)
        R_out = r_max * 1.02
        ax.plot([0, R_out * np.cos(ang)], [0, R_out * np.sin(ang)],
                color="white", linewidth=0.8, alpha=0.4, linestyle="--", zorder=3)

    # Region value annotations + knit direction arrow
    for r in range(N_REGIONS):
        cx_r, cy_r = reg_cx[r], reg_cy[r]
        val = sf_vals[r]
        kd  = np.radians(knit_dirs[r])
        dx, dy = L_arrow * np.cos(kd), L_arrow * np.sin(kd)
        ax.annotate("", xy=(cx_r + dx, cy_r + dy), xytext=(cx_r - dx, cy_r - dy),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.0), zorder=5)
        zone = "in" if r < 8 else "out"
        ax.text(cx_r, cy_r - L_arrow * 0.7,
                f"{zone}{r%8}\n{val:.4f}",
                ha="center", va="top", color="white", fontsize=6.5, fontweight="bold",
                zorder=6, bbox=dict(facecolor="#00000088", edgecolor="none", pad=1))

    add_colorbar(ax, cmap, norm, label)


# ── Deviation map (best call) ─────────────────────────────────────────────────
style_ax(ax_dev, f"Deviation |V_sim − V_target|  (best call #{best_call})", equal=True)

norm_dev = Normalize(vmin=0, vmax=dev_best.max())
cmap_dev = plt.cm.inferno
fill_faces(ax_dev, V_target[:, 0], V_target[:, 1], F, face_dev_best,
           norm_dev, cmap_dev, edge_alpha=0.03)

# Hoop ring + spokes overlay
theta = np.linspace(0, 2 * np.pi, 300)
ax_dev.plot(R_HOOP * np.cos(theta), R_HOOP * np.sin(theta),
            color="#00d4ff", linewidth=1.2, alpha=0.5, zorder=4, linestyle="--")
for k in range(8):
    ang = np.radians(k * 45)
    ax_dev.plot([0, r_max * 1.02 * np.cos(ang)], [0, r_max * 1.02 * np.sin(ang)],
                color="#00d4ff", linewidth=0.6, alpha=0.35, linestyle="--", zorder=3)

add_colorbar(ax_dev, cmap_dev, norm_dev, "Deviation (m)")

# Annotate best-result box
sp = sf_params["symmetric_params"]
ax_dev.text(0.02, 0.98,
            f"Best call #{best_call}: RMSE={rmses[best_idx]*1000:.2f} mm  "
            f"max_dev={dev_best.max()*1000:.2f} mm\n"
            f"sf_w_in={sp['sf_wale_in']:.4f}  sf_c_in={sp['sf_course_in']:.4f}  "
            f"sf_w_out={sp['sf_wale_out']:.4f}  sf_c_out={sp['sf_course_out']:.4f}\n"
            f"Si={sp['scale_Si']:.4f}  So={sp['scale_So']:.4f}  Ha={sp['scale_Ha']:.4f}",
            transform=ax_dev.transAxes, va="top", ha="left",
            color="white", fontsize=7.5, fontfamily="monospace",
            bbox=dict(facecolor="#0d0d1a", edgecolor="#444", alpha=0.9, pad=4))


# ── Convergence history ───────────────────────────────────────────────────────
ax_hist.set_facecolor(BG)
ax_hist.set_title("Convergence history (D8-symmetric, 7 params)", color="white",
                  fontsize=10, pad=5)
ax_hist.set_xlabel("FEM call #", color="white", fontsize=8)
ax_hist.set_ylabel("RMSE (mm)", color="#ff6b35", fontsize=8)
ax_hist.tick_params(colors="#ff6b35", labelsize=7)
ax_hist.spines["left"].set_color("#ff6b35")
for sp_name in ["top", "bottom", "right"]:
    ax_hist.spines[sp_name].set_color("#555")

rmses_mm = rmses * 1000
ax_hist.plot(calls, rmses_mm, color="#ff6b35", linewidth=1.2, zorder=3, label="RMSE")
ax_hist.scatter([best_call], [rmses_mm[best_idx]], color="#ffdd00", s=80, zorder=5,
                label=f"Best #{best_call}: {rmses_mm[best_idx]:.2f} mm")
ax_hist.scatter([calls[-1]], [rmses_mm[-1]], color="#00d4ff", s=60, zorder=5,
                marker="s", label=f"Final #{calls[-1]}: {rmses_mm[-1]:.2f} mm")

ax2 = ax_hist.twinx()
ax2.plot(calls, crowns, color="#b0e87c", linewidth=1.0, linestyle="--",
         alpha=0.8, label="crown h")
ax2.axhline(V_target[:, 2].max(), color="#b0e87c", linewidth=0.8,
            linestyle=":", alpha=0.5, label=f"target {V_target[:,2].max():.4f} m")
ax2.set_ylabel("Crown height (m)", color="#b0e87c", fontsize=8)
ax2.tick_params(colors="#b0e87c", labelsize=7)
ax2.spines["right"].set_color("#b0e87c")
ax2.spines["left"].set_color("#ff6b35")

lines1, lab1 = ax_hist.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax_hist.legend(lines1 + lines2, lab1 + lab2,
               loc="upper right", facecolor=BG, labelcolor="white", fontsize=7)
ax_hist.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

offset = max(5, int(len(calls) * 0.05))
ax_hist.annotate(f"#{best_call}\n{rmses_mm[best_idx]:.2f} mm",
                 xy=(best_call, rmses_mm[best_idx]),
                 xytext=(best_call + offset, rmses_mm[best_idx] + rmses_mm.std() * 0.5),
                 color="#ffdd00", fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="#ffdd00", lw=0.8))


# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
