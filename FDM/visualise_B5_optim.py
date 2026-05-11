"""
Visualise B5 FEM optimisation result:
  - Left:  9-region top-down map with sf_wale / sf_course / knit_dir per region
  - Right: RMSE convergence history
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
PARAMS_JSON = os.path.join(HERE, "optimisation", "B5_optimised_params.json")
OUT_PNG     = os.path.join(HERE, "B5_optim_result.png")
BG          = "#0f0f1a"

# ── Load optimised params ─────────────────────────────────────────────────────
with open(PARAMS_JSON) as f:
    params = json.load(f)

regions_data = {(r["row"], r["col"]): r for r in params["regions"]}
history      = params["history"]
best_rmse    = params["loss_rmse_m"]
n_calls      = params["n_calls"]
t_crown      = params["target_crown_m"]
converged    = params["converged"]

# ── Load FDM mesh ─────────────────────────────────────────────────────────────
from compas.datastructures import Mesh
with open(RESULT_JSON) as _f:
    _d = json.load(_f)

mesh = Mesh()
for vkey, vdata in _d["vertex"].items():
    mesh.add_vertex(int(vkey), x=vdata["x"], y=vdata["y"], z=vdata["z"])
for fverts in _d["face"].values():
    mesh.add_face(fverts)

# Build qpre lookup from edgedata (keys stored as "(u, v)" strings)
_qpre = {}
for k, v in _d.get("edgedata", {}).items():
    u, w = [int(x) for x in k.strip("()").split(",")]
    _qpre[(u, w)] = _qpre[(w, u)] = v.get("qpre", 0.0)

verts = list(mesh.vertices())
pts   = np.array([mesh.vertex_coordinates(v) for v in verts])
edges = list(mesh.edges())
faces = list(mesh.faces())

q_arr = np.array([_qpre.get((u, v), 0.0) for u, v in edges])
l_arr = np.array([np.linalg.norm(pts[u] - pts[v]) for u, v in edges])
f_arr = q_arr * l_arr

# ── Cable paths (Dijkstra, same logic as visualise_B5_9regions.py) ─────────────
import heapq

bdry     = set(mesh.vertices_on_boundary())
bdry_pts = {v: pts[v] for v in bdry}

def high_force_path(src, tgt):
    cost = 1.0 / (f_arr + 1e-6)
    adj  = {v: [] for v in mesh.vertices()}
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, cost[i], i))
        adj[v].append((u, cost[i], i))
    dist = {v: np.inf for v in mesh.vertices()}
    prev = {}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == tgt and u != src:
            path_v, path_e = [], []
            cur = u
            while cur in prev:
                pv, ei = prev[cur]
                path_v.append(cur); path_e.append(ei); cur = pv
            path_v.append(src)
            return list(reversed(path_v)), list(reversed(path_e))
        for v, w, ei in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = (u, ei)
                heapq.heappush(pq, (nd, v))
    return [], []

def closest_bdry(xy):
    return min(bdry_pts, key=lambda v: np.linalg.norm(pts[v, :2] - np.array(xy)))

R = 10.0
cable_specs = [
    (closest_bdry([-R/3, -R]), closest_bdry([-R/3,  R]), "#00d4ff"),
    (closest_bdry([ R/3, -R]), closest_bdry([ R/3,  R]), "#00d4ff"),
    (closest_bdry([-R,  -R/3]), closest_bdry([ R, -R/3]), "#ff6b35"),
    (closest_bdry([-R,   R/3]), closest_bdry([ R,  R/3]), "#ff6b35"),
]
cable_paths = []
for src, tgt, col in cable_specs:
    pv, pe = high_force_path(src, tgt)
    cable_paths.append((pv, pe, col))

# ── Region assignment ─────────────────────────────────────────────────────────
x_A = np.median([pts[v, 0] for v in cable_paths[0][0]])
x_B = np.median([pts[v, 0] for v in cable_paths[1][0]])
y_C = np.median([pts[v, 1] for v in cable_paths[2][0]])
y_D = np.median([pts[v, 1] for v in cable_paths[3][0]])

def face_region(cx, cy):
    col = 0 if cx < x_A else (1 if cx < x_B else 2)
    row = 0 if cy < y_C else (1 if cy < y_D else 2)
    return row, col

face_cx = np.array([np.mean([mesh.vertex_coordinates(v)[0] for v in mesh.face_vertices(f)]) for f in faces])
face_cy = np.array([np.mean([mesh.vertex_coordinates(v)[1] for v in mesh.face_vertices(f)]) for f in faces])
face_regs = [face_region(face_cx[i], face_cy[i]) for i in range(len(faces))]

reg_pts = defaultdict(list)
for i, rc in enumerate(face_regs):
    reg_pts[rc].append([face_cx[i], face_cy[i]])
reg_centroids = {k: np.mean(v, axis=0) for k, v in reg_pts.items()}

# ── Colour regions by sf_wale ─────────────────────────────────────────────────
sf_w_vals = np.array([regions_data[(r,c)]["sf_wale"]  for r in range(3) for c in range(3)])
sf_c_vals = np.array([regions_data[(r,c)]["sf_course"] for r in range(3) for c in range(3)])
vmin_w, vmax_w = sf_w_vals.min(), sf_w_vals.max()
vmin_c, vmax_c = sf_c_vals.min(), sf_c_vals.max()

cmap_w = plt.cm.Blues
cmap_c = plt.cm.Oranges
norm_w = Normalize(vmin=vmin_w - 0.01, vmax=vmax_w + 0.01)
norm_c = Normalize(vmin=vmin_c - 0.01, vmax=vmax_c + 0.01)

def region_fill_color(row, col, mode):
    rd = regions_data[(row, col)]
    if mode == "wale":
        return cmap_w(norm_w(rd["sf_wale"]))
    else:
        return cmap_c(norm_c(rd["sf_course"]))

# ── Figure layout: 1 row × 3 cols ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 7), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.35)
ax_w = fig.add_subplot(gs[0, 0])   # sf_wale map
ax_c = fig.add_subplot(gs[0, 1])   # sf_course map
ax_h = fig.add_subplot(gs[0, 2])   # RMSE history

fig.suptitle(
    f"B5 FEM Optimisation  |  RMSE={best_rmse:.4f} m  |  calls={n_calls}  "
    f"|  target crown={t_crown:.3f} m  |  converged={converged}",
    color="white", fontsize=12, y=1.01)

def setup_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.set_xlabel("x (m)", color="white", fontsize=8)
    ax.set_ylabel("y (m)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#555")

setup_ax(ax_w, "sf_wale per region")
setup_ax(ax_c, "sf_course per region")

for ax, mode, cmap, norm in [
    (ax_w, "wale",   cmap_w, norm_w),
    (ax_c, "course", cmap_c, norm_c),
]:
    # Shade mesh faces by region colour
    for i, f in enumerate(faces):
        row, col = face_regs[i]
        fv_xy = [mesh.vertex_coordinates(v)[:2] for v in mesh.face_vertices(f)]
        c = region_fill_color(row, col, mode)
        ax.fill([p[0] for p in fv_xy], [p[1] for p in fv_xy],
                color=c, alpha=0.75, zorder=1)

    # Mesh edges (faint)
    for u, v in edges:
        ax.plot([pts[u,0], pts[v,0]], [pts[u,1], pts[v,1]],
                color="white", alpha=0.05, linewidth=0.3, zorder=2)

    # Cable paths
    for pv, pe, col in cable_paths:
        if not pv: continue
        ax.plot([pts[v,0] for v in pv], [pts[v,1] for v in pv],
                color=col, linewidth=2.5, zorder=4, solid_capstyle="round")

    # Region labels: value + knit dir arrow — sizes scale with geometry
    R = max(pts[:,0].max() - pts[:,0].min(), pts[:,1].max() - pts[:,1].min()) / 2
    L          = 0.13 * R   # arrow half-length
    label_off  = 0.02 * R   # text offset below centroid
    for (row, col), ctr in reg_centroids.items():
        rd  = regions_data[(row, col)]
        val = rd["sf_wale"] if mode == "wale" else rd["sf_course"]
        kd  = np.radians(rd["knit_dir_deg"])
        dx, dy = L * np.cos(kd), L * np.sin(kd)
        ax.annotate("", xy=(ctr[0]+dx, ctr[1]+dy), xytext=(ctr[0]-dx, ctr[1]-dy),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.2), zorder=5)
        lbl_r = {0: "B", 1: "M", 2: "T"}[row]
        lbl_c = {0: "L", 1: "C", 2: "R"}[col]
        ax.text(ctr[0], ctr[1] - label_off,
                f"{lbl_r}{lbl_c}\n{val:.4f}\n{rd['knit_dir_deg']}°",
                ha="center", va="top", color="white", fontsize=7, fontweight="bold",
                zorder=6, bbox=dict(facecolor="#00000080", edgecolor="none", pad=1.5))

    # Colorbar
    sm  = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb  = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    cb.outline.set_edgecolor("white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    cb.set_label("sf_" + mode, color="white", fontsize=8)

# ── RMSE convergence history ──────────────────────────────────────────────────
ax_h.set_facecolor(BG)
ax_h.set_title("RMSE convergence history", color="white", fontsize=10, pad=6)
ax_h.set_xlabel("FEM call #", color="white", fontsize=8)
ax_h.set_ylabel("RMSE (m)", color="white", fontsize=8)
ax_h.tick_params(colors="white", labelsize=7)
for sp in ax_h.spines.values(): sp.set_color("#555")

calls  = [h["call"]   for h in history]
losses = [h["loss"]   for h in history]
crowns = [h["crown_height"] for h in history]

ax_h.plot(calls, losses, color="#00d4ff", linewidth=1.5, zorder=3, label="RMSE")
ax_h.scatter(calls, losses, color="#00d4ff", s=20, zorder=4)

# Best point
best_idx = int(np.argmin(losses))
ax_h.scatter(calls[best_idx], losses[best_idx],
             color="#ff6b35", s=80, zorder=5, label=f"Best {losses[best_idx]:.4f} m")

# Crown height on twin axis
ax2 = ax_h.twinx()
ax2.plot(calls, crowns, color="#b0e87c", linewidth=1.2, linestyle="--", alpha=0.8, label="crown h")
ax2.axhline(t_crown, color="#b0e87c", linewidth=0.8, linestyle=":", alpha=0.5)
ax2.set_ylabel("Crown height (m)", color="#b0e87c", fontsize=8)
ax2.tick_params(colors="#b0e87c", labelsize=7)
ax2.spines["right"].set_color("#b0e87c")
ax2.text(calls[-1] + 0.3, t_crown + 0.01, f"target {t_crown:.2f}",
         color="#b0e87c", fontsize=7, va="bottom")

# Combined legend
lines1, lab1 = ax_h.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax_h.legend(lines1 + lines2, lab1 + lab2,
            loc="upper right", facecolor=BG, labelcolor="white", fontsize=7)

ax_h.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.6)

# ── Summary table below history plot ─────────────────────────────────────────
rows_text = [f"{'Reg':>3}  {'sf_wale':>9}  {'sf_course':>10}  {'knit':>5}"]
rows_text += ["-" * 36]
for row in range(3):
    for col in range(3):
        rd = regions_data[(row, col)]
        lbl = f"{'BTM'[2-row]}{'LCR'[col]}"
        rows_text.append(f"{lbl:>3}  {rd['sf_wale']:>9.4f}  {rd['sf_course']:>10.4f}  {rd['knit_dir_deg']:>4}°")
summary = "\n".join(rows_text)
fig.text(0.67, -0.02, summary, ha="left", va="top", color="white",
         fontsize=7.5, fontfamily="monospace",
         bbox=dict(facecolor="#0d0d1a", edgecolor="#444", alpha=0.9, pad=5))

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {OUT_PNG}")
