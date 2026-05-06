"""Render the C5 remesh (.obj, 1309v/2137f) with cable paths overlaid.

cable_paths_C5.json indexes the coarser .off (961v); since the .obj is denser,
each cable vertex is remapped to its nearest .obj vertex (xyz nearest neighbour).
The remapped indices are written next to the script for reuse by the directional
field generator.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE      = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH  = os.path.join(HERE, "data", "C5", "C5_remeshed.obj")
OFF_PATH  = os.path.join(HERE, "..", "data", "C5_remeshed.off")
CABLES    = os.path.join(HERE, "..", "data", "cable_paths_C5.json")
OUT_PNG   = os.path.join(HERE, "data", "C5", "C5_obj_cables.png")
OUT_JSON  = os.path.join(HERE, "data", "C5", "cable_paths_C5_obj.json")


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t: continue
            if t[0] == "v":
                V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                # 1-based, may be 'a/b/c' — strip after slash
                F.append([int(s.split("/")[0]) - 1 for s in t[1:4]])
    return np.array(V), np.array(F)


def load_off(path):
    with open(path) as f:
        lines = f.read().strip().splitlines()
    nv, nf, _ = map(int, lines[1].split())
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


# ── Load both meshes + cable JSON ─────────────────────────────────────────────
V_obj, F_obj = load_obj(OBJ_PATH)
V_off, _     = load_off(OFF_PATH)
with open(CABLES) as f:
    cables_off = json.load(f)
print(f"OBJ : {len(V_obj)} verts, {len(F_obj)} faces  ({OBJ_PATH})")
print(f"OFF : {len(V_off)} verts                     ({OFF_PATH})")
print(f"Cables: {len(cables_off)} spokes")

# ── The .obj is unit-normalised; the .off is at full scale. Rescale before NN.
obj_span = float(V_obj[:, 0].max() - V_obj[:, 0].min())
off_span = float(V_off[:, 0].max() - V_off[:, 0].min())
scale = obj_span / off_span
V_off_scaled = V_off * scale
print(f"Scale ratio off→obj: {scale:.5f}  (x-span {off_span:.3f} → {obj_span:.3f})")

# ── Remap cable indices from .off → .obj via nearest-neighbour on xyz ─────────
def nearest_obj_idx(p):
    return int(np.argmin(np.sum((V_obj - p) ** 2, axis=1)))

cables_obj = {}
max_err = 0.0
for name, idx_list in cables_off.items():
    new_idx, errs = [], []
    for i in idx_list:
        j = nearest_obj_idx(V_off_scaled[i])
        new_idx.append(j)
        errs.append(float(np.linalg.norm(V_obj[j] - V_off_scaled[i])))
    cables_obj[name] = new_idx
    max_err = max(max_err, max(errs))
print(f"Max remap error: {max_err:.5f} (in obj units; ≈{max_err/scale:.3f} m at full scale)")

with open(OUT_JSON, "w") as f:
    json.dump(cables_obj, f, indent=2)
print(f"Remapped cables saved: {OUT_JSON}")

# ── Plot ──────────────────────────────────────────────────────────────────────
face_verts = [[V_obj[F_obj[i, j]] for j in range(3)] for i in range(len(F_obj))]

fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor("#1a1a2e")

# 3D
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_facecolor("#1a1a2e")
zmax = max(V_obj[:, 2].max(), 1e-6)
poly = Poly3DCollection(
    face_verts, alpha=0.55,
    facecolor=cm.viridis(V_obj[F_obj].mean(axis=1)[:, 2] / zmax),
    edgecolor="white", linewidth=0.12)
ax1.add_collection3d(poly)

for name, path in cables_obj.items():
    pts = V_obj[path]
    ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2],
             color="red", linewidth=2.0, alpha=0.95)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                color="orange", s=14, edgecolors="red", linewidths=0.4, zorder=5)

ax1.set_title(f"C5 remesh (.obj) — {len(V_obj)} verts / {len(F_obj)} faces  "
              f"(red = 8 cables remapped from .off)",
              color="white", fontsize=11)
ax1.set_xlabel("x (m)", color="white", fontsize=8)
ax1.set_ylabel("y (m)", color="white", fontsize=8)
ax1.set_zlabel("z (m)", color="white", fontsize=8)
ax1.tick_params(colors="white", labelsize=7)
for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
    pane.fill = False
xr = float(np.abs(V_obj[:, :2]).max()) * 1.05
ax1.set_xlim(-xr, xr); ax1.set_ylim(-xr, xr); ax1.set_zlim(0, zmax * 1.1)
ax1.view_init(elev=25, azim=-50)

# Top-down
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_facecolor("#1a1a2e")
ax2.set_aspect("equal")

for face in F_obj:
    p = V_obj[face]
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        ax2.plot([p[i, 0], p[j, 0]], [p[i, 1], p[j, 1]],
                 color="white", linewidth=0.18, alpha=0.28)

for name, path in cables_obj.items():
    pts = V_obj[path]
    ax2.plot(pts[:, 0], pts[:, 1], color="red", linewidth=2.4, alpha=0.95)
    ax2.scatter(pts[:, 0], pts[:, 1], color="orange", s=18,
                edgecolors="red", linewidths=0.5, zorder=5)

bdry_idx = np.where(np.abs(V_obj[:, 2]) < 1e-6)[0]
ax2.scatter(V_obj[bdry_idx, 0], V_obj[bdry_idx, 1], c="cyan", s=18, zorder=4,
            label=f"z=0 boundary ({len(bdry_idx)})")

# spoke endpoint stars + angle labels
for name, path in cables_obj.items():
    p = V_obj[path[-1]]
    ax2.scatter(p[0], p[1], c="yellow", s=90, zorder=6, marker="*",
                edgecolors="red", linewidths=1)
    angle = float(name.split("_")[1].rstrip("deg"))
    ax2.annotate(f"{angle:.1f}°", xy=(p[0], p[1]),
                 xytext=(p[0]*1.13, p[1]*1.13), color="red", fontsize=8,
                 ha="center", va="center")

ax2.set_title("Top-down view — cables on .obj mesh",
              color="white", fontsize=11)
ax2.set_xlabel("x (m)", color="white", fontsize=8)
ax2.set_ylabel("y (m)", color="white", fontsize=8)
ax2.tick_params(colors="white", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("white")
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)

fig.suptitle(f"C5 dense remesh (.obj) with stiffener spokes — "
             f"max nearest-neighbour error = {max_err:.4f} m",
             color="white", fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT_PNG}")
