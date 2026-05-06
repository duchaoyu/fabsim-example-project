"""C5 cable visualiser — user-specified cables on OBJ mesh.

Cable endpoints are 0-based OBJ vertex indices (C5_remeshed.obj, 1309 verts).
Layout: 8 inner spokes (apex→hoop) + 8 outer spokes (hoop→boundary) + hoop ring
= 24 cable sections.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import deque
import colorsys

HERE     = os.path.dirname(os.path.abspath(__file__))
OBJ_PATH = os.path.join(HERE, "data", "C5", "C5_remeshed.obj")
OUT_PNG  = os.path.join(HERE, "data", "C5", "C5_dense_cables.png")
OUT_JSON = os.path.join(HERE, "data", "C5", "cable_paths_C5_dense.json")

# ── Correct cable waypoints (0-based OBJ indices) ─────────────────────────────
CABLES_DEF = [
    (1091,  995,  949),
    (1091, 1059, 1032),
    (1091, 1138, 1193),
    (1091, 1187, 1218),
    (1091, 1170, 1212),
    (1091, 1105, 1128),
    (1091, 1028,  973),
    (1091,  977,  942),
]
HOOP_VERTS_OBJ = [995, 1059, 1138, 1187, 1170, 1105, 1028, 977]

# ── Load OBJ ──────────────────────────────────────────────────────────────────
def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t: continue
            if t[0] == "v":  V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f": F.append([int(s.split("/")[0])-1 for s in t[1:4]])
    return np.array(V), np.array(F, dtype=int)

V_obj, F_obj = load_obj(OBJ_PATH)
N_obj = len(V_obj)
print(f"OBJ: {N_obj} verts, {len(F_obj)} faces")

# ── Build cable paths from waypoints ──────────────────────────────────────────
cables = {}
for k, (v_src, v_hoop, v_dst) in enumerate(CABLES_DEF):
    theta = float(np.degrees(np.arctan2(V_obj[v_dst,1], V_obj[v_dst,0])) % 360)
    cables[f"Si{k}_{theta:.1f}"] = [v_src, v_hoop]   # inner spoke
    cables[f"So{k}_{theta:.1f}"] = [v_hoop, v_dst]   # outer spoke

hoop_ring_cable = HOOP_VERTS_OBJ + [HOOP_VERTS_OBJ[0]]
cables["H0_hoop"] = hoop_ring_cable

print(f"Cables: {len(cables)} paths, "
      f"{sum(len(p) for p in cables.values())} waypoints")

with open(OUT_JSON, "w") as f:
    json.dump(cables, f, indent=2)
print(f"Saved {OUT_JSON}")

# ── 16 connected-component regions ───────────────────────────────────────────
adj_obj = [set() for _ in range(N_obj)]
for tri in F_obj:
    for i, j in [(0,1),(1,2),(2,0)]:
        adj_obj[tri[i]].add(tri[j]); adj_obj[tri[j]].add(tri[i])

visited, comps = set(), []
for s in range(N_obj):
    if s in visited: continue
    comp = set([s]); q = deque([s])
    while q:
        u = q.popleft()
        for v in adj_obj[u]:
            if v not in comp: comp.add(v); q.append(v)
    visited |= comp; comps.append(sorted(comp))
comps.sort(key=lambda c: float(np.linalg.norm(V_obj[c, :2], axis=1).mean()))
print(f"Connected components: {len(comps)}")

r_hoop_mid = float(np.linalg.norm(V_obj[HOOP_VERTS_OBJ, :2], axis=1).mean())
region_info = []
for comp in comps:
    xy    = V_obj[comp, :2]
    r_m   = float(np.linalg.norm(xy, axis=1).mean())
    th_m  = float(np.degrees(np.arctan2(xy[:,1].mean(), xy[:,0].mean())) % 360)
    wedge = int(th_m // 45)
    rtype = "inner" if r_m < r_hoop_mid else "outer"
    region_info.append({"comp": comp, "wedge": wedge, "type": rtype})

def wedge_color(wedge, rtype):
    h = wedge / 8
    s = 0.55 if rtype == "inner" else 0.95
    v = 0.80 if rtype == "inner" else 0.65
    return colorsys.hsv_to_rgb(h, s, v)

region_colors = [wedge_color(r["wedge"], r["type"]) for r in region_info]

vert_comp = np.full(N_obj, -1, dtype=int)
for ci, ri in enumerate(region_info):
    for v in ri["comp"]: vert_comp[v] = ci
face_comp = np.full(len(F_obj), -1, dtype=int)
for fi, tri in enumerate(F_obj):
    c = vert_comp[tri[0]]
    if vert_comp[tri[1]] == c and vert_comp[tri[2]] == c:
        face_comp[fi] = c

# ── Plot ──────────────────────────────────────────────────────────────────────
BG          = "#0d0d1a"
xr          = float(np.abs(V_obj[:, :2]).max()) * 1.08
zmax        = float(V_obj[:, 2].max())
face_verts  = [[V_obj[F_obj[i,j]] for j in range(3)] for i in range(len(F_obj))]

def cable_style(name):
    if name.startswith("H"):  return "#00ff88", 3.0   # hoop
    if name.startswith("Si"): return "#ffdd00", 2.6   # inner spoke
    return                           "#ffdd00", 2.6   # outer spoke

face_colors = [region_colors[face_comp[fi]] if face_comp[fi] >= 0 else (0.08,0.08,0.12)
               for fi in range(len(F_obj))]

fig = plt.figure(figsize=(18, 9))
fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.08, wspace=0.05)
ax2 = fig.add_subplot(gs[:, 0])           # top-down
ax1 = fig.add_subplot(gs[0, 1], projection="3d")
ax3 = fig.add_subplot(gs[1, 1], projection="3d")

# ── Top-down ──────────────────────────────────────────────────────────────────
ax2.set_facecolor(BG); ax2.set_aspect("equal")

for fi, tri in enumerate(F_obj):
    ci = face_comp[fi]
    if ci < 0: continue
    poly2d = plt.Polygon(V_obj[tri, :2], facecolor=region_colors[ci],
                         edgecolor=region_colors[ci], linewidth=0.1, alpha=0.82)
    ax2.add_patch(poly2d)

for name, path in cables.items():
    pts = V_obj[path]
    col, lw = cable_style(name)
    ax2.plot(pts[:,0], pts[:,1], color=col, linewidth=lw,
             solid_capstyle="round", zorder=5)

# apex + hoop junction labels
ax2.scatter(*V_obj[1091, :2], color="white", s=120, zorder=10, marker="*")
ax2.annotate("1091", V_obj[1091, :2], xytext=(V_obj[1091,0]+0.015, V_obj[1091,1]+0.015),
             color="white", fontsize=6.5, zorder=11)
for k, (v_src, v_hoop, v_dst) in enumerate(CABLES_DEF):
    hx, hy = V_obj[v_hoop, :2]
    ax2.scatter(hx, hy, color="white", s=30, zorder=9)
    ax2.annotate(str(v_hoop), (hx, hy), xytext=(hx*0.80, hy*0.80),
                 color="white", fontsize=6.0, ha="center", va="center", zorder=11)
    bx, by = V_obj[v_dst, :2]
    ax2.scatter(bx, by, color="#ffdd00", s=30, zorder=9)
    ax2.annotate(str(v_dst), (bx, by), xytext=(bx*1.14, by*1.14),
                 color="#ffdd00", fontsize=6.0, ha="center", va="center",
                 fontweight="bold", zorder=11)

ax2.set_xlim(-xr, xr); ax2.set_ylim(-xr, xr)
ax2.tick_params(colors="#445566", labelsize=7)
for sp in ax2.spines.values(): sp.set_color("#223344")
ax2.set_xlabel("x", color="#445566", fontsize=8)
ax2.set_ylabel("y", color="#445566", fontsize=8)
ax2.set_title("16 regions coloured by wedge  (light=inner · saturated=outer)",
              color="white", fontsize=10, pad=5)
ax2.legend(handles=[mpatches.Patch(color="#ffdd00", label="spokes"),
                    mpatches.Patch(color="#00ff88", label="hoop")],
           loc="lower right", fontsize=8, facecolor="#111827",
           labelcolor="white", edgecolor="#334455", framealpha=0.85)

# ── 3D views ──────────────────────────────────────────────────────────────────
for ax, elev, azim in [(ax1, 28, -55), (ax3, 88, -90)]:
    ax.set_facecolor(BG)
    poly3 = Poly3DCollection(face_verts, alpha=0.88, facecolor=face_colors,
                             edgecolor=(0,0,0,0.15), linewidth=0.05)
    ax.add_collection3d(poly3)
    for name, path in cables.items():
        pts = V_obj[path]
        col, lw = cable_style(name)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], color=col, linewidth=lw, alpha=1.0)
    ax.scatter(*V_obj[1091], color="white", s=40, zorder=10)
    ax.tick_params(colors="#334455", labelsize=5)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#1a2a3a")
    ax.set_xlim(-xr, xr); ax.set_ylim(-xr, xr); ax.set_zlim(0, zmax*1.1)
    ax.view_init(elev=elev, azim=azim)

ax1.set_title("3D oblique", color="white", fontsize=9, pad=3)
ax3.set_title("3D top",     color="white", fontsize=9, pad=3)

fig.suptitle("C5  ·  8 inner spokes + 8 outer spokes + hoop ring = 24 sections",
             color="white", fontsize=13, y=1.01)
plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {OUT_PNG}")
