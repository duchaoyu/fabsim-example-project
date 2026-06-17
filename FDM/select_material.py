"""
Material selection from the FDM tension field (B5).

Process (see chat for the reasoning):
  1. Load FDM result  -> per-edge force density q  + equilibrium geometry.
  2. Recover a per-REGION membrane stress tensor from the edge forces
     (virial / discrete-network stress:  sigma = (1/A) * sum_e q_e * (e (x) e) ).
  3. Resolve onto the knit directions -> demanded line tension (T_wale, T_course).
  4. Aggregate to the 9 B5 regions (same split as visualise_B5_9regions.py).
  5. For each pool material, the reachable tension box ceiling per direction is
        T_max,dir = (E*t)_dir * (sf_max - 1)
     (the E*t comes straight from a uniaxial strip tensile test).
  6. Overlay region demand cloud + material boxes, rank by suitability.

Output: FDM/material_selection_B5.png  (scatter + boxes  |  ranked table)

EDIT THE TWO CONFIG BLOCKS BELOW with your measured material data and the
pressure calibration, then re-run:  python FDM/select_material.py
"""
import os, heapq, json, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")
OUT_PNG     = os.path.join(HERE, "material_selection_B5.png")
BG          = "#1a1a2e"

# ════════════════════════════════════════════════════════════════════════════
# CONFIG 1 — MATERIAL POOL   (drop in your tensile-test numbers)
# ----------------------------------------------------------------------------
# Et_wale, Et_course = membrane stiffness E*t [N/m] = slope of the line-tension
# vs strain curve from a uniaxial strip test, measured in each knit direction.
# (For a knit, t is ambiguous, so E*t in N/m is the right quantity to use.)
# Defaults below = your 5 motifs with t = 1.0  (so E*t == E).
# ════════════════════════════════════════════════════════════════════════════
SF_MAX = 2.0          # max stretch factor  -> eps_max = SF_MAX - 1
EPS_MAX = SF_MAX - 1.0

MATERIALS = [
    # name,           Et_wale,  Et_course,  color
    ("M1 course-stiff",  5000.0,  12507.0,  "#00d4ff"),
    ("M2",               5000.0,   8000.0,  "#4dd0a0"),
    ("M3 isotropic",     5000.0,   5000.0,  "#ffd166"),
    ("M4",               8000.0,   5000.0,  "#ff9f4d"),
    ("M5 wale-stiff",   12507.0,   5000.0,  "#ff6b6b"),
]

# ════════════════════════════════════════════════════════════════════════════
# CONFIG 2 — UNIT CALIBRATION  (FDM tension is in FDM units; E*t is in N/m)
# ----------------------------------------------------------------------------
# fofin_B5.py runs at PRESSURE = 1.0 (dimensionless). To compare against the
# tensile test, scale the recovered tension so units match N/m. The honest
# anchor is your real forming pressure:  scale = REAL_PRESSURE / FDM_PRESSURE.
# Anisotropy (ratios) and region-to-region comparison are INDEPENDENT of this
# scale (and of the virial constant) — only the absolute box-fit depends on it.
# ════════════════════════════════════════════════════════════════════════════
FDM_PRESSURE  = 1.0
REAL_PRESSURE = 1000.0          # <-- set to your forming pressure (same basis as E*t)
TENSION_SCALE = REAL_PRESSURE / FDM_PRESSURE

# Knit directions (B5 uses uniform 0deg: wale || global x, course || global y).
WALE_AXIS   = np.array([1.0, 0.0, 0.0])
COURSE_AXIS = np.array([0.0, 1.0, 0.0])

R = 10.0   # half-span used for cable seeding (matches visualise_B5_9regions.py)

# ── Load FDM result (legacy compas json parsed directly) ────────────────────
_d      = json.load(open(RESULT_JSON))
n_v     = max(int(k) for k in _d["vertex"]) + 1
pts     = np.zeros((n_v, 3))
anchor  = np.zeros(n_v, dtype=bool)
for k, vd in _d["vertex"].items():
    i = int(k)
    pts[i] = [vd["x"], vd["y"], vd["z"]]
    anchor[i] = bool(vd.get("is_anchor", False))

faces = [[int(x) for x in fv] for fv in _d["face"].values()]   # list of vertex-index lists

edges, q_arr = [], []
for ek, ed in _d["edgedata"].items():
    u, v = ast.literal_eval(ek)
    edges.append((int(u), int(v)))
    q_arr.append(ed["qpre"])
q_arr = np.array(q_arr)
l_arr = np.array([np.linalg.norm(pts[u] - pts[v]) for u, v in edges])
f_arr = q_arr * l_arr   # axial force per edge

def tri_area(fv):
    p = pts[fv]
    if len(fv) == 3:
        return 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    # general polygon: fan triangulation
    a = 0.0
    for i in range(1, len(fv) - 1):
        a += 0.5 * np.linalg.norm(np.cross(p[i] - p[0], p[i + 1] - p[0]))
    return a

# ── 9-region split (replicated from visualise_B5_9regions.py) ────────────────
bdry     = set(np.where(anchor)[0].tolist())
bdry_pts = {v: pts[v] for v in bdry}
all_verts = list(range(n_v))

def high_force_path(src, tgt):
    cost = 1.0 / (f_arr + 1e-6)
    adj  = {v: [] for v in all_verts}
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, cost[i])); adj[v].append((u, cost[i]))
    dist = {v: np.inf for v in all_verts}; prev = {}
    dist[src] = 0.0; pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == tgt and u != src:
            path = []; cur = u
            while cur in prev: path.append(cur); cur = prev[cur]
            path.append(src); return list(reversed(path))
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]: dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    return []

def closest_bdry(xy):
    return min(bdry_pts, key=lambda v: np.linalg.norm(pts[v, :2] - np.array(xy)))

cable_seeds = [
    (closest_bdry([-R/3, -R]), closest_bdry([-R/3,  R])),  # A  x~-3.3
    (closest_bdry([ R/3, -R]), closest_bdry([ R/3,  R])),  # B  x~+3.3
    (closest_bdry([-R, -R/3]), closest_bdry([ R, -R/3])),  # C  y~-3.3
    (closest_bdry([-R,  R/3]), closest_bdry([ R,  R/3])),  # D  y~+3.3
]
cables = [high_force_path(s, t) for s, t in cable_seeds]
x_A = np.median([pts[v, 0] for v in cables[0]])
x_B = np.median([pts[v, 0] for v in cables[1]])
y_C = np.median([pts[v, 1] for v in cables[2]])
y_D = np.median([pts[v, 1] for v in cables[3]])

REGION_NAMES = {
    (2,0): "R1 TL", (2,1): "R2 TC", (2,2): "R3 TR",
    (1,0): "R4 ML", (1,1): "R5 MC", (1,2): "R6 MR",
    (0,0): "R7 BL", (0,1): "R8 BC", (0,2): "R9 BR",
}
def region_of(cx, cy):
    col = 0 if cx < x_A else (1 if cx < x_B else 2)
    row = 0 if cy < y_C else (1 if cy < y_D else 2)
    return row, col

# ── Per-region virial membrane stress -> demanded (T_wale, T_course) ─────────
# Assign each edge to a region by its midpoint; accumulate sigma = sum q (e (x) e),
# divide by region area (sum of face areas in the region).
reg_sigma = defaultdict(lambda: np.zeros((3, 3)))
reg_area  = defaultdict(float)

for fv in faces:
    cx  = np.mean([pts[v, 0] for v in fv]); cy = np.mean([pts[v, 1] for v in fv])
    reg_area[region_of(cx, cy)] += tri_area(fv)

for i, (u, v) in enumerate(edges):
    e   = pts[v] - pts[u]
    mid = 0.5 * (pts[u] + pts[v])
    reg = region_of(mid[0], mid[1])
    reg_sigma[reg] += q_arr[i] * np.outer(e, e)

def resolve(sigma, d):
    d = d / np.linalg.norm(d)
    return float(d @ sigma @ d)

rows = []  # (region, T_wale, T_course)
for key in sorted(REGION_NAMES, key=lambda k: REGION_NAMES[k]):
    A  = max(reg_area[key], 1e-9)
    sg = reg_sigma[key] / A * TENSION_SCALE
    Tw = max(resolve(sg, WALE_AXIS),   0.0)
    Tc = max(resolve(sg, COURSE_AXIS), 0.0)
    rows.append((REGION_NAMES[key], Tw, Tc))

Tw_all = np.array([r[1] for r in rows])
Tc_all = np.array([r[2] for r in rows])

print(f"\nFDM tension demand per region (scaled, N/m)  [TENSION_SCALE={TENSION_SCALE:g}]")
print(f"{'region':10s} {'T_wale':>10s} {'T_course':>10s} {'ratio c/w':>10s}")
for name, Tw, Tc in rows:
    print(f"{name:10s} {Tw:10.1f} {Tc:10.1f} {Tc/max(Tw,1e-6):10.2f}")

# ── Material suitability ─────────────────────────────────────────────────────
aniso_demand = np.median(Tc_all / np.maximum(Tw_all, 1e-6))

table = []  # name,color,ceil_w,ceil_c,feasible,max_util,mean_util,aniso_mat,mismatch,score,crit_P
for name, Etw, Etc, col in MATERIALS:
    ceil_w = Etw * EPS_MAX
    ceil_c = Etc * EPS_MAX
    util_w = Tw_all / ceil_w
    util_c = Tc_all / ceil_c
    max_util  = float(max(util_w.max(), util_c.max()))
    mean_util = float(np.mean(np.concatenate([util_w, util_c])))
    feasible  = max_util <= 1.0
    aniso_mat = Etc / Etw
    mismatch  = abs(np.log(aniso_demand / aniso_mat))   # 0 = perfect directional match
    # critical forming pressure: demand scales linearly w/ pressure, so the
    # material still fits all regions up to  REAL_PRESSURE / max_util.
    crit_P    = REAL_PRESSURE / max_util if max_util > 0 else np.inf
    # score: feasibility gate * directional fit * centredness (util near ~0.5 ideal)
    fit    = max(0.0, 1.0 - min(mismatch, 1.0))
    centre = max(0.0, 1.0 - abs(mean_util - 0.5) * 2.0)
    score  = (0.6 * fit + 0.4 * centre) * (1.0 if feasible else 0.25)
    table.append([name, col, ceil_w, ceil_c, feasible, max_util,
                  mean_util, aniso_mat, mismatch, score, crit_P])

table.sort(key=lambda r: r[-2], reverse=True)

print(f"\nDemanded course/wale anisotropy (median): {aniso_demand:.2f}")
print(f"\n{'material':16s} {'ceil_w':>8s} {'ceil_c':>8s} {'feas':>5s} "
      f"{'maxU':>6s} {'meanU':>6s} {'a_mat':>6s} {'maxP':>8s} {'score':>6s}")
for r in table:
    print(f"{r[0]:16s} {r[2]:8.0f} {r[3]:8.0f} {str(r[4]):>5s} "
          f"{r[5]:6.2f} {r[6]:6.2f} {r[7]:6.2f} {r[10]:8.0f} {r[9]:6.2f}")

# ── Per-region preferred material (anisotropy match) -> graded-material map ──
mat_aniso = [(name, Etc / Etw) for name, Etw, Etc, _ in MATERIALS]
print("\nPer-region preferred material (by anisotropy c/w):")
print(f"{'region':10s} {'aniso c/w':>10s}  best-match material")
region_pref = {}
for name, Tw, Tc in rows:
    a = Tc / max(Tw, 1e-6)
    best = min(mat_aniso, key=lambda m: abs(np.log(a / m[1])))
    region_pref[name] = best[0]
    print(f"{name:10s} {a:10.2f}  {best[0]}")
n_distinct = len(set(region_pref.values()))
print(f"\n-> {n_distinct} distinct materials preferred across 9 regions "
      f"({'single material OK' if n_distinct == 1 else 'graded / multi-material indicated'})")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE : scatter + boxes  |  ranked table
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 7.5)); fig.patch.set_facecolor(BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.18)

# ── Left: demand cloud + reachable boxes ─────────────────────────────────────
ax = fig.add_subplot(gs[0]); ax.set_facecolor(BG)
for sp in ax.spines.values(): sp.set_color("white")
ax.tick_params(colors="white", labelsize=9)
ax.set_xlabel("demanded wale tension  T_wale  [N/m]", color="white", fontsize=10)
ax.set_ylabel("demanded course tension  T_course  [N/m]", color="white", fontsize=10)
ax.set_title("FDM tension demand vs. material reachable boxes",
             color="white", fontsize=12)

# material reachable boxes (origin -> ceiling)
for r in table:
    name, col, cw, cc = r[0], r[1], r[2], r[3]
    ax.add_patch(Rectangle((0, 0), cw, cc, fill=False, edgecolor=col,
                           lw=2.2 if r[4] else 1.2,
                           ls="-" if r[4] else "--", alpha=0.9))
    ax.plot([], [], color=col, lw=2.5,
            ls="-" if r[4] else "--",
            label=f"{name}  ({'fits' if r[4] else 'too small'})")

# region demand points + labels
ax.scatter(Tw_all, Tc_all, s=90, c="white", edgecolor="black",
           zorder=5, label="region demand")
for name, Tw, Tc in rows:
    ax.annotate(name.split()[0], (Tw, Tc), textcoords="offset points",
                xytext=(6, 4), color="white", fontsize=8, zorder=6)

# guide line: demanded anisotropy
xm = max(Tw_all.max(), max(r[2] for r in table)) * 1.05
ax.plot([0, xm], [0, xm * aniso_demand], color="white", ls=":", lw=1,
        alpha=0.5, label=f"demand aniso c/w={aniso_demand:.2f}")

ax.set_xlim(0, xm)
ax.set_ylim(0, max(Tc_all.max(), max(r[3] for r in table)) * 1.05)
ax.legend(loc="upper right", facecolor="#0d0d1a", labelcolor="white",
          fontsize=8, framealpha=0.85)

# ── Right: ranked suitability table ──────────────────────────────────────────
axt = fig.add_subplot(gs[1]); axt.set_facecolor(BG); axt.axis("off")
axt.set_title("Material suitability  (ranked)", color="white", fontsize=12, pad=14)

col_labels = ["material", "E·t wale", "E·t course", "fits?",
              "max\nutil", "aniso\nc/w", "max P\n(fits)", "score"]
best_score = max(t[9] for t in table)
cell_text, cell_colors = [], []
for r in table:
    name, col, cw, cc, feas, mu, meu, amat, mism, sc, cP = r
    cell_text.append([name, f"{cw:.0f}", f"{cc:.0f}",
                      "yes" if feas else "no",
                      f"{mu:.2f}", f"{amat:.2f}", f"{cP:.0f}", f"{sc:.2f}"])
    base = "#16213e" if feas else "#3a1f2a"
    cell_colors.append([col, base, base,
                        "#1f6f4a" if feas else "#7a2b2b",
                        "#7a2b2b" if mu > 1 else base, base, base,
                        "#1f6f4a" if sc == best_score else base])

tbl = axt.table(cellText=cell_text, colLabels=col_labels,
                cellColours=cell_colors, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.1)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor("#444")
    cell.get_text().set_color("white")
    if ri == 0:
        cell.set_facecolor("#0d0d1a"); cell.get_text().set_fontweight("bold")

note = (
    f"Reachable box ceiling = E·t · (sf_max-1),  sf_max={SF_MAX}\n"
    f"Tension scale = REAL_PRESSURE/FDM_PRESSURE = {TENSION_SCALE:g}  (CALIBRATE)\n"
    f"score = 0.6·directional-fit + 0.4·centredness, gated by feasibility\n"
    f"Best fit: {table[0][0]}   |   demanded aniso c/w = {aniso_demand:.2f}"
)
axt.text(0.5, -0.02, note, transform=axt.transAxes, ha="center", va="top",
         color="white", fontsize=8.5, family="monospace",
         bbox=dict(facecolor="#0d0d1a", edgecolor="white", alpha=0.7, pad=6))

fig.suptitle("B5 — material selection from FDM tension field",
             color="white", fontsize=14, y=0.99)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\nSaved: {OUT_PNG}")
