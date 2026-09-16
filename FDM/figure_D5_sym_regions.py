"""
D5, 10 regions under mirror_x symmetry: the region layout and its stretch factors.

Left  : plan view of the FEM mesh, faces shaded by pair (0-4).  The two regions
        of a pair are mirror images across x and share one parameter set, so the
        shading is an ordinal ramp, not 10 arbitrary hues.
Right : sf_wale and sf_course per pair on one shared axis (same unit).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

HERE   = os.path.dirname(os.path.abspath(__file__))
MESH   = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
MAP    = os.path.join(HERE, "optimisation", "d5_sym_v1_region_map.json")
PARAMS = os.path.join(HERE, "optimisation", "d5_sym_v1_optimised.json")
OUT    = os.path.join(HERE, "data", "D5", "D5_sym_regions_stretch.png")

# ordinal ramp, palette.md blue steps 250..650 (light mode floor is step 250)
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
WALE, COURSE = "#2a78d6", "#eb6834"
INK, INK2, GRIDC = "#0b0b0b", "#52514e", "#d8d7d2"


def load_off(path):
    L = open(path).readlines()
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in l.split()[:3]] for l in L[2:2 + nv]])
    F = [[int(x) for x in l.split()[1:]] for l in L[2 + nv:2 + nv + nf]]
    return V, F


V, F = load_off(MESH)
fr   = np.array(json.load(open(MAP))["face_regions"])
P    = json.load(open(PARAMS))
R    = P["regions"]
n_pairs = P["n_pairs"]

pair_w = {}; pair_c = {}
for r in R:
    p = r.get("pair", r["region_id"] % n_pairs)
    pair_w[p] = r["sf_wale"]; pair_c[p] = r["sf_course"]
pairs = sorted(pair_w)

fig = plt.figure(figsize=(14.5, 6.4))

# ── Panel 1: region layout ────────────────────────────────────────────────────
ax = fig.add_subplot(121)
face_pair = fr % n_pairs
polys  = [V[f][:, :2] for f in F]
colors = [RAMP[p] for p in face_pair]
ax.add_collection(PolyCollection(polys, facecolors=colors,
                                 edgecolors="#ffffff", linewidths=0.08))
# label each pair at the centroid of its faces (right-hand half only, then mirror)
for p in pairs:
    sel = np.where(face_pair == p)[0]
    cen = np.array([V[F[i]].mean(axis=0) for i in sel])
    for side in (cen[cen[:, 0] >= 0], cen[cen[:, 0] < 0]):
        if len(side) == 0:
            continue
        c = side.mean(axis=0)
        ax.text(c[0], c[1], str(p), ha="center", va="center", fontsize=13,
                color=INK, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.30", fc="white", ec=INK, lw=1.1))
ax.axvline(0.0, color=INK, lw=1.0, ls="--", alpha=0.6, zorder=4)
ax.text(0.0, 0.645, "mirror_x", ha="center", fontsize=8, color=INK2)
ax.set_xlim(-0.68, 0.68); ax.set_ylim(-0.68, 0.68); ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("y (m)", color=INK2)
ax.set_title(f"{P['n_regions']} regions as {n_pairs} mirror pairs "
             f"(both halves share one parameter set)", fontsize=10, color=INK)

# ── Panel 2: the stretch factors ──────────────────────────────────────────────
ax2 = fig.add_subplot(122)
y = np.arange(len(pairs))
w = np.array([pair_w[p] for p in pairs])
c = np.array([pair_c[p] for p in pairs])
for i in y:
    ax2.plot([c[i], w[i]], [i, i], color=GRIDC, lw=2, zorder=1, solid_capstyle="round")
ax2.scatter(w, y, s=95, color=WALE,   zorder=3, label="sf_wale")
ax2.scatter(c, y, s=95, color=COURSE, zorder=3, label="sf_course")
for i in y:
    ax2.annotate(f"{w[i]:.4f}", (w[i], i), (0, 11), textcoords="offset points",
                 ha="center", fontsize=8.5, color=INK)
    ax2.annotate(f"{c[i]:.4f}", (c[i], i), (0, -17), textcoords="offset points",
                 ha="center", fontsize=8.5, color=INK)
ax2.set_yticks(y, [f"pair {p}" for p in pairs], color=INK2)
ax2.invert_yaxis()
ax2.set_xlabel("rest-shape stretch factor", color=INK2)
ax2.set_xlim(1.00, 1.26)
ax2.grid(axis="x", color=GRIDC, lw=0.6)
ax2.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax2.spines[sp].set_visible(False)
ax2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.02, 0.62))
ax2.set_title(f"wale nearly constant ({w.min():.4f}-{w.max():.4f}); "
              f"course graded ({c.min():.4f}-{c.max():.4f})",
              fontsize=10, color=INK)

fig.suptitle(f"D5 mirror_x, {P['n_regions']} regions / {n_pairs} free pairs - "
             f"RMSE {P['rmse_mm']:.3f} mm, converged, {P['n_calls']} calls, "
             f"cable scale fixed {P['cable_scale_fixed']}",
             fontsize=11, color=INK, y=0.99)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", OUT)
