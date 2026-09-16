"""
Diagram of the B5 9-region grid and its optimised stretch factors.

Panel 1: plan view of the FEM mesh with the 3x3 region grid (cuts at +-R/3),
         each cell labelled with its region id and the crown marked.
Panel 2: sf_wale per region.   Panel 3: sf_course per region.
Panel 4: max z per region (target surface, with the optimised value beneath).

Both stretch-factor panels are sequential one-hue ramps (blue / orange) over
the *actual* value range, which is very narrow - the absolute span is printed
on each colourbar so the colour contrast is not mistaken for a large spread.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MESH   = os.path.join(ROOT, "data", "B5_remeshed_shared.off")
OPTIM  = os.path.join(ROOT, "data", "B5_optimised.off")
PARAMS = os.path.join(HERE, "optimisation", "B5_optimised_params.json")
OUT    = os.path.join(HERE, "data", "B5", "B5_regions_stretch_factors.png")

# palette (references/palette.md): blue ramp = sequential slot 1, orange = slot 2
BLUE   = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fce3d6", "#f8c3a6", "#f4a078", "#eb6834", "#c94f22", "#a03c19", "#742a11"]
AQUA   = ["#d6f2e7", "#a8e2ca", "#6fcda7", "#1baf7a", "#159063", "#10724e", "#0a4e36"]
CMAP_W = LinearSegmentedColormap.from_list("wale", BLUE)
CMAP_C = LinearSegmentedColormap.from_list("course", ORANGE)
CMAP_Z = LinearSegmentedColormap.from_list("maxz", AQUA)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#b9b8b3"


def load_off(path):
    L = open(path).readlines()
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in L[2:2 + nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in L[2 + nv:2 + nv + nf]])
    return V, F


V, F = load_off(MESH)
Vo, _ = load_off(OPTIM)
P = json.load(open(PARAMS))
R = max(np.ptp(V[:, 0]), np.ptp(V[:, 1])) / 2.0
cut = R / 3.0                                    # grid cuts at +-0.2 m

w = np.zeros((3, 3)); c = np.zeros((3, 3))
for r in P["regions"]:
    w[r["row"], r["col"]] = r["sf_wale"]
    c[r["row"], r["col"]] = r["sf_course"]

def region_of(x, y):
    col_ = 0 if x < -cut else (1 if x < cut else 2)
    row_ = 0 if y < -cut else (1 if y < cut else 2)
    return row_, col_

zt = np.full((3, 3), -np.inf); zo = np.full((3, 3), -np.inf)
for vt, vo in zip(V, Vo):
    r_, c_ = region_of(vt[0], vt[1])
    zt[r_, c_] = max(zt[r_, c_], vt[2])
    zo[r_, c_] = max(zo[r_, c_], vo[2])

fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.4))

# ── Panel 1: the region grid over the mesh plan ───────────────────────────────
ax = axes[0]
fc = V[F].mean(axis=1)                            # face centroids
col = np.where(fc[:, 0] < -cut, 0, np.where(fc[:, 0] < cut, 1, 2))
row = np.where(fc[:, 1] < -cut, 0, np.where(fc[:, 1] < cut, 1, 2))
rid = row * 3 + col
# alternating neutral tint: separates cells without encoding identity by colour
tint = np.where((row + col) % 2 == 0, "#e9e8e3", "#f7f6f3")
ax.add_collection(PolyCollection([V[f][:, :2] for f in F], facecolors=tint,
                                 edgecolors="#cfcec9", linewidths=0.18))
for t in (-cut, cut):
    ax.axvline(t, color=INK, lw=1.6, zorder=4)
    ax.axhline(t, color=INK, lw=1.6, zorder=4)
for r_ in range(3):
    for c_ in range(3):
        cx = (-R + cut) / 1.0 if False else [-(R + cut) / 2, 0.0, (R + cut) / 2][c_]
        cy = [-(R + cut) / 2, 0.0, (R + cut) / 2][r_]
        ax.text(cx, cy, str(r_ * 3 + c_), ha="center", va="center",
                fontsize=17, color=INK, zorder=6,
                bbox=dict(boxstyle="circle,pad=0.32", fc="white", ec=INK, lw=1.2))
        ax.text(cx, cy - 0.085, f"r{r_} c{c_}", ha="center", va="center",
                fontsize=7.5, color=INK2, zorder=6)
i_crown = V[:, 2].argmax()
ax.plot(V[i_crown, 0], V[i_crown, 1], marker="*", ms=15, color="#e34948",
        mec="white", mew=0.9, zorder=7)
ax.annotate("crown 0.2317 m", (V[i_crown, 0], V[i_crown, 1]), (0.135, -0.30),
            fontsize=8, color=INK2, zorder=7,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlim(-0.66, 0.66); ax.set_ylim(-0.66, 0.66); ax.set_aspect("equal")
ax.set_xlabel("x (m)", color=INK2); ax.set_ylabel("y (m)", color=INK2)
ax.set_title(f"9-region grid   (cuts at $\\pm R/3 = \\pm{cut:.1f}$ m)",
             fontsize=10, color=INK)

# ── Panels 2 & 3: the stretch factors ────────────────────────────────────────
for ax, M, cmap, name in ((axes[1], w, CMAP_W, "sf_wale"),
                          (axes[2], c, CMAP_C, "sf_course"),
                          (axes[3], zt, CMAP_Z, "max z")):
    im = ax.imshow(M, cmap=cmap, origin="lower", extent=[-0.5, 2.5, -0.5, 2.5])
    lo, hi = M.min(), M.max()
    for r_ in range(3):
        for c_ in range(3):
            v = M[r_, c_]
            dark = (v - lo) / (hi - lo) > 0.55
            ink = "white" if dark else INK
            if name == "max z":
                ax.text(c_, r_ + 0.20, f"{v:.4f} m", ha="center", va="center",
                        fontsize=11, color=ink)
                ax.text(c_, r_ + 0.02, f"opt {zo[r_, c_]:.4f} m", ha="center",
                        va="center", fontsize=7.5, color=ink, alpha=0.8)
                ax.text(c_, r_ - 0.11, f"{1000*(zo[r_, c_]-v):+.1f} mm", ha="center",
                        va="center", fontsize=7.5, color=ink, alpha=0.8)
            else:
                ax.text(c_, r_ + 0.10, f"{v:.6f}", ha="center", va="center",
                        fontsize=11, color=ink)
            ax.text(c_, r_ - 0.27, f"region {r_*3+c_}", ha="center", va="center",
                    fontsize=7.5, color=ink, alpha=0.75)
    ax.set_xticks([0, 1, 2], ["col 0", "col 1", "col 2"], color=INK2)
    ax.set_yticks([0, 1, 2], ["row 0", "row 1", "row 2"], color=INK2)
    if name == "max z":
        ax.set_title("max z per region   (target; optimised beneath)",
                     fontsize=10, color=INK)
        ax.plot(1, 1, marker="*", ms=14, color="#e34948", mec="white", mew=0.9,
                zorder=6)
    else:
        ax.set_title(f"{name}   (span {hi-lo:.2e}, {100*(hi-lo)/M.mean():.2f}%)",
                     fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    cb.outline.set_visible(False)
    cb.set_label(f"{lo:.4f} – {hi:.4f} m" if name == "max z"
                 else f"{lo:.6f} – {hi:.6f}", fontsize=8, color=INK2)

fig.suptitle("B5 inverse optimisation — 9 regions and their rest-shape stretch factors "
             f"(converged, RMSE {1000*P['loss_rmse_m']:.2f} mm, knit direction 0° everywhere)",
             fontsize=11, color=INK, y=1.0)
fig.text(0.5, -0.035,
         "Both fields vary by well under 0.3% — the colour ramps are stretched over a very narrow "
         "range.\nThe optimiser effectively converged on a single uniform ~4.14% scale-up; "
         "the shape work is done by the cables.",
         ha="center", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print("saved", OUT)
