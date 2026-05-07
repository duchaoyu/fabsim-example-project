"""
FDM form-finding for D5_remeshed.obj.

Key implementation detail: adjoint gradient — 3 sparse solves per iteration
instead of 3*n_e, giving ~800x speedup over the naive formulation.
"""
import os, datetime, json
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(HERE, "data", "D5", "D5_remeshed.obj")
OUTDIR = os.path.join(HERE, "data", "D5")
BG     = "#0f0f1a"

PRESSURE   = 1.0
Q_INIT     = 1.0
Q_MIN      = 0.01
INFLATE_IT = 5
MAXITER    = 2000


# ── Load OBJ ──────────────────────────────────────────────────────────────────
def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.array(verts, float), np.array(faces, int)


V_target, F = load_obj(INPUT)
n_v, n_f = len(V_target), len(F)

# ── Boundary = edges belonging to exactly one face ────────────────────────────
edge_count = {}
for tri in F:
    for i in range(3):
        e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
        edge_count[e] = edge_count.get(e, 0) + 1
boundary_verts = {v for e, c in edge_count.items() if c == 1 for v in e}

fixed = sorted(boundary_verts)
free  = [v for v in range(n_v) if v not in boundary_verts]
n_fix, n_fr = len(fixed), len(free)

edges = sorted(edge_count.keys())
n_e   = len(edges)

# ── Connectivity matrices ─────────────────────────────────────────────────────
rows, cols, data = [], [], []
for ei, (a, b) in enumerate(edges):
    rows += [ei, ei]; cols += [a, b]; data += [1.0, -1.0]
C   = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_e, n_v))
Ci  = C[:, free]
Cf  = C[:, fixed]
Cit = Ci.T.tocsr()

xyz_fix = V_target[fixed]
S_free  = V_target[free]
span    = V_target[:, 0].max() - V_target[:, 0].min()
height  = V_target[:, 2].max()

print(f"D5 FDM: {n_fr} free, {n_e} edges, {n_f} faces")
print(f"Span={span:.3f} m  crown={height:.4f} m  p={PRESSURE}  q0={Q_INIT}")
print(f"Solver: L-BFGS-B  adjoint gradient  (max {MAXITER} iters)\n")


# ── Vectorised vertex normals (area-weighted) ─────────────────────────────────
def vertex_normals_areas(xyz):
    v0, v1, v2 = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)          # (n_f, 3) unnormalised face normals
    fa = 0.5 * np.linalg.norm(fn, axis=1)    # (n_f,)   face areas
    vn = np.zeros((n_v, 3))
    va = np.zeros(n_v)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
        np.add.at(va, F[:, k], fa)
    va /= 3.0
    return vn, va


# ── FDM equilibrium with pressure (returns free-node positions) ───────────────
def inflate(q_vec, x_init=None):
    """INFLATE_IT Picard steps. LU factored once per call (q fixed)."""
    Q  = scipy.sparse.diags(q_vec)
    Dn = Cit.dot(Q).dot(Ci).tocsc()
    lu = scipy.sparse.linalg.splu(Dn)

    xyz = np.zeros((n_v, 3))
    xyz[fixed] = xyz_fix
    xyz[free]  = S_free if x_init is None else x_init

    for _ in range(INFLATE_IT):
        vn, va = vertex_normals_areas(xyz)
        norms  = np.linalg.norm(vn, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        p_free = vn[free] / norms[free] * va[free, np.newaxis] * PRESSURE
        rhs    = p_free - Cit.dot(Q).dot(Cf).dot(xyz_fix)
        xyz[free] = lu.solve(rhs)

    return xyz[free], lu   # return LU for gradient reuse


# ── Objective + adjoint gradient ──────────────────────────────────────────────
# Adjoint trick: instead of solving Dn @ (dX/dq_i) for each of n_e edges,
# solve Dn @ λ = diff once per axis (3 solves total).
# grad_q[i] = -2 * λ^T @ (dDn/dq_i) @ X_free
#           = -2 * λ^T @ Cit[:,i] * (Ci @ X_free)[i]
#           = -2 * (Cit^T @ λ)[i] * (Ci @ X_free)[i]
#           = -2 * (Ci @ λ)[i]   *  (Ce_x)[i]
# where Ce_x = C @ xyz_full (edge differences).
# Summing over axes: grad_q = -2 * sum_axis (Ci @ λ_axis) ⊙ (C @ xyz)_axis

_iter  = [0]
_x_prev = [None]   # warm-start inflate

def obj_and_grad(q_vec):
    X_free, lu = inflate(q_vec, _x_prev[0])
    _x_prev[0] = X_free.copy()

    diff = X_free - S_free                  # (n_fr, 3)
    obj  = float(np.sum(diff ** 2))

    xyz = np.zeros((n_v, 3))
    xyz[fixed] = xyz_fix
    xyz[free]  = X_free
    Ce_xyz = C.dot(xyz)                     # (n_e, 3)  edge differences

    grad_q = np.zeros(n_e)
    for axis in range(3):
        lam     = lu.solve(diff[:, axis])           # (n_fr,) — adjoint variable
        Ce_lam  = Ci.dot(lam)                       # (n_e,)
        grad_q -= 2.0 * Ce_lam * Ce_xyz[:, axis]

    _iter[0] += 1
    if _iter[0] % 20 == 0:
        rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        print(f"  iter {_iter[0]:4d}  RMSE={rmse:.5f} m  obj={obj:.4f}")
    return obj, grad_q


# ── Optimise ──────────────────────────────────────────────────────────────────
q0     = np.full(n_e, Q_INIT)
result = minimize(obj_and_grad, q0, jac=True, method="L-BFGS-B",
                  bounds=[(Q_MIN, None)] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-8, "gtol": 1e-8})

print(f"\nConverged: {result.success}  |  {result.message}")
print(f"Final obj={result.fun:.6f}  iters={_iter[0]}")
q_opt = result.x

# ── Final result ──────────────────────────────────────────────────────────────
X_final, _ = inflate(q_opt)
diff        = X_final - S_free
rmse        = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
print(f"RMSE={rmse:.5f} m  ({100*rmse/span:.3f}% of span)")

V_result         = V_target.copy()
V_result[free]   = X_final

# ── Save JSON ─────────────────────────────────────────────────────────────────
ts  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out = os.path.join(OUTDIR, f"D5_fdm_{ts}.json")
with open(out, "w") as f:
    json.dump({"geometry": "D5", "n_verts": n_v, "n_faces": n_f,
               "span_m": float(span), "target_crown_m": float(height),
               "fdm_crown_m": float(V_result[:, 2].max()),
               "rmse_m": float(rmse), "pressure": PRESSURE,
               "converged": bool(result.success),
               "q_min": float(q_opt.min()), "q_max": float(q_opt.max()),
               "verts": V_result.tolist(), "q": q_opt.tolist()}, f)
print(f"Saved JSON: {out}")

# ── Visualise ─────────────────────────────────────────────────────────────────
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

norm_q = Normalize(vmin=q_opt.min(), vmax=np.percentile(q_opt, 98))
cmap_q = plt.cm.hot

fig = plt.figure(figsize=(20, 8), facecolor=BG)
gs  = fig.add_gridspec(1, 3, wspace=0.08, left=0.03, right=0.97, top=0.90, bottom=0.06)
ax3d = fig.add_subplot(gs[0, 0], projection="3d")
ax2d = fig.add_subplot(gs[0, 1])
ax_h = fig.add_subplot(gs[0, 2])

fig.suptitle(
    f"D5 FDM  |  force densities q  |  q range [{q_opt.min():.2f}, {q_opt.max():.2f}]  |  "
    f"crown={V_result[:,2].max():.4f} m  (target {height:.4f} m)",
    color="white", fontsize=11, y=0.97)

# Left: 3D perspective
ax3d.set_facecolor(BG)
cr = Poly3DCollection([V_result[t] for t in F], zsort="min", linewidth=0)
cr.set_facecolor(np.full((n_f, 4), [0.12, 0.12, 0.15, 0.7]))
cr.set_edgecolor("none")
ax3d.add_collection3d(cr)
lc3d = Line3DCollection([[V_result[a], V_result[b]] for (a, b) in edges],
                         linewidths=0.7, cmap=cmap_q, norm=norm_q)
lc3d.set_array(q_opt)
ax3d.add_collection3d(lc3d)
ax3d.set_xlim(V_result[:, 0].min(), V_result[:, 0].max())
ax3d.set_ylim(V_result[:, 1].min(), V_result[:, 1].max())
ax3d.set_zlim(0, V_result[:, 2].max() * 1.15)
ax3d.set_xlabel("x", color="white", fontsize=7, labelpad=1)
ax3d.set_ylabel("y", color="white", fontsize=7, labelpad=1)
ax3d.set_zlabel("z", color="white", fontsize=7, labelpad=1)
ax3d.tick_params(colors="white", labelsize=6, pad=1)
ax3d.set_title("3D perspective", color="white", fontsize=9, pad=4)
ax3d.view_init(elev=30, azim=-60)
ax3d.set_box_aspect([1, 1, 0.5])
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False; pane.set_edgecolor("#333")
ax3d.grid(color="#333", linewidth=0.4)

# Middle: top-down plan view (XY projection)
ax2d.set_facecolor(BG)
ax2d.set_aspect("equal")
ax2d.set_title("Top view — edges coloured by force density q", color="white", fontsize=9, pad=4)
ax2d.set_xlabel("x (m)", color="white", fontsize=8)
ax2d.set_ylabel("y (m)", color="white", fontsize=8)
ax2d.tick_params(colors="white", labelsize=7)
for sp in ax2d.spines.values(): sp.set_color("#444")

segs2d = [[(V_result[a, 0], V_result[a, 1]), (V_result[b, 0], V_result[b, 1])]
          for (a, b) in edges]
lc2d = LineCollection(segs2d, linewidths=0.8, cmap=cmap_q, norm=norm_q, zorder=3)
lc2d.set_array(q_opt)
ax2d.add_collection(lc2d)
ax2d.set_xlim(V_result[:, 0].min() - 0.02, V_result[:, 0].max() + 0.02)
ax2d.set_ylim(V_result[:, 1].min() - 0.02, V_result[:, 1].max() + 0.02)

sm = ScalarMappable(cmap=cmap_q, norm=norm_q)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax2d, fraction=0.035, pad=0.03, shrink=0.85)
cb.set_label("Force density q", color="white", fontsize=7)
cb.ax.yaxis.set_tick_params(color="white", labelsize=6)
cb.outline.set_edgecolor("white")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

# Right: histogram
ax_h.set_facecolor(BG)
ax_h.set_title("Force density distribution", color="white", fontsize=9, pad=4)
ax_h.set_xlabel("q", color="white", fontsize=8)
ax_h.set_ylabel("edge count", color="white", fontsize=8)
ax_h.tick_params(colors="white", labelsize=7)
for sp in ax_h.spines.values(): sp.set_color("#555")

counts, bin_edges = np.histogram(q_opt, bins=60)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ax_h.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
         color=cmap_q(norm_q(bin_centers)), edgecolor="none", alpha=0.9)
ax_h.axvline(q_opt.mean(), color="white", linewidth=1.0, linestyle="--", alpha=0.7,
             label=f"mean = {q_opt.mean():.3f}")
ax_h.legend(facecolor=BG, labelcolor="white", fontsize=8)
ax_h.grid(color="#333", linestyle="--", linewidth=0.5, alpha=0.4, axis="y")

ax_h.text(0.97, 0.97,
          f"n edges:  {n_e}\n"
          f"q min:    {q_opt.min():.4f}\n"
          f"q max:    {q_opt.max():.4f}\n"
          f"q mean:   {q_opt.mean():.4f}\n"
          f"q median: {np.median(q_opt):.4f}\n"
          f"\ncrown target: {height*1000:.1f} mm\n"
          f"crown FDM:    {V_result[:,2].max()*1000:.1f} mm\n"
          f"RMSE:         {rmse*1000:.2f} mm",
          transform=ax_h.transAxes, va="top", ha="right",
          color="white", fontsize=8, fontfamily="monospace",
          bbox=dict(facecolor="#0d0d1a", edgecolor="#444", alpha=0.9, pad=4))

png_out = os.path.join(OUTDIR, "D5_fdm_result.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved PNG: {png_out}")
