"""
FDM form-finding for D5 on a Laplacian-smoothed target.

Same adjoint pipeline as fofin_D5.py, but the target geometry is smoothed
first (umbrella Laplacian, both boundaries pinned - the outer rim and the
oculus).  The unsmoothed run produced a patchy q field whose high-force
edges did not join up into continuous paths; local noise in the target
forces neighbouring edges to disagree.  Smoothing the target removes that
noise so the force paths can become connected.

  --lap-iters N   smoothing sweeps (default 10)
  --lap-lambda L  step per sweep   (default 0.5)
  --no-smooth     skip smoothing, i.e. reproduce fofin_D5.py

Writes D5_fdm_smooth_<ts>.json plus a q-field comparison figure.
"""
import argparse
import os, datetime, json, time
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


_ap = argparse.ArgumentParser()
_ap.add_argument("--lap-iters",  type=int,   default=10)
_ap.add_argument("--lap-lambda", type=float, default=0.5)
_ap.add_argument("--no-smooth",  action="store_true")
ARGS = _ap.parse_args()

V_target, F = load_obj(INPUT)
n_v, n_f = len(V_target), len(F)
V_raw = V_target.copy()

# ── Boundary = edges belonging to exactly one face ────────────────────────────
edge_count = {}
for tri in F:
    for i in range(3):
        e = tuple(sorted([tri[i], tri[(i+1) % 3]]))
        edge_count[e] = edge_count.get(e, 0) + 1
boundary_verts = {v for e, c in edge_count.items() if c == 1 for v in e}

fixed = sorted(boundary_verts)
free  = [v for v in range(n_v) if v not in boundary_verts]


# ── Laplacian smoothing of the target (boundaries pinned) ─────────────────────
def laplacian_smooth(V, F, n_iter, lam, pinned):
    """Uniform (umbrella) Laplacian: V_i <- V_i + lam * (mean(neighbours) - V_i)."""
    nbr = [set() for _ in range(len(V))]
    for tri in F:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            nbr[a].add(b); nbr[b].add(a)
    nbr = [np.fromiter(s, int) for s in nbr]
    movable = np.array([i for i in range(len(V)) if i not in pinned])
    V = V.copy()
    for _ in range(n_iter):
        Vn = V.copy()
        for i in movable:
            Vn[i] = V[i] + lam * (V[nbr[i]].mean(axis=0) - V[i])
        V = Vn
    return V


if not ARGS.no_smooth:
    V_target = laplacian_smooth(V_target, F, ARGS.lap_iters, ARGS.lap_lambda,
                                boundary_verts)
    shift = np.linalg.norm(V_target - V_raw, axis=1)
    print(f"Laplacian: {ARGS.lap_iters} sweeps, lambda={ARGS.lap_lambda}  "
          f"-> moved vertices by mean {shift.mean()*1000:.3f} mm, "
          f"max {shift.max()*1000:.3f} mm")
else:
    print("Laplacian: skipped (--no-smooth)")
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
t_opt0 = time.perf_counter()
result = minimize(obj_and_grad, q0, jac=True, method="L-BFGS-B",
                  bounds=[(Q_MIN, None)] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-8, "gtol": 1e-8})
t_opt  = time.perf_counter() - t_opt0

print(f"\nConverged: {result.success}  |  {result.message}")
print(f"Final obj={result.fun:.6f}  iters={_iter[0]}")
print(f"Elapsed:   {t_opt:.2f} s for {_iter[0]} iters "
      f"({1e3*t_opt/max(_iter[0],1):.1f} ms/iter, {n_e} design variables)")
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
tag = "D5_fdm" if ARGS.no_smooth else "D5_fdm_smooth"
out = os.path.join(OUTDIR, f"{tag}_{ts}.json")
with open(out, "w") as f:
    json.dump({"geometry": "D5",
               "laplacian_iters": 0 if ARGS.no_smooth else ARGS.lap_iters,
               "laplacian_lambda": 0.0 if ARGS.no_smooth else ARGS.lap_lambda, "n_verts": n_v, "n_faces": n_f,
               "span_m": float(span), "target_crown_m": float(height),
               "fdm_crown_m": float(V_result[:, 2].max()),
               "rmse_m": float(rmse), "pressure": PRESSURE,
               "converged": bool(result.success),
               "n_edges": int(n_e), "iters": int(_iter[0]),
               "elapsed_s": float(t_opt),
               "q_min": float(q_opt.min()), "q_max": float(q_opt.max()),
               "verts": V_result.tolist(), "q": q_opt.tolist()}, f)
print(f"Saved JSON: {out}")

# ── q-field comparison figure ─────────────────────────────────────────────────
from matplotlib.collections import LineCollection

P = np.array([[V_result[a], V_result[b]] for a, b in edges])
ref_path = os.path.join(OUTDIR, "D5_fdm_20260507143057.json")
panels = [(q_opt, P, f"smoothed target ({ARGS.lap_iters} sweeps)"
           if not ARGS.no_smooth else "unsmoothed")]
if os.path.exists(ref_path) and not ARGS.no_smooth:
    ref = json.load(open(ref_path))
    Vr  = np.array(ref["verts"]); qr = np.array(ref["q"])
    if len(qr) == n_e:
        panels.insert(0, (qr, np.array([[Vr[a], Vr[b]] for a, b in edges]),
                          "original (no smoothing)"))

fig, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 7.0))
if len(panels) == 1:
    axes = [axes]
for ax, (qq, PP, name) in zip(axes, panels):
    ref98 = np.percentile(qq, 98)
    lw = 0.12 + 4.2 * np.clip(qq / ref98, 0, 1)
    ax.add_collection(LineCollection(PP[:, :, :2], linewidths=lw,
                                     colors="#1b2a4a", alpha=0.85))
    ax.set_xlim(-0.66, 0.66); ax.set_ylim(-0.66, 0.66); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{name}\nq {qq.min():.3f}-{qq.max():.3f}, "
                 f"{100*np.mean(qq <= Q_MIN*1.01):.0f}% at Q_MIN", fontsize=10)
fig.suptitle(f"D5 FDM force densities - line thickness proportional to q "
             f"(RMSE {rmse*1000:.3f} mm)", fontsize=11, y=0.99)
fig.tight_layout()
png = os.path.join(OUTDIR, "D5_fdm_smooth_q.png" if not ARGS.no_smooth
                   else "D5_fdm_unsmoothed_q.png")
fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
print(f"Saved figure: {png}")
