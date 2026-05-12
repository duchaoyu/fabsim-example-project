"""
FDM form-finding for C5_remeshed_smooth.obj (961v / 1856f).

Same pipeline as fofin_C5.py / fofin_C5_dense.py but targets the
Laplacian-smoothed remesh.  Saves:
  data/C5/mesh_out_C5_smooth_<timestamp>.json   (full result)
  data/C5/mesh_out_C5_smooth_latest.json        (fixed-name for downstream)
  data/C5/C5_smooth_fdm_result.png              (matplotlib visualisation)
"""
import os, datetime
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh
from compas.matrices import connectivity_matrix

HERE  = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "data", "C5", "C5_remeshed_smooth.obj")
DATA  = os.path.join(HERE, "data", "C5")

PRESSURE     = 1.0
Q_INIT       = 1.0
Q_MIN        = 0.01
INFLATE_IT   = 5
INFLATE_DAMP = 1.0
MAXITER      = 2000

# ── Load + flatten ────────────────────────────────────────────────────────────
mesh_target = Mesh.from_obj(INPUT)
mesh_target.update_default_vertex_attributes(is_anchor=False, px=0.0, py=0.0, pz=0.0, residual=None)
mesh_target.update_default_edge_attributes(qpre=Q_INIT)

for vkey in mesh_target.vertices_on_boundary():
    mesh_target.vertex_attribute(vkey, "is_anchor", True)

target_xyz = {v: mesh_target.vertex_coordinates(v) for v in mesh_target.vertices()}

mesh  = mesh_target.copy()
fixed = list(mesh.vertices_where({"is_anchor": True}))
free  = [v for v in mesh.vertices() if v not in fixed]
for v in free:
    mesh.vertex_attribute(v, "z", 0.0)

edges = list(mesh.edges())
n_e   = len(edges)
n_v   = mesh.number_of_vertices()

C   = connectivity_matrix(edges, "csr")
Ci  = C[:, free]; Cf = C[:, fixed]; Cit = Ci.T
xyz_fixed = np.array([mesh.vertex_coordinates(v) for v in fixed], dtype=float)
S_free    = np.array([target_xyz[v] for v in free], dtype=float)


def inflate(xyz_full, q_vec, pressure):
    xyz   = xyz_full.copy()
    loads = np.zeros_like(xyz)
    for _ in range(INFLATE_IT):
        for v in free + fixed:
            mesh.vertex_attributes(v, ["x", "y", "z"], xyz[v].tolist())
        for v in free:
            n = np.array(mesh.vertex_normal(v))
            a = mesh.vertex_area(v)
            loads[v] = n * a * pressure
        Q  = scipy.sparse.diags(q_vec)
        Dn = Cit.dot(Q).dot(Ci)
        pf = loads[free] - Cit.dot(Q).dot(Cf).dot(xyz_fixed)
        xyz_free_new = scipy.sparse.linalg.spsolve(Dn, pf)
        for i, v in enumerate(free):
            xyz[v] = (1.0 - INFLATE_DAMP) * xyz[v] + INFLATE_DAMP * xyz_free_new[i]
    return xyz


_call = [0]

def obj_grad(q_vec):
    xyz_full = np.zeros((n_v, 3), dtype=float)
    for i, v in enumerate(fixed):
        xyz_full[v] = xyz_fixed[i]
    xyz_eq = inflate(xyz_full, q_vec, PRESSURE)

    X_free = np.array([xyz_eq[v] for v in free])
    diff   = X_free - S_free
    obj    = float(np.sum(diff ** 2))

    Q   = scipy.sparse.diags(q_vec)
    Dn  = Cit.dot(Q).dot(Ci)
    xyz_arr = np.array([xyz_eq[v] for v in range(n_v)])

    grad = np.zeros(n_e)
    for ax in range(3):
        b     = Cit.dot(scipy.sparse.diags(C.dot(xyz_arr[:, ax])))
        dX_dq = -scipy.sparse.linalg.spsolve(Dn, b)
        grad += 2.0 * (diff[:, ax] @ dX_dq)

    _call[0] += 1
    if _call[0] % 20 == 0:
        rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        print(f"  iter {_call[0]:4d}  obj={obj:.6f}  RMSE={rmse:.5f} m", flush=True)
    return obj, grad


span   = max(p[0] for p in target_xyz.values()) - min(p[0] for p in target_xyz.values())
height = max(p[2] for p in target_xyz.values())
print(f"C5 smooth FDM: {len(free)} free, {n_e} edges  ({INPUT})", flush=True)
print(f"Target span={span:.3f} m  height={height:.3f} m  pressure={PRESSURE}", flush=True)
print(f"Solver: L-BFGS-B (max {MAXITER} iters)\n", flush=True)

q0     = np.full(n_e, Q_INIT)
result = minimize(obj_grad, q0, jac=True, method="L-BFGS-B",
                  bounds=[(Q_MIN, None)] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-8, "gtol": 1e-8, "disp": True})

print(f"\nConverged: {result.success}  |  {result.message}", flush=True)
print(f"obj={result.fun:.6f}  calls={_call[0]}", flush=True)

q_opt = result.x
xyz_full = np.zeros((n_v, 3), dtype=float)
for i, v in enumerate(fixed):
    xyz_full[v] = xyz_fixed[i]
xyz_final = inflate(xyz_full, q_opt, PRESSURE)

mesh_out = mesh_target.copy()
for v in mesh_out.vertices():
    mesh_out.vertex_attributes(v, ["x", "y", "z"], xyz_final[v].tolist())
for i, e in enumerate(mesh_out.edges()):
    mesh_out.edge_attribute(e, "qpre", float(q_opt[i]))

X_free_final = np.array([xyz_final[v] for v in free])
rmse = float(np.sqrt(np.mean(np.sum((X_free_final - S_free) ** 2, axis=1))))
print(f"Final RMSE: {rmse:.5f} m  ({100*rmse/span:.2f}% of span)", flush=True)

# ── Save JSON ─────────────────────────────────────────────────────────────────
os.makedirs(DATA, exist_ok=True)
ts  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out = os.path.join(DATA, f"mesh_out_C5_smooth_{ts}.json")
mesh_out.to_json(out)
out_latest = os.path.join(DATA, "mesh_out_C5_smooth_latest.json")
mesh_out.to_json(out_latest)
print(f"Saved: {out}", flush=True)
print(f"Saved: {out_latest}", flush=True)

# ── Matplotlib PNG (server-safe) ──────────────────────────────────────────────
vkeys  = list(mesh_out.vertices())
v_idx  = {v: i for i, v in enumerate(vkeys)}
V_out  = np.array([mesh_out.vertex_coordinates(v) for v in vkeys])
V_tgt  = np.array([target_xyz[v] for v in vkeys])
faces  = [[v_idx[v] for v in mesh_out.face_vertices(f)] for f in mesh_out.faces()]

q_max = q_opt.max()

fig = plt.figure(figsize=(14, 5))

# Row 1, col 1: FDM result surface
ax1 = fig.add_subplot(131, projection="3d")
tris_out = [V_out[f] for f in faces if len(f) == 3]
mc = Poly3DCollection(tris_out, alpha=0.15, facecolor="tomato", edgecolor="none")
ax1.add_collection3d(mc)
ax1.set_box_aspect([1, 1, 0.55]); ax1.view_init(elev=30, azim=-60)
ax1.set_title("FDM result", fontsize=9)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
for v, xyz in zip(vkeys, V_out):
    ax1.scatter(*xyz, s=0.5, c="tomato", depthshade=False)

# Row 1, col 2: target surface
ax2 = fig.add_subplot(132, projection="3d")
tris_tgt = [V_tgt[f] for f in faces if len(f) == 3]
mc2 = Poly3DCollection(tris_tgt, alpha=0.15, facecolor="steelblue", edgecolor="none")
ax2.add_collection3d(mc2)
ax2.set_box_aspect([1, 1, 0.55]); ax2.view_init(elev=30, azim=-60)
ax2.set_title("Target (C5 smooth)", fontsize=9)
ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")

# Row 1, col 3: top-down q heatmap
ax3 = fig.add_subplot(133)
q_edge = np.array([mesh_out.edge_attribute(e, "qpre") for e in mesh_out.edges()])
# Colour each edge by q
for idx, e in enumerate(mesh_out.edges()):
    u, v2 = e
    p1 = mesh_out.vertex_coordinates(u)
    p2 = mesh_out.vertex_coordinates(v2)
    t = float(q_opt[idx]) / q_max
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]],
             color=plt.cm.plasma(t), linewidth=0.5, alpha=0.7)
ax3.set_aspect("equal"); ax3.set_title("Force densities q (top view)", fontsize=9)
ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")

sm = plt.cm.ScalarMappable(cmap="plasma",
                           norm=plt.Normalize(vmin=q_opt.min(), vmax=q_max))
sm.set_array([])
fig.colorbar(sm, ax=ax3, label="q", shrink=0.8)

fig.suptitle(f"C5 smooth FDM  RMSE={rmse*1000:.1f} mm  span={span:.2f} m",
             fontsize=10, y=1.01)
fig.tight_layout()
png_out = os.path.join(DATA, "C5_smooth_fdm_result.png")
fig.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_out}", flush=True)
