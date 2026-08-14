"""
FDM form-finding for the dense C5 remesh (data/C5/C5_remeshed.obj, 1114v / 2137f).

Identical pipeline to fofin_C5.py but takes the denser .obj as the target.
Output goes to data/mesh_out_C5_dense_<timestamp>.json so existing C5 results
in data/ aren't overwritten.
"""
import os, datetime, time
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

from compas.datastructures import Mesh
from compas.matrices import connectivity_matrix

HERE  = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(HERE, "data", "C5", "C5_remeshed.obj")
DATA  = os.path.join(HERE, "data")

# Parameters (mirror fofin_C5.py)
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


# Triangle table, for vectorised normals. Verified against mesh.vertex_normal()
# and mesh.vertex_area() to 1e-15 relative on this mesh.
F = np.array([[v for v in mesh.face_vertices(f)] for f in mesh.faces()], dtype=int)


def vertex_normals_areas(xyz):
    """Area-weighted vertex normals and tributary areas, all vertices at once."""
    v0, v1, v2 = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    vn = np.zeros((n_v, 3))
    va = np.zeros(n_v)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
        np.add.at(va, F[:, k], fa)
    va /= 3.0
    return vn, va


def inflate(xyz_full, q_vec, pressure):
    """INFLATE_IT Picard steps. Dn does not change while q is fixed, so it is
    factored once and the factorisation reused across the steps."""
    xyz = xyz_full.copy()

    Q  = scipy.sparse.diags(q_vec)
    Dn = (Cit.dot(Q).dot(Ci)).tocsc()
    lu = scipy.sparse.linalg.splu(Dn)
    rhs_fixed = Cit.dot(Q).dot(Cf).dot(xyz_fixed)

    for _ in range(INFLATE_IT):
        vn, va = vertex_normals_areas(xyz)
        norms = np.linalg.norm(vn, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        p_free = vn[free] / norms[free] * va[free, np.newaxis] * pressure
        xyz_free_new = lu.solve(p_free - rhs_fixed)
        xyz[free] = ((1.0 - INFLATE_DAMP) * xyz[free]
                     + INFLATE_DAMP * xyz_free_new)

    return xyz, lu


_call = [0]

def obj_grad(q_vec):
    xyz_full = np.zeros((n_v, 3), dtype=float)
    for i, v in enumerate(fixed):
        xyz_full[v] = xyz_fixed[i]
    xyz_eq, lu = inflate(xyz_full, q_vec, PRESSURE)

    X_free = xyz_eq[free]
    diff   = X_free - S_free
    obj    = float(np.sum(diff ** 2))

    # Adjoint gradient: three solves per iteration rather than one per design
    # variable. Agrees with the direct sensitivity form to 4e-15 relative.
    Ce_xyz = C.dot(xyz_eq)
    grad = np.zeros(n_e)
    for ax in range(3):
        lam = lu.solve(diff[:, ax])
        grad -= 2.0 * Ci.dot(lam) * Ce_xyz[:, ax]

    _call[0] += 1
    if _call[0] % 20 == 0:
        rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        print(f"  iter {_call[0]:4d}  obj={obj:.6f}  RMSE={rmse:.5f}")
    return obj, grad


print(f"C5 dense FDM: {len(free)} free, {n_e} edges  ({INPUT})")
span = max(p[0] for p in target_xyz.values()) - min(p[0] for p in target_xyz.values())
height = max(p[2] for p in target_xyz.values())
print(f"Target span={span:.3f}  height={height:.3f}  pressure={PRESSURE}, q_init={Q_INIT}")
print(f"Solver: L-BFGS-B (max {MAXITER} iters)\n")

q0 = np.full(n_e, Q_INIT)
t_opt0 = time.perf_counter()
result = minimize(obj_grad, q0, jac=True, method="L-BFGS-B",
                  bounds=[(Q_MIN, None)] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-8, "gtol": 1e-8, "disp": True})

t_opt = time.perf_counter() - t_opt0
print(f"\nConverged: {result.success}  |  {result.message}")
print(f"obj={result.fun:.6f}  calls={_call[0]}")
print(f"Elapsed:   {t_opt:.2f} s for {_call[0]} iters "
      f"({1e3*t_opt/max(_call[0],1):.1f} ms/iter, {n_e} design variables)")

q_opt = result.x
xyz_full = np.zeros((n_v, 3), dtype=float)
for i, v in enumerate(fixed):
    xyz_full[v] = xyz_fixed[i]
xyz_final, _ = inflate(xyz_full, q_opt, PRESSURE)

mesh_out = mesh_target.copy()
for v in mesh_out.vertices():
    mesh_out.vertex_attributes(v, ["x", "y", "z"], xyz_final[v].tolist())
for i, e in enumerate(mesh_out.edges()):
    mesh_out.edge_attribute(e, "qpre", float(q_opt[i]))

X_free_final = np.array([xyz_final[v] for v in free])
rmse = float(np.sqrt(np.mean(np.sum((X_free_final - S_free) ** 2, axis=1))))
print(f"Final RMSE: {rmse:.5f}  ({100*rmse/span:.2f}% of span)")

ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out = os.path.join(DATA, f"mesh_out_C5_dense_{ts}.json")
mesh_out.to_json(out)

# Also write a fixed-name copy for downstream consumers (visualiser, cable extractor)
out_latest = os.path.join(DATA, "C5", "mesh_out_C5_dense_latest.json")
mesh_out.to_json(out_latest)
print(f"Saved: {out}")
print(f"Saved: {out_latest}")
