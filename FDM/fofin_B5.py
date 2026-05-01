"""
FDM form-finding for B5.obj target geometry.

Uses scipy.optimize.minimize (L-BFGS-B) with analytical gradient (thesis eq. 11-14)
instead of hand-coded gradient descent — adaptive step size, much faster convergence.

Pipeline:
  1. Load B5.obj as target surface S
  2. Flatten interior to z=0, fix boundary as anchors
  3. Optimise force densities q so that inflated FDM equilibrium ≈ S
  4. Save result to data/ and visualise
"""
import os, datetime
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

from compas.datastructures import Mesh
from compas.numerical.matrices import connectivity_matrix
from compas.numerical import fd_numpy

HERE    = os.path.dirname(os.path.abspath(__file__))
INPUT   = os.path.join(HERE, "input", "B5.obj")
DATA    = os.path.join(HERE, "data")

# ── Parameters ────────────────────────────────────────────────────────────────
PRESSURE   = 1.0     # inflation pressure (dimensionless FDM units)
Q_INIT     = 1.0     # uniform starting force density
Q_MIN      = 0.01    # lower bound (tension-only network)
INFLATE_IT = 5       # inner iterations to update pressure loads
MAXITER    = 500     # L-BFGS-B iterations

# ── Load mesh ─────────────────────────────────────────────────────────────────
mesh_target = Mesh.from_obj(INPUT)

mesh_target.update_default_vertex_attributes(is_anchor=False, px=0.0, py=0.0, pz=0.0, residual=None)
mesh_target.update_default_edge_attributes(qpre=Q_INIT)

# Boundary = fixed (anchors)
for vkey in mesh_target.vertices_on_boundary():
    mesh_target.vertex_attribute(vkey, "is_anchor", True)

# Store target vertex positions before flattening
target_xyz = {vkey: mesh_target.vertex_coordinates(vkey)
              for vkey in mesh_target.vertices()}

# Working mesh: flatten interior to z=0 as initial guess
mesh = mesh_target.copy()
fixed  = list(mesh.vertices_where({"is_anchor": True}))
free   = [v for v in mesh.vertices() if v not in fixed]
for vkey in free:
    mesh.vertex_attribute(vkey, "z", 0.0)

edges  = list(mesh.edges())
n_e    = len(edges)
n_v    = mesh.number_of_vertices()

C  = connectivity_matrix(edges, "csr")
Ci = C[:, free]
Cf = C[:, fixed]
Cit = Ci.T

# Fixed positions (boundary stays constant)
xyz_fixed = np.array([mesh.vertex_coordinates(v) for v in fixed], dtype=float)

# Target positions for free nodes only
S_free = np.array([target_xyz[v] for v in free], dtype=float)  # (n_free, 3)


# ── Inflation helper ──────────────────────────────────────────────────────────
def inflate(xyz_full, q_vec, pressure):
    """Solve FDM equilibrium iteratively (updates pressure loads each step)."""
    xyz = xyz_full.copy()
    q   = q_vec.copy()
    loads = np.zeros_like(xyz)

    for _ in range(INFLATE_IT):
        # Update vertex normals & areas from current geometry
        for i, vkey in enumerate(free):
            mesh.vertex_attributes(vkey, ["x", "y", "z"], xyz[vkey].tolist())
        for i, vkey in enumerate(fixed):
            mesh.vertex_attributes(vkey, ["x", "y", "z"], xyz[vkey].tolist())

        for vkey in free:
            n   = np.array(mesh.vertex_normal(vkey))
            a   = mesh.vertex_area(vkey)
            loads[vkey] = n * a * pressure

        # Solve equilibrium
        Q   = scipy.sparse.diags(q)
        Dn  = Cit.dot(Q).dot(Ci)
        p_f = loads[free] - Cit.dot(Q).dot(Cf).dot(xyz_fixed)

        xyz_free_new = scipy.sparse.linalg.spsolve(Dn, p_f)  # (n_free, 3)
        for i, vkey in enumerate(free):
            xyz[vkey] = xyz_free_new[i]

    return xyz


# ── Objective + gradient (thesis eq. 11–14) ───────────────────────────────────
_call_count = [0]

def objective_and_gradient(q_vec):
    # Build full xyz (fixed nodes stay put)
    xyz_full = np.zeros((n_v, 3), dtype=float)
    for i, vkey in enumerate(fixed):
        xyz_full[vkey] = xyz_fixed[i]

    xyz_eq = inflate(xyz_full, q_vec, PRESSURE)

    X_free = np.array([xyz_eq[v] for v in free])  # (n_free, 3)
    diff   = X_free - S_free                        # (n_free, 3)
    obj    = float(np.sum(diff**2))

    # Gradient of equilibrium w.r.t. q  (eq. 12-14)
    Q  = scipy.sparse.diags(q_vec)
    Dn = Cit.dot(Q).dot(Ci)

    xyz_arr = np.array([xyz_eq[v] for v in range(n_v)])  # (n_v, 3)

    grad_q = np.zeros(n_e)
    for axis in range(3):
        b     = Cit.dot(scipy.sparse.diags(C.dot(xyz_arr[:, axis])))
        dX_dq = -scipy.sparse.linalg.spsolve(Dn, b)  # (n_free, n_e)
        grad_q += 2.0 * (diff[:, axis] @ dX_dq)

    _call_count[0] += 1
    if _call_count[0] % 20 == 0:
        rmse = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        print(f"  iter {_call_count[0]:4d}  obj={obj:.4f}  RMSE={rmse:.4f} m")

    return obj, grad_q


# ── Optimise ──────────────────────────────────────────────────────────────────
print(f"B5 FDM optimisation: {len(free)} free nodes, {n_e} edges")
print(f"Target span: 20 m, height: 3.86 m  |  pressure={PRESSURE}, q_init={Q_INIT}")
print(f"Solver: L-BFGS-B (max {MAXITER} iters)\n")

q0     = np.full(n_e, Q_INIT)
bounds = [(Q_MIN, None)] * n_e

result = minimize(
    objective_and_gradient,
    q0,
    jac=True,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-8, "disp": True},
)

print(f"\nConverged: {result.success}  |  {result.message}")
print(f"Final obj={result.fun:.6f}  calls={_call_count[0]}")

q_opt = result.x

# ── Build final equilibrium mesh ──────────────────────────────────────────────
xyz_full = np.zeros((n_v, 3), dtype=float)
for i, vkey in enumerate(fixed):
    xyz_full[vkey] = xyz_fixed[i]
xyz_final = inflate(xyz_full, q_opt, PRESSURE)

mesh_out = mesh_target.copy()
for vkey in mesh_out.vertices():
    mesh_out.vertex_attributes(vkey, ["x", "y", "z"], xyz_final[vkey].tolist())
for i, edge in enumerate(mesh_out.edges()):
    mesh_out.edge_attribute(edge, "qpre", float(q_opt[i]))

# RMSE report
X_free_final = np.array([xyz_final[v] for v in free])
rmse = float(np.sqrt(np.mean(np.sum((X_free_final - S_free)**2, axis=1))))
print(f"Final RMSE to target: {rmse:.4f} m  (span 20 m → {100*rmse/20:.2f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
ts   = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out  = os.path.join(DATA, f"mesh_out_B5_{ts}.json")
mesh_out.to_json(out)
print(f"Saved: {out}")

# ── Visualise ─────────────────────────────────────────────────────────────────
try:
    from compas_view2.app import App
    from compas.geometry import Line

    viewer = App(width=900, height=700)
    viewer.view.camera.rz = -300
    viewer.view.camera.rx = -600
    viewer.view.camera.distance = 60

    viewer.add(mesh_out,    color=(0.9, 0.3, 0.3))   # red  = FDM result
    viewer.add(mesh_target, color=(0.3, 0.8, 0.3))   # green = target (B5)

    q_max = q_opt.max()
    for i, edge in enumerate(mesh_out.edges()):
        line = Line(*mesh_out.edge_coordinates(*edge))
        t = float(q_opt[i]) / q_max
        viewer.add(line, linecolor=(t, 0, 1 - t), linewidth=1 + 3 * t)

    viewer.show()
except ImportError:
    print("compas_view2 not available — skipping viewer (result saved to data/)")
