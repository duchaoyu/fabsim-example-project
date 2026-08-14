"""Time the crease (2part) form finding of FDM/fofin_butt_steps.py.

The original cannot run here: it imports compas.numerical (gone in compas 2) and
compas_view2 (a viewer, not installed, and it draws inside the step loop). This
harness reproduces its computation exactly - same 10 gradient-descent steps, same
time step, same pressure, same 5-step fd_inflate, same direct-sensitivity
gradient - with fd_numpy taken from compas_fd and every viewer call dropped. No
arithmetic is changed.
"""
import json
import os
import time

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import diags

from compas.datastructures import Mesh
from compas.matrices import connectivity_matrix
from compas_fd.solvers import fd_numpy

HERE = os.path.dirname(os.path.abspath(__file__))
NETWORK = os.path.join(HERE, "data", "2part.json")

steps = 10          # as in fofin_butt_steps.py
pressure = 1.2
time_step = 1.0


def load_compas1_mesh(path):
    """2part.json is compas 1.x serialisation; compas 2 will not read it."""
    d = json.load(open(path))
    keys = sorted(d["vertex"], key=int)
    xyz = [[d["vertex"][k]["x"], d["vertex"][k]["y"], d["vertex"][k]["z"]]
           for k in keys]
    remap = {int(k): i for i, k in enumerate(keys)}
    faces = [[remap[v] for v in d["face"][k]] for k in sorted(d["face"], key=int)]
    return Mesh.from_vertices_and_faces(xyz, faces)


def fd_inflate(mesh, pressure):
    """Identical to the original, with compas_fd's fd_numpy."""
    xyz = mesh.vertices_attributes(["x", "y", "z"])
    edges = list(mesh.edges())
    fixed = list(mesh.vertices_where({"is_anchor": True}))
    free = list(mesh.vertices_where({"is_anchor": False}))

    for _ in range(5):
        for vkey in free:
            v_n = mesh.vertex_normal(vkey)
            v_area = mesh.vertex_area(vkey)
            mesh.vertex_attributes(vkey, ["px", "py", "pz"],
                                   [v_n[i] * v_area * pressure for i in range(3)])
        load = -240 * 0.005 * 0.1
        for vkey in free:
            pz = mesh.vertex_attribute(vkey, "pz")
            mesh.vertex_attribute(vkey, "pz", pz + mesh.vertex_area(vkey) * load)

        loads = mesh.vertices_attributes(["px", "py", "pz"])
        qpre = mesh.edges_attribute("qpre")

        res = fd_numpy(vertices=xyz, fixed=fixed, edges=edges,
                       forcedensities=qpre, loads=loads)
        xyz = res.vertices
        for key in mesh.vertices():
            mesh.vertex_attributes(key, ["x", "y", "z"], xyz[key])
            mesh.vertex_attribute(key, "residual", res.residuals[key])
        for i, (u, v) in enumerate(mesh.edges()):
            mesh.edge_attribute((u, v), "fpre", res.forces[i])
            mesh.edge_attribute((u, v), "qpre", qpre[i])
            mesh.edge_attribute((u, v), "lpre", res.lengths[i])


mesh_0 = load_compas1_mesh(NETWORK)
mesh_0.update_default_vertex_attributes(is_anchor=False, residual=None,
                                        px=0, py=0, pz=0)
mesh_0.update_default_edge_attributes(qpre=2.0)
for key in mesh_0.vertices_on_boundary():
    mesh_0.vertex_attribute(key, "is_anchor", True)

n_v, n_f, n_e = (mesh_0.number_of_vertices(), mesh_0.number_of_faces(),
                 mesh_0.number_of_edges())
n_fixed = len(list(mesh_0.vertices_where({"is_anchor": True})))
print(f"crease FDM: {n_v - n_fixed} free of {n_v} v, {n_f} f, {n_e} edges")
print(f"Solver: hand-coded gradient descent, {steps} steps, "
      f"time_step={time_step}, pressure={pressure}")

t_all0 = time.perf_counter()
fd_inflate(mesh_0, pressure=pressure)
t_init = time.perf_counter() - t_all0

target_xyzs = {v: mesh_0.vertex_coordinates(v) for v in mesh_0.vertices()}

t_grad = t_inf = 0.0
t_opt0 = time.perf_counter()
for it in range(steps):
    mesh_0_pts = mesh_0.vertices_attributes("xyz")
    qpre = mesh_0.edges_attribute("qpre")
    fixed = list(mesh_0.vertices_where({"is_anchor": True}))
    edges = list(mesh_0.edges())
    free = list(set(range(len(mesh_0_pts))) - set(fixed))
    xyz = np.asarray(mesh_0_pts, dtype=np.float64).reshape((-1, 3))
    q = np.asarray(qpre, dtype=float).reshape((-1, 1))

    t = time.perf_counter()
    C = connectivity_matrix(edges, "csr")
    Ci, Cf, Cit = C[:, free], C[:, fixed], C[:, free].transpose()
    Dn = Cit.dot(diags([q.flatten()], [0])).dot(Ci)

    dx_q = scipy.sparse.linalg.spsolve(Dn, Cit.dot(diags(C.dot(xyz[:, 0]))))
    dy_q = scipy.sparse.linalg.spsolve(Dn, Cit.dot(diags(C.dot(xyz[:, 1]))))
    dz_q = scipy.sparse.linalg.spsolve(Dn, Cit.dot(diags(C.dot(xyz[:, 2]))))

    sum_gradient = np.zeros(q.shape)
    for i, key in enumerate(free):
        x0, y0, z0 = target_xyzs[key]
        x, y, z = xyz[key]
        sum_gradient += (2 * (x - x0) * dx_q[i] + 2 * (y - y0) * dy_q[i]
                         + 2 * (z - z0) * dz_q[i]).T
    # Original: qpre = qpre + time_step * sum_gradient, with qpre a list of
    # length n_e and sum_gradient (n_e, 1) -> broadcasts to (n_e, n_e) and
    # assigns a whole row as each edge's force density. Flattened here to the
    # elementwise update the surrounding code implies. Cost per step unchanged.
    g = np.asarray(sum_gradient).reshape(-1)      # dense+sparse -> np.matrix
    qpre = (q.flatten() + time_step * g).tolist()
    t_grad += time.perf_counter() - t

    t = time.perf_counter()
    mesh_1 = mesh_0.copy()
    for i, edge in enumerate(mesh_1.edges()):
        mesh_1.edge_attribute(edge, "qpre", qpre[i])
    fd_inflate(mesh_1, pressure=pressure)
    t_inf += time.perf_counter() - t

    dev = np.array([np.linalg.norm(np.array(mesh_1.vertex_coordinates(v))
                                   - np.array(target_xyzs[v]))
                    for v in mesh_1.vertices()])
    print(f"  step {it + 1:2d}  RMS deviation from target {dev.mean():.5f}")

    if it != steps - 1:
        mesh_0 = mesh_1.copy()

t_opt = time.perf_counter() - t_opt0
print(f"\nElapsed:   {t_opt:.2f} s for {steps} steps "
      f"({1e3 * t_opt / steps:.1f} ms/step, {n_e} design variables)")
print(f"  initial fd_inflate      {t_init:7.2f} s")
print(f"  gradient (direct)       {t_grad:7.2f} s  ({100 * t_grad / t_opt:.0f}%)")
print(f"  fd_inflate per step     {t_inf:7.2f} s  ({100 * t_inf / t_opt:.0f}%)")
