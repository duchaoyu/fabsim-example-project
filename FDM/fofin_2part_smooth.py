"""
FDM best-fit form-finding for the new 2-part (middle-crease) model.

Same pipeline as fofin_C5_smooth.py: the target surface is used both as the
network topology and as the fitting target; boundary vertices are anchors;
the interior is flattened, inflated under a uniform pressure, and the edge
force densities q are optimised with L-BFGS-B so the equilibrium net
best-fits the target vertices.

Input is triangulated (the reference 2part trimesh used for the FEM
optimisation, data/2part/2part_opt_simu_m.off, is 341v / 634 triangles) and
scaled so its footprint diameter is TARGET_DIAMETER = 1.2 m, matching
FDM/scale_geometry.py.

Saves into FDM/data/2part/:
  mesh_out_2part_smooth_<timestamp>.json   full result (coords + qpre)
  mesh_out_2part_smooth_latest.json        fixed name for downstream scripts
  2parts_smooth_tri_m.off                  scaled trimesh target (FEM input)
  2parts_smooth_fdm.off                    FDM equilibrium surface
  2part_smooth_fdm_result.png              visualisation

Usage:
  .venv/bin/python FDM/fofin_2part_smooth.py [input.obj]
"""
import os, sys, datetime, time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from compas.datastructures import Mesh
from compas.matrices import connectivity_matrix

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(HERE, ".."))
DATA  = os.path.join(HERE, "data", "2part")
INPUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "2part", "2parts_smooth.obj")
REF_TRI = os.path.join(ROOT, "data", "2part", "2part_opt_simu_m.off")

TARGET_DIAMETER = 1.2      # m, max(x-span, y-span); same convention as scale_geometry.py
PRESSURE        = 1.0
Q_INIT          = 1.0
Q_MIN           = 0.01
INFLATE_IT      = 5
INFLATE_DAMP    = 1.0
MAXITER         = int(os.environ.get("MAXITER", 2000))

os.makedirs(DATA, exist_ok=True)

def write_off(mesh, path):
    """compas 2.15 to_off() is broken (compas.PRECISION removed) - write it directly."""
    vkeys = list(mesh.vertices())
    idx   = {v: i for i, v in enumerate(vkeys)}
    faces = [[idx[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]
    with open(path, "w") as fh:
        fh.write("OFF\n")
        fh.write(f"{len(vkeys)} {len(faces)} 0\n")
        for v in vkeys:
            x, y, z = mesh.vertex_coordinates(v)
            fh.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
        for f in faces:
            fh.write(str(len(f)) + " " + " ".join(str(i) for i in f) + "\n")



# ── Load, triangulate, scale ──────────────────────────────────────────────────
mesh_target = Mesh.from_obj(INPUT)
n_f_raw = mesh_target.number_of_faces()
if not mesh_target.is_trimesh():
    mesh_target.quads_to_triangles()
    print(f"Triangulated: {n_f_raw} faces -> {mesh_target.number_of_faces()} triangles", flush=True)

V = np.array([mesh_target.vertex_coordinates(v) for v in mesh_target.vertices()])
diam = max(np.ptp(V[:, 0]), np.ptp(V[:, 1]))
scale = TARGET_DIAMETER / diam
if abs(scale - 1.0) > 1e-9:
    for v in mesh_target.vertices():
        x, y, z = mesh_target.vertex_coordinates(v)
        mesh_target.vertex_attributes(v, ["x", "y", "z"], [x * scale, y * scale, z * scale])
print(f"Input: {INPUT}", flush=True)
print(f"  {mesh_target.number_of_vertices()} vertices, {mesh_target.number_of_faces()} triangles", flush=True)
print(f"  native diameter {diam:.4f} -> scaled by {scale:.6f} to {TARGET_DIAMETER} m", flush=True)

# Orient upright: a hanging/funicular export has its interior below the
# supports, so flip z to get the compression dome (base z = 0, crown up),
# matching data/2part/2part_opt_simu_m.off.
_bdr = set(mesh_target.vertices_on_boundary())
z_bdr = np.mean([mesh_target.vertex_attribute(v, "z") for v in _bdr])
z_int = np.mean([mesh_target.vertex_attribute(v, "z") for v in mesh_target.vertices() if v not in _bdr])
if z_int < z_bdr:
    print("  hanging form detected -> flipping z to an upright dome", flush=True)
    for v in mesh_target.vertices():
        mesh_target.vertex_attribute(v, "z", -mesh_target.vertex_attribute(v, "z"))

# Put the base at z = 0 so the anchors sit on the springing plane.
zmin = min(mesh_target.vertex_attribute(v, "z") for v in mesh_target.vertices())
for v in mesh_target.vertices():
    mesh_target.vertex_attribute(v, "z", mesh_target.vertex_attribute(v, "z") - zmin)

mesh_target.update_default_vertex_attributes(is_anchor=False, px=0.0, py=0.0, pz=0.0, residual=None)
mesh_target.update_default_edge_attributes(qpre=Q_INIT)
for vkey in mesh_target.vertices_on_boundary():
    mesh_target.vertex_attribute(vkey, "is_anchor", True)

target_xyz = {v: mesh_target.vertex_coordinates(v) for v in mesh_target.vertices()}

if os.path.exists(REF_TRI):
    ref = Mesh.from_off(REF_TRI)
    Vr = np.array([ref.vertex_coordinates(v) for v in ref.vertices()])
    print(f"  reference 2part trimesh: {ref.number_of_vertices()}v / {ref.number_of_faces()}f, "
          f"span {np.ptp(Vr[:,0]):.3f} x {np.ptp(Vr[:,1]):.3f} m, height {np.ptp(Vr[:,2]):.3f} m", flush=True)

# ── Flatten the interior as the FDM starting point ────────────────────────────
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
    """Fixed-point pneumatic form finding: pressure loads follow the surface."""
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
_hist = []

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
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    _hist.append(rmse)
    if _call[0] % 20 == 0:
        print(f"  iter {_call[0]:4d}  obj={obj:.6f}  RMSE={rmse:.5f} m", flush=True)
    return obj, grad


span   = max(p[0] for p in target_xyz.values()) - min(p[0] for p in target_xyz.values())
height = max(p[2] for p in target_xyz.values())
print(f"\n2part smooth FDM: {len(free)} free, {len(fixed)} anchors, {n_e} edges", flush=True)
print(f"Target span={span:.3f} m  height={height:.3f} m  pressure={PRESSURE}", flush=True)
print(f"Solver: L-BFGS-B (max {MAXITER} iters)\n", flush=True)

q0     = np.full(n_e, Q_INIT)
t_opt0 = time.perf_counter()
result = minimize(obj_grad, q0, jac=True, method="L-BFGS-B",
                  bounds=[(Q_MIN, None)] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-8, "gtol": 1e-8, "disp": True})
t_opt = time.perf_counter() - t_opt0

print(f"\nConverged: {result.success}  |  {result.message}", flush=True)
print(f"obj={result.fun:.6f}  calls={_call[0]}", flush=True)
print(f"Elapsed:   {t_opt:.2f} s for {_call[0]} iters "
      f"({1e3*t_opt/max(_call[0],1):.1f} ms/iter, {n_e} design variables)", flush=True)

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
dev  = np.linalg.norm(X_free_final - S_free, axis=1)
rmse = float(np.sqrt(np.mean(dev ** 2)))
print(f"Final RMSE: {rmse:.5f} m  ({100*rmse/span:.2f}% of span), max dev {dev.max()*1000:.1f} mm", flush=True)
print(f"q range: {q_opt.min():.4f} .. {q_opt.max():.4f}", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
ts  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out = os.path.join(DATA, f"mesh_out_2part_smooth_{ts}.json")
mesh_out.to_json(out)
mesh_out.to_json(os.path.join(DATA, "mesh_out_2part_smooth_latest.json"))
write_off(mesh_target, os.path.join(DATA, "2parts_smooth_tri_m.off"))
write_off(mesh_out, os.path.join(DATA, "2parts_smooth_fdm.off"))
print(f"Saved: {out}", flush=True)
print(f"Saved: {os.path.join(DATA, 'mesh_out_2part_smooth_latest.json')}", flush=True)
print(f"Saved: {os.path.join(DATA, '2parts_smooth_tri_m.off')}  (scaled trimesh target)", flush=True)
print(f"Saved: {os.path.join(DATA, '2parts_smooth_fdm.off')}    (FDM surface)", flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
vkeys = list(mesh_out.vertices())
v_idx = {v: i for i, v in enumerate(vkeys)}
V_out = np.array([mesh_out.vertex_coordinates(v) for v in vkeys])
V_tgt = np.array([target_xyz[v] for v in vkeys])
faces = [[v_idx[v] for v in mesh_out.face_vertices(f)] for f in mesh_out.faces()]

fig = plt.figure(figsize=(18, 5))

ax1 = fig.add_subplot(141, projection="3d")
ax1.add_collection3d(Poly3DCollection([V_out[f] for f in faces], alpha=0.25,
                                      facecolor="tomato", edgecolor="k", linewidths=0.1))
ax1.set_box_aspect([1, 1, 0.55]); ax1.view_init(elev=30, azim=-60)
ax1.set_xlim(V_tgt[:,0].min(), V_tgt[:,0].max()); ax1.set_ylim(V_tgt[:,1].min(), V_tgt[:,1].max())
ax1.set_zlim(0, V_tgt[:,2].max())
ax1.set_title("FDM result", fontsize=9)

ax2 = fig.add_subplot(142, projection="3d")
ax2.add_collection3d(Poly3DCollection([V_tgt[f] for f in faces], alpha=0.25,
                                      facecolor="steelblue", edgecolor="k", linewidths=0.1))
ax2.set_box_aspect([1, 1, 0.55]); ax2.view_init(elev=30, azim=-60)
ax2.set_xlim(V_tgt[:,0].min(), V_tgt[:,0].max()); ax2.set_ylim(V_tgt[:,1].min(), V_tgt[:,1].max())
ax2.set_zlim(0, V_tgt[:,2].max())
ax2.set_title("Target (2parts smooth, Ø1.2 m)", fontsize=9)

ax3 = fig.add_subplot(143)
q_max = q_opt.max()
for idx, e in enumerate(mesh_out.edges()):
    p1 = mesh_out.vertex_coordinates(e[0]); p2 = mesh_out.vertex_coordinates(e[1])
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]],
             color=plt.cm.plasma(float(q_opt[idx]) / q_max), linewidth=0.6, alpha=0.8)
ax3.set_aspect("equal"); ax3.set_title("Force densities q (top view)", fontsize=9)
ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=q_opt.min(), vmax=q_max))
sm.set_array([])
fig.colorbar(sm, ax=ax3, label="q", shrink=0.8)

ax4 = fig.add_subplot(144)
dev_all = np.linalg.norm(V_out - V_tgt, axis=1) * 1000.0
sc = ax4.tripcolor(V_tgt[:, 0], V_tgt[:, 1],
                   [f for f in faces if len(f) == 3], dev_all,
                   shading="gouraud", cmap="viridis")
ax4.set_aspect("equal"); ax4.set_title("Deviation from target (mm)", fontsize=9)
ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)")
fig.colorbar(sc, ax=ax4, label="mm", shrink=0.8)

fig.suptitle(f"2part smooth FDM best-fit — RMSE={rmse*1000:.1f} mm, max={dev.max()*1000:.1f} mm, "
             f"span={span:.2f} m, {n_v}v / {mesh_out.number_of_faces()}f",
             fontsize=10, y=1.02)
fig.tight_layout()
png_out = os.path.join(DATA, "2part_smooth_fdm_result.png")
fig.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_out}", flush=True)
