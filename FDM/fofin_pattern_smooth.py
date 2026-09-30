"""
FDM best-fit form-finding for data/pattern_smooth.obj.

Same pipeline as fofin_C5_smooth.py: the target surface is used both as the
network topology and as the fitting target; vertices at z = 0 are anchors
(the remaining boundary is a free edge); the free vertices are flattened,
inflated under a uniform pressure, and the edge
force densities q are optimised with L-BFGS-B so the equilibrium net
best-fits the target vertices.

Input is triangulated and
scaled so its footprint diameter is TARGET_DIAMETER = 1.2 m, matching
FDM/scale_geometry.py.

Saves into FDM/data/pattern/:
  mesh_out_pattern_smooth[_TAG]_<timestamp>.json   full result (coords + qpre)
  mesh_out_pattern_smooth[_TAG]_latest.json        fixed name for downstream scripts
  pattern_smooth_tri_m.off                        scaled trimesh target (FEM input)
  pattern_smooth_fdm[_TAG].off                    FDM equilibrium surface
  pattern_smooth_fdm[_TAG]_result.png             visualisation

Usage:
  [LAMBDA_L=1.0] [R_MIN=0.9] [R_MAX=1.1] [MU_CAP=1e3] [TAG=...] .venv/bin/python FDM/fofin_pattern_smooth.py [input.obj]
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm

from compas.datastructures import Mesh
from compas.matrices import connectivity_matrix

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.abspath(os.path.join(HERE, ".."))
DATA  = os.environ.get("FDM_OUT_DIR", os.path.join(HERE, "data", "pattern"))
INPUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "pattern_smooth.obj")
REF_TRI = ""   # no reference trimesh for this geometry

TARGET_DIAMETER = 1.2      # m, max(x-span, y-span); same convention as scale_geometry.py
PRESSURE        = 1.0
Q_INIT          = 1.0
Q_MIN           = 0.01
Q_MAX           = 100.0
INFLATE_IT      = 5
INFLATE_DAMP    = 1.0
MAXITER         = int(os.environ.get("MAXITER", 2000))
# Weight of the edge-length term, mean((L/L0 - 1)^2), against the position
# term, mean(|x - x_target|^2) / l0^2 (l0 = mean target edge length). It keeps
# the net's layout close to the target mesh; 0 is a plain vertex best fit.
LAMBDA_L        = float(os.environ.get("LAMBDA_L", 1.0))
# Band on edge length: edges outside R_MIN * L0 .. R_MAX * L0 pay a stiff
# quadratic penalty MU_CAP * mean(dist(L/L0, [R_MIN, R_MAX])^2).
R_MIN           = float(os.environ.get("R_MIN", 0.0))
R_MAX           = float(os.environ.get("R_MAX", "inf"))
MU_CAP          = float(os.environ.get("MU_CAP", 1e3))
TAG             = os.environ.get("TAG", "")          # suffix for output files

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

# The OBJ is exported upright with its supports on the ground plane: anchor
# only the vertices at z = 0 (a line support plus corner points); the rest of
# the boundary is a free edge.
zmin = min(mesh_target.vertex_attribute(v, "z") for v in mesh_target.vertices())
for v in mesh_target.vertices():
    mesh_target.vertex_attribute(v, "z", mesh_target.vertex_attribute(v, "z") - zmin)
z_tol = 1e-6 * TARGET_DIAMETER

mesh_target.update_default_vertex_attributes(is_anchor=False, px=0.0, py=0.0, pz=0.0, residual=None)
mesh_target.update_default_edge_attributes(qpre=Q_INIT)
for vkey in mesh_target.vertices():
    if mesh_target.vertex_attribute(vkey, "z") < z_tol:
        mesh_target.vertex_attribute(vkey, "is_anchor", True)
n_anchor = len(list(mesh_target.vertices_where({"is_anchor": True})))
print(f"  anchors: {n_anchor} vertices at z = 0", flush=True)

target_xyz = {v: mesh_target.vertex_coordinates(v) for v in mesh_target.vertices()}

if os.path.exists(REF_TRI):
    ref = Mesh.from_off(REF_TRI)
    Vr = np.array([ref.vertex_coordinates(v) for v in ref.vertices()])
    print(f"  reference trimesh: {ref.number_of_vertices()}v / {ref.number_of_faces()}f, "
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


T_full = np.array([target_xyz[v] for v in range(n_v)], dtype=float)
L0     = np.linalg.norm(C.dot(T_full), axis=1)       # target edge lengths
l0     = float(L0.mean())
free_idx = np.array(free)

_call = [0]
_hist = []

def equilibrium(q_vec):
    xyz_full = np.zeros((n_v, 3), dtype=float)
    for i, v in enumerate(fixed):
        xyz_full[v] = xyz_fixed[i]
    return inflate(xyz_full, q_vec, PRESSURE)


def obj_grad(s_vec):
    """Objective in s = log q. The gradient uses the adjoint of Dn x = p with
    the pressure loads held fixed at the last inflation step."""
    q_vec  = np.exp(s_vec)
    xyz_eq = equilibrium(q_vec)
    diff   = xyz_eq[free_idx] - S_free

    # position term, normalised by the mean target edge length
    J_x   = float(np.sum(diff ** 2)) / (len(free) * l0 ** 2)
    g_X   = 2.0 * diff / (len(free) * l0 ** 2)                 # dJ/dX_free

    # edge-length term: keeps the layout of the target mesh
    U     = C.dot(xyz_eq)
    L     = np.linalg.norm(U, axis=1)
    r     = L / L0
    J_L   = float(np.mean((r - 1.0) ** 2))
    coef  = 2.0 * LAMBDA_L * (r - 1.0) / (L0 * np.maximum(L, 1e-12) * n_e)
    # band on edge length: signed distance outside [R_MIN, R_MAX]
    over  = np.maximum(r - R_MAX, 0.0) - np.maximum(R_MIN - r, 0.0)
    J_cap = float(np.mean(over ** 2))
    coef += 2.0 * MU_CAP * over / (L0 * np.maximum(L, 1e-12) * n_e)
    g_X  += C.T.dot(coef[:, None] * U)[free_idx]

    Dn    = Cit.dot(scipy.sparse.diags(q_vec)).dot(Ci).tocsc()
    solve = scipy.sparse.linalg.factorized(Dn)
    grad_q = np.zeros(n_e)
    for ax in range(3):
        adj = solve(g_X[:, ax])
        grad_q -= Ci.dot(adj) * U[:, ax]

    _call[0] += 1
    rmse = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    _hist.append(rmse)
    if _call[0] % 50 == 0:
        print(f"  iter {_call[0]:4d}  J_x={J_x:.5f}  J_L={J_L:.5f}  L/L0 {r.min():.3f}..{r.max():.3f}  "
              f"RMSE={rmse*1e3:.2f} mm", flush=True)
    return J_x + LAMBDA_L * J_L + MU_CAP * J_cap, grad_q * q_vec


span   = max(p[0] for p in target_xyz.values()) - min(p[0] for p in target_xyz.values())
height = max(p[2] for p in target_xyz.values())
print(f"\npattern smooth FDM: {len(free)} free, {len(fixed)} anchors, {n_e} edges", flush=True)
print(f"Target span={span:.3f} m  height={height:.3f} m  pressure={PRESSURE}  "
      f"mean edge {l0*1e3:.1f} mm  LAMBDA_L={LAMBDA_L}  L/L0 band [{R_MIN}, {R_MAX}]", flush=True)
print(f"Solver: L-BFGS-B on log q (max {MAXITER} iters)\n", flush=True)

s0     = np.full(n_e, np.log(Q_INIT))
t_opt0 = time.perf_counter()
result = minimize(obj_grad, s0, jac=True, method="L-BFGS-B",
                  bounds=[(np.log(Q_MIN), np.log(Q_MAX))] * n_e,
                  options={"maxiter": MAXITER, "ftol": 1e-10, "gtol": 1e-8})
t_opt = time.perf_counter() - t_opt0

print(f"\nConverged: {result.success}  |  {result.message}", flush=True)
print(f"obj={result.fun:.6f}  calls={_call[0]}", flush=True)
print(f"Elapsed:   {t_opt:.2f} s for {_call[0]} iters "
      f"({1e3*t_opt/max(_call[0],1):.1f} ms/iter, {n_e} design variables)", flush=True)

q_opt = np.exp(result.x)
xyz_final = equilibrium(q_opt)

mesh_out = mesh_target.copy()
for v in mesh_out.vertices():
    mesh_out.vertex_attributes(v, ["x", "y", "z"], xyz_final[v].tolist())
for i, e in enumerate(mesh_out.edges()):
    mesh_out.edge_attribute(e, "qpre", float(q_opt[i]))

X_free_final = np.array([xyz_final[v] for v in free])
dev  = np.linalg.norm(X_free_final - S_free, axis=1)
rmse = float(np.sqrt(np.mean(dev ** 2)))
print(f"Final RMSE: {rmse:.5f} m  ({100*rmse/span:.2f}% of span), max dev {dev.max()*1000:.1f} mm", flush=True)
dxy = np.linalg.norm((X_free_final - S_free)[:, :2], axis=1)
dz  = np.abs((X_free_final - S_free)[:, 2])
r_final = np.linalg.norm(C.dot(xyz_final), axis=1) / L0
print(f"  in-plane dev mean {dxy.mean()*1e3:.1f} / max {dxy.max()*1e3:.1f} mm, "
      f"vertical dev mean {dz.mean()*1e3:.1f} / max {dz.max()*1e3:.1f} mm", flush=True)
print(f"  edge length L/L0: p1 {np.percentile(r_final, 1):.3f}  p50 {np.median(r_final):.3f}  "
      f"p99 {np.percentile(r_final, 99):.3f}  min {r_final.min():.3f}  max {r_final.max():.3f}  "
      f"({int(np.sum((r_final > R_MAX + 0.01) | (r_final < R_MIN - 0.01)))} edges outside band by > 0.01)", flush=True)
print(f"q range: {q_opt.min():.4f} .. {q_opt.max():.4f}  (median {np.median(q_opt):.3f}, "
      f"{int(np.sum(q_opt <= Q_MIN * 1.01))} edges at Q_MIN)", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
sfx = f"_{TAG}" if TAG else ""
ts  = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
out = os.path.join(DATA, f"mesh_out_pattern_smooth{sfx}_{ts}.json")
mesh_out.to_json(out)
mesh_out.to_json(os.path.join(DATA, f"mesh_out_pattern_smooth{sfx}_latest.json"))
write_off(mesh_target, os.path.join(DATA, "pattern_smooth_tri_m.off"))
write_off(mesh_out, os.path.join(DATA, f"pattern_smooth_fdm{sfx}.off"))
print(f"Saved: {out}", flush=True)
print(f"Saved: {os.path.join(DATA, f'mesh_out_pattern_smooth{sfx}_latest.json')}", flush=True)
print(f"Saved: {os.path.join(DATA, 'pattern_smooth_tri_m.off')}  (scaled trimesh target)", flush=True)
print(f"Saved: {os.path.join(DATA, f'pattern_smooth_fdm{sfx}.off')}    (FDM surface)", flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
vkeys = list(mesh_out.vertices())
v_idx = {v: i for i, v in enumerate(vkeys)}
V_out = np.array([mesh_out.vertex_coordinates(v) for v in vkeys])
V_tgt = np.array([target_xyz[v] for v in vkeys])
faces = [[v_idx[v] for v in mesh_out.face_vertices(f)] for f in mesh_out.faces()]

fig = plt.figure(figsize=(24, 5))

ax1 = fig.add_subplot(151, projection="3d")
ax1.add_collection3d(Poly3DCollection([V_out[f] for f in faces], alpha=0.25,
                                      facecolor="tomato", edgecolor="k", linewidths=0.1))
ax1.set_box_aspect([1, 1, 0.55]); ax1.view_init(elev=30, azim=-60)
ax1.set_xlim(V_tgt[:,0].min(), V_tgt[:,0].max()); ax1.set_ylim(V_tgt[:,1].min(), V_tgt[:,1].max())
ax1.set_zlim(0, V_tgt[:,2].max())
ax1.set_title("FDM result", fontsize=9)

ax2 = fig.add_subplot(152, projection="3d")
ax2.add_collection3d(Poly3DCollection([V_tgt[f] for f in faces], alpha=0.25,
                                      facecolor="steelblue", edgecolor="k", linewidths=0.1))
ax2.set_box_aspect([1, 1, 0.55]); ax2.view_init(elev=30, azim=-60)
ax2.set_xlim(V_tgt[:,0].min(), V_tgt[:,0].max()); ax2.set_ylim(V_tgt[:,1].min(), V_tgt[:,1].max())
ax2.set_zlim(0, V_tgt[:,2].max())
ax2.set_title("Target (pattern smooth, Ø1.2 m)", fontsize=9)

# q spans orders of magnitude, so colour it on a log scale
ax3 = fig.add_subplot(153)
seg = [[V_out[e[0], :2], V_out[e[1], :2]] for e in mesh_out.edges()]
q_norm = LogNorm(vmin=max(q_opt.min(), Q_MIN), vmax=q_opt.max())
lc = LineCollection(seg, cmap="plasma", norm=q_norm, linewidths=0.7)
lc.set_array(q_opt)
ax3.add_collection(lc); ax3.autoscale()
ax3.set_aspect("equal"); ax3.set_title("Force densities q (top view, log scale)", fontsize=9)
ax3.set_xlabel("x (m)"); ax3.set_ylabel("y (m)")
fig.colorbar(lc, ax=ax3, label="q", shrink=0.8)

ax4 = fig.add_subplot(154)
r_lo = min(0.8, R_MIN) if R_MIN > 0 else 0.8
r_hi = max(1.2, R_MAX) if np.isfinite(R_MAX) else 1.2
lr = LineCollection(seg, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=1.0, vmin=r_lo, vmax=r_hi),
                    linewidths=0.7)
lr.set_array(np.clip(r_final, r_lo, r_hi))
ax4.add_collection(lr); ax4.autoscale()
ax4.set_aspect("equal"); ax4.set_title("Edge length L / L0 (top view)", fontsize=9)
ax4.set_xlabel("x (m)"); ax4.set_ylabel("y (m)")
fig.colorbar(lr, ax=ax4, label="L / L0 (clipped)", shrink=0.8)

ax5 = fig.add_subplot(155)
dev_all = np.linalg.norm(V_out - V_tgt, axis=1) * 1000.0
sc = ax5.tripcolor(V_tgt[:, 0], V_tgt[:, 1],
                   [f for f in faces if len(f) == 3], dev_all,
                   shading="gouraud", cmap="viridis")
ax5.set_aspect("equal"); ax5.set_title("Deviation from target (mm)", fontsize=9)
ax5.set_xlabel("x (m)"); ax5.set_ylabel("y (m)")
fig.colorbar(sc, ax=ax5, label="mm", shrink=0.8)

fig.suptitle(f"pattern smooth FDM best-fit (λ_L={LAMBDA_L:g}, L/L0 in [{R_MIN:g}, {R_MAX:g}]) — RMSE={rmse*1000:.1f} mm, max={dev.max()*1000:.1f} mm, "
             f"span={span:.2f} m, {n_v}v / {mesh_out.number_of_faces()}f",
             fontsize=10, y=1.02)
fig.tight_layout()
png_out = os.path.join(DATA, f"pattern_smooth_fdm{sfx}_result.png")
fig.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_out}", flush=True)
