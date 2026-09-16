"""
Turn the D5 form-finding output into a proper compas mesh.

fofin_D5.py writes a plain dict (verts + q + scalars), not a serialised mesh,
and names it D5_fdm_<ts>.json instead of mesh_out_*.  So Mesh.from_json cannot
open it and it does not sit with the other FDM results.

This borrows the face topology from D5_remeshed_fem.off (847v / 1563f, the same
mesh the form-finding ran on), attaches q as 'qpre' per edge, and writes:

  mesh_out_D5_<ts>.json           compas 2.x  (Mesh.from_json)
  mesh_out_D5_<ts>_compas1.json   compas 1.x  (Rhino / Grasshopper)
  D5_fdm_<ts>.off / .obj          geometry only

Usage:  python FDM/convert_D5_fdm_to_mesh.py [D5_fdm_*.json]
"""
import os, sys, json
import numpy as np
from compas.datastructures import Mesh

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "D5")
SRC  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "D5_fdm_20260507143057.json")
TOPO = os.path.join(DATA, "D5_remeshed_fem.off")


def load_off(path):
    L = open(path).readlines()
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in l.split()[:3]] for l in L[2:2 + nv]])
    F = [[int(x) for x in l.split()[1:]] for l in L[2 + nv:2 + nv + nf]]
    return V, F


d = json.load(open(SRC))
V = np.array(d["verts"], dtype=float)
q = np.array(d["q"], dtype=float)
V_topo, F = load_off(TOPO)

if len(V) != len(V_topo):
    sys.exit(f"vertex count mismatch: {SRC} has {len(V)}, {TOPO} has {len(V_topo)}")

mesh = Mesh()
mesh.update_default_vertex_attributes(x=0.0, y=0.0, z=0.0)
mesh.update_default_edge_attributes(qpre=1.0)
for i, (x, y, z) in enumerate(V):
    mesh.add_vertex(key=i, x=float(x), y=float(y), z=float(z))
for f in F:
    mesh.add_face(f)

edges = list(mesh.edges())
if len(edges) != len(q):
    print(f"WARNING: {len(edges)} edges but {len(q)} q values - qpre left at default")
else:
    for e, qv in zip(edges, q):
        mesh.edge_attribute(e, "qpre", float(qv))

# the form-finding pinned the boundary; record it like the other results do
mesh.update_default_vertex_attributes(is_anchor=False)
for v in mesh.vertices_on_boundary():
    mesh.vertex_attribute(v, "is_anchor", True)

ts   = os.path.basename(SRC).replace("D5_fdm_", "").replace(".json", "")
stem = os.path.join(DATA, f"mesh_out_D5_{ts}")
mesh.to_json(stem + ".json")

flat = {
    "attributes": dict(mesh.attributes),
    "dva": dict(mesh.default_vertex_attributes),
    "dea": dict(mesh.default_edge_attributes),
    "dfa": dict(mesh.default_face_attributes),
    "vertex":   {str(v): dict(mesh.vertex_attributes(v)) for v in mesh.vertices()},
    "face":     {str(f): [int(v) for v in mesh.face_vertices(f)] for f in mesh.faces()},
    "facedata": {str(f): dict(mesh.face_attributes(f)) for f in mesh.faces()},
    "edgedata": {str((int(u), int(w))): dict(mesh.edge_attributes((u, w)))
                 for u, w in mesh.edges()},
    "max_vertex": int(max(mesh.vertices(), default=-1)),
    "max_face":   int(max(mesh.faces(), default=-1)),
}
with open(stem + "_compas1.json", "w") as f:
    json.dump(flat, f)

geo = os.path.join(DATA, f"D5_fdm_{ts}")
with open(geo + ".off", "w") as f:
    f.write("OFF\n%d %d 0\n" % (len(V), len(F)))
    for v in V:
        f.write("%.8f %.8f %.8f\n" % tuple(v))
    for fc in F:
        f.write(str(len(fc)) + " " + " ".join(str(i) for i in fc) + "\n")
with open(geo + ".obj", "w") as f:
    f.write(f"# D5 FDM result, topology from {os.path.basename(TOPO)}\n")
    for v in V:
        f.write("v %.8f %.8f %.8f\n" % tuple(v))
    for fc in F:
        f.write("f " + " ".join(str(i + 1) for i in fc) + "\n")

print(f"source : {SRC}")
print(f"topo   : {TOPO}")
print(f"mesh   : {mesh.number_of_vertices()}v / {mesh.number_of_faces()}f / "
      f"{mesh.number_of_edges()}e   anchors {len(list(mesh.vertices_on_boundary()))}")
print(f"span   : %.4f x %.4f m   crown %.6f m" % (np.ptp(V[:, 0]), np.ptp(V[:, 1]), V[:, 2].max()))
print(f"q      : %.5f .. %.5f (median %.5f)" % (q.min(), q.max(), np.median(q)))
print(f"recorded: rmse {d['rmse_m']*1000:.4f} mm, converged={d['converged']}, pressure={d['pressure']}")
for p in (stem + ".json", stem + "_compas1.json", geo + ".off", geo + ".obj"):
    print("wrote  :", p)

check = Mesh.from_json(stem + ".json")
print(f"verify : Mesh.from_json OK - {check.number_of_vertices()}v / {check.number_of_faces()}f")
