"""
Convert an FDM result JSON into a compas 2 Mesh (and re-save it so that a
plain Mesh.from_json works on it afterwards).

The 2024 results are compas 1.x serialisation, which compas 2.15 refuses.
This reads either format via mesh_io.load_mesh_json, normalises qpre to a
plain float (the 2024 files store it as a 1-element list), and writes:

  <name>_compas2.json    compas 2 serialisation, loads with Mesh.from_json
  <name>.obj / .off      optional, with --obj / --off

Usage:
  python FDM/convert_fdm_to_compas.py [result.json] [--obj] [--off]
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_io import load_mesh_json, edge_q

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "data", "mesh_out_B5_20260501213116.json")

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default=DEFAULT)
ap.add_argument("--obj", action="store_true", help="also write .obj")
ap.add_argument("--off", action="store_true", help="also write .off")
args = ap.parse_args()

src = os.path.abspath(args.path)
mesh = load_mesh_json(src)

# normalise qpre so downstream code never needs the [0] subscript
n_fixed = 0
for e in mesh.edges():
    raw = mesh.edge_attribute(e, "qpre")
    if isinstance(raw, (list, tuple)):
        n_fixed += 1
    mesh.edge_attribute(e, "qpre", edge_q(mesh, e))

stem = os.path.splitext(src)[0]
out_json = stem + "_compas2.json"
mesh.to_json(out_json)

print(f"source : {src}")
print(f"mesh   : {mesh.number_of_vertices()} vertices, {mesh.number_of_faces()} faces, "
      f"{mesh.number_of_edges()} edges")
q = np.array([mesh.edge_attribute(e, "qpre") for e in mesh.edges()], dtype=float)
V = np.array([mesh.vertex_coordinates(v) for v in mesh.vertices()])
print(f"q      : {q.min():.5f} .. {q.max():.5f}  (median {np.median(q):.5f})"
      + (f"   [unwrapped {n_fixed} list values]" if n_fixed else ""))
print(f"bbox   : span {np.ptp(V[:,0]):.4f} x {np.ptp(V[:,1]):.4f} m, "
      f"z {V[:,2].min():.4f} .. {V[:,2].max():.4f} m")
print(f"wrote  : {out_json}")

if args.obj:
    out = stem + ".obj"
    vk = list(mesh.vertices()); idx = {v: i for i, v in enumerate(vk)}
    with open(out, "w") as f:
        for v in vk:
            f.write("v %.8f %.8f %.8f\n" % tuple(mesh.vertex_coordinates(v)))
        for fk in mesh.faces():
            f.write("f " + " ".join(str(idx[v] + 1) for v in mesh.face_vertices(fk)) + "\n")
    print(f"wrote  : {out}")

if args.off:
    out = stem + ".off"
    vk = list(mesh.vertices()); idx = {v: i for i, v in enumerate(vk)}
    faces = [[idx[v] for v in mesh.face_vertices(fk)] for fk in mesh.faces()]
    with open(out, "w") as f:
        f.write("OFF\n%d %d 0\n" % (len(vk), len(faces)))
        for v in vk:
            f.write("%.8f %.8f %.8f\n" % tuple(mesh.vertex_coordinates(v)))
        for fc in faces:
            f.write(str(len(fc)) + " " + " ".join(str(i) for i in fc) + "\n")
    print(f"wrote  : {out}")

# verify the round trip
from compas.datastructures import Mesh
check = Mesh.from_json(out_json)
print(f"verify : Mesh.from_json OK - {check.number_of_vertices()}v / {check.number_of_faces()}f")
