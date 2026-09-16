"""
Loader for the mesh JSONs in this repo, which come in two incompatible formats.

The 2024 scripts (fofin_butt_steps.py, fofin_cross.py, fofin_seismic.py, ...)
wrote compas 1.x serialisation: 'vertex' / 'face' / 'edgedata' at the top level.
compas 2.x refuses those with

    TypeError: The data in the file is not a <class '...Mesh'>

The 2026 scripts (fofin_B5.py, fofin_C5*.py, fofin_2part_smooth.py) write
compas 2.x: the same payload nested under 'data', with a 'dtype'/'guid' wrapper.

load_mesh_json() reads either.  Of the 103 mesh JSONs in the repo only 6 load
with a bare Mesh.from_json; this reads all 103.

    from FDM.mesh_io import load_mesh_json, edge_q
    mesh = load_mesh_json(path)
"""
import json
from compas.datastructures import Mesh


def load_mesh_json(path):
    """Return a compas 2 Mesh from either serialisation format."""
    with open(path) as f:
        d = json.load(f)

    if "data" in d and isinstance(d["data"], dict) and "vertex" in d["data"]:
        return Mesh.from_json(path)              # already compas 2.x

    if "vertex" not in d or "face" not in d:
        raise ValueError(f"{path} does not look like a serialised mesh")

    # compas 1.x -> rebuild explicitly, keeping vertex keys and edge attributes
    mesh = Mesh()
    mesh.update_default_vertex_attributes(d.get("dva", {}))
    mesh.update_default_edge_attributes(d.get("dea", {}))
    mesh.update_default_face_attributes(d.get("dfa", {}))

    for key, attr in d["vertex"].items():
        mesh.add_vertex(key=int(key), **attr)
    for key, verts in d["face"].items():
        mesh.add_face([int(v) for v in verts], fkey=int(key))

    for key, attr in d.get("edgedata", {}).items():
        u, v = (int(t) for t in key.strip("()").split(","))
        if mesh.has_edge((u, v)) or mesh.has_edge((v, u)):
            for name, value in attr.items():
                mesh.edge_attribute((u, v), name, value)
    return mesh


def edge_q(mesh, edge, name="qpre"):
    """Force density as a float, whether it was stored as a float or [float].

    The 2024 results assigned qpre from an (n,1) numpy column, so each value
    serialised as a 1-element list; the 2026 ones cast with float().
    """
    q = mesh.edge_attribute(edge, name)
    return float(q[0]) if isinstance(q, (list, tuple)) else float(q)
