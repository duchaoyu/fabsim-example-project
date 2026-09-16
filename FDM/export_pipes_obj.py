"""
Export FDM edges as solid pipes whose radius scales with q, as a mesh .obj.

Rhino only has to read triangles and quads - no line/curve import, no Pipe
command, no display-pipeline tricks.

  python FDM/export_pipes_obj.py <mesh.json> [options]

Options:
  --sides N        polygon sides per pipe        (default 8)
  --rmin M         radius of the weakest edge    (default 0.0005 m)
  --rmax M         radius at --qref              (default 0.0040 m)
  --qref P         percentile of q mapped to rmax (default 98)
  --caps           close the pipe ends with a fan
  --joints         add a ball at every vertex, sized to its largest pipe
  --out PATH       output .obj

Radius is linear in q, clipped at the --qref percentile so a few extreme
edges do not flatten everything else.
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mesh_io import load_mesh_json, edge_q

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?",
                default=os.path.join(HERE, "data", "D5",
                                     "mesh_out_D5_smooth_20260916135202.json"))
ap.add_argument("--sides",  type=int,   default=8)
ap.add_argument("--rmin",   type=float, default=0.0005)
ap.add_argument("--rmax",   type=float, default=0.0040)
ap.add_argument("--qref",   type=float, default=98.0)
ap.add_argument("--caps",   action="store_true")
ap.add_argument("--joints", action="store_true")
ap.add_argument("--out",    type=str,   default=None)
a = ap.parse_args()

mesh = load_mesh_json(os.path.abspath(a.path))
E = list(mesh.edges())
q = np.array([edge_q(mesh, e) for e in E], dtype=float)
P0 = np.array([mesh.vertex_coordinates(u) for u, _ in E], dtype=float)
P1 = np.array([mesh.vertex_coordinates(v) for _, v in E], dtype=float)

qref = np.percentile(q, a.qref)
rad  = a.rmin + (a.rmax - a.rmin) * np.clip(q / qref, 0.0, 1.0)

verts, faces = [], []
ring = np.arange(a.sides)
ang  = 2.0 * np.pi * ring / a.sides
cos_t, sin_t = np.cos(ang), np.sin(ang)


def frame(d):
    """Two unit vectors perpendicular to d."""
    n = np.linalg.norm(d)
    if n == 0:
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    d = d / n
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(d, ref); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


n_skipped = 0
for k in range(len(E)):
    p0, p1, r = P0[k], P1[k], rad[k]
    d = p1 - p0
    if np.linalg.norm(d) < 1e-12:
        n_skipped += 1
        continue
    u, v = frame(d)
    base = len(verts)
    off = r * (cos_t[:, None] * u[None, :] + sin_t[:, None] * v[None, :])
    verts.extend((p0 + off).tolist())
    verts.extend((p1 + off).tolist())
    for i in range(a.sides):
        j = (i + 1) % a.sides
        faces.append([base + i, base + j, base + a.sides + j, base + a.sides + i])
    if a.caps:
        c0 = len(verts); verts.append(p0.tolist())
        c1 = len(verts); verts.append(p1.tolist())
        for i in range(a.sides):
            j = (i + 1) % a.sides
            faces.append([c0, base + j, base + i])
            faces.append([c1, base + a.sides + i, base + a.sides + j])

if a.joints:
    # ball radius = largest pipe meeting at that vertex
    vr = {}
    for k, (uu, vv) in enumerate(E):
        vr[uu] = max(vr.get(uu, 0.0), rad[k])
        vr[vv] = max(vr.get(vv, 0.0), rad[k])
    n_lat = max(3, a.sides // 2)
    for vk, r in vr.items():
        c = np.array(mesh.vertex_coordinates(vk))
        base = len(verts)
        for i in range(1, n_lat):
            phi = np.pi * i / n_lat
            z, rr = np.cos(phi) * r, np.sin(phi) * r
            for t in range(a.sides):
                th = 2 * np.pi * t / a.sides
                verts.append([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), c[2] + z])
        top = len(verts); verts.append([c[0], c[1], c[2] + r])
        bot = len(verts); verts.append([c[0], c[1], c[2] - r])
        for i in range(n_lat - 2):
            for t in range(a.sides):
                t2 = (t + 1) % a.sides
                A = base + i * a.sides + t;  B = base + i * a.sides + t2
                C = base + (i + 1) * a.sides + t2; D = base + (i + 1) * a.sides + t
                faces.append([A, B, C, D])
        for t in range(a.sides):
            t2 = (t + 1) % a.sides
            faces.append([top, base + t2, base + t])
            faces.append([bot, base + (n_lat - 2) * a.sides + t,
                          base + (n_lat - 2) * a.sides + t2])

out = a.out or os.path.splitext(a.path)[0] + "_pipes.obj"
with open(out, "w") as f:
    f.write(f"# pipes from {os.path.basename(a.path)}\n")
    f.write(f"# radius {a.rmin*1000:.2f}-{a.rmax*1000:.2f} mm linear in q, "
            f"clipped at the p{a.qref:.0f} value q={qref:.4f}\n")
    f.write(f"# {len(E)} edges, q {q.min():.5f}-{q.max():.5f}\n")
    f.write("g fdm_pipes\n")
    for p in verts:
        f.write("v %.8f %.8f %.8f\n" % tuple(p))
    for fc in faces:
        f.write("f " + " ".join(str(i + 1) for i in fc) + "\n")

print(f"source : {a.path}")
print(f"edges  : {len(E)}   q {q.min():.5f} .. {q.max():.5f}   p{a.qref:.0f} = {qref:.5f}")
print(f"radius : {a.rmin*1000:.2f} .. {a.rmax*1000:.2f} mm "
      f"({a.sides}-sided{', capped' if a.caps else ''}{', with joints' if a.joints else ''})")
if n_skipped:
    print(f"skipped: {n_skipped} zero-length edges")
print(f"output : {out}")
print(f"         {len(verts)} vertices, {len(faces)} faces, "
      f"{os.path.getsize(out)/1e6:.1f} MB")
