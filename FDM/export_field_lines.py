"""
Export a directional (cross) field as OBJ line segments for Rhino.

Generalises export_C5_field_lines.py to any geometry whose field JSON holds
per-face d1 / d2 / centroid.  Segments are centred on the face centroid, so
d1 and d2 overlay as a cross.

Length is SCALE x the mesh's mean edge length, then hard-capped at
--max-length (strictly below it, not equal).

  python FDM/export_field_lines.py D5 [--scale 0.8] [--max-length 0.02]

Writes, next to the field JSON:
  <geom>_field_d1.obj      first direction only
  <geom>_field_d2.obj      second direction only
  <geom>_field_cross.obj   both, in two named groups
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

GEOM = {
    "D5": ("data/D5/directional_field_D5.json", "data/D5/D5_remeshed.obj",
           "d1_radial", "d2_circumferential"),
    "C5": ("data/C5/directional_field_C5.json", "data/C5/C5_remeshed.obj",
           "d1_radial", "d2_circumferential"),
}

ap = argparse.ArgumentParser()
ap.add_argument("geometry", nargs="?", default="D5", choices=sorted(GEOM))
ap.add_argument("--scale",      type=float, default=0.8,
                help="segment length as a multiple of mean edge length")
ap.add_argument("--max-length", type=float, default=0.02,
                help="hard cap; every segment comes out strictly below this")
a = ap.parse_args()

field_rel, mesh_rel, n1, n2 = GEOM[a.geometry]
FIELD = os.path.join(HERE, field_rel)
MESH  = os.path.join(HERE, mesh_rel)
OUTD  = os.path.dirname(FIELD)


def load_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            i = [int(t.split("/")[0]) - 1 for t in l.split()[1:]]
            F.append(i[:3])
            if len(i) == 4:
                F.append([i[0], i[2], i[3]])
    return np.array(V), F


V, F = load_obj(MESH)
field = json.load(open(FIELD))

elen = [np.linalg.norm(V[f[i]] - V[f[(i + 1) % len(f)]])
        for f in F for i in range(len(f))]
mean_edge = float(np.mean(elen))

L = mean_edge * a.scale
capped = False
# Coordinates are written with 8 decimals, so a length sitting exactly on the
# cap can round up to just above it.  Back off by 1e-6 m (0.001 mm), which is
# far larger than the ~2e-8 rounding error and visually irrelevant.
CAP_MARGIN = 1e-6
if L >= a.max_length - CAP_MARGIN:
    L = a.max_length - CAP_MARGIN
    capped = True

keys = sorted(field, key=int)
C  = np.array([field[k]["centroid"] for k in keys])
D1 = np.array([field[k]["d1"] for k in keys], dtype=float)
D2 = np.array([field[k]["d2"] for k in keys], dtype=float)
D1 /= np.linalg.norm(D1, axis=1, keepdims=True)
D2 /= np.linalg.norm(D2, axis=1, keepdims=True)


def write_obj(path, groups, header):
    n = 0
    lens = []
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        f.write(f"# segment length {L:.6f} m"
                + (f" (capped at {a.max_length})\n" if capped
                   else f" ({a.scale} x mean edge {mean_edge:.6f})\n"))
        f.write("# OBJ 'l' polylines - Rhino imports these as curves\n")
        for name, Cg, Dg in groups:
            f.write(f"g {name}\n")
            p = Cg - 0.5 * L * Dg
            q = Cg + 0.5 * L * Dg
            lens.extend(np.linalg.norm(q - p, axis=1).tolist())
            for pp, qq in zip(p, q):
                f.write("v %.8f %.8f %.8f\n" % tuple(pp))
                f.write("v %.8f %.8f %.8f\n" % tuple(qq))
            for _ in range(len(Cg)):
                f.write(f"l {n+1} {n+2}\n")
                n += 2
    lens = np.array(lens)
    print("  %-26s %5d lines   length %.8f m   max %.8f  all < %.3f: %s"
          % (os.path.basename(path), n // 2, lens.mean(), lens.max(),
             a.max_length, bool(np.all(lens < a.max_length))))


print(f"{a.geometry}: {len(keys)} faces, mean edge {mean_edge:.6f} m")
print(f"segment length {L:.8f} m" + ("  (capped)" if capped else ""))
write_obj(os.path.join(OUTD, f"{a.geometry}_field_d1.obj"),
          [(n1, C, D1)], f"{a.geometry} directional field - {n1}")
write_obj(os.path.join(OUTD, f"{a.geometry}_field_d2.obj"),
          [(n2, C, D2)], f"{a.geometry} directional field - {n2}")
write_obj(os.path.join(OUTD, f"{a.geometry}_field_cross.obj"),
          [(n1, C, D1), (n2, C, D2)], f"{a.geometry} directional field - cross")
