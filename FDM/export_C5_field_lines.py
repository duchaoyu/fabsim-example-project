"""
Export the C5 directional (cross) field as OBJ line segments for Rhino.

Reads FDM/data/C5/directional_field_C5.json (per-face d1 / d2 / centroid,
2137 faces on C5_remeshed.obj, Ø1.2 m) and writes polylines using the OBJ
`l` element, which Rhino imports as curves.

  C5_field_d1.obj      radial / wale direction only
  C5_field_d2.obj      circumferential / course direction only
  C5_field_cross.obj   both, in two named groups (d1_radial, d2_circumferential)

Each segment is centred on its face centroid, so the pair reads as a cross.
Length is SCALE x the mean edge length of the mesh; pass --scale to change it.
"""
import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "C5")
FIELD = os.path.join(DATA, "directional_field_C5.json")
MESH  = os.path.join(DATA, "C5_remeshed.obj")

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.8,
                help="segment length as a multiple of mean edge length")
args = ap.parse_args()


def load_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith("v "):   V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "): F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:]])
    return np.array(V), F


V, F = load_obj(MESH)
field = json.load(open(FIELD))

# mean edge length sets the tick length, so the field reads at the mesh's own scale
elen = []
for f in F:
    for i in range(len(f)):
        elen.append(np.linalg.norm(V[f[i]] - V[f[(i + 1) % len(f)]]))
L = float(np.mean(elen)) * args.scale

keys = sorted(field, key=int)
C  = np.array([field[k]["centroid"] for k in keys])
D1 = np.array([field[k]["d1"] for k in keys])
D2 = np.array([field[k]["d2"] for k in keys])
# normalise defensively - the solve can leave slightly off-unit vectors
D1 /= np.linalg.norm(D1, axis=1, keepdims=True)
D2 /= np.linalg.norm(D2, axis=1, keepdims=True)


def write_obj(path, groups, header):
    """groups: list of (name, centroids, directions)."""
    n = 0
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        f.write(f"# segment length {L:.6f} m ({args.scale} x mean edge length)\n")
        f.write("# OBJ 'l' polylines - Rhino imports these as curves\n")
        for name, Cg, Dg in groups:
            f.write(f"g {name}\n")
            a = Cg - 0.5 * L * Dg
            b = Cg + 0.5 * L * Dg
            for p, q in zip(a, b):
                f.write("v %.8f %.8f %.8f\n" % tuple(p))
                f.write("v %.8f %.8f %.8f\n" % tuple(q))
            for _ in range(len(Cg)):
                f.write(f"l {n+1} {n+2}\n")
                n += 2
    print("wrote %-42s %5d lines" % (os.path.basename(path), n // 2))


write_obj(os.path.join(DATA, "C5_field_d1.obj"),
          [("d1_radial", C, D1)], "C5 directional field - d1 (radial / wale)")
write_obj(os.path.join(DATA, "C5_field_d2.obj"),
          [("d2_circumferential", C, D2)], "C5 directional field - d2 (circumferential / course)")
write_obj(os.path.join(DATA, "C5_field_cross.obj"),
          [("d1_radial", C, D1), ("d2_circumferential", C, D2)],
          "C5 directional field - full cross field")
print("faces %d  mean edge %.5f m  segment %.5f m" % (len(keys), L / args.scale, L))
