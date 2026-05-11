"""
Export the FEM-simulated geometry at the optimum as .off and .obj.

Reads the latest run_*_verts.csv (the final FEM evaluation written by the
optimiser at the converged sf+cable parameters) and emits:

  data/B5_optimised.off
  data/B5_optimised.obj

The face topology is taken from data/B5_remeshed_shared.off (unchanged).
"""
import os, csv, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OPT  = os.path.join(HERE, "optimisation")

REF_OFF = os.path.join(DATA, "B5_remeshed_shared.off")
OUT_OFF = os.path.join(DATA, "B5_optimised.off")
OUT_OBJ = os.path.join(DATA, "B5_optimised.obj")

def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

def latest_verts():
    files = sorted(glob.glob(os.path.join(OPT, "run_*_verts.csv")))
    if not files:
        raise FileNotFoundError("No run_*_verts.csv in optimisation/")
    return files[-1]

def write_off(path, V, F):
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in F:
            f.write(f"{len(tri)} " + " ".join(str(int(i)) for i in tri) + "\n")

def write_obj(path, V, F):
    with open(path, "w") as f:
        for v in V:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in F:
            # OBJ is 1-indexed
            f.write("f " + " ".join(str(int(i) + 1) for i in tri) + "\n")

V_ref, F = load_off(REF_OFF)

verts_csv = latest_verts()
print(f"Source: {os.path.basename(verts_csv)}")

V_opt = np.zeros_like(V_ref)
with open(verts_csv) as fh:
    for row in csv.DictReader(fh):
        i = int(row["vid"])
        V_opt[i] = [float(row["x"]), float(row["y"]), float(row["z"])]

# Sanity: check we got every vertex
n_filled = (V_opt != 0).any(axis=1).sum()
print(f"Filled {n_filled}/{len(V_ref)} vertices  (zero-rows = boundary at z=0 OK)")

# Stats vs target reference
target_verts = np.load(os.path.join(DATA, "B5_remeshed_target_verts.npy"))
interior     = np.load(os.path.join(DATA, "B5_remeshed_interior_idx.npy"))
diff = V_opt[interior] - target_verts[interior]
d3   = np.linalg.norm(diff, axis=1)
print(f"Geometry stats:")
print(f"  vertices: {len(V_opt)}, faces: {len(F)}")
print(f"  span: x={V_opt[:,0].max()-V_opt[:,0].min():.4f}m  "
      f"y={V_opt[:,1].max()-V_opt[:,1].min():.4f}m  "
      f"z={V_opt[:,2].max()-V_opt[:,2].min():.4f}m")
print(f"  crown sim={V_opt[:,2].max():.4f}m   target={target_verts[:,2].max():.4f}m")
print(f"  vs target (interior): RMSE={np.sqrt(np.mean(d3**2))*1000:.3f}mm  "
      f"mean={d3.mean()*1000:.3f}mm  max={d3.max()*1000:.3f}mm")

write_off(OUT_OFF, V_opt, F)
write_obj(OUT_OBJ, V_opt, F)
print(f"\nSaved: {OUT_OFF}")
print(f"Saved: {OUT_OBJ}")
