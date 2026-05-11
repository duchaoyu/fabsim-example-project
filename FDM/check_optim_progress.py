"""
Live progress checker: read the latest run_*_verts.csv files and compute
RMSE / mean / max deviation against the target. Shows the last N runs and
the running best.
"""
import os, csv, glob, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OPT  = os.path.join(HERE, "optimisation")

V_target    = np.load(os.path.join(DATA, "B5_remeshed_target_verts.npy"))
interior    = np.load(os.path.join(DATA, "B5_remeshed_interior_idx.npy"))

def deviation(verts_csv):
    V = np.zeros_like(V_target)
    with open(verts_csv) as f:
        for row in csv.DictReader(f):
            i = int(row["vid"])
            V[i] = [float(row["x"]), float(row["y"]), float(row["z"])]
    diff = V[interior] - V_target[interior]
    d3   = np.linalg.norm(diff, axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(d3**2))),
        "mean": float(d3.mean()),
        "max":  float(d3.max()),
        "crown_sim": float(V[:,2].max()),
    }

ap = argparse.ArgumentParser()
ap.add_argument("--last", type=int, default=10, help="show last N runs")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(OPT, "run_*_verts.csv")))
if not files:
    print("No run_*_verts.csv yet"); raise SystemExit

print(f"Total FEM calls so far: {len(files)}")
print(f"Target crown: {V_target[:,2].max():.4f} m  |  interior verts: {len(interior)}\n")

print(f"{'call':>6}  {'RMSE (mm)':>9}  {'mean (mm)':>9}  {'max (mm)':>8}  {'crown (m)':>10}")
print("-" * 56)

best = None
for f in files:
    d   = deviation(f)
    n   = int(os.path.basename(f).split("_")[1])
    if best is None or d["rmse"] < best[1]["rmse"]:
        best = (n, d)

# print last N
for f in files[-args.last:]:
    d = deviation(f)
    n = int(os.path.basename(f).split("_")[1])
    star = " <-- best" if n == best[0] else ""
    print(f"{n:6d}  {d['rmse']*1000:9.3f}  {d['mean']*1000:9.3f}  {d['max']*1000:8.3f}  {d['crown_sim']:10.4f}{star}")

print("-" * 56)
print(f"Best so far: call {best[0]}  RMSE={best[1]['rmse']*1000:.3f} mm  "
      f"mean={best[1]['mean']*1000:.3f} mm  max={best[1]['max']*1000:.3f} mm")
