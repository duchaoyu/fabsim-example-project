"""
Export the FEM geometry of a given c5_p2 call as .obj (and .off).

  call 34  -> 0.0828 mm interior RMSE, the hand-picked best used by
              export_C5_optim_result.py
  call 32  -> 0.3608 mm, the optimiser's own phase-2 final point
              (phase2_rmse_m in C5_16region_optimised_sym.json)

Usage:  python FDM/export_C5_call_obj.py [call_number]
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OFF  = os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off")
OPT  = os.path.join(HERE, "optimisation")
OUTD = os.path.join(HERE, "data", "C5")

call = int(sys.argv[1]) if len(sys.argv) > 1 else 32


def load_off(path):
    L = open(path).readlines()
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in l.split()[:3]] for l in L[2:2 + nv]])
    F = [[int(x) for x in l.split()[1:]] for l in L[2 + nv:2 + nv + nf]]
    return V, F


V_t, F = load_off(OFF)
V = np.loadtxt(os.path.join(OPT, f"c5_p2_{call:05d}_verts.csv"),
               delimiter=",", skiprows=1)[:, 1:4]

r = np.hypot(V_t[:, 0], V_t[:, 1])
interior = np.where(r <= 0.98 * r.max())[0]
d = np.linalg.norm(V - V_t, axis=1)
rmse_int = float(np.sqrt(np.mean(np.sum((V - V_t)[interior] ** 2, axis=1))))

stem = os.path.join(OUTD, f"C5_optim_call{call:05d}")
with open(stem + ".obj", "w") as f:
    f.write(f"# C5 FEM result, call {call}\n")
    f.write(f"# interior RMSE {rmse_int*1000:.4f} mm, max deviation {d.max()*1000:.4f} mm\n")
    f.write(f"# crown {V[:,2].max():.6f} m (target {V_t[:,2].max():.6f} m)\n")
    for v in V:
        f.write("v %.8f %.8f %.8f\n" % tuple(v))
    for fc in F:
        f.write("f " + " ".join(str(i + 1) for i in fc) + "\n")
with open(stem + ".off", "w") as f:
    f.write("OFF\n%d %d 0\n" % (len(V), len(F)))
    for v in V:
        f.write("%.8f %.8f %.8f\n" % tuple(v))
    for fc in F:
        f.write(str(len(fc)) + " " + " ".join(str(i) for i in fc) + "\n")

print(f"call {call}: {len(V)} verts / {len(F)} faces")
print(f"  interior RMSE {rmse_int*1000:.4f} mm   max dev {d.max()*1000:.4f} mm")
print(f"  crown {V[:,2].max():.6f} m  (target {V_t[:,2].max():.6f} m, "
      f"{1000*(V[:,2].max()-V_t[:,2].max()):+.3f} mm)")
print(f"  wrote {stem}.obj")
print(f"  wrote {stem}.off")
