"""
Simulate a dome where the right 1/3 of the circular boundary is replaced
by a steel cable (free to lift) while the left 2/3 remains fixed.

Sweeps cable pre-tension (L_rest as fraction of natural arc length).

Outputs to: sensitivity_analysis/data/boundary_cable/
"""
import os, sys, json, subprocess, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MESH_PATH, FEM_BINARY
from curvature import read_off

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "boundary_cable")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CABLE_EA  = 150000.0
PRESSURE  = 1000.0
MOTIF     = 1
SF        = 1.0
KNIT_DIR  = 0.0
FREE_ARC  = np.pi / 3   # ±60° → 120° = 1/3 of boundary

# L_rest as fraction of natural arc length: 1.0 = no pre-stress, <1 = pre-tensioned
L_REST_FRACS = [1.0, 0.95, 0.90, 0.85, 0.80]

# ── Load mesh and classify boundary vertices ──────────────────────────────────
V0, F = read_off(MESH_PATH)

edge_cnt = defaultdict(int)
for f in F:
    for i in range(3):
        e = tuple(sorted([int(f[i]), int(f[(i+1)%3])]))
        edge_cnt[e] += 1
bv_set = set()
for e, c in edge_cnt.items():
    if c == 1:
        bv_set.update(e)
all_bv = sorted(bv_set)

angles  = np.arctan2(V0[all_bv, 1], V0[all_bv, 0])
in_arc  = np.array([abs(a) <= FREE_ARC for a in angles])
arc_bv  = [all_bv[i] for i in range(len(all_bv)) if     in_arc[i]]
left_bv = [all_bv[i] for i in range(len(all_bv)) if not in_arc[i]]

arc_ordered = [v for v, _ in sorted(
    zip(arc_bv, [angles[i] for i in range(len(all_bv)) if in_arc[i]]),
    key=lambda x: x[1])]

junction     = [arc_ordered[0], arc_ordered[-1]]
fixed_verts  = sorted(set(left_bv + junction))

L_natural = sum(
    np.linalg.norm(V0[arc_ordered[k+1]] - V0[arc_ordered[k]])
    for k in range(len(arc_ordered) - 1))

print(f"Boundary total: {len(all_bv)}  |  fixed: {len(fixed_verts)}  "
      f"|  cable arc: {len(arc_ordered)}")
print(f"Natural arc length: {L_natural*1000:.1f} mm\n")

# Write fixed-vertices file (shared across all runs)
fv_path = os.path.join(OUT_DIR, "fixed_verts.txt")
with open(fv_path, "w") as f:
    for v in fixed_verts:
        f.write(f"{v}\n")

# ── Run sweep ─────────────────────────────────────────────────────────────────
results = []
for frac in L_REST_FRACS:
    L_rest   = frac * L_natural
    tag      = f"lr{frac:.2f}"
    prefix   = os.path.join(OUT_DIR, tag)
    scalars_path = prefix + "_scalars.csv"

    if not os.path.exists(scalars_path):
        cable_spec = {"indices": arc_ordered, "EA": CABLE_EA,
                      "L_rest": float(L_rest)}
        cpath = os.path.join(OUT_DIR, f"cable_{tag}.json")
        with open(cpath, "w") as f:
            json.dump(cable_spec, f)

        cmd = [FEM_BINARY, MESH_PATH,
               f"{SF:.4f}", f"{SF:.4f}", f"{KNIT_DIR:.2f}", f"{PRESSURE:.1f}",
               str(MOTIF), cpath, prefix, fv_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  FAILED frac={frac}: {r.stderr[:200]}")
            continue

    with open(scalars_path) as f:
        row = next(csv.DictReader(f))
    h  = float(row["crown_height"]) * 1000
    T  = float(row["cable_tension"])
    print(f"  frac={frac:.2f}  L_rest={L_rest*1000:.0f}mm  "
          f"h={h:.1f}mm  T={T:.0f}N")
    results.append({"frac": frac, "L_rest_mm": L_rest*1000,
                    "crown_height_mm": h, "cable_tension_N": T,
                    "prefix": prefix})

# ── Run reference (fully fixed, no cable) ────────────────────────────────────
ref_prefix = os.path.join(OUT_DIR, "result_nocable")
if not os.path.exists(ref_prefix + "_scalars.csv"):
    subprocess.run([FEM_BINARY, MESH_PATH,
                    f"{SF:.4f}", f"{SF:.4f}", f"{KNIT_DIR:.2f}", f"{PRESSURE:.1f}",
                    str(MOTIF), "none", ref_prefix],
                   capture_output=True, text=True, timeout=300)
with open(ref_prefix + "_scalars.csv") as f:
    ref_row = next(csv.DictReader(f))
ref_h = float(ref_row["crown_height"]) * 1000
print(f"\n  reference (no cable, fully fixed): h={ref_h:.1f}mm")

# ── Save OBJs ─────────────────────────────────────────────────────────────────
for res in results:
    verts = pd.read_csv(res["prefix"] + "_verts.csv").sort_values("vid")[["x","y","z"]].values
    obj_path = res["prefix"] + ".obj"
    with open(obj_path, "w") as f:
        f.write(f"# frac={res['frac']:.2f}  T={res['cable_tension_N']:.0f}N\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in F:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"\nOBJs saved to {OUT_DIR}/")

# ── Figure: 3D views for each L_rest fraction ────────────────────────────────
n = len(results)
fig = plt.figure(figsize=(4.5*n, 5.5))
fig.suptitle(
    f"Right-1/3 boundary cable (steel EA=150 kN, p={PRESSURE:.0f} Pa, motif {MOTIF})\n"
    "Orange = fixed boundary  |  Blue = cable arc  |  decreasing L_rest → more pre-tension",
    fontsize=9)

fixed_edge_set = {e for e, c in edge_cnt.items() if c == 1
                  and e[0] in set(fixed_verts) and e[1] in set(fixed_verts)}

for idx, res in enumerate(results):
    verts = pd.read_csv(res["prefix"] + "_verts.csv").sort_values("vid")[["x","y","z"]].values
    ax = fig.add_subplot(1, n, idx+1, projection="3d")

    # Mesh surface
    tris = [[verts[f[i]] for i in range(3)] for f in F]
    pc   = Poly3DCollection(tris, alpha=0.20, linewidth=0,
                            facecolor="steelblue", edgecolor="none")
    ax.add_collection3d(pc)

    # Fixed boundary edges
    for e in fixed_edge_set:
        a, b = e
        ax.plot([verts[a,0], verts[b,0]],
                [verts[a,1], verts[b,1]],
                [verts[a,2], verts[b,2]], color="#CC6600", lw=2.0)

    # Cable arc
    cx = [verts[v,0] for v in arc_ordered]
    cy = [verts[v,1] for v in arc_ordered]
    cz = [verts[v,2] for v in arc_ordered]
    ax.plot(cx, cy, cz, color="#1155CC", lw=2.5)

    frac  = res["frac"]
    T     = res["cable_tension_N"]
    h     = res["crown_height_mm"]
    ax.set_title(f"L_rest={frac*100:.0f}%  T={T:.0f}N\nh={h:.0f}mm",
                 fontsize=8)
    R = 0.68
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, R)
    ax.set_xlabel("x(m)", fontsize=7); ax.set_ylabel("y(m)", fontsize=7)
    ax.set_zlabel("z(m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=28, azim=-55)

plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "figR_boundary_cable.pdf")
fig.savefig(out_fig, bbox_inches="tight")
fig.savefig(out_fig.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
print(f"Saved: {out_fig}")
