"""
Free-form shell with open boundaries (pattern_smooth, rm2, 10 strips) with GFRP
edge splines under the follower load: result surface, deviation map, stretch
factors per region and sections against the target.

Takes the best valid evaluation of a run (its calls log) or its result JSON,
re-solves it with the binary (the driver deletes per-call shapes) and draws it.

    .venv/bin/python FDM/figure_pattern_smooth_spline.py [prefix]

prefix defaults to pattern_smooth_rm2_10strip_spline10_sI_fol; writes
figures/<prefix>.png.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from optimise_4part import load_off

OPT = os.path.join(HERE, "optimisation")
BINARY = os.path.join(HERE, "..", "build-linux", "fem_batch_nregion")
SEED = os.path.join(OPT, "pattern_smooth_rm2_10strip_lap_p3_result.json")   # mesh, cables, supports, knit

prefix = sys.argv[1] if len(sys.argv) > 1 else "pattern_smooth_rm2_10strip_spline10_sI_fol"
rj = os.path.join(OPT, f"{prefix}_result.json")
if os.path.exists(rj):
    best = json.load(open(rj)); label = "final"
else:
    recs = [json.loads(l) for l in open(os.path.join(OPT, f"{prefix}_calls.jsonl"))]
    best = min((r for r in recs if r.get("rmse_mm")), key=lambda r: r["rmse_mm"])
    label = f"in progress, best of {len(recs)} calls (call {best['call']})"

S = json.load(open(SEED))
mesh = os.path.join(HERE, S["mesh"]); rmap = os.path.join(OPT, f"../{S['region_map']}")
V, F = load_off(mesh)
C = json.load(open(os.path.join(HERE, S["cable_file"])))
fixed = S["fixed_vertices"]
free = np.array(sorted(set(range(len(V))) - set(fixed)))
face_region = np.array(json.load(open(rmap))["face_regions"])
knit = S["knit_dir_deg_region_means"]
d = 0.010
E03 = C["E03a"] + C["E03b"][1:] + C["E03c"][1:]
splines = [C["E00"], C["E02"], E03]
cables = ["C00", "C01a", "C01b", "C02"]
p = dict(pressure=1000.0, motif=1, cable_ea=157000.0, E1=12500.0, E2=5000.0, nu=0.198,
         cable_paths=[C[k] for k in cables], fixed_vertices=fixed, newton_reg_max=1e6,
         regions=[dict(sf_wale=best["sf_wale"][r], sf_course=best["sf_course"][r],
                       knit_dir_deg=knit[r]) for r in range(len(knit))],
         cable_rest_scales=best["cable_rest_scales"], spline_paths=splines,
         spline_EA=40e9 * np.pi * d**2 / 4, spline_EI=40e9 * np.pi * d**4 / 64, spline_rest=1)
tmp = tempfile.mkdtemp()
json.dump(p, open(os.path.join(tmp, "p.json"), "w"))
env = dict(os.environ, FEM_PRESSURE="follower", FEM_FOLLOWER_START="volume")
r = subprocess.run([BINARY, mesh, rmap, os.path.join(tmp, "p.json"), os.path.join(tmp, "o")],
                   capture_output=True, text=True, env=env)
status = [l for l in r.stderr.splitlines() if l.startswith("SOLVER_STATUS")][-1].split()[1]
X = np.loadtxt(os.path.join(tmp, "o_verts.csv"), delimiter=",", skiprows=1)[:, 1:]
dv = np.linalg.norm(X - V, axis=1) * 1e3
rmse, mx = np.sqrt((dv[free] ** 2).mean()), dv[free].max()
span = np.ptp(V[:, 0]) * 1e3
print(f"{status}: RMSE {rmse:.2f} mm, mean {dv[free].mean():.2f}, max {mx:.2f} mm")

fig = plt.figure(figsize=(17, 10))
fig.suptitle(f"Free-form shell, 10 mm GFRP edge splines, follower load 1000 Pa, structure I "
             f"({label}; solve {status})", fontsize=12)

# (a) 3D result surface coloured by deviation
ax = fig.add_subplot(2, 3, 1, projection="3d")
ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=F, cmap="magma_r",
                array=dv[F].mean(axis=1), linewidth=0.05, edgecolor="0.6", alpha=0.95)
for c in splines:
    ax.plot(*X[c].T, color="tab:blue", lw=3)
for k in cables:
    ax.plot(*X[C[k]].T, color="tab:green", lw=1.5)
ax.set_box_aspect((np.ptp(V[:, 0]), np.ptp(V[:, 1]), np.ptp(V[:, 2]) * 1.5))
ax.view_init(28, -60); ax.set_axis_off()
ax.set_title("(a) result surface, splines blue, cables green")

# (b) deviation map
ax = fig.add_subplot(2, 3, 2)
t = ax.tripcolor(V[:, 0], V[:, 1], F, dv, shading="gouraud", cmap="magma_r")
for c in splines:
    ax.plot(*V[c, :2].T, color="tab:blue", lw=3)
for k in cables:
    ax.plot(*V[C[k], :2].T, color="tab:green", lw=1.2)
ax.plot(*V[fixed, :2].T, "k.", ms=4)
i = int(free[np.argmax(dv[free])]); ax.plot(*V[i, :2], "c*", ms=14)
plt.colorbar(t, ax=ax, label="distance to target (mm)")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"(b) RMSE {rmse:.2f} mm ({100*rmse/span:.2f}% of span), max {mx:.1f} mm "
             f"({100*mx/span:.2f}%)\nsupports black, max at star")

# (c, d) stretch factors per face
for k, (key, name) in enumerate([("sf_wale", "wale"), ("sf_course", "course")]):
    ax = fig.add_subplot(2, 3, 3 + 3 * k if k else 3)
    vals = np.asarray(best[key])[face_region]
    pc = PolyCollection(V[F][:, :, :2], array=vals, cmap="viridis", edgecolors="none")
    pc.set_clim(1.0, max(1.05, vals.max())); ax.add_collection(pc)
    for kk in cables:
        ax.plot(*V[C[kk], :2].T, color="w", lw=1)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(pc, ax=ax, label=f"sf_{name}")
    ax.set_title(f"({'cd'[k]}) {name} stretch factor per region "
                 f"({vals.min():.3f}-{vals.max():.3f})")

# (e) sections: target solid, result dashed
ax = fig.add_subplot(2, 3, (4, 5))
for y0, col in [(-0.35, "tab:blue"), (-0.6, "tab:orange"), (-0.85, "tab:green")]:
    m = np.abs(V[:, 1] - y0) < 0.02
    o = np.argsort(V[m, 0])
    ax.plot(V[m, 0][o] * 1e3, V[m, 2][o] * 1e3, "-", color=col, label=f"y = {y0} m, target")
    ax.plot(X[m, 0][o] * 1e3, X[m, 2][o] * 1e3, "--", color=col, label=f"y = {y0} m, result")
ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=3)
ax.set_aspect("equal")
ax.set_title("(e) sections across the shell (vertices within 20 mm of the plane)")

out = os.path.join(HERE, "figures", f"{prefix}.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print(out)
