"""
The continuous 2-part cable layout with the force densities shown: each cable
segment is coloured AND thickened by its edge q, over the whole FDM net drawn
the same way at low contrast.

    .venv/bin/python FDM/figure_2part_cable_forces.py
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")
OUT  = os.path.join(DATA, "2part_cables_continuous_forces.png")

L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])

fdm = json.load(open(os.path.join(DATA, "mesh_out_2part_smooth_latest.json")))
g = fdm.get("data", fdm)
qe = {}
for k, d in g["edgedata"].items():
    u, v = eval(k)
    qv = d.get("qpre", 1.0)
    qe[tuple(sorted((int(u), int(v))))] = float(qv[0] if isinstance(qv, (list, tuple)) else qv)

cab = json.load(open(os.path.join(DATA, "cable_paths_2part_continuous.json")))
q_all = np.array(list(qe.values()))
vmax = q_all.max()
print(f"net q: {q_all.min():.3f} .. {vmax:.3f}, median {np.median(q_all):.3f}")

PRESSURE_SCALE = 1000.0   # q is stored at the FDM's PRESSURE = 1.0


def seg_q(path):
    return np.array([qe.get(tuple(sorted((path[i], path[i + 1]))), 0.0)
                     for i in range(len(path) - 1)])


fig, axes = plt.subplots(1, 3, figsize=(19, 6.3),
                         gridspec_kw={"width_ratios": [1, 1, 0.9]})

# ── the whole net, thickness and colour by q ─────────────────────────────────
ax = axes[0]
segs, ws, cs = [], [], []
for (u, v), q in qe.items():
    segs.append([V[u, :2], V[v, :2]]); ws.append(0.25 + 3.2 * (q / vmax)); cs.append(q)
lc = LineCollection(segs, linewidths=ws, array=np.array(cs), cmap="inferno_r",
                    norm=plt.Normalize(0, vmax))
ax.add_collection(lc)
ax.set_aspect("equal"); ax.autoscale(); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("the whole FDM net\nwidth and colour = q", fontsize=10)

# ── the cables alone, same encoding ──────────────────────────────────────────
ax = axes[1]
ax.triplot(V[:, 0], V[:, 1], F, color="0.93", lw=0.25, zorder=0)
segs, ws, cs = [], [], []
for name, p in cab.items():
    q = seg_q(p)
    for i in range(len(p) - 1):
        segs.append([V[p[i], :2], V[p[i + 1], :2]])
        ws.append(1.0 + 6.0 * (q[i] / vmax)); cs.append(q[i])
lc = LineCollection(segs, linewidths=ws, array=np.array(cs), cmap="inferno_r",
                    norm=plt.Normalize(0, vmax), zorder=3)
ax.add_collection(lc)
for name, p in cab.items():
    ax.scatter(V[p][[0, -1], 0], V[p][[0, -1], 1], s=40, color="#2a78d6",
               ec="k", lw=0.5, zorder=5)
ax.set_aspect("equal"); ax.autoscale(); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"the {len(cab)} continuous cables\nblue = boundary anchors", fontsize=10)
cb = fig.colorbar(lc, ax=axes[:2], fraction=0.021, pad=0.02)
cb.set_label("q  (at the FDM's pressure = 1.0)")

# ── q along each cable ───────────────────────────────────────────────────────
ax = axes[2]
COL = {"K0": "#e34948", "K1": "#f2a93b", "K2": "#2a78d6", "K3": "#3fa46a"}
for name, p in cab.items():
    P = V[p]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    q = seg_q(p)
    ax.step(0.5 * (s[:-1] + s[1:]), q, where="mid", color=COL[name], lw=1.8,
            label=f"{name}  mean {q.mean():.2f}  min {q.min():.2f}")
ax.axhline(np.median(q_all), color="0.5", ls="--", lw=1,
           label=f"net median {np.median(q_all):.2f}")
ax.set_xlabel("arc length along the cable, m"); ax.set_ylabel("q")
ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
ax.set_title("q along each cable, anchor to anchor", fontsize=10)

fig.suptitle("2-part smooth — continuous cable layout with force densities "
             f"(x{PRESSURE_SCALE:.0f} for q in N/m at 1000 Pa)", fontsize=12)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("Saved", OUT)
for name, p in cab.items():
    q = seg_q(p)
    print(f"  {name}: q mean {q.mean():.3f}  min {q.min():.3f}  max {q.max():.3f}   "
          f"at 1000 Pa: {q.mean()*PRESSURE_SCALE:7.1f} N/m mean")
