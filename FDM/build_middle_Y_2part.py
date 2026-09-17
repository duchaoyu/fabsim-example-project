"""
Build the middle Y of the 2-part smooth shape as ONE clean cable: three simple
branches meeting at a single junction, each ending exactly once.

The extracted cables in cable_paths_2part.json are out-and-back walks, not
simple paths, so they cannot be used as-is:

  C14  v49 -> v40 -> v49      retraces itself; its 1.364 m is twice the real stem
  C00  v494 -> v370 -> v464   doubles back and leaves on a DIFFERENT boundary
                              vertex, which is the spurious second touch
  C05  v370 -> v49 -> v185    the only simple path of the three

So each branch is taken as a half-path from the junction v49 = (0, -0.368):

  stem   v49 -> v40   up the crease at x = 0
  left   v49 -> v185  out to the boundary at (-0.282, -0.530)
  right  v49 -> v464  the exact mirror of the left arm

v494 is dropped on purpose: it is the other end of C00's doubling-back, and
keeping it would give the right side two boundary touches and the left side one.

    .venv/bin/python FDM/build_middle_Y_2part.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "2part")
OUT_JSON = os.path.join(DATA, "cable_middle_Y_2part.json")
OUT_PNG  = os.path.join(DATA, "2part_middle_Y_clean.png")

L = open(os.path.join(DATA, "2parts_smooth_tri_m.off")).read().split("\n")
nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
cab = json.load(open(os.path.join(DATA, "cable_paths_2part.json")))

JUNCTION = 49
stem  = cab["C14"][0:12]                      # v49 .. v40, the forward half only
left  = cab["C05"][1:9]                       # v49 .. v185
right = [JUNCTION] + cab["C00"][6:0:-1] + [cab["C00"][12]]   # v49 .. v464

branches = {"stem": stem, "left": left, "right": right}


def plen(idx):
    return float(np.linalg.norm(np.diff(V[idx], axis=0), axis=1).sum())


# ── checks ───────────────────────────────────────────────────────────────────
bd_count = {}
for tri in F:
    for k in range(3):
        e = tuple(sorted((tri[k], tri[(k + 1) % 3])))
        bd_count[e] = bd_count.get(e, 0) + 1
boundary = {v for e, c in bd_count.items() if c == 1 for v in e}

print("branch   verts   length      start -> end                   end on boundary")
for nm, b in branches.items():
    assert len(set(b)) == len(b), f"{nm} revisits a vertex"
    assert b[0] == JUNCTION, f"{nm} does not start at the junction"
    a, z = V[b[0]], V[b[-1]]
    print(f"{nm:7s} {len(b):5d}  {plen(b):6.3f} m   "
          f"({a[0]:6.3f},{a[1]:6.3f}) -> ({z[0]:6.3f},{z[1]:6.3f})     "
          f"{b[-1] in boundary}")

shared = set(stem[1:]) & set(left[1:]) | set(stem[1:]) & set(right[1:]) \
         | set(left[1:]) & set(right[1:])
print(f"\nbranches share no vertex except the junction : {not shared}")
mirror_ok = np.allclose(V[left][:, 0], -V[right][:, 0], atol=1e-9) and \
            np.allclose(V[left][:, 1:], V[right][:, 1:], atol=1e-9)
print(f"left and right are exact mirrors in x        : {mirror_ok}")
total = sum(plen(b) for b in branches.values())
print(f"total cable length                           : {total:.3f} m")

json.dump({"junction_vertex": JUNCTION,
           "branches": {k: [int(i) for i in v] for k, v in branches.items()},
           "lengths_m": {k: plen(v) for k, v in branches.items()},
           "total_length_m": total,
           "note": "half-paths from the junction; the out-and-back duplicates "
                   "of C14/C00 and the spurious boundary end v494 are removed"},
          open(OUT_JSON, "w"), indent=2)
print(f"\nSaved {OUT_JSON}")

# ── figure ───────────────────────────────────────────────────────────────────
COL = {"stem": "#e34948", "left": "#2a78d6", "right": "#f2a93b"}
fig = plt.figure(figsize=(14.5, 5.8))

ax = fig.add_subplot(1, 3, 1)
ax.triplot(V[:, 0], V[:, 1], F, color="0.9", lw=0.3)
for k, p in cab.items():
    ax.plot(V[p][:, 0], V[p][:, 1], color="#dcdfe4", lw=1.1)
for nm, b in branches.items():
    ax.plot(V[b][:, 0], V[b][:, 1], color=COL[nm], lw=2.8, label=nm, zorder=3)
    ax.scatter(*V[b[-1], :2], s=42, color=COL[nm], ec="k", lw=0.6, zorder=5)
ax.scatter(*V[JUNCTION, :2], s=70, marker="D", color="k", zorder=6, label="junction")
ax.scatter(*V[494, :2], s=55, marker="x", color="#8a8f99", lw=1.8, zorder=5,
           label="v494 dropped")
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
ax.set_title("plan — one junction, three ends", fontsize=10)

ax = fig.add_subplot(1, 3, 2)
ax.triplot(V[:, 0], V[:, 1], F, color="0.93", lw=0.3)
for nm, b in branches.items():
    ax.plot(V[b][:, 0], V[b][:, 1], color=COL[nm], lw=3.0, zorder=3)
ax.scatter(*V[JUNCTION, :2], s=70, marker="D", color="k", zorder=6)
ax.set_xlim(-0.40, 0.40); ax.set_ylim(-0.60, -0.28)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("the fork, zoomed — both arms touch once", fontsize=10)

ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                color="0.85", alpha=0.35, edgecolor="none")
for nm, b in branches.items():
    ax.plot(V[b][:, 0], V[b][:, 1], V[b][:, 2], color=COL[nm], lw=3.2)
ax.scatter(*V[JUNCTION], s=60, marker="D", color="k")
ax.set_box_aspect([1, 1, 0.45]); ax.view_init(elev=28, azim=-70)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_title("on the surface", fontsize=10)

fig.suptitle(f"2-part smooth — the middle Y as one cable "
             f"({total:.3f} m: stem {plen(stem):.3f} + arms 2 x {plen(left):.3f})",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PNG}")
