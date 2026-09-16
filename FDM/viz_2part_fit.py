"""Visualise a 2-part FEM best-fit: the fitted surface coloured by deviation
from the target, plus the target and the cable network for reference.

    python3 FDM/viz_2part_fit.py optimisation/2part_p2_result.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))


def load_off(path):
    L = [l for l in open(path).read().split("\n") if l.strip()]
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def load_obj(path):
    V, F = [], []
    for l in open(path):
        if l.startswith("v "):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in l.split()[1:4]])
    return np.array(V), np.array(F)


def main():
    res_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "optimisation", "2part_p2_result.json")
    if not os.path.isabs(res_path):
        res_path = os.path.join(HERE, res_path)
    R = json.load(open(res_path))
    prefix = os.path.basename(res_path).replace("_result.json", "")
    obj_path = os.path.join(HERE, "optimisation", f"{prefix}_best_fit.obj")
    out_png = os.path.join(HERE, "data", "2part", f"{prefix}_fit.png")

    Vt, F = load_off(os.path.join(HERE, R["mesh"]))
    Vf, _ = load_obj(obj_path)
    cables = json.load(open(os.path.join(HERE, R["cable_file"])))
    dev_mm = np.linalg.norm(Vf - Vt, axis=1) * 1000.0
    face_dev = dev_mm[F].mean(axis=1)
    vmax = float(np.percentile(dev_mm, 99))

    fig = plt.figure(figsize=(16.5, 5.8), facecolor="#fcfcfb")

    ax = fig.add_subplot(1, 3, 1)
    tp = ax.tripcolor(Vt[:, 0], Vt[:, 1], F, facecolors=face_dev,
                      cmap="magma_r", vmin=0, vmax=vmax)
    for name, p in sorted(cables.items()):
        ax.plot(Vt[p, 0], Vt[p, 1], color="#2a78d6", lw=1.1, alpha=0.85)
    ax.set_aspect("equal"); ax.set_axis_off()
    plt.colorbar(tp, ax=ax, shrink=0.75, label="deviation from target, mm")
    ax.set_title(f"a  plan, deviation |FEM - target|\n"
                 f"interior RMSE {R['rmse_interior_mm']:.2f} mm, "
                 f"max {R['max_interior_deviation_mm']:.2f} mm",
                 fontsize=10, loc="left")

    ax = fig.add_subplot(1, 3, 2, projection="3d", computed_zorder=False)
    ax.set_proj_type("ortho")
    cmap = plt.get_cmap("magma_r")
    cols = cmap(np.clip(face_dev / max(vmax, 1e-9), 0, 1))
    ax.add_collection3d(Poly3DCollection([Vf[t] for t in F], facecolors=cols,
                                         edgecolors="none", alpha=0.98))
    # Line3DCollection wants equal-length segments, so a polyline is split into
    # its edges rather than passed whole
    ax.add_collection3d(Line3DCollection(
        [[Vf[a], Vf[b]] for p in cables.values() for a, b in zip(p[:-1], p[1:])],
        colors="#2a78d6", linewidths=1.4))
    ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.62, 0.62); ax.set_zlim(0, 0.42)
    ax.set_box_aspect((1, 1, 0.42), zoom=1.3)
    ax.view_init(elev=24, azim=-62)
    ax.set_axis_off()
    ax.set_title(f"b  the fitted FEM surface\ncrown {R['crown_m']:.4f} m "
                 f"(target {R['target_crown_m']:.4f} m)", fontsize=10, loc="left")

    ax = fig.add_subplot(1, 3, 3)
    # sections across the crease (y = const) and along it (x = 0)
    for yv, c in [(-0.30, "#e34948"), (0.0, "#2a78d6"), (0.30, "#3a8f4a")]:
        m = np.abs(Vt[:, 1] - yv) < 0.035
        o = np.argsort(Vt[m, 0])
        ax.plot(Vt[m, 0][o], Vt[m, 2][o], color=c, lw=2.0, alpha=0.45)
        ax.plot(Vf[m, 0][o], Vf[m, 2][o], color=c, lw=1.2, ls="--")
        ax.plot([], [], color=c, lw=1.8, label=f"y = {yv:+.2f} m")
    ax.set_xlabel("x, m"); ax.set_ylabel("z, m")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("c  sections across the crease\nthick = target, dashed = FEM fit",
                 fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=165, facecolor="#fcfcfb")
    print(f"wrote {os.path.relpath(out_png, HERE)}")
    print(f"  interior RMSE {R['rmse_interior_mm']:.3f} mm   "
          f"max interior dev {R['max_interior_deviation_mm']:.3f} mm   "
          f"max disp {R['max_disp_from_rest_mm']:.2f} mm "
          f"(floor {R['min_disp_floor_mm']:.2f}, at_floor={R['at_floor']})")


if __name__ == "__main__":
    main()
