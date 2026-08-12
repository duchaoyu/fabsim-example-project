"""
Why the 2part equilibrium misses its target by 24.6 mm.

Block A treated that residual as a limitation of the uniform two-parameter
pre-strain model.  That explanation does not survive looking at where the target
came from.  The target is an FDM form-finding result, and force-density
form-finding assigns a stiffness per EDGE: in the final iteration
(mesh_out_2parts_20240802010532.json) q runs from 0.1 to 14.7, and the stiffest
decile is concentrated 9x on the line x = 0.  The target was form-found with a
stiffened tie along the valley between the two lobes.

The FEM runs carry no cable at all.  A pressurised membrane with no line element
cannot hold a crease — a crease needs a line load — so the mismatch is a MISSING
STRUCTURAL ELEMENT, not a parameterisation deficiency.  No refinement of the
stretch-factor field, per-region or otherwise, would close it: a smooth pre-strain
field cannot produce a kink.

This script makes that case in one figure by showing that the stiff-edge line and
the mismatch stripe are the same line.

    python3 diagnose_2part_target.py     # needs run_block_A.py --geometry 2part --keep-verts
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)

import imperfection_config as cfg
import geometry as geom
import fem_runner
import mesh_tools

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})
C_Q, C_DEV = "#0077BB", "#EE7733"

# The form-finding iteration that matches 2part_opt_simu_m.off most closely
# (mean vertex discrepancy 2.5 mm after the 0.1 scale; the others are 4-29 mm).
FDM_JSON = os.path.join(REPO, "FDM", "data",
                        "mesh_out_2parts_20240802010532.json")
FDM_SCALE = 0.1   # the OFF is the FDM mesh scaled by exactly 0.1, same indexing


def load_fdm_edges(path):
    """(edge index pairs, q per edge, vertex positions in OFF units)."""
    with open(path) as f:
        d = json.load(f)
    V = {int(k): v for k, v in d["vertex"].items()}
    P = np.array([[V[i]["x"], V[i]["y"], V[i]["z"]]
                  for i in range(len(V))]) * FDM_SCALE
    edges, q = [], []
    for k, v in d["edgedata"].items():
        if not isinstance(v, dict) or "qpre" not in v:
            continue
        a, b = (int(t) for t in k.strip("()").replace(" ", "").split(","))
        edges.append((a, b))
        q.append(float(np.ravel(v["qpre"])[0]))
    return np.array(edges), np.array(q), P


def main():
    g = geom.get("2part")
    V0, F = mesh_tools.load_off(g.mesh)
    V_target = V0
    base_p = os.path.join(cfg.DATA_DIR, "runs_A_2part", "A0_verts.csv")
    if not os.path.exists(base_p):
        sys.exit("run: python3 run_block_A.py --geometry 2part --keep-verts")
    V_base = fem_runner.read_verts(base_p)

    edges, q, P = load_fdm_edges(FDM_JSON)
    dev = (V_base[:, 2] - V_target[:, 2]) * 1e3          # signed, mm
    tri = Triangulation(V0[:, 0], V0[:, 1], F)

    # Enrichment of the stiff decile on the valley line, the number the argument
    # rests on.
    mid = 0.5 * (P[edges[:, 0]] + P[edges[:, 1]])
    hi = q > np.percentile(q, 90)
    band = 0.008                       # m, a narrow band about x = 0
    share_hi = float((np.abs(mid[hi, 0]) < band).mean())
    share_all = float((np.abs(mid[:, 0]) < band).mean())

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

    # (a) the form-finding stiffness field.  Magnitude over a domain -> one hue,
    # light to dark; log-scaled because q spans 147x.
    ax = axes[0]
    segs = np.stack([P[edges[:, 0], :2], P[edges[:, 1], :2]], axis=1)
    lc = LineCollection(segs, cmap="Blues", norm=matplotlib.colors.LogNorm(
        vmin=max(q.min(), 1e-2), vmax=q.max()), linewidths=1.4)
    lc.set_array(q)
    ax.add_collection(lc)
    ax.set_xlim(P[:, 0].min() * 1.05, P[:, 0].max() * 1.05)
    ax.set_ylim(P[:, 1].min() * 1.05, P[:, 1].max() * 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(a) FDM force density $q$ per edge\n"
                 "the target was form-found with a valley tie")
    fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="$q$ (log)")

    # (b) the mismatch the FEM baseline leaves.  Signed -> diverging.
    ax = axes[1]
    lim = np.abs(dev).max()
    im = ax.tripcolor(tri, facecolors=dev[F].mean(axis=1), cmap="RdBu_r",
                      vmin=-lim, vmax=lim, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(b) FEM baseline − target\n"
                 "no cable in the model: the crease never forms")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="signed dev. (mm)")

    # (c) both collapsed onto x.  Two quantities with different units, so both are
    # normalised to their own maximum and share one axis — never a second y-scale.
    ax = axes[2]
    bins = np.linspace(V0[:, 0].min(), V0[:, 0].max(), 26)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    q_prof = np.array([np.percentile(q[(mid[:, 0] >= a) & (mid[:, 0] < b)], 90)
                       if ((mid[:, 0] >= a) & (mid[:, 0] < b)).sum() > 3 else np.nan
                       for a, b in zip(bins[:-1], bins[1:])])
    d_prof = np.array([np.abs(dev[(V0[:, 0] >= a) & (V0[:, 0] < b)]).mean()
                       if ((V0[:, 0] >= a) & (V0[:, 0] < b)).sum() > 0 else np.nan
                       for a, b in zip(bins[:-1], bins[1:])])
    ax.plot(ctr, q_prof / np.nanmax(q_prof), color=C_Q, lw=1.6,
            marker="o", ms=3.5, label="edge stiffness $q$ (90th pct)")
    ax.plot(ctr, d_prof / np.nanmax(d_prof), color=C_DEV, lw=1.6,
            marker="s", ms=3.5, label="|baseline − target|")
    ax.axvline(0.0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("normalised to own maximum")
    ax.set_title("(c) the stiff line and the mismatch\n"
                 "are the same line")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    fig.suptitle("The 24.6 mm 2part mismatch is a missing valley tie, "
                 "not a parameterisation limit", fontsize=11)
    fig.tight_layout()

    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    path = os.path.join(cfg.FIG_DIR, "2part_target_diagnosis.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"FDM q: {q.min():.2f} to {q.max():.2f} over {len(q)} edges "
          f"({q.max() / q.min():.0f}x)")
    print(f"stiffest decile within {band * 1e3:.0f} mm of x=0: "
          f"{100 * share_hi:.0f}%  (all edges: {100 * share_all:.0f}%) "
          f"-> {share_hi / share_all:.1f}x enrichment")
    print(f"mismatch peaks at x = {ctr[np.nanargmax(d_prof)]:+.3f} m, "
          f"q peaks at x = {ctr[np.nanargmax(q_prof)]:+.3f} m")
    print(f"wrote {path} (+ .png)")


if __name__ == "__main__":
    main()
