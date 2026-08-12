"""
What the imperfection does to the surface, rather than to a scalar.

The sensitivity figure says how far the shape moved; this one says where.  That
matters for the 2part in particular, because its two lobes and the saddle between
them do not respond alike, and a single RMS number hides that.

Panels:
  (a) the baseline equilibrium, coloured by signed deviation from the design
      target — the model's own standing mismatch, before any imperfection
  (b) the same surface coloured by |deviation from baseline| under the dominant
      factor at +delta — the imperfection's own footprint
  (c) section through the lobes: target, baseline, and the dominant factor +/- delta
  (d) section through the saddle, same four curves

Needs the runs kept:
    python3 run_block_A.py --geometry 2part --keep-verts
    python3 plot_shape.py --geometry 2part
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imperfection_config as cfg
import geometry as geom
import fem_runner
import mesh_tools

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
C_TARGET, C_BASE = "#333333", "#0077BB"
C_PLUS, C_MINUS  = "#EE7733", "#009988"


def section(tri_rest, z, axis, level, extent, n=400):
    """Profile of `z` along the straight cut {axis = level}, in rest coordinates.

    Interpolated on the rest-mesh triangulation rather than gathered from a slab
    of vertices.  A slab has to be wide enough to catch vertices on an
    unstructured 341-vertex mesh, and at that width it straddles the saddle:
    vertices from either side of it get sorted by the along-cut coordinate and the
    profile comes out as a sawtooth that is an artefact of the sampling, not a
    feature of the surface.  Interpolating on a line has no such width.

    The cut is taken in rest (material) coordinates, so the same material line is
    compared across configurations — which is what makes the four curves in one
    panel commensurable.
    """
    from matplotlib.tri import LinearTriInterpolator
    t = np.linspace(*extent, n)
    pts = np.empty((n, 2))
    pts[:, axis] = level
    pts[:, 1 - axis] = t
    zi = LinearTriInterpolator(tri_rest, z)(pts[:, 0], pts[:, 1])
    return t, zi, 1 - axis


def face_values(F, vertex_values):
    """Per-face means, since plot_trisurf and tripcolor colour faces, not
    vertices; handing them a per-vertex array silently produces a flat surface."""
    return vertex_values[F].mean(axis=1)


def dominant_factor(sens_csv):
    """The factor with the largest L_pos, read from the sensitivity table."""
    with open(sens_csv) as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda r: float(r["L_pos_mean"]))["factor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="2part", choices=geom.NAMES)
    ap.add_argument("--band", type=float, default=0.05,
                    help="half-thickness of the section slab, metres")
    args = ap.parse_args()

    g = geom.get(args.geometry)
    run_dir = os.path.join(cfg.DATA_DIR, f"runs_A_{g.name}")
    sens = os.path.join(cfg.DATA_DIR, f"block_A_{g.name}_sensitivity.csv")
    factor = dominant_factor(sens)
    ids = {sign: rid for rid, f, sign in cfg.BLOCK_A if f == factor}

    V0, F = mesh_tools.load_off(g.mesh)
    need = {"A0": None, ids[+1]: None, ids[-1]: None}
    for rid in need:
        p = os.path.join(run_dir, f"{rid}_verts.csv")
        if not os.path.exists(p):
            sys.exit(f"{p} missing — re-run run_block_A.py with --keep-verts")
        need[rid] = fem_runner.read_verts(p)
    V_base, V_plus, V_minus = need["A0"], need[ids[+1]], need[ids[-1]]
    V_target = mesh_tools.load_off(g.target)[0] if g.target else V_base

    # Rest coordinates are the common domain for every panel: the deviation fields
    # are plotted in plan view over the rest mesh, and the sections are cut in it.
    tri = Triangulation(V0[:, 0], V0[:, 1], F)
    dev_target = (V_base[:, 2] - V_target[:, 2]) * 1e3          # signed, mm
    dev_pert   = np.linalg.norm(V_plus - V_base, axis=1) * 1e3  # magnitude, mm

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))

    # Plan view rather than a 3D render: the job of (a) and (b) is to show a
    # magnitude over a spatial domain, and a heatmap reads that off directly where
    # a perspective surface hides half the field behind its own relief.  z contours
    # carry the shape context that the 3D view would have provided.
    def field_panel(ax, values, cmap, label, title, symmetric):
        lim = np.abs(values).max()
        clim = (-lim, lim) if symmetric else (0, lim)
        im = ax.tripcolor(tri, facecolors=face_values(F, values), cmap=cmap,
                          vmin=clim[0], vmax=clim[1], edgecolors="none")
        cs = ax.tricontour(tri, V_base[:, 2] * 1e3, levels=8, colors="#444444",
                           linewidths=0.4, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    field_panel(axes[0, 0], dev_target, "RdBu_r", "signed dev. (mm)",
                "(a) baseline − design target,\nwith baseline z contours (mm)",
                symmetric=True)
    field_panel(axes[0, 1], dev_pert, "YlOrRd", "|displacement| (mm)",
                f"(b) {factor} $+\\delta$ displacement\nfrom baseline",
                symmetric=False)

    # Sections: one through the highest point, one through the plane of symmetry.
    # On the 2part those are the lobe line and the saddle.
    apex = V0[np.argmax(V_target[:, 2])]
    xr = (V0[:, 0].min(), V0[:, 0].max())
    yr = (V0[:, 1].min(), V0[:, 1].max())
    cuts = [(1, float(apex[1]), xr, "through the lobes"),
            (0, 0.0, yr, "through the saddle")]
    for k, (axis, level, extent, label) in enumerate(cuts):
        ax = axes[1, k]
        for V, colour, lw, ls, name in (
                (V_target, C_TARGET, 1.6, "-",  "design target"),
                (V_base,   C_BASE,   1.4, "-",  "baseline"),
                (V_plus,   C_PLUS,   1.0, "--", f"{factor} $+\\delta$"),
                (V_minus,  C_MINUS,  1.0, ":",  f"{factor} $-\\delta$")):
            t, zi, other = section(tri, V[:, 2], axis, level, extent)
            ax.plot(t, zi * 1e3, color=colour, lw=lw, ls=ls, label=name)
        ax.set_xlabel(f"{'xy'[1 - axis]} (m), rest coordinate")
        ax.set_ylabel("z (mm)")
        ax.set_title(f"({'cd'[k]}) section {label}  "
                     f"({'xy'[axis]} = {level:+.3f} m)")
        ax.grid(linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        if k == 0:
            ax.legend(frameon=False, loc="lower center", ncol=2)

    fig.suptitle(f"Block A on {g.name}: where the tolerance shows up, not just "
                 f"how much", fontsize=11)
    fig.tight_layout()

    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    path = os.path.join(cfg.FIG_DIR, f"blockA_{g.name}_shape.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"dominant factor: {factor}")
    print(f"standing mismatch with target: "
          f"{np.sqrt(np.mean(dev_target ** 2)):.2f} mm RMS, "
          f"{np.abs(dev_target).max():.2f} mm peak")
    print(f"{factor} +delta displacement:  "
          f"{np.sqrt(np.mean(dev_pert ** 2)):.2f} mm RMS, "
          f"{dev_pert.max():.2f} mm peak")

    # Do the design error and the fabrication error live in the same mode?  If the
    # imperfection displaced the surface along the mismatch, tightening the
    # tolerance would move the shape toward its target; if the two are orthogonal,
    # no tolerance in this list can address the mismatch at all.  Compared as
    # vertex-displacement fields over the free vertices, both signed and in 3D.
    free = mesh_tools.interior_mask(V0, F)
    u_mismatch = (V_base - V_target)[free].ravel()
    u_pert     = (V_plus - V_base)[free].ravel()
    cos = float(u_mismatch @ u_pert
                / (np.linalg.norm(u_mismatch) * np.linalg.norm(u_pert)))
    print(f"\ncosine(mismatch field, {factor} displacement field) = {cos:+.3f}")
    if abs(cos) < 0.35:
        print("  -> nearly orthogonal: the fabrication tolerance and the design")
        print("     mismatch occupy different modes.  Tightening this tolerance")
        print("     moves the surface, but not toward its target — the mismatch")
        print("     needs a richer parameterisation, not better fabrication.")
    else:
        print("  -> substantially aligned: this tolerance moves the surface along")
        print("     the same mode as the standing mismatch, so controlling it does")
        print("     bear on how closely the target is met.")
    print(f"\nwrote {path} (+ .png)")


if __name__ == "__main__":
    main()
