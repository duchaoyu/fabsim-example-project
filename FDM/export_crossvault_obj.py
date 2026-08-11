"""Write the cross vault and its crease lines to a single OBJ.

The surface comes from the converged FDM result; the crease lines are the cable
trajectories extracted by figure_cable_extraction (same parameters, so the two stay
in step). Surface and cables share one vertex block, so a crease vertex is the same
point as the surface vertex it lies on — no snapping needed downstream.

Everything is grouped and named, so in Rhino the cables arrive as selectable groups
and in Blender as separate objects:

    o crossvault_surface     169 v, 168 f  (quads and tris, as the FDM mesh)
    o crease_01_diagonal     polyline via 'l'
    ...

    python FDM/export_crossvault_obj.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_cable_extraction as fce

OUT = os.path.join(fce.HERE, "data", "crossvault", "crossvault_creases.obj")


def main():
    V, F, E = fce.load()
    boundary = fce.boundary_vertices(F)
    apex = fce.apex_point(V)
    s2 = {frozenset((u, w)): fce.sin2_theta(V, apex, u, w) for u, w, _ in E}
    q_of = {frozenset((u, w)): qq for u, w, qq in E}
    res = fce.extract(V, E, s2, boundary, fce.LAMBDA, fce.Q_THRESHOLD)
    cables = fce.trace_cables(V, res["edges"], q_of, res["hi_edges"],
                              res["tip_routes"])

    keys = sorted(V)                       # OBJ is 1-indexed, in this order
    idx = {k: i + 1 for i, k in enumerate(keys)}
    P = np.array([V[k] for k in keys])
    lo, hi = P.min(axis=0), P.max(axis=0)

    def kind(c):
        both_anchored = (c["path"][0] in boundary and c["path"][-1] in boundary)
        return "diagonal" if both_anchored else "arch"

    lines = [
        "# Cross vault with cable crease lines",
        f"# source      {os.path.relpath(fce.RESULT, fce.HERE)}",
        f"# generated   FDM/{os.path.basename(__file__)}",
        f"# plan        {hi[0]-lo[0]:.3f} x {hi[1]-lo[1]:.3f} m, rise "
        f"{hi[2]-lo[2]:.3f} m (rise/span 1:{(hi[0]-lo[0])/(hi[2]-lo[2]):.2f})",
        f"# surface     {len(keys)} vertices, {len(F)} faces",
        f"# creases     {len(cables)} cables, "
        f"{sum(1 for c in cables if kind(c) == 'diagonal')} diagonal + "
        f"{sum(1 for c in cables if kind(c) == 'arch')} arch",
        f"# extraction  lambda={fce.LAMBDA:g}  mu={fce.ANCHOR_COST:g}  "
        f"q>={fce.Q_THRESHOLD:g}  theta={fce.THETA_MODE}  "
        f"ties_outward={fce.TIES_OUTWARD}",
        "# units are metres",
        "",
    ]
    for k in keys:
        x, y, z = V[k]
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    lines += ["", "o crossvault_surface", "g crossvault_surface"]
    for f in F:
        lines.append("f " + " ".join(str(idx[v]) for v in f))

    for i, c in enumerate(cables, 1):
        name = f"crease_{i:02d}_{kind(c)}"
        lines += ["", f"o {name}", f"g {name}",
                  f"# {c['edges']} segments, {c['length']:.4f} m, "
                  f"q {c['q_min']:.2f}-{c['q_max']:.2f}",
                  "l " + " ".join(str(idx[v]) for v in c["path"])]
    lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(lines))

    print(f"wrote {os.path.relpath(OUT, fce.HERE)}  "
          f"({os.path.getsize(OUT)/1024:.0f} kB)")
    print(f"  surface  {len(keys)} v, {len(F)} f")
    for i, c in enumerate(cables, 1):
        print(f"  crease_{i:02d}_{kind(c):8s} {c['edges']:>2} segments  "
              f"{c['length']:.4f} m")


if __name__ == "__main__":
    main()
