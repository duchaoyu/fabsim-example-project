"""Visualise a pattern_smooth FEM inverse result against the target.

Takes the best valid call from a calls log (so it works while a run is still
going) or a finished <prefix>_result.json, re-runs that one FEM solve, and
draws: the FEM surface over the target, |deviation|, vertical and in-plane
deviation in plan, and the parameters.

    python3 FDM/viz_pattern_smooth_fit.py optimisation/pattern_smooth_p3_calls.jsonl
    python3 FDM/viz_pattern_smooth_fit.py optimisation/pattern_smooth_p3_result.json
    python3 FDM/viz_pattern_smooth_fit.py <calls.jsonl> E00 [out.png]   # E00 supported
    ... --variant rm                     # the sketch remesh
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import optimise_pattern_smooth as ops

HERE = os.path.dirname(os.path.abspath(__file__))


def best_params(path):
    if path.endswith(".jsonl"):
        recs = [json.loads(l) for l in open(path) if l.strip()]
        recs = [r for r in recs if r.get("valid") and r.get("rmse_mm") is not None]
        r = min(recs, key=lambda r: r["rmse_mm"])
        return r, f"best of {len(recs)} valid calls so far (call {r['call']})"
    d = json.load(open(path))
    return d, f"final result, {d['n_calls']} calls"


REG_MAX = [0.0]


def main():
    if "--reg-max" in sys.argv:
        i = sys.argv.index("--reg-max")
        REG_MAX[0] = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    if "--variant" in sys.argv:
        i = sys.argv.index("--variant")
        ops.set_variant(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    src = sys.argv[1]
    r, note = best_params(os.path.join(HERE, src) if not os.path.isabs(src) else src)
    if r.get("variant"):
        ops.set_variant(r["variant"])
    if r.get("region_map"):
        ops.REGION_MAP = os.path.join(HERE, r["region_map"])
    if r.get("cable_file"):
        ops.CABLE_FILE = os.path.join(HERE, r["cable_file"])
    n_reg = len(r["sf_wale"])
    V, F = ops.o2.load_off(ops.MESH_PATH)
    cab = json.load(open(ops.CABLE_FILE))
    fix_edges = r.get("fixed_edges") or [e for e in (sys.argv[2] if len(sys.argv) > 2
                                                        else "").split(",") if e]
    fixed = sorted(json.load(open(ops.CABLE_META))["supports"])
    for e in fix_edges:
        fixed = sorted(set(fixed) | set(cab[e]))
    names = [k for k in sorted(cab) if k not in fix_edges]
    params = {"pressure": r.get("pressure", 1000.0), "motif": 1, "cable_ea": ops.CABLE_EA,
              **ops.MATERIAL, "cable_paths": [cab[k] for k in names],
              "fixed_vertices": fixed,
              "regions": [{"sf_wale": r["sf_wale"][i], "sf_course": r["sf_course"][i],
                           "knit_dir_deg": 0.0} for i in range(n_reg)],
              "cable_rest_scales": r["cable_rest_scales"]}
    # the solver cap the run used: a calls-log record does not carry it, so
    # fall back to the --reg-max argument, then to the result JSON's value
    reg = r.get("newton_reg_max") or REG_MAX[0]
    if reg:
        params["newton_reg_max"] = float(reg)
    tmp = tempfile.mkdtemp()
    pj = os.path.join(tmp, "p.json")
    json.dump(params, open(pj, "w"))
    res = subprocess.run([ops.BINARY, ops.MESH_PATH, ops.REGION_MAP, pj,
                          os.path.join(tmp, "o")], capture_output=True, text=True, check=True)
    status = [l.split()[1] for l in res.stderr.splitlines() if l.startswith("SOLVER_STATUS")]
    if not status or status[-1] != "success":
        sys.exit(f"FEM re-solve did not converge ({status[-1] if status else 'no status'}); "
                 f"not plotting a non-equilibrium shape. Pass --reg-max if the run used one.")
    X = np.loadtxt(os.path.join(tmp, "o_verts.csv"), delimiter=",", skiprows=1)[:, 1:]
    free = np.array(sorted(set(range(len(V))) - set(fixed)))
    d = X - V
    dn = np.linalg.norm(d, axis=1)
    rmse = float(np.sqrt(np.mean(dn[free] ** 2)))
    region = np.array(json.load(open(ops.REGION_MAP))["face_regions"])

    fig = plt.figure(figsize=(25, 10.5))
    fig.suptitle(f"pattern_smooth FEM inverse — RMSE {rmse*1e3:.1f} mm "
                 f"({100*rmse/np.ptp(V[:, 0]):.1f} % of span), max {dn[free].max()*1e3:.0f} mm, "
                 f"crown {X[:, 2].max():.3f} m (target {V[:, 2].max():.3f})   [{note}]",
                 fontsize=12)

    # 3D: FEM over target
    for k, (elev, azim) in enumerate([(24, -62), (24, 118)]):
        ax = fig.add_subplot(2, 4, 1 + 4 * k, projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection([V[f] for f in F], facecolors="#9fb9d6",
                                             alpha=0.25, edgecolors="none"))
        ax.add_collection3d(Poly3DCollection([X[f] for f in F], facecolors="#e8a38c",
                                             alpha=0.85, edgecolors="#b07060",
                                             linewidths=0.1))
        for n in names:
            p = X[cab[n]]
            ax.plot(*p.T, color="#b3261e" if n.startswith("C") else "#6b1f1a",
                    lw=1.8 if n.startswith("C") else 1.2)
        c = 0.5 * (V.min(0) + V.max(0))
        ax.set_xlim(c[0] - 0.6, c[0] + 0.6); ax.set_ylim(c[1] - 0.6, c[1] + 0.6)
        ax.set_zlim(0, 0.35); ax.set_box_aspect((1, 1, 0.35), zoom=1.2)
        ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
        ax.set_title("FEM (orange) over target (blue)" if k == 0 else "opposite view",
                     fontsize=10)

    def plan(ax, val, title, cmap, sym=False):
        kw = dict(vmin=-np.abs(val).max(), vmax=np.abs(val).max()) if sym else {}
        tc = ax.tripcolor(V[:, 0], V[:, 1], F, val, shading="gouraud", cmap=cmap, **kw)
        plt.colorbar(tc, ax=ax, shrink=0.8, label="mm")
        for n in names:
            ax.plot(*V[cab[n]][:, :2].T, "k-", lw=0.9)
        ax.plot(*V[fixed][:, :2].T, "o", color="#2a78d6", ms=3, ls="none")
        ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(title, fontsize=10)

    plan(fig.add_subplot(2, 4, 2), dn * 1e3, "|deviation| from target", "viridis")
    plan(fig.add_subplot(2, 4, 3), d[:, 2] * 1e3,
         "vertical deviation (FEM − target)", "RdBu_r", sym=True)
    plan(fig.add_subplot(2, 4, 6), np.linalg.norm(d[:, :2], axis=1) * 1e3,
         "in-plane deviation", "viridis")

    # the regions, coloured, each labelled with its stretch factors
    from matplotlib.collections import PolyCollection
    ax = fig.add_subplot(2, 4, 4)
    cmap = plt.get_cmap("tab10" if n_reg <= 10 else "tab20")
    ax.add_collection(PolyCollection([V[f][:, :2] for f in F],
                                     facecolors=[cmap(g % cmap.N) for g in region],
                                     edgecolors="white", linewidths=0.1, alpha=0.75))
    for n in names:
        ax.plot(*V[cab[n]][:, :2].T, "k-", lw=1.0)
    ax.plot(*V[fixed][:, :2].T, "o", color="#2a78d6", ms=3, ls="none")
    cen = V[F].mean(1)
    for i in range(n_reg):
        m = region == i
        # stagger alternate labels up / down so narrow strips do not overlap
        dy = (0.12 if i % 2 else -0.12) if n_reg > 6 else 0.0
        ax.text(cen[m, 0].mean(), cen[m, 1].mean() + dy,
                f"R{i}\nw {r['sf_wale'][i]:.3f}\nc {r['sf_course'][i]:.3f}",
                ha="center", va="center", fontsize=7 if n_reg > 6 else 9,
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
    ax.set_aspect("equal"); ax.autoscale(); ax.set_axis_off()
    ax.set_title(f"{n_reg} regions: stretch factors (w = wale, c = course)", fontsize=10)

    # stretch factors per region, as a profile
    ax = fig.add_subplot(2, 4, 7)
    xs = np.arange(n_reg)
    ax.plot(xs, r["sf_wale"], "o-", color="#b3261e", label="sf_wale")
    ax.plot(xs, r["sf_course"], "s-", color="#2a78d6", label="sf_course")
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([f"R{i}" for i in xs], fontsize=8)
    ax.set_ylabel("stretch factor"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("stretch factor per region (< 1 = slack rest shape)", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax = fig.add_subplot(2, 4, 8); ax.set_axis_off()
    rows = [f"R{i}  ({(region == i).sum():3d} faces)   sf_wale {r['sf_wale'][i]:.4f}   "
            f"sf_course {r['sf_course'][i]:.4f}" for i in range(n_reg)]
    rows += [""] + [f"{n}  rest scale {s:.4f}" for n, s in zip(names, r["cable_rest_scales"])]
    rows += ["", f"E1 {ops.MATERIAL['E1']:g}  E2 {ops.MATERIAL['E2']:g}  nu {ops.MATERIAL['nu']}"
                 f"   p {params['pressure']:g} Pa   EA {ops.CABLE_EA:g}",
             f"{len(fixed)} fixed vertices (blue)"
             + (f", incl. edge(s) {','.join(fix_edges)}" if fix_edges else "")
             + "; knit fixed from the field"]
    ax.text(0.0, 1.0, "\n".join(rows), va="top", family="monospace", fontsize=10 if len(rows) < 20 else 7.5)

    out = (sys.argv[3] if len(sys.argv) > 3 else
           os.path.join(ops.OUT_DIR, "pattern_smooth_fit.png"))
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"RMSE {rmse*1e3:.2f} mm  -> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
