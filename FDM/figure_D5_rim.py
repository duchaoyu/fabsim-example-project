"""
Figure: support of the opening rim on the shell with opening (D5).

The published D5 fits clamp the whole boundary, the opening rim included, so
the inner cable carries nothing.  Each column here is a fit with a different
rim, re-optimised for that rim (optimise_D5_symmetric.py --rim ...), and
re-solved at its optimum:

  clamped : rim fixed (the published model, d5_sym_pfix)
  cable   : rim free, inner cable only
  ring<d> : rim free, GFRP rod of d mm formed to the rim

Row 1: distance to the target surface (top view).  Row 2: the rim seen
through the opening (x-z), target vs result.  Row 3: the mid-section x = 0
(y-z), target vs result.

Usage: python3 figure_D5_rim.py [--cases clamped cable ring10 ring20]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import optimise_D5_symmetric as S
from check_D5_free_rim import run, OUT, RMAP, E_GFRP, rim_order_from_ground
from deviation_tools import closest_dist
from figure_case_results import SEQ, INK, INK2, CABLE

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.join(HERE, "optimisation")
FIG = os.path.join(HERE, "figures", "D5_rim")
LABEL = {"clamped": "rim clamped (published model)", "cable": "rim free, cable only"}


def section(X, F, x0=0.0):
    """Points where the mesh edges cross the plane x = x0, as (y, z) sorted by y."""
    E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    a, b = X[E[:, 0]], X[E[:, 1]]
    s = (x0 - a[:, 0]) / np.where(np.abs(b[:, 0] - a[:, 0]) < 1e-12, np.nan, b[:, 0] - a[:, 0])
    m = (s >= 0) & (s <= 1)
    P = a[m] + s[m, None] * (b[m] - a[m])
    return P[np.argsort(P[:, 1])][:, 1:]


def gaps(P, h=0.1):
    """Split a sorted section polyline where consecutive points are > h apart."""
    cut = np.where(np.linalg.norm(np.diff(P, axis=0), axis=1) > h)[0] + 1
    return np.split(P, cut)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["clamped", "cable", "ring10", "ring20"])
    args = ap.parse_args()

    V, F = S.load_off(S.MESH)
    field = json.load(open(S.FIELD_J))
    cable = json.load(open(S.CABLE_J))["vertex_indices"]
    rim = rim_order_from_ground(cable, V)
    fixed = sorted(set(int(v) for v in np.where(V[:, 2] < 1e-6)[0]) | {rim[0]})
    rmap = json.load(open(RMAP))
    rmap["face_knit_dirs_deg"] = [
        float(np.degrees(np.arctan2(field[str(f)]["d1"][1], field[str(f)]["d1"][0])) % 180)
        if str(f) in field else 0.0 for f in range(len(F))]
    os.makedirs(OUT, exist_ok=True)
    rmap_path = os.path.join(OUT, "rmap.json")
    json.dump(rmap, open(rmap_path, "w"))
    interior = np.hypot(V[:, 0], V[:, 1]) <= np.hypot(V[:, 0], V[:, 1]).max() * 0.98

    results = {}
    for c in args.cases:
        jf = os.path.join(OPT, "d5_sym_pfix_optimised.json" if c == "clamped"
                          else f"d5_rim_{c}_optimised.json")
        best = json.load(open(jf))
        p = {"pressure": S.PRESSURE, "motif": 5, "newton_reg_max": 1e6,
             "regions": [{k: r[k] for k in ("sf_wale", "sf_course", "knit_dir_deg")}
                         for r in best["regions"]]}
        if c in ("clamped", "cable"):
            p.update(cable_ea=S.CABLE_EA, cable_paths=[cable + [cable[0]]],
                     cable_rest_scales=[S.CABLE_SCALE])
        else:
            d = float(c[4:]) / 1000.0
            p.update(spline_paths=[rim], spline_EA=E_GFRP * np.pi * d**2 / 4,
                     spline_EI=E_GFRP * np.pi * d**4 / 64, spline_rest=1)
        if c != "clamped":
            p["fixed_vertices"] = fixed
        X, res, _ = run(f"fig_{c}", p, S.MESH, rmap_path, follower=(c != "clamped"))
        u = np.linalg.norm(X - V, axis=1) * 1000
        d = closest_dist(X, V, F) * 1000
        results[c] = dict(X=X, d=d, res=res,
                          rmse_mm=float(np.sqrt(np.mean(u[interior]**2))),
                          rim_max_mm=float(u[cable].max()), max_mm=float(d.max()),
                          sf_wale=[r["sf_wale"] for r in best["regions"][:best["n_pairs"]]],
                          sf_course=[r["sf_course"] for r in best["regions"][:best["n_pairs"]]],
                          n_calls=best.get("n_calls"))
        print(f"{c:8s} res {res:.1e}  RMSE {results[c]['rmse_mm']:5.2f} mm  "
              f"rim max {results[c]['rim_max_mm']:5.1f} mm  closest max {d.max():5.1f} mm")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "text.color": INK})
    n = len(args.cases)
    fig, axes = plt.subplots(3, n, figsize=(3.6 * n, 8.6),
                             gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    vmax = max(5.0, np.ceil(max(r["d"].max() for r in results.values()) / 5) * 5)
    tri = Triangulation(V[:, 0], V[:, 1], F)
    rimc = rim
    for j, c in enumerate(args.cases):
        r, X = results[c], results[c]["X"]
        ax = axes[0, j]
        tp = ax.tripcolor(tri, r["d"], cmap=SEQ, vmin=0, vmax=vmax, shading="gouraud")
        ax.plot(X[rimc, 0], X[rimc, 1], color=CABLE if c == "cable" else INK,
                lw=1.2 if c != "clamped" else 0.6)
        ax.set_aspect("equal"); ax.axis("off")
        name = LABEL.get(c, f"rim free, {c[4:]} mm GFRP ring")
        ax.set_title(f"{name}\nRMSE {r['rmse_mm']:.2f} mm, max {r['max_mm']:.1f} mm\n"
                     f"rim max {r['rim_max_mm']:.1f} mm", loc="left", fontsize=8)

        ax = axes[1, j]
        ax.plot(V[rimc, 0] * 1000, V[rimc, 2] * 1000, color=INK2, lw=1, ls="--", label="target")
        ax.plot(X[rimc, 0] * 1000, X[rimc, 2] * 1000, color=CABLE, lw=1.4, label="result")
        ax.set_aspect("equal"); ax.set_xlim(-200, 200); ax.set_ylim(-5, 230)
        ax.set_xlabel("x (mm)", color=INK2)
        if j == 0:
            ax.set_ylabel("z (mm)", color=INK2); ax.legend(frameon=False, fontsize=7)
            ax.set_title("rim, seen through the opening", loc="left", fontsize=8)

        ax = axes[2, j]
        T, R = section(V, F), section(X, F)
        for P, kw in ((T, dict(color=INK2, lw=1, ls="--")), (R, dict(color=CABLE, lw=1.4))):
            for seg in gaps(P):              # no line across the opening
                ax.plot(seg[:, 0] * 1000, seg[:, 1] * 1000, **kw)
        ax.set_aspect("equal"); ax.set_ylim(-5, 230)
        ax.set_xlabel("y (mm)", color=INK2)
        if j == 0:
            ax.set_ylabel("z (mm)", color=INK2)
            ax.set_title("mid-section x = 0 (opening on the left)", loc="left", fontsize=8)
        for a in axes[1:, j]:
            a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7)

    cb = fig.colorbar(tp, ax=axes[0, :], orientation="horizontal", fraction=0.04, pad=0.03,
                      aspect=60)
    cb.set_label("distance to target surface (mm)", color=INK2)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "D5_rim_support")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    json.dump({c: {k: v for k, v in r.items() if k not in ("X", "d")} for c, r in results.items()},
              open(out + ".json", "w"), indent=1)
    print("->", out + ".png")


if __name__ == "__main__":
    main()
