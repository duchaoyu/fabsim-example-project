"""
Block A analysis: first-order sensitivities, the linearity check, and the
predicted crown-height spread that §6.1.3's measured deviation is compared to.

Three things come out of this:

1. The sensitivity table.  Per factor: the response at +/- one tolerance, the
   normalised sensitivity (elasticity, dimensionless and delta-free), and the
   half-range that goes into the §6.5.2 budget.

2. The linearity check, which is what makes Block A a *method* check rather than
   a result.  Asymmetry = |y+ + y- - 2*y0| / |y+ - y-| compares the second
   difference to the first.  Small asymmetry means the response is linear over the
   tolerance, which licenses (a) linearising in Blocks B-E and (b) rescaling these
   numbers when an estimated tolerance is replaced by a measured one.  Large
   asymmetry means the tolerance has to be right before the run, not after.

3. The predicted spread, as the root-sum-of-squares of the per-factor half-ranges.
   For a SCALAR output such as crown height, RSS is the correct first-order
   standard deviation whenever the factor errors are independent, and nothing
   about the shape of the response can change that.  For a FIELD NORM such as
   L_pos it is not: L_pos is the length of a sum of displacement fields, so it
   depends on the angles between them.  --check-overlap measures those angles as
   the pairwise cosine between the per-factor fields, and the two limits are
   opposite:

     orthogonal fields  -> L_pos is deterministic at the RSS; signs cannot cancel
     parallel fields    -> L_pos is |sum of signed contributions|, so its mean
                           falls to about 0.8 of the RSS and individual draws can
                           very nearly cancel

   So the cosine matrix predicts the *shape* of the Block D distribution for
   L_pos, not a correction to the crown-height budget.

Usage:
    python3 analyse_block_A.py [--measured-deviation-mm X] [--check-overlap]
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imperfection_config as cfg
from tolerances import TOLERANCES

# House style, matching sensitivity_analysis/visualization.py so the figures sit
# with the rest of the chapter.
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})
# Paul Tol bright pair, already the house colours (visualization.py:88).
# Checked for CVD separation: dE_OKLab 33 normal vision, 36 under deuteranopia
# and protanopia, against floors of 15 and 8.
C_PLUS, C_MINUS = "#0077BB", "#EE7733"
C_GREY = "#666666"

# Units and display scaling per output.
UNITS = {
    "L_pos":        ("mm",    1e3),
    "crown_height": ("mm",    1e3),
    "max_stress":   ("N/m",   1.0),
    "mean_stress":  ("N/m",   1.0),
    "H_apex":       ("1/m",   1.0),
}

# Outputs for which a signed central difference is meaningful.  L_pos is a
# distance from the baseline, so it is zero at the baseline and non-negative on
# both sides; a central difference of it would be nonsense.
SIGNED = ["crown_height", "max_stress", "H_apex"]


def load_runs(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        for k, v in list(r.items()):
            if k not in ("run", "factor", "mesh"):
                r[k] = float(v) if v not in ("", "nan") else float("nan")
        out[r["run"]] = r
    return out


def load_floor(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}


def sensitivities(runs):
    """Per-factor table rows."""
    base = runs["A0"]
    pairs = {}
    for run_id, factor, sign in cfg.BLOCK_A:
        if factor is None:
            continue
        pairs.setdefault(factor, {})[sign] = runs[run_id]

    table = []
    for factor in cfg.BLOCK_A_FACTORS:
        rp, rm = pairs[factor][+1], pairs[factor][-1]
        tol = TOLERANCES[factor]
        x0 = cfg.NOMINAL[factor]
        rel = tol.rel_at(x0)

        row = {"factor": factor,
               "nominal": x0,
               "delta_abs": tol.absolute(x0),
               "delta_rel": rel,
               "kind": "+".join(tol.kind),
               "status": tol.status}

        for out in SIGNED:
            y0, yp, ym = base[out], rp[out], rm[out]
            half = 0.5 * (yp - ym)
            second = yp + ym - 2.0 * y0
            row[f"{out}_base"]  = y0
            row[f"{out}_plus"]  = yp - y0
            row[f"{out}_minus"] = ym - y0
            row[f"{out}_half"]  = half
            row[f"{out}_elast"] = (half / y0) / rel if y0 != 0 and rel != 0 else np.nan
            row[f"{out}_asym"]  = abs(second) / abs(half) if half != 0 else np.nan

        lp, lm = rp["L_pos"], rm["L_pos"]
        row["L_pos_plus"]  = lp
        row["L_pos_minus"] = lm
        row["L_pos_mean"]  = 0.5 * (lp + lm)
        row["L_pos_asym"]  = abs(lp - lm) / (0.5 * (lp + lm)) if (lp + lm) else np.nan
        row["L_pos_shape_mean"] = 0.5 * (rp["L_pos_shape"] + rm["L_pos_shape"])
        table.append(row)
    return table


def overlap_matrix(runs, run_dir):
    """Pairwise cosine between the per-factor displacement fields.

    The field for a factor is the half-difference (u+ - u-)/2 of the deformed
    positions, i.e. the linear part of the response, with the second-order part
    differenced out.  A cosine near zero means the two factors move the surface in
    unrelated ways, so their contributions to L_pos cannot cancel; a cosine near
    +/-1 means they move it along the same mode, so contributions add or cancel
    algebraically and L_pos acquires a broad, right-skewed distribution.  See the
    module docstring: this bears on the field norm, not on the crown-height RSS.
    """
    import fem_runner
    fields = {}
    for factor in cfg.BLOCK_A_FACTORS:
        ids = {sign: rid for rid, f, sign in cfg.BLOCK_A if f == factor}
        paths = {s: os.path.join(run_dir, f"{ids[s]}_verts.csv") for s in (+1, -1)}
        if not all(os.path.exists(p) for p in paths.values()):
            return None, None
        Vp = fem_runner.read_verts(paths[+1])
        Vm = fem_runner.read_verts(paths[-1])
        fields[factor] = (0.5 * (Vp - Vm)).ravel()

    names = cfg.BLOCK_A_FACTORS
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            na, nb = np.linalg.norm(fields[a]), np.linalg.norm(fields[b])
            M[i, j] = float(fields[a] @ fields[b] / (na * nb)) if na and nb else np.nan
    return names, M


def print_tables(table, floor, measured_mm):
    est = [r["factor"] for r in table if r["status"] == "estimate"]

    print("=" * 92)
    print("BLOCK A — method check on the circular dome")
    print("=" * 92)
    print(f"nominal: s_wale={cfg.S_WALE_NOM}  s_course={cfg.S_COURSE_NOM}  "
          f"p={cfg.PRESSURE_NOM:.0f} Pa  E1={cfg.E1_NOM:.0f} N/m  "
          f"E2/E1={cfg.R_RATIO_NOM:.4f}  nu={cfg.NU_NOM}  R={cfg.R_BOUNDARY_NOM} m")
    print(f"baseline: h_crown={table[0]['crown_height_base'] * 1e3:.2f} mm  "
          f"sigma_max={table[0]['max_stress_base']:.1f} N/m  "
          f"H_apex={table[0]['H_apex_base']:.4f} 1/m")
    if floor:
        # The probe re-solves the baseline along a different continuation path.
        # It comes back identical to the 1e-8 m precision the solver prints, so
        # the floor is a bound rather than a measured value.
        lp_um = floor.get("L_pos_m", 0.0) * 1e6
        dh_um = floor.get("d_crown_height_m", 0.0) * 1e6
        print(f"numerical floor: L_pos {'< 0.01' if lp_um == 0 else f'= {lp_um:.3f}'} um, "
              f"crown {'< 0.01' if dh_um == 0 else f'= {dh_um:.3f}'} um "
              f"(continuation-path probe, bounded by the 1e-8 m output precision)")

    # ── Sensitivity table ─────────────────────────────────────────────────────
    print("\n" + "-" * 92)
    print("First-order response at +/- one tolerance")
    print("-" * 92)
    print(f"{'factor':10s} {'delta':>12s} {'dh(+)':>9s} {'dh(-)':>9s} "
          f"{'dh half':>9s} {'elast':>7s} {'asym':>6s} {'L_pos':>8s} {'L_asym':>7s}")
    print(f"{'':10s} {'':>12s} {'mm':>9s} {'mm':>9s} {'mm':>9s} {'-':>7s} "
          f"{'%':>6s} {'mm':>8s} {'%':>7s}")
    for r in sorted(table, key=lambda r: -abs(r["crown_height_half"])):
        print(f"{r['factor']:10s} {r['delta_abs']:12.5g} "
              f"{r['crown_height_plus'] * 1e3:9.3f} "
              f"{r['crown_height_minus'] * 1e3:9.3f} "
              f"{r['crown_height_half'] * 1e3:9.3f} "
              f"{r['crown_height_elast']:7.3f} "
              f"{100 * r['crown_height_asym']:6.1f} "
              f"{r['L_pos_mean'] * 1e3:8.3f} "
              f"{100 * r['L_pos_asym']:7.1f}")

    # ── Linearity verdict ─────────────────────────────────────────────────────
    worst = max(table, key=lambda r: r["crown_height_asym"])
    print("\n" + "-" * 92)
    print("Linearity over the tolerance")
    print("-" * 92)
    print(f"largest asymmetry: {worst['factor']} at "
          f"{100 * worst['crown_height_asym']:.1f}% of its own first difference")
    if worst["crown_height_asym"] < 0.15:
        print("  -> the response is linear over these tolerances.  Blocks B-E may")
        print("     linearise about the nominal point, and these numbers rescale")
        print("     linearly when an estimated tolerance is replaced by a measured one.")
    else:
        print("  -> the response is NOT linear over these tolerances.  The magnitudes")
        print("     must be right before the run; rescaling an estimated delta is not")
        print("     valid, and the offending factor needs a smaller-delta re-run.")

    # ── Spread budget ─────────────────────────────────────────────────────────
    halves = np.array([abs(r["crown_height_half"]) for r in table])
    rss    = float(np.sqrt(np.sum(halves ** 2)))
    worst_case = float(np.sum(halves))
    lp = np.array([r["L_pos_mean"] for r in table])
    lp_rss = float(np.sqrt(np.sum(lp ** 2)))

    print("\n" + "-" * 92)
    print("Predicted spread, all six factors at one tolerance")
    print("-" * 92)
    print(f"crown height, RSS (independent factors):  {rss * 1e3:8.3f} mm  "
          f"({100 * rss / table[0]['crown_height_base']:.2f}% of h)")
    print(f"crown height, worst case (all aligned):   {worst_case * 1e3:8.3f} mm  "
          f"({100 * worst_case / table[0]['crown_height_base']:.2f}% of h)")
    print(f"L_pos, RSS:                               {lp_rss * 1e3:8.3f} mm")

    print("\n" + "-" * 92)
    print("Reality check against §6.1.3")
    print("-" * 92)
    if measured_mm is None:
        print("  measured deviation not supplied (--measured-deviation-mm).")
        print(f"  When §6.1.3 gives a number, compare it against {rss * 1e3:.3f} mm.")
        print("  If the measurement is much larger, every later band is a loose")
        print("  lower bound and §6.5.2 must say so.")
    else:
        ratio = measured_mm / (rss * 1e3)
        print(f"  measured  {measured_mm:.3f} mm   predicted (RSS)  {rss * 1e3:.3f} mm")
        print(f"  ratio measured/predicted = {ratio:.2f}")
        if ratio > 2.0:
            print("  -> the tolerances explain only a fraction of the observed")
            print("     deviation.  Every later band is a loose LOWER bound, and")
            print("     §6.5.2 must state that.  The gap is either a tolerance not")
            print("     in this list or a model error.")
        elif ratio < 0.5:
            print("  -> the predicted spread exceeds what was measured: the assumed")
            print("     tolerances are pessimistic, most likely delta_s.")
        else:
            print("  -> the predicted spread is the same order as the measurement, so")
            print("     the tolerance list accounts for the observed deviation.")

    if est:
        print("\n" + "!" * 92)
        print(f"{len(est)} of {len(table)} tolerances are ESTIMATES, not measurements: "
              f"{', '.join(est)}")
        print("§6.5.2 has to declare this.  delta_s is the one that matters most —")
        print("it is the largest single contributor below, and it has no measurement.")
        print("!" * 92)

    return rss, worst_case, lp_rss


def make_figure(table, rss, measured_mm, overlap, path):
    order = sorted(table, key=lambda r: abs(r["crown_height_half"]))
    names = [cfg.FACTOR_LABELS[r["factor"]] for r in order]
    y = np.arange(len(order))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))

    # (a) tornado of crown-height response
    ax = axes[0, 0]
    dp = np.array([r["crown_height_plus"] for r in order]) * 1e3
    dm = np.array([r["crown_height_minus"] for r in order]) * 1e3
    ax.barh(y + 0.18, dp, height=0.34, color=C_PLUS, label=r"$+\delta$")
    ax.barh(y - 0.18, dm, height=0.34, color=C_MINUS, label=r"$-\delta$")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y, names)
    ax.set_xlabel(r"$\Delta h_\mathrm{crown}$ (mm)")
    ax.set_title("(a) crown height at one tolerance")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    # (b) position loss.  One series, so no legend: the axis names it.
    ax = axes[0, 1]
    lp = np.array([r["L_pos_mean"] for r in order]) * 1e3
    ax.barh(y, lp, height=0.6, color=C_PLUS)
    for yi, r in zip(y, order):
        ax.plot([r["L_pos_plus"] * 1e3, r["L_pos_minus"] * 1e3], [yi, yi],
                marker="|", markersize=6, linestyle="none", color=C_GREY)
    ax.set_yticks(y, names)
    ax.set_xlabel(r"$\mathcal{L}_\mathrm{pos}$ (mm)")
    ax.set_title(r"(b) surface deviation; ticks are $\pm\delta$ separately")
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    # (c) linearity check
    ax = axes[1, 0]
    asym = np.array([100 * r["crown_height_asym"] for r in order])
    ax.barh(y, asym, height=0.6,
            color=[C_MINUS if a > 15 else C_PLUS for a in asym])
    ax.axvline(15.0, color=C_GREY, linewidth=0.8, linestyle="--")
    ax.text(15.0, len(order) - 0.4, " 15%", color=C_GREY, fontsize=7,
            va="top", ha="left")
    ax.set_yticks(y, names)
    ax.set_xlabel("asymmetry, % of first difference")
    ax.set_title("(c) linearity over the tolerance")
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    # (d) either the field-overlap matrix or the spread budget
    ax = axes[1, 1]
    if overlap is not None:
        onames, M = overlap
        labels = [cfg.FACTOR_LABELS[n] for n in onames]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
        ax.set_title("(d) displacement-field cosine")
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        halves = np.array([abs(r["crown_height_half"]) for r in order]) * 1e3
        ax.barh(y, halves, height=0.6, color=C_PLUS)
        ax.axvline(rss * 1e3, color="black", linewidth=1.0,
                   label=f"RSS = {rss * 1e3:.2f} mm")
        if measured_mm is not None:
            ax.axvline(measured_mm, color=C_MINUS, linewidth=1.0, linestyle="--",
                       label=f"§6.1.3 measured = {measured_mm:.2f} mm")
        ax.set_yticks(y, names)
        ax.set_xlabel(r"$|\Delta h_\mathrm{crown}|$ half-range (mm)")
        ax.set_title("(d) spread budget")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(axis="x", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(cfg.DATA_DIR, "block_A.csv"))
    ap.add_argument("--measured-deviation-mm", type=float, default=None,
                    help="crown-height deviation measured in §6.1.3, for the "
                         "reality check")
    ap.add_argument("--check-overlap", action="store_true",
                    help="compute the displacement-field cosine matrix; needs "
                         "the runs to have been kept with --keep-verts")
    args = ap.parse_args()

    runs  = load_runs(args.runs)
    floor = load_floor(os.path.join(cfg.DATA_DIR, "block_A_floor.csv"))
    table = sensitivities(runs)

    rss, worst_case, lp_rss = print_tables(table, floor,
                                           args.measured_deviation_mm)

    overlap = None
    if args.check_overlap:
        overlap = overlap_matrix(runs, os.path.join(cfg.DATA_DIR, "runs_A"))
        if overlap[0] is None:
            print("\n(--check-overlap: vertex files absent; re-run run_block_A.py "
                  "with --keep-verts)")
            overlap = None
        else:
            onames, M = overlap
            off = np.abs(M - np.eye(len(onames)))
            i, j = np.unravel_index(np.argmax(off), M.shape)
            print("\n" + "-" * 92)
            print("Displacement-field overlap")
            print("-" * 92)
            print(f"most aligned pair: {onames[i]} / {onames[j]}  "
                  f"cosine = {M[i, j]:+.3f}")
            offdiag = M[~np.eye(len(onames), dtype=bool)]
            med = float(np.median(np.abs(offdiag)))
            print(f"median |cosine| off the diagonal: {med:.3f}")
            if med > 0.8:
                print("  -> the factors are nearly DEGENERATE on this geometry: they")
                print("     excite one and the same mode, differing in sign and")
                print("     amplitude only.  Two consequences.  (i) Block A cannot")
                print("     identify which factor moved a measured surface, only by")
                print("     how much — attribution needs the multi-region case study,")
                print("     where the fields separate.  (ii) L_pos contributions add")
                print("     and cancel algebraically, so the joint L_pos of Block D")
                print("     will be broad and right-skewed with a mean below the RSS,")
                print("     while the crown-height RSS is unaffected.")
            else:
                print("  -> the factors move the surface in largely distinct ways, so")
                print("     L_pos contributions cannot cancel and the RSS is a tight")
                print("     prediction of the joint spread.")

    os.makedirs(cfg.FIG_DIR, exist_ok=True)
    out_csv = os.path.join(cfg.DATA_DIR, "block_A_sensitivity.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    fig_path = make_figure(table, rss, args.measured_deviation_mm, overlap,
                           os.path.join(cfg.FIG_DIR, "blockA_sensitivity.pdf"))
    print(f"\nwrote {out_csv}")
    print(f"wrote {fig_path} (+ .png)")


if __name__ == "__main__":
    main()
