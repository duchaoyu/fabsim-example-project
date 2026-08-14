"""Block A at a range of tolerance magnitudes, not just at one.

Block A asks what each factor does at plus and minus one assumed tolerance. Five
of the six tolerances are estimates, so the more useful question is what happens
when an estimate turns out to be half, or twice, what was assumed. This sweeps
each factor over a range of multiples of its assumed delta and records the same
metrics, so the answer can be read off a curve instead of rescaled by hand -
which matters because the +/- responses are not perfectly symmetric.

It reuses run_block_A.run_params and run_block_A.metrics, so the sweep and the
Block A table cannot disagree: at a multiple of 1.0 it reproduces the table.

    python3 run_tolerance_sweep.py --geometry disc
    python3 run_tolerance_sweep.py --geometry 2part

Writes data/tolerance_sweep_<geometry>.csv.
"""

import argparse
import csv
import os

import numpy as np

import geometry as geom
import imperfection_config as cfg
import mesh_tools
import fem_runner
from run_block_A import (measured_radius, metrics, nominal_dict, run_params)
from tolerances import TOLERANCES

MULTIPLES = (0.25, 0.5, 1.0, 1.5, 2.0)

# --percent mode: the same fractional error applied to every parameter, so the
# six are compared on a common footing rather than each at its own assumed
# tolerance. This is the axis that shows R to be the most damaging factor per
# unit relative error, which its 0.33% tolerance hides.
PERCENTS = (0.25, 0.5, 1.0, 1.5, 2.0)

# --absolute mode: the two geometric parameters in millimetres, which is the
# unit they are actually specified and measured in. A boundary ring is anchored
# to a distance and a cable is cut to a length; percent of nominal is a detour
# through a number nobody sets.
ABS_MM = (2.5, 5.0, 10.0, 15.0, 20.0)
ABS_FACTORS = ("R", "cable_L")


def sweep_params(g, nom, factor, mult):
    """As run_params, but at a fractional multiple of the tolerance.

    Tol.perturb scales linearly in `sign`, so a fractional sign is a fractional
    tolerance. The radius case needs its own mesh per magnitude, hence the tag.
    """
    if factor != "R":
        return run_params(g, nom, factor, mult)

    val = TOLERANCES["R"].perturb(nom["R"], mult)
    tag = f"{g.name}_R{'p' if mult > 0 else 'm'}{abs(mult):g}".replace(".", "")
    mesh, _ = mesh_tools.radius_variant(
        g.mesh, val / nom["R"], cfg.MESH_DIR, tag)
    return dict(g.params()), mesh


PARAM_KEY = {"s_wale": "sf_wale", "s_course": "sf_course",
             "pressure": "pressure", "E1": "E1", "r": "r_ratio", "nu": "nu"}

# Tolerance-table key for factors whose study name differs.
TOL_KEY = {"cable_L": "rho"}


def factors_for(g):
    """The six of Block A, plus Poisson's ratio, plus the cable if there is one.

    Block A left nu out because it enters the case study from Block B, and the
    cable out because its geometries had none. Neither reason applies once the
    case study is run as designed, and both are cheap to include.
    """
    fs = list(cfg.BLOCK_A_FACTORS) + ["nu"]
    if g.cable is not None:
        fs.append("cable_L")
    return fs


def absolute_params(g, nom, factor, mm):
    """Perturb a geometric factor by mm millimetres of its own length."""
    p = dict(g.params())
    d = mm / 1000.0
    if factor == "R":
        tag = f"{g.name}_mm{'p' if mm > 0 else 'm'}{abs(mm):g}".replace(".", "")
        mesh, _ = mesh_tools.radius_variant(
            g.mesh, (nom["R"] + d) / nom["R"], cfg.MESH_DIR, tag)
        return p, mesh
    if factor == "cable_L":
        L0 = fem_runner.cable_length(g.mesh, g.cable["indices"])
        p["cable"] = dict(g.cable, L_rest=L0 + d)
        return p, g.mesh
    raise ValueError(f"{factor} is not a length")


def percent_params(g, nom, factor, pct):
    """Perturb `factor` by pct percent of its own nominal value."""
    p = dict(g.params())
    scale = 1.0 + pct / 100.0
    if factor == "R":
        tag = f"{g.name}_pc{'p' if pct > 0 else 'm'}{abs(pct):g}".replace(".", "")
        mesh, _ = mesh_tools.radius_variant(g.mesh, scale, cfg.MESH_DIR, tag)
        return p, mesh
    if factor == "cable_L":
        # The rest length the cable is built to, as a fraction of the length it
        # would take on the rest mesh: what channel seating and anchorage
        # take-up get wrong.
        L0 = fem_runner.cable_length(g.mesh, g.cable["indices"])
        p["cable"] = dict(g.cable, L_rest=L0 * scale)
        return p, g.mesh
    p[PARAM_KEY[factor]] = nom[factor] * scale
    return p, g.mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc", choices=geom.NAMES)
    ap.add_argument("--absolute", action="store_true",
                    help="sweep the geometric factors R and cable rest length "
                         "in millimetres rather than percent")
    ap.add_argument("--percent", action="store_true",
                    help="sweep a common percentage of nominal rather than "
                         "multiples of each factor's assumed tolerance")
    args = ap.parse_args()

    g = geom.get(args.geometry)
    os.makedirs(cfg.MESH_DIR, exist_ok=True)
    run_dir = os.path.join(cfg.DATA_DIR, f"runs_sweep_{g.name}")
    os.makedirs(run_dir, exist_ok=True)

    V_ref, F_ref = mesh_tools.load_off(g.mesh)
    nom = nominal_dict(g)
    free = mesh_tools.interior_mask(V_ref, F_ref)
    V_target = mesh_tools.load_off(g.target)[0] if g.target else None

    p0, mesh0 = run_params(g, nom, None, 0)
    radius0 = measured_radius(mesh0)
    out0 = fem_runner.run(mesh0, os.path.join(run_dir, "S0"), **p0)
    V_base = out0["verts"]
    base = metrics(out0, radius0, V_base, radius0, V_ref, V_target, free)

    print(f"geometry {g.name}: {len(V_ref)} v, {int(free.sum())} interior")
    print(f"baseline crown {1e3 * base['crown_height']:.1f} mm, "
          f"L_target {1e3 * base['L_target']:.2f} mm"
          if V_target is not None else
          f"baseline crown {1e3 * base['crown_height']:.1f} mm")

    if args.absolute:
        steps, key = ABS_MM, "mm"
    elif args.percent:
        steps, key = PERCENTS, "percent"
    else:
        steps, key = MULTIPLES, "multiple"
    rows = [dict(factor="baseline", **{key: 0.0}, delta_abs=0.0, **base)]
    n = 0
    if args.absolute:
        loop_factors = [f for f in ABS_FACTORS if f != "cable_L" or g.cable]
    elif args.percent:
        loop_factors = factors_for(g)
    else:
        loop_factors = list(cfg.BLOCK_A_FACTORS)
    for factor in loop_factors:
        tol = TOLERANCES[TOL_KEY.get(factor, factor)]
        for step in steps:
            for sign in (+1, -1):
                s = sign * step
                if args.absolute:
                    p, mesh = absolute_params(g, nom, factor, s)
                    delta_abs = s / 1000.0
                elif args.percent:
                    p, mesh = percent_params(g, nom, factor, s)
                    delta_abs = nom.get(factor, float("nan")) * s / 100.0
                else:
                    p, mesh = sweep_params(g, nom, factor, s)
                    delta_abs = s * tol.absolute(nom[factor])
                radius = measured_radius(mesh)
                tag = (f"{factor}_{'p' if sign > 0 else 'm'}{step:g}"
                       + ("mm" if args.absolute else
                          "pc" if args.percent else ""))
                out = fem_runner.run(mesh, os.path.join(run_dir, tag), **p)
                r = metrics(out, radius, V_base, radius0, V_ref, V_target, free)
                rows.append(dict(factor=factor, **{key: s},
                                 delta_abs=delta_abs, **r))
                n += 1
                print(f"  {tag:18s} crown {1e3 * r['crown_height']:8.2f} mm  "
                      f"L_pos {1e3 * r['L_pos']:6.2f} mm" +
                      (f"  L_target {1e3 * r['L_target']:6.2f} mm"
                       if V_target is not None else ""))

    suffix = ("_absolute" if args.absolute else
              "_percent" if args.percent else "")
    out_csv = os.path.join(cfg.DATA_DIR,
                           f"tolerance_sweep_{g.name}{suffix}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{n} runs\nwrote {out_csv}")


if __name__ == "__main__":
    main()
