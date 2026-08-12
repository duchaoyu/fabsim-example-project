"""
Block A — method check on the circular dome, 13 runs.

    A0        baseline at nominal
    A1-A2     s_wale     +/- delta_s   (uniform)
    A3-A4     s_course   +/- delta_s   (uniform)
    A5-A6     p          +/- delta_p
    A7-A8     E1         +/- delta_E
    A9-A10    E2/E1      +/- delta_r
    A11-A12   boundary radius +/- delta_R

Plus a numerical-reproducibility probe: A0 re-solved along a different
continuation path (stretch factors ramped in N warm-started steps instead of
applied directly).  Both paths converge to the same equilibrium up to the Newton
tolerance, so the difference between them is a floor on what this study can
resolve.  A response smaller than that floor is not a response.  It is reported
separately and is not one of the 13.

Usage:
    python3 run_block_A.py [--out data/block_A.csv] [--keep-verts]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imperfection_config as cfg
import fem_runner
import mesh_tools
from tolerances import TOLERANCES

# apex_curvature lives with the sensitivity study; it is the estimator that
# survives directional anisotropy (the section estimators average |kappa| along a
# diameter and are nearly blind to it), and it accepts a reference mesh so two
# runs are fitted from an identical vertex set.
sys.path.insert(0, cfg.SA_DIR)
from apex_curvature import apex_curvature


def run_params(factor, sign):
    """Solver arguments for one Block A run, and the mesh it needs.

    The returned radius is always *measured* off the mesh actually used, never
    the nominal design value, so the baseline and the rescaled runs are
    normalised the same way.  On the reference disc the two differ: the design
    radius is 600.0 mm but the mesher placed the boundary vertices between 597.8
    and 600.0 mm, so the realised radius is 598.5 mm.
    """
    p = {
        "sf_wale":      cfg.S_WALE_NOM,
        "sf_course":    cfg.S_COURSE_NOM,
        "knit_dir_deg": cfg.KNIT_DIR_NOM,
        "pressure":     cfg.PRESSURE_NOM,
        "E1":           cfg.E1_NOM,
        "r_ratio":      cfg.R_RATIO_NOM,
        "nu":           cfg.NU_NOM,
    }
    mesh = cfg.BASE_MESH

    if factor is None:
        return p, mesh

    val = cfg.perturbed(factor, sign)
    if factor == "s_wale":
        p["sf_wale"] = val
    elif factor == "s_course":
        p["sf_course"] = val
    elif factor == "pressure":
        p["pressure"] = val
    elif factor == "E1":
        p["E1"] = val
    elif factor == "r":
        p["r_ratio"] = val
    elif factor == "nu":
        p["nu"] = val
    elif factor == "R":
        # A relative in-plane rescale by delta_R / R_nominal.  Expressed as a
        # ratio rather than as an absolute target radius, so it does not matter
        # whether the mesh realises the nominal radius exactly.
        tag = f"R{'p' if sign > 0 else 'm'}"
        mesh, _ = mesh_tools.radius_variant(
            cfg.BASE_MESH, val / cfg.R_BOUNDARY_NOM, cfg.MESH_DIR, tag)
    else:
        raise ValueError(f"unknown factor {factor}")
    return p, mesh


def measured_radius(mesh):
    V, _ = mesh_tools.load_off(mesh)
    return mesh_tools.boundary_radius(V)


def metrics(out, radius, V_base, radius_base, V_ref):
    """Outputs for one run.

    V_base is the baseline deformed surface, V_ref the *undeformed nominal* mesh.

    L_pos       RMS vertex distance from the baseline equilibrium, in metres —
                the position loss of the §6.5.2 table.  For Block A the baseline
                stands in for the target, since the circular dome has no
                optimisation target of its own.
    L_pos_shape the same after normalising each surface by its own boundary
                radius.  For the radius runs this separates "the same dome, a
                different size" from "a different dome": a purely self-similar
                response would show up in L_pos and vanish here.
    """
    V = out["verts"]
    d = np.linalg.norm(V - V_base, axis=1)

    dn = np.linalg.norm(V / radius - V_base / radius_base, axis=1)

    m = {
        "crown_height":           out["crown_height"],
        "h_over_R":               out["crown_height"] / radius,
        "max_stress":             out["max_stress"],
        "mean_stress":            out["mean_stress"],
        "stress_ratio":           out["max_stress"] / out["mean_stress"],
        "boundary_reaction_mean": out["boundary_reaction_mean"],
        "L_pos":                  float(np.sqrt(np.mean(d ** 2))),
        "L_pos_max":              float(d.max()),
        "L_pos_shape":            float(np.sqrt(np.mean(dn ** 2))),
    }

    # Curvature at the crown.  The reference mesh is the *unscaled* nominal disc
    # for every run, including the radius runs, so all 13 fits use the same
    # vertex set and are comparable.
    k = apex_curvature(V, cfg.KNIT_DIR_NOM, ref_verts=V_ref)
    if k is None:
        m.update({key: float("nan") for key in
                  ("H_apex", "k_min", "k_max", "k_ratio", "fit_rms_mm")})
    else:
        m.update({key: k[key] for key in
                  ("H_apex", "k_min", "k_max", "k_ratio", "fit_rms_mm")})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(cfg.DATA_DIR, "block_A.csv"))
    ap.add_argument("--keep-verts", action="store_true",
                    help="keep the per-run *_verts.csv (needed for shape plots)")
    ap.add_argument("--ramp-probe", type=int, default=4,
                    help="continuation steps for the reproducibility probe")
    args = ap.parse_args()

    fem_runner.check_binary()
    os.makedirs(cfg.DATA_DIR, exist_ok=True)
    os.makedirs(cfg.MESH_DIR, exist_ok=True)
    run_dir = os.path.join(cfg.DATA_DIR, "runs_A")
    os.makedirs(run_dir, exist_ok=True)

    V_ref, F_ref = mesh_tools.load_off(cfg.BASE_MESH)
    r_ring = np.linalg.norm(V_ref[:, :2], axis=1)
    ring   = r_ring > 0.99 * r_ring.max()
    print(f"mesh      {cfg.BASE_MESH}")
    print(f"          {len(V_ref)} vertices, {len(F_ref)} faces")
    print(f"          design radius {cfg.R_BOUNDARY_NOM * 1e3:.1f} mm, realised "
          f"{mesh_tools.boundary_radius(V_ref) * 1e3:.1f} mm")
    # The reference disc is not exactly round: its boundary vertices scatter over
    # 2.2 mm, which is 44% of delta_R.  So the baseline already carries an
    # out-of-round imperfection comparable to the tolerance being tested, and a
    # dedicated out-of-round block has to be measured against this, not against a
    # perfect circle.
    print(f"          boundary vertices span {r_ring[ring].min() * 1e3:.1f}-"
          f"{r_ring[ring].max() * 1e3:.1f} mm (std "
          f"{r_ring[ring].std() * 1e3:.2f} mm) — the mesh is already "
          f"{100 * (r_ring[ring].max() - r_ring[ring].min()) / (2 * TOLERANCES['R'].absolute(cfg.R_BOUNDARY_NOM)):.0f}% "
          f"of delta_R out of round")
    print(f"nominal   s_wale={cfg.S_WALE_NOM}  s_course={cfg.S_COURSE_NOM}  "
          f"theta={cfg.KNIT_DIR_NOM}  p={cfg.PRESSURE_NOM} Pa")
    print(f"          E1={cfg.E1_NOM} N/m  E2/E1={cfg.R_RATIO_NOM:.4f}  "
          f"nu={cfg.NU_NOM}")
    print("\ntolerances applied")
    for f in cfg.BLOCK_A_FACTORS:
        t = TOLERANCES[f]
        print(f"  {f:9s} +/- {t.absolute(cfg.NOMINAL[f]):.6g} "
              f"({100 * t.rel_at(cfg.NOMINAL[f]):.2f}%)  "
              f"{'+'.join(t.kind):22s} {t.status:8s} {t.source}")
    print()

    # Baseline first: every other run is measured against it.
    p0, mesh0 = run_params(None, 0)
    radius0 = measured_radius(mesh0)
    out0 = fem_runner.run(mesh0, os.path.join(run_dir, "A0"), **p0)
    V_base = out0["verts"]

    rows = []
    for run_id, factor, sign in cfg.BLOCK_A:
        p, mesh = run_params(factor, sign)
        radius = measured_radius(mesh)
        out = (out0 if factor is None
               else fem_runner.run(mesh, os.path.join(run_dir, run_id), **p))
        m = metrics(out, radius, V_base, radius0, V_ref)

        row = {
            "run": run_id,
            "factor": factor if factor else "baseline",
            "sign": sign,
            "value": cfg.NOMINAL[factor] if factor else float("nan"),
            "perturbed": cfg.perturbed(factor, sign) if factor else float("nan"),
            "mesh": os.path.basename(mesh),
            "radius_m": radius,
            "sf_wale": p["sf_wale"],
            "sf_course": p["sf_course"],
            "pressure": p["pressure"],
            "E1": p["E1"],
            "r_ratio": p["r_ratio"],
            "nu": p["nu"],
            **m,
        }
        rows.append(row)
        print(f"  {run_id:4s} {row['factor']:9s} {sign:+d}  "
              f"h={m['crown_height'] * 1000:7.2f} mm  "
              f"L_pos={m['L_pos'] * 1000:6.3f} mm  "
              f"sig_max={m['max_stress']:8.1f}  H={m['H_apex']:.4f}")

    # ── Numerical-reproducibility floor ───────────────────────────────────────
    out_r = fem_runner.run(mesh0, os.path.join(run_dir, "A0_ramp"),
                           sf_ramp_steps=args.ramp_probe, **p0)
    m_r = metrics(out_r, radius0, V_base, radius0, V_ref)
    print(f"\nreproducibility probe (sf ramped in {args.ramp_probe} steps)")
    print(f"  dh    = {abs(m_r['crown_height'] - out0['crown_height']) * 1e6:.3f} um")
    print(f"  L_pos = {m_r['L_pos'] * 1e6:.3f} um   "
          f"(floor: responses below this are numerical, not physical)")

    import csv as _csv
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    floor_path = os.path.join(cfg.DATA_DIR, "block_A_floor.csv")
    with open(floor_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["metric", "value"])
        w.writeheader()
        w.writerow({"metric": "ramp_steps", "value": args.ramp_probe})
        w.writerow({"metric": "d_crown_height_m",
                    "value": abs(m_r["crown_height"] - out0["crown_height"])})
        w.writerow({"metric": "L_pos_m", "value": m_r["L_pos"]})
        w.writerow({"metric": "d_H_apex_1perm",
                    "value": abs(m_r["H_apex"] - rows[0]["H_apex"])})

    if not args.keep_verts:
        for name in os.listdir(run_dir):
            if name.endswith("_verts.csv") or name.endswith("_stress.csv"):
                os.unlink(os.path.join(run_dir, name))

    print(f"\nwrote {args.out}")
    print(f"wrote {floor_path}")


if __name__ == "__main__":
    main()
