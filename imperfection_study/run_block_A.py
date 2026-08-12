"""
Block A — method check, 13 runs on one geometry.

    A0        baseline at nominal
    A1-A2     s_wale     +/- delta_s   (uniform)
    A3-A4     s_course   +/- delta_s   (uniform)
    A5-A6     p          +/- delta_p
    A7-A8     E1         +/- delta_E
    A9-A10    E2/E1      +/- delta_r
    A11-A12   boundary   +/- delta_R   (uniform in-plane rescale; see geometry.py)

Plus a numerical-reproducibility probe: A0 re-solved along a different
continuation path (stretch factors ramped in N warm-started steps instead of
applied directly).  Both paths converge to the same equilibrium up to the Newton
tolerance, so the difference between them is a floor on what this study can
resolve.  A response smaller than that floor is not a response.  It is reported
separately and is not one of the 13.

Usage:
    python3 run_block_A.py [--geometry disc|2part] [--keep-verts]
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import imperfection_config as cfg
import geometry as geom
import fem_runner
import mesh_tools
from tolerances import TOLERANCES

# apex_curvature lives with the sensitivity study; it is the estimator that
# survives directional anisotropy (the section estimators average |kappa| along a
# diameter and are nearly blind to it), and it accepts a reference mesh so two
# runs are fitted from an identical vertex set.
sys.path.insert(0, cfg.SA_DIR)
from apex_curvature import apex_curvature


def nominal_dict(g):
    """Nominal value of every Block A factor for this geometry.

    "R" is the characteristic in-plane radius of the clamped boundary ring,
    measured off the mesh rather than taken from the design intent — see
    geometry.Geometry.r_char.
    """
    return {"s_wale": g.s_wale, "s_course": g.s_course, "pressure": g.pressure,
            "E1": g.E1, "r": g.r_ratio, "nu": g.nu, "R": g.r_char()}


def run_params(g, nom, factor, sign):
    """Solver arguments for one Block A run, and the mesh it needs."""
    p = dict(g.params())
    mesh = g.mesh
    if factor is None:
        return p, mesh

    val = TOLERANCES[factor].perturb(nom[factor], sign)
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
        # A relative in-plane rescale by delta_R / R_char, expressed as a ratio so
        # it does not matter whether the mesh realises its design radius exactly.
        tag = f"{g.name}_R{'p' if sign > 0 else 'm'}"
        mesh, _ = mesh_tools.radius_variant(
            g.mesh, val / nom["R"], cfg.MESH_DIR, tag)
    else:
        raise ValueError(f"unknown factor {factor}")
    return p, mesh


def measured_radius(mesh):
    V, _ = mesh_tools.load_off(mesh)
    return mesh_tools.boundary_radius(V)


def metrics(out, radius, V_base, radius_base, V_ref, V_target, free):
    """Outputs for one run.

    V_base   the baseline deformed surface
    V_ref    the undeformed nominal mesh, used only to pick the curvature
             neighbourhood so every run is fitted from the same vertices
    V_target the design target, or None

    L_pos       RMS vertex distance from the baseline equilibrium, in metres —
                the position loss of the §6.5.2 table, measured over interior
                vertices only.  The boundary is clamped, so its deviation is zero
                by construction and averaging it in would deflate the loss by a
                factor that depends on nothing but boundary discretisation.
    L_target    the same distance from the design target, where one exists.  This
                is the quantity that says whether the as-built structure holds its
                intended shape; L_pos says only how far the imperfection moved it.
    L_pos_shape L_pos after normalising each surface by its own boundary radius,
                which separates "the same shape at a different size" from "a
                different shape" for the radius runs.
    """
    V = out["verts"]
    d = np.linalg.norm(V[free] - V_base[free], axis=1)
    dn = np.linalg.norm((V / radius)[free] - (V_base / radius_base)[free], axis=1)

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
    if V_target is not None:
        dt = np.linalg.norm(V[free] - V_target[free], axis=1)
        m["L_target"]     = float(np.sqrt(np.mean(dt ** 2)))
        m["L_target_max"] = float(dt.max())
    else:
        m["L_target"] = m["L_target_max"] = float("nan")

    k = apex_curvature(V, cfg.KNIT_DIR_NOM, ref_verts=V_ref)
    keys = ("H_apex", "k_min", "k_max", "k_ratio", "fit_rms_mm")
    m.update({key: (float("nan") if k is None else k[key]) for key in keys})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="disc", choices=geom.NAMES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--keep-verts", action="store_true",
                    help="keep the per-run *_verts.csv (needed for --check-overlap)")
    ap.add_argument("--ramp-probe", type=int, default=4,
                    help="continuation steps for the reproducibility probe")
    args = ap.parse_args()

    g = geom.get(args.geometry)
    out_csv = args.out or os.path.join(cfg.DATA_DIR, f"block_A_{g.name}.csv")

    fem_runner.check_binary()
    os.makedirs(cfg.MESH_DIR, exist_ok=True)
    run_dir = os.path.join(cfg.DATA_DIR, f"runs_A_{g.name}")
    os.makedirs(run_dir, exist_ok=True)

    V_ref, F_ref = mesh_tools.load_off(g.mesh)
    nom = nominal_dict(g)
    # Interior vertices: the boundary ring is clamped, so it carries no signal.
    free = mesh_tools.interior_mask(V_ref, F_ref)
    V_target = mesh_tools.load_off(g.target)[0] if g.target else None

    r_ring = np.linalg.norm(V_ref[:, :2], axis=1)
    ring   = r_ring > 0.99 * r_ring.max()
    print(f"geometry  {g.name}")
    print(f"          {g.mesh}")
    print(f"          {len(V_ref)} vertices ({int(free.sum())} interior), "
          f"{len(F_ref)} faces")
    print(f"          {g.note}")
    print(f"          R_char {nom['R'] * 1e3:.1f} mm; boundary ring spans "
          f"{r_ring[ring].min() * 1e3:.1f}-{r_ring[ring].max() * 1e3:.1f} mm "
          f"(std {r_ring[ring].std() * 1e3:.2f} mm, "
          f"{100 * (r_ring[ring].max() - r_ring[ring].min()) / (2 * TOLERANCES['R'].absolute(nom['R'])):.0f}% "
          f"of delta_R)")
    print(f"nominal   s_wale={g.s_wale:.4f}  s_course={g.s_course:.4f}  "
          f"theta={g.knit_dir_deg}  p={g.pressure:.0f} Pa")
    print(f"          E1={g.E1:.0f} N/m  E2/E1={g.r_ratio:.4f}  nu={g.nu}")
    print(f"          source: {g.nominal_source}")
    print("\ntolerances applied")
    for f in cfg.BLOCK_A_FACTORS:
        t = TOLERANCES[f]
        print(f"  {f:9s} +/- {t.absolute(nom[f]):.6g} "
              f"({100 * t.rel_at(nom[f]):.2f}%)  "
              f"{'+'.join(t.kind):22s} {t.status:8s} {t.source}")
    print()

    # Baseline first: every other run is measured against it.
    p0, mesh0 = run_params(g, nom, None, 0)
    radius0 = measured_radius(mesh0)
    out0 = fem_runner.run(mesh0, os.path.join(run_dir, "A0"), **p0)
    V_base = out0["verts"]

    rows = []
    for run_id, factor, sign in cfg.BLOCK_A:
        p, mesh = run_params(g, nom, factor, sign)
        radius = measured_radius(mesh)
        out = (out0 if factor is None
               else fem_runner.run(mesh, os.path.join(run_dir, run_id), **p))
        m = metrics(out, radius, V_base, radius0, V_ref, V_target, free)

        rows.append({
            "run": run_id, "geometry": g.name,
            "factor": factor if factor else "baseline", "sign": sign,
            "value": nom[factor] if factor else float("nan"),
            "perturbed": (TOLERANCES[factor].perturb(nom[factor], sign)
                          if factor else float("nan")),
            "mesh": os.path.basename(mesh), "radius_m": radius,
            "sf_wale": p["sf_wale"], "sf_course": p["sf_course"],
            "pressure": p["pressure"], "E1": p["E1"],
            "r_ratio": p["r_ratio"], "nu": p["nu"],
            **m,
        })
        tgt = ("" if V_target is None
               else f"  L_tgt={m['L_target'] * 1000:6.2f} mm")
        print(f"  {run_id:4s} {rows[-1]['factor']:9s} {sign:+d}  "
              f"h={m['crown_height'] * 1000:7.2f} mm  "
              f"L_pos={m['L_pos'] * 1000:6.3f} mm  "
              f"sig_max={m['max_stress']:8.1f}  H={m['H_apex']:.4f}{tgt}")

    # ── Numerical-reproducibility floor ───────────────────────────────────────
    out_r = fem_runner.run(mesh0, os.path.join(run_dir, "A0_ramp"),
                           sf_ramp_steps=args.ramp_probe, **p0)
    m_r = metrics(out_r, radius0, V_base, radius0, V_ref, V_target, free)
    print(f"\nreproducibility probe (sf ramped in {args.ramp_probe} steps)")
    print(f"  dh    = {abs(m_r['crown_height'] - out0['crown_height']) * 1e6:.3f} um")
    print(f"  L_pos = {m_r['L_pos'] * 1e6:.3f} um   "
          f"(floor: responses below this are numerical, not physical)")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    floor_path = os.path.join(cfg.DATA_DIR, f"block_A_{g.name}_floor.csv")
    with open(floor_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"])
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

    print(f"\nwrote {out_csv}")
    print(f"wrote {floor_path}")


if __name__ == "__main__":
    main()
