"""
Fit the nominal stretch factors for a geometry whose rest mesh IS its design
target, so that Block A perturbs about a working point rather than an arbitrary one.

The loss is the RMS distance between the inflated equilibrium and the target,
over INTERIOR vertices only.  The boundary is clamped, so its deviation is zero by
construction and including it would deflate the loss by a factor that depends on
nothing but how finely the boundary happens to be discretised.

Two modes:

  isotropic    one variable, sf_wale = sf_course.  This is the default, because a
               nominal for Block A has to be interior to the validity box: a
               nominal sitting on a bound cannot be perturbed symmetrically, and
               the +/- responses that Block A compares would differ only because
               one side was clipped.

  anisotropic  two variables.  On the 2part this saturates at both bounds
               (sf_wale -> 1.400, sf_course -> 0.950), which is worth recording:
               a uniform two-parameter pre-strain cannot reach this target, which
               is precisely why the case-study pipelines use 9 or 16 regions.  Do
               not use the saturated point as a Block A nominal.

Usage:
    python3 fit_nominal.py --geometry 2part [--mode isotropic] [--maxiter 60]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import geometry as geom
import fem_runner
import mesh_tools

# The Chapter 6 model-validity box: below sf = 0.95 a membrane element goes into
# compression, which is outside the formulation rather than merely noisy.
SF_BOUNDS = (0.95, 1.4)


def interior_mask(V, F):
    """Free vertices: everything off the clamped boundary ring."""
    return mesh_tools.interior_mask(V, F)


def make_loss(g, V_target, free, prefix):
    calls = [0]
    history = []

    def loss(p):
        calls[0] += 1
        sf_w, sf_c = (p[0], p[0]) if len(p) == 1 else (p[0], p[1])
        out = fem_runner.run(g.mesh, prefix,
                             sf_wale=sf_w, sf_course=sf_c,
                             knit_dir_deg=g.knit_dir_deg, pressure=g.pressure,
                             E1=g.E1, r_ratio=g.r_ratio, nu=g.nu)
        d = np.linalg.norm(out["verts"][free] - V_target[free], axis=1)
        val = float(np.sqrt(np.mean(d ** 2)))
        history.append({"call": calls[0], "sf_wale": sf_w, "sf_course": sf_c,
                        "L_target_m": val,
                        "crown_height": out["crown_height"]})
        print(f"  [{calls[0]:3d}] sf=({sf_w:.4f}, {sf_c:.4f})  "
              f"L_target = {val * 1000:7.3f} mm  "
              f"crown = {out['crown_height']:.4f} m")
        return val

    return loss, calls, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="2part", choices=geom.NAMES)
    ap.add_argument("--mode", default="isotropic",
                    choices=["isotropic", "anisotropic"])
    ap.add_argument("--maxiter", type=int, default=60)
    args = ap.parse_args()

    g = geom.get(args.geometry)
    if g.target is None:
        print(f"{g.name} has no design target — nothing to fit.")
        return

    fem_runner.check_binary()
    V_target, F_target = mesh_tools.load_off(g.target)
    free = interior_mask(V_target, F_target)
    print(f"geometry  {g.name}  ({g.mesh})")
    print(f"target    {g.target}")
    print(f"          {len(V_target)} vertices, {int(free.sum())} interior, "
          f"crown {V_target[:, 2].max():.4f} m")
    print(f"mode      {args.mode}, bounds {SF_BOUNDS}\n")

    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"fit_{g.name}")

    loss, calls, history = make_loss(g, V_target, free, prefix)
    p0 = [g.s_wale] if args.mode == "isotropic" else [g.s_wale, g.s_course]
    bounds = [SF_BOUNDS] * len(p0)

    res = minimize(loss, p0, method="L-BFGS-B", bounds=bounds,
                   options={"eps": 0.01, "ftol": 1e-12, "gtol": 1e-9,
                            "maxiter": args.maxiter})

    sf_w, sf_c = (res.x[0], res.x[0]) if len(res.x) == 1 else (res.x[0], res.x[1])
    on_bound = [n for n, v in (("sf_wale", sf_w), ("sf_course", sf_c))
                if min(abs(v - SF_BOUNDS[0]), abs(v - SF_BOUNDS[1])) < 1e-4]

    print(f"\nconverged: {res.success}  |  {res.message}")
    print(f"sf_wale = {sf_w:.4f}   sf_course = {sf_c:.4f}")
    print(f"L_target = {res.fun * 1000:.3f} mm over {int(free.sum())} interior "
          f"vertices   ({calls[0]} FEM calls)")
    if on_bound:
        print(f"\nWARNING: {', '.join(on_bound)} sits on a validity bound.")
        print("  A nominal on a bound cannot be perturbed symmetrically, so this")
        print("  point is NOT usable as a Block A nominal — the +/- responses")
        print("  would differ because one side was clipped, not because the")
        print("  physics is asymmetric.  It also says a uniform pre-strain cannot")
        print("  reach this target: the case study needs per-region factors.")

    record = {
        "geometry": g.name, "mode": args.mode,
        "sf_wale": sf_w, "sf_course": sf_c,
        "knit_dir_deg": g.knit_dir_deg, "pressure": g.pressure,
        "E1": g.E1, "r_ratio": g.r_ratio, "nu": g.nu,
        "L_target_mm": res.fun * 1000.0,
        "n_interior": int(free.sum()),
        "n_calls": calls[0],
        "on_bound": on_bound,
        "converged": bool(res.success),
        "history": history,
    }
    # Only an interior point may be installed as the nominal.
    path = os.path.join(out_dir, f"nominal_{g.name}.json")
    if on_bound:
        path = os.path.join(out_dir, f"fit_{g.name}_{args.mode}_saturated.json")
        print(f"\n  -> written to {os.path.basename(path)} as a record, NOT "
              f"installed as the nominal.")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {path}")

    for suffix in ("_verts.csv", "_scalars.csv", "_stress.csv"):
        p = prefix + suffix
        if os.path.exists(p):
            os.unlink(p)


if __name__ == "__main__":
    main()
