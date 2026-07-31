"""
Validate FEA runs from their saved vertex/stress output.

Two datasets:
  sweep    data/uniform_sf_sweep/   — the uniform-sf diagonal (run_uniform_sf_sweep.py)
  samples  data/{id:05d}_*.csv      — the Sobol samples behind results_with_sections.csv

Per-run checks:
  bc        boundary vertices unmoved (Dirichlet respected).  Because the
            boundary is clamped to the input mesh, this doubles as a provenance
            check: if a run was produced on a different mesh than the one
            config.MESH_PATH now points at, its boundary will not sit on the
            current mesh's boundary.  `bc_scale` reports the uniform radial
            scale that best maps the current mesh's boundary onto the run's —
            1.0 means same mesh, anything else means a different (scaled) mesh.
  dome      z >= 0, apex near the centre, profile monotone from centre to rim
  elem      no collapsed or inverted triangles (deformed/rest area ratio)
  rim       fraction of faces tipped past vertical — happens at the outermost
            ring when a strongly pre-stretched dome bulges past the clamp
  laplace   membrane equilibrium at the apex:  p_eff = N_wale*k_wale + N_course*k_course
            using the solver's own tensions and a quadric fit to the apex.
            Curvatures are taken along the knit frame, so this works at any
            knit_dir.

NOTE on the Laplace check: p_eff comes out at 3.00x the *nominal* pressure
argument for every run.  That is not a per-run defect — fabsim's pressure work
term is p * (x1+x2+x3).(e1 x e2) / 6 per triangle (OrthotropicStVKElement.cpp,
energy()), and expanding that triple product gives 3 * p * V_enclosed rather
than p * V_enclosed.  So the equilibria are exact for an inflation pressure
three times the value passed in.  A run is judged consistent here if
p_eff / p_nominal is within LAPLACE_TOL of 3.

Usage:
  python3 validate_fem_runs.py            # both datasets
  python3 validate_fem_runs.py sweep
  python3 validate_fem_runs.py samples [--max-runs N]
"""

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off

V0, F = read_off(MESH_PATH)
R_RIM = float(np.linalg.norm(V0[:, :2], axis=1).max())

# Boundary vertices = on an edge used by one triangle (same rule as the C++ side)
_ec = collections.Counter()
for f in F:
    for i in range(3):
        _ec[tuple(sorted((f[i], f[(i + 1) % 3])))] += 1
BDRY = np.array(sorted({v for e, n in _ec.items() if n == 1 for v in e}))

APEX_R       = 0.18    # radius (m) of the apex patch for the quadric/tension fit
PRESSURE_FAC = 3.0     # fabsim's pressure work term is 3x p*V (see module docstring)
LAPLACE_TOL  = 0.15    # accept |p_eff/(3 p) - 1| below this
BC_TOL_MM    = 1e-3    # boundary movement above this is a real BC violation
AREA_MIN     = 0.5     # deformed/rest area below this = collapsed element
SPIKE_MAX    = 4.0     # shape jump vs local median, for the sweep continuity test


def _apex_curvatures(V, knit_dir_deg):
    """Curvature along the wale and course directions at the apex, from a
    quadric fit z = c0 + c1 x + c2 y + c3 x^2 + c4 y^2 + c5 xy."""
    m = np.linalg.norm(V0[:, :2], axis=1) < APEX_R
    if m.sum() < 8:
        return np.nan, np.nan
    x, y, z = V[m, 0], V[m, 1], V[m, 2]
    A = np.column_stack([np.ones(m.sum()), x, y, x**2, y**2, x * y])
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    # curvature along unit direction d is -d' H d with H the Hessian of z
    H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    th = np.radians(knit_dir_deg)
    d_wale   = np.array([np.sin(th),  np.cos(th)])    # knit_dir convention in the binary
    d_course = np.array([np.cos(th), -np.sin(th)])
    return (float(-d_wale @ H @ d_wale), float(-d_course @ H @ d_course))


def check_run(verts_path, stress_path, knit_dir_deg, pressure):
    out = {}
    V = pd.read_csv(verts_path).sort_values("vid")[["x", "y", "z"]].values

    out["bc_max_move_mm"] = float(np.abs(V[BDRY] - V0[BDRY]).max() * 1000)
    # Uniform radial scale that best maps the current mesh's boundary onto this
    # run's boundary; 1.0 = same mesh (see module docstring).
    b0, b1 = V0[BDRY, :2], V[BDRY, :2]
    out["bc_scale"] = float((b1 * b0).sum() / (b0 * b0).sum())
    out["bc_resid_after_scale_mm"] = float(
        np.abs(b1 - out["bc_scale"] * b0).max() * 1000)
    out["z_min_mm"] = float(V[:, 2].min() * 1000)
    out["z_max_mm"] = float(V[:, 2].max() * 1000)
    out["apex_offset_mm"] = float(
        np.linalg.norm(V0[np.argmax(V[:, 2]), :2]) * 1000)

    rad = np.linalg.norm(V0[:, :2], axis=1)
    nb = 10
    edges = np.linspace(0, rad.max(), nb + 1)
    zb = np.array([V[(rad >= edges[k]) & (rad < edges[k + 1]), 2].mean()
                   for k in range(nb)])
    zb = zb[np.isfinite(zb)]
    out["profile_monotone"] = bool(np.all(np.diff(zb) <= 1e-6))

    p = V[F]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    p0 = V0[F]
    area0 = np.linalg.norm(np.cross(p0[:, 1] - p0[:, 0],
                                    p0[:, 2] - p0[:, 0]), axis=1) / 2
    out["min_area_ratio"] = float((area / area0).min())
    out["max_area_ratio"] = float((area / area0).max())

    cen = p.mean(axis=1)
    tipped = n[:, 2] < 0
    out["n_tipped"] = int(tipped.sum())
    out["tipped_min_radius_frac"] = (
        float(np.linalg.norm(cen[tipped, :2], axis=1).min() / R_RIM)
        if tipped.any() else np.nan)
    out["max_radius_frac"] = float(
        np.linalg.norm(V[:, :2], axis=1).max() / R_RIM)

    k_w, k_c = _apex_curvatures(V, knit_dir_deg)
    sdf = pd.read_csv(stress_path).sort_values("face")
    cen_s = V[F[sdf["face"].values.astype(int)]].mean(axis=1)
    ap = np.linalg.norm(cen_s[:, :2], axis=1) < APEX_R
    if ap.any() and np.isfinite(k_w) and np.isfinite(k_c):
        N_w = float(sdf["T_wale_Nm"].values[ap].mean())
        N_c = float(sdf["T_course_Nm"].values[ap].mean())
        out["k_wale"], out["k_course"] = k_w, k_c
        out["N_wale"], out["N_course"] = N_w, N_c
        out["p_eff"] = N_w * k_w + N_c * k_c
        out["p_eff_over_p"] = out["p_eff"] / pressure
        out["laplace_err"] = out["p_eff_over_p"] / PRESSURE_FAC - 1.0
    return out, V


def _report(res, label, continuity=False):
    ok  = res[~res["flagged"].astype(bool)]
    bad = res[res["flagged"].astype(bool)]
    print(f"\n=== {label}: {len(res)} runs "
          f"({len(ok)} converged, {len(bad)} flagged) ===")
    if not len(ok):
        return

    def line(name, series, limit, mode="max"):
        v = pd.to_numeric(series, errors="coerce").dropna()
        if v.empty:
            print(f"  {name:26s} no data")
            return
        worst = v.max() if mode == "max" else v.min()
        fail = (v > limit) if mode == "max" else (v < limit)
        tag = "PASS" if not fail.any() else f"FAIL ({int(fail.sum())} runs)"
        print(f"  {name:26s} worst {worst:11.4g}   limit {limit:<8g} {tag}")

    line("bc max move (mm)",   ok["bc_max_move_mm"], BC_TOL_MM)
    sc = pd.to_numeric(ok["bc_scale"], errors="coerce").dropna()
    if not sc.empty:
        off = (sc - 1.0).abs() > 1e-6
        tag = ("PASS (same mesh)" if not off.any() else
               f"DIFFERENT MESH ({int(off.sum())} runs)")
        print(f"  {'boundary radial scale':26s} median {sc.median():.6f}  "
              f"range {sc.min():.6f}-{sc.max():.6f}   {tag}")
        if off.any():
            rs = pd.to_numeric(ok.loc[off, "bc_resid_after_scale_mm"],
                               errors="coerce").max()
            print(f"  {'  residual after rescale':26s} {rs:.3g} mm "
                  f"(small = the run's mesh is this one scaled uniformly)")
    line("z_min (mm)",         ok["z_min_mm"],       -1e-3, mode="min")
    line("apex offset (mm)",   ok["apex_offset_mm"], 0.15 * R_RIM * 1000)
    line("min area ratio",     ok["min_area_ratio"], AREA_MIN, mode="min")
    nm = int((~ok["profile_monotone"].astype(bool)).sum())
    print(f"  {'profile monotone':26s} {len(ok)-nm}/{len(ok)} runs "
          f"{'PASS' if nm == 0 else f'FAIL ({nm} runs)'}")

    tip = ok[ok["n_tipped"] > 0]
    if len(tip):
        print(f"  {'rim faces past vertical':26s} {len(tip)} runs, "
              f"innermost at r/R = {tip['tipped_min_radius_frac'].min():.3f}, "
              f"max bulge r/R = {tip['max_radius_frac'].max():.3f}  "
              f"{'(outer ring only — OK)' if tip['tipped_min_radius_frac'].min() > 0.9 else 'CHECK'}")
    else:
        print(f"  {'rim faces past vertical':26s} none")

    if continuity and "dV_spike" in ok:
        line("shape spike vs neighbours", ok["dV_spike"], SPIKE_MAX)

    e = pd.to_numeric(ok["laplace_err"], errors="coerce").abs().dropna()
    ratio = pd.to_numeric(ok["p_eff_over_p"], errors="coerce").dropna()
    print(f"  {'Laplace p_eff/p_nominal':26s} "
          f"median {ratio.median():.3f}  range {ratio.min():.3f}-{ratio.max():.3f}"
          f"   (expected {PRESSURE_FAC:.2f})")
    n_off = int((e > LAPLACE_TOL).sum())
    print(f"  {'  deviation from 3x':26s} median {e.median()*100:4.1f}%  "
          f"max {e.max()*100:4.1f}%   "
          f"{'PASS' if n_off == 0 else f'FAIL ({n_off} runs)'}")

    if len(bad):
        print(f"  flagged runs: z_max max {pd.to_numeric(bad['z_max_mm'], errors='coerce').max():.3g} mm "
              f"(undeformed output — solver did not converge)")


def validate_sweep():
    csv = os.path.join(DATA_DIR, "uniform_sf_sweep.csv")
    df = pd.read_csv(csv)
    rows = []
    for motif, sub in df.groupby("motif"):
        sub = sub.sort_values("sf").reset_index(drop=True)
        prev_V = prev_sf = None
        steps = []
        for _, r in sub.iterrows():
            pre = os.path.join(DATA_DIR, "uniform_sf_sweep",
                               f"m{int(motif)}_sf{r['sf']:.3f}")
            if not os.path.exists(pre + "_verts.csv"):
                rows.append({"motif": motif, "sf": r["sf"], "flagged": True,
                             "missing": True})
                continue
            out, V = check_run(pre + "_verts.csv", pre + "_stress.csv",
                               r.get("knit_dir", 0.0), r["pressure"])
            out.update({"motif": int(motif), "sf": r["sf"],
                        "flagged": bool(r["sim_failed"])})
            if not out["flagged"]:
                if prev_V is not None:
                    d = np.abs(V - prev_V).max() / (r["sf"] - prev_sf)
                    out["dV_rate"] = float(d)
                    steps.append(float(d))
                prev_V, prev_sf = V, r["sf"]
            rows.append(out)
        med = np.median(steps) if steps else np.nan
        for row in rows:
            if row.get("motif") == int(motif) and "dV_rate" in row:
                row["dV_spike"] = row["dV_rate"] / med
    res = pd.DataFrame(rows)
    _report(res, "uniform-sf sweep", continuity=True)
    return res


def validate_samples(max_runs=None):
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    rows = []
    n = 0
    for _, r in df.iterrows():
        sid = int(r["sample_id"])
        vp = os.path.join(DATA_DIR, f"{sid:05d}_verts.csv")
        sp = os.path.join(DATA_DIR, f"{sid:05d}_stress.csv")
        if not (os.path.exists(vp) and os.path.exists(sp)):
            continue
        out, _ = check_run(vp, sp, r["knit_dir"], r["pressure"])
        out.update({"sample_id": sid, "group": r["group"],
                    "flagged": bool(r.get("sim_failed", False))})
        rows.append(out)
        n += 1
        if max_runs and n >= max_runs:
            break
    res = pd.DataFrame(rows)
    if not len(res):
        print("\nNo per-sample vertex files found in data/ — skipping samples.")
        return res
    _report(res, "Sobol samples (all groups)")
    for g, sub in res.groupby("group"):
        _report(sub, f"Sobol samples / {g}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="both",
                    choices=["sweep", "samples", "both"])
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    if args.dataset in ("sweep", "both"):
        r = validate_sweep()
        r.to_csv(os.path.join(DATA_DIR, "validation_sweep.csv"), index=False)
    if args.dataset in ("samples", "both"):
        r = validate_samples(args.max_runs)
        if len(r):
            r.to_csv(os.path.join(DATA_DIR, "validation_samples.csv"), index=False)
    print()
