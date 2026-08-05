"""
Validate every FEA run in the uniform-sf sweep.

Checks, per run, using only the saved vertex/stress output:

  bc        boundary vertices unmoved (Dirichlet respected)
  dome      z >= 0 everywhere, single apex, profile monotone from centre to rim
  elem      no inverted or collapsed triangles; principal stretches sane
  cont      shape changes continuously w.r.t. the neighbouring sf run
  laplace   membrane equilibrium at the apex: p = N_wale*k_wale + N_course*k_course
            must recover the applied pressure.  This is an independent physics
            check — it uses the solver's own tensions and the geometry, and
            involves no material model or surrogate.

Usage:  python3 validate_uniform_sf_sweep.py
"""

import os
import sys
import collections

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off

SWEEP_CSV = os.path.join(DATA_DIR, "uniform_sf_sweep.csv")
SWEEP_DIR = os.path.join(DATA_DIR, "uniform_sf_sweep")

V0, F = read_off(MESH_PATH)

# Boundary vertices = those on an edge used by a single triangle (same rule as
# findBoundaryVertices in the C++ binary).
_ec = collections.Counter()
for f in F:
    for i in range(3):
        _ec[tuple(sorted((f[i], f[(i + 1) % 3])))] += 1
BDRY = np.array(sorted({v for e, n in _ec.items() if n == 1 for v in e}))

APEX_R = 0.15          # radius (m) of the apex patch used for the Laplace check
LAPLACE_TOL = 0.20     # accept |p_est/p - 1| below this


def _verts(prefix):
    return pd.read_csv(prefix + "_verts.csv").sort_values("vid")[["x", "y", "z"]].values


def _apex_curvature(V, axis):
    """Curvature at the apex of the section cut at `axis`=0, by fitting a
    parabola to vertices near that plane and near the centre."""
    other = 1 - axis
    near = (np.abs(V0[:, axis]) < 0.06) & (np.abs(V0[:, other]) < APEX_R)
    if near.sum() < 5:
        return np.nan
    s = V[near, other]
    z = V[near, 2]
    c = np.polyfit(s, z, 2)
    return float(-2.0 * c[0])          # z = c0 s^2 + ... -> k = -2 c0


def validate():
    df = pd.read_csv(SWEEP_CSV)
    rows = []

    for motif, sub in df.groupby("motif"):
        sub = sub.sort_values("sf").reset_index(drop=True)
        prev_V, prev_sf = None, None

        for _, r in sub.iterrows():
            sf = r["sf"]
            prefix = os.path.join(SWEEP_DIR, f"m{int(motif)}_sf{sf:.3f}")
            out = {"motif": int(motif), "sf": sf,
                   "flagged": bool(r["sim_failed"])}
            if not os.path.exists(prefix + "_verts.csv"):
                out["missing"] = True
                rows.append(out)
                continue

            V = _verts(prefix)

            # ── boundary condition ────────────────────────────────────────────
            out["bc_max_move_mm"] = float(
                np.abs(V[BDRY] - V0[BDRY]).max() * 1000)

            # ── dome shape ────────────────────────────────────────────────────
            out["z_min_mm"] = float(V[:, 2].min() * 1000)
            out["z_max_mm"] = float(V[:, 2].max() * 1000)
            rad = np.linalg.norm(V0[:, :2], axis=1)
            # profile monotonicity: bin by rest radius, z should decrease outward
            nb = 10
            edges = np.linspace(0, rad.max(), nb + 1)
            zb = [V[(rad >= edges[k]) & (rad < edges[k + 1]), 2].mean()
                  for k in range(nb)]
            zb = np.array([v for v in zb if np.isfinite(v)])
            out["profile_monotone"] = bool(np.all(np.diff(zb) <= 1e-6))
            # apex should be near the centre
            out["apex_offset_mm"] = float(
                np.linalg.norm(V0[np.argmax(V[:, 2]), :2]) * 1000)

            # ── element quality ───────────────────────────────────────────────
            p0 = V[F]
            n = np.cross(p0[:, 1] - p0[:, 0], p0[:, 2] - p0[:, 0])
            area = np.linalg.norm(n, axis=1) / 2
            p0r = V0[F]
            area0 = np.linalg.norm(np.cross(p0r[:, 1] - p0r[:, 0],
                                            p0r[:, 2] - p0r[:, 0]), axis=1) / 2
            out["min_area_ratio"] = float((area / area0).min())
            out["max_area_ratio"] = float((area / area0).max())
            # consistent orientation of the deformed normals (no flipped faces)
            out["normals_up_frac"] = float((n[:, 2] > 0).mean())

            # ── continuity in sf ──────────────────────────────────────────────
            if prev_V is not None and not out["flagged"]:
                out["dV_per_dsf_mm"] = float(
                    np.abs(V - prev_V).max() * 1000 / (sf - prev_sf))
            if not out["flagged"]:
                prev_V, prev_sf = V, sf

            # ── Laplace equilibrium at the apex ───────────────────────────────
            # section cut at x=0 runs along y = wale direction (knit_dir = 0)
            k_wale   = _apex_curvature(V, axis=0)
            k_course = _apex_curvature(V, axis=1)
            sdf = pd.read_csv(prefix + "_stress.csv").sort_values("face")
            cen = V[F[sdf["face"].values.astype(int)]].mean(axis=1)
            apex = np.linalg.norm(cen[:, :2], axis=1) < APEX_R
            if apex.any():
                N_wale   = float(sdf["T_wale_Nm"].values[apex].mean())
                N_course = float(sdf["T_course_Nm"].values[apex].mean())
                out["k_wale"], out["k_course"] = k_wale, k_course
                out["N_wale"], out["N_course"] = N_wale, N_course
                out["p_est"] = N_wale * k_wale + N_course * k_course
                out["p_err"] = out["p_est"] / r["pressure"] - 1.0
            rows.append(out)

    res = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, "uniform_sf_sweep_validation.csv")
    res.to_csv(out_csv, index=False)

    ok = res[~res["flagged"]]
    bad = res[res["flagged"]]
    print(f"Runs: {len(res)}  (converged {len(ok)}, flagged {len(bad)})\n")

    def report(name, series, limit, mode="max"):
        v = series.dropna()
        if v.empty:
            print(f"  {name:22s} no data")
            return
        worst = v.max() if mode == "max" else v.min()
        fail = (v > limit) if mode == "max" else (v < limit)
        status = "PASS" if not fail.any() else f"FAIL ({fail.sum()} runs)"
        print(f"  {name:22s} worst {worst:10.4g}   limit {limit:<8g} {status}")

    print("Converged runs:")
    report("bc max move (mm)",    ok["bc_max_move_mm"],  1e-6)
    report("z_min (mm)",          ok["z_min_mm"],        -1e-3, mode="min")
    report("apex offset (mm)",    ok["apex_offset_mm"],  60)
    report("min area ratio",      ok["min_area_ratio"],  0.5,  mode="min")
    report("max area ratio",      ok["max_area_ratio"],  4.0)
    report("normals up frac",     ok["normals_up_frac"], 1.0,  mode="min")
    report("d|V|/dsf (mm)",       ok["dV_per_dsf_mm"],   400)
    nm = int((~ok["profile_monotone"]).sum())
    print(f"  {'profile monotone':22s} {len(ok)-nm}/{len(ok)} runs "
          f"{'PASS' if nm == 0 else f'FAIL ({nm} runs)'}")

    pe = ok["p_err"].abs().dropna()
    print(f"\n  Laplace p = N_w*k_w + N_c*k_c  vs applied p:")
    print(f"    median error {pe.median()*100:5.1f}%   "
          f"90th pct {pe.quantile(0.9)*100:5.1f}%   max {pe.max()*100:5.1f}%   "
          f"{'PASS' if pe.quantile(0.9) < LAPLACE_TOL else 'CHECK'}")

    if len(bad):
        print(f"\nFlagged runs (solver failed): sf ranges")
        for m, s in bad.groupby("motif"):
            print(f"    motif{m}: {s.sf.min():.2f}-{s.sf.max():.2f} "
                  f"({len(s)} runs), z_max max {s.z_max_mm.max():.3g} mm")

    print(f"\nSaved {out_csv}")
    return res


if __name__ == "__main__":
    validate()
