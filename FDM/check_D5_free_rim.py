"""
D5 (shell with opening): is the opening rim really held by the inner cable?

The D5 drivers never pass "fixed_vertices" to fem_batch_nregion, so the binary
fixes the WHOLE boundary, including the 42 rim vertices of the opening.  The
inner cable is then inert and the rim is clamped at its target position.

This script re-solves the published symmetric best fit (d5_sym_pfix) with the
rim released (only the z = 0 ground ring and the rim's ground-contact vertex
fixed) and compares the rim-support options:

  clamped   : the published model (whole boundary fixed)
  cable     : rim free, inner cable only (EA 157 kN, rest scale 0.95)
  ring<d>   : rim free, GFRP rod of diameter d mm (E 40 GPa) formed to the rim,
              pinned at its ground-contact vertex (an arch from ground to ground)

Follower pressure throughout for the released cases (volume work is wrong at
free edges, see fabsim pressure note), 1000 Pa.

Usage: python3 check_D5_free_rim.py [--rings 6 10 16 20]
"""
import argparse, json, os, subprocess, tempfile
import numpy as np
import optimise_D5_symmetric as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "optimisation", "d5_free_rim")
BEST = os.path.join(HERE, "optimisation", "d5_sym_pfix_optimised.json")
RMAP = os.path.join(HERE, "optimisation", "d5_sym_pfix_region_map.json")
E_GFRP = 40e9


def rim_order_from_ground(cable_idx, V):
    i0 = int(np.argmin(V[cable_idx, 2]))
    return cable_idx[i0:] + cable_idx[:i0 + 1]          # closed, starts/ends at ground


def run(name, params, mesh, rmap_path, follower):
    os.makedirs(OUT, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=OUT) as pf:
        json.dump(params, pf)
    env = dict(os.environ)
    if follower:
        env.update(FEM_PRESSURE="follower", FEM_FOLLOWER_START="volume")
    prefix = os.path.join(OUT, name)
    r = subprocess.run([S.BINARY, mesh, rmap_path, pf.name, prefix],
                       capture_output=True, text=True, timeout=900, env=env)
    os.unlink(pf.name)
    res = [float(l.split("max=")[1].split()[0]) for l in r.stderr.splitlines()
           if l.startswith("SOLVER_RESIDUAL")]
    vp = prefix + "_verts.csv"
    X = np.loadtxt(vp, delimiter=",", skiprows=1)[:, 1:] if os.path.exists(vp) else None
    return X, (res[-1] if res else np.nan), r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rings", type=float, nargs="*", default=[6, 10, 16, 20])
    args = ap.parse_args()

    V, F = S.load_off(S.MESH)
    field = json.load(open(S.FIELD_J))
    cable = json.load(open(S.CABLE_J))["vertex_indices"]
    best = json.load(open(BEST))
    regions = [{"sf_wale": r["sf_wale"], "sf_course": r["sf_course"],
                "knit_dir_deg": r["knit_dir_deg"]} for r in best["regions"]]

    rmap = json.load(open(RMAP))
    rmap["face_knit_dirs_deg"] = [
        float(np.degrees(np.arctan2(field[str(f)]["d1"][1], field[str(f)]["d1"][0])) % 180)
        if str(f) in field else 0.0 for f in range(len(F))]
    os.makedirs(OUT, exist_ok=True)
    rmap_path = os.path.join(OUT, "rmap.json")
    json.dump(rmap, open(rmap_path, "w"))

    ground = [int(v) for v in np.where(V[:, 2] < 1e-6)[0]]
    rim = rim_order_from_ground(cable, V)
    fixed = sorted(set(ground) | {rim[0]})
    rim_set = set(cable)
    interior = [v for v in range(len(V)) if v not in set(ground)]
    print(f"{len(V)} verts, {len(ground)} ground, rim {len(cable)} (ground contact v{rim[0]}, "
          f"z {V[rim[0], 2]*1000:.1f} mm)")

    base = {"pressure": S.PRESSURE, "motif": 5, "regions": regions, "newton_reg_max": 1e6}
    cable_p = {"cable_ea": S.CABLE_EA, "cable_paths": [cable + [cable[0]]],
               "cable_rest_scales": [S.CABLE_SCALE]}
    cases = [("clamped", {**base, **cable_p}, False),
             ("cable", {**base, **cable_p, "fixed_vertices": fixed}, True)]
    for d in args.rings:
        dm = d / 1000.0
        cases.append((f"ring{d:g}", {**base, "fixed_vertices": fixed, "spline_paths": [rim],
                                     "spline_EA": E_GFRP * np.pi * dm**2 / 4,
                                     "spline_EI": E_GFRP * np.pi * dm**4 / 64,
                                     "spline_rest": 1}, True))

    summary = {}
    for name, p, fol in cases:
        X, res, err = run(name, p, S.MESH, rmap_path, fol)
        if X is None:
            print(f"{name:9s} FAILED\n{err[-600:]}"); continue
        d = np.linalg.norm(X - V, axis=1)
        rm = np.array(cable)
        row = dict(residual=res, converged=bool(res < 1e-3),
                   rmse_mm=float(np.sqrt(np.mean(d[interior]**2)) * 1000),
                   max_mm=float(d[interior].max() * 1000),
                   rim_rmse_mm=float(np.sqrt(np.mean(d[rm]**2)) * 1000),
                   rim_max_mm=float(d[rm].max() * 1000),
                   rim_max_out_mm=float(((X[rm] - V[rm]) @ np.array([0, -1, 0])).max() * 1000))
        summary[name] = row
        print(f"{name:9s} res {res:.1e} {'ok ' if row['converged'] else 'NOT'}  "
              f"RMSE {row['rmse_mm']:6.2f}  max {row['max_mm']:6.1f}  | rim RMSE "
              f"{row['rim_rmse_mm']:6.2f}  rim max {row['rim_max_mm']:6.1f} mm")
    json.dump({"fixed_vertices": fixed, "rim_path": rim, "cases": summary},
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
