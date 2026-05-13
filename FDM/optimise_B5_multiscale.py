"""
B5 scale comparison: 9-region FEM inverse optimisation at 1.2 m, 1.5 m, 2.0 m, 3.0 m.

For each diameter the base B5_remeshed_shared.off is uniformly scaled, written to
data/B5_remeshed_shared_<tag>.off, and the full L-BFGS-B optimisation is run.
Cable paths (vertex indices) are scale-independent and reused unchanged.

Results saved to:
  FDM/optimisation/B5_<tag>_optimised_params.json   — per-scale result
  FDM/optimisation/B5_<tag>_final_stress.csv         — face-level tension at optimum
  FDM/optimisation/B5_multiscale_summary.json        — combined table

A summary comparison table including knit tensions is printed at the end.

Usage:
    python3 optimise_B5_multiscale.py [--motif 1] [--pressure 1000] [--maxiter 200]
    python3 optimise_B5_multiscale.py --diameters 1.5 2.0   # subset
    python3 optimise_B5_multiscale.py --postprocess          # add tension to 1.2m result
"""
import argparse, csv, json, os, shutil, sys, subprocess, tempfile
import numpy as np
from scipy.optimize import minimize

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(HERE, "..", "data")
OUT_DIR   = os.path.join(HERE, "optimisation")

BASE_MESH = os.path.join(DATA_DIR, "B5_remeshed_shared.off")
CABLE_PATHS_FILE = os.environ.get(
    "FEM_CABLE_PATHS", os.path.join(DATA_DIR, "cable_paths_B5_remeshed.json"))
BINARY = os.environ.get(
    "FEM_BINARY_NREGION", os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))

KNIT_DIR = {(r, c): 0 for r in range(3) for c in range(3)}
ALL_DIAMETERS = [1.2, 1.5, 2.0, 3.0]

# ── OFF helpers ────────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    F = np.array([[int(x) for x in l.split()[1:]] for l in lines[2+nv:2+nv+nf]])
    return V, F

def save_off(path, V, F):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in F:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def diameter_tag(d):
    return f"{d:.1f}m".replace(".", "p")

def scaled_mesh_path(diameter):
    tag = diameter_tag(diameter)
    if abs(diameter - 1.2) < 1e-6:
        return BASE_MESH
    return os.path.join(DATA_DIR, f"B5_remeshed_shared_{tag}.off")

def ensure_scaled_mesh(diameter):
    path = scaled_mesh_path(diameter)
    if os.path.exists(path):
        return path
    V0, F0 = load_off(BASE_MESH)
    base_span = V0[:,0].max() - V0[:,0].min()
    V_scaled  = V0 * (diameter / base_span)
    save_off(path, V_scaled, F0)
    print(f"  Created {os.path.basename(path)}")
    return path

# ── Region map ────────────────────────────────────────────────────────────────
def make_region_map(V, F, x_lo, x_hi, y_lo, y_hi):
    face_cx = V[F[:,0],0]/3 + V[F[:,1],0]/3 + V[F[:,2],0]/3
    face_cy = V[F[:,0],1]/3 + V[F[:,1],1]/3 + V[F[:,2],1]/3
    regions = []
    for cx, cy in zip(face_cx, face_cy):
        col = 0 if cx < x_lo else (1 if cx < x_hi else 2)
        row = 0 if cy < y_lo else (1 if cy < y_hi else 2)
        regions.append(row * 3 + col)
    return regions

# ── Cable paths ───────────────────────────────────────────────────────────────
def load_cable_paths(path):
    if not os.path.exists(path):
        print(f"WARNING: cable paths not found: {path}")
        return {}
    with open(path) as f:
        d = json.load(f)
    return d if isinstance(d, dict) else {str(i): v for i, v in enumerate(d)}

# ── FEM runner ────────────────────────────────────────────────────────────────
_call_count = [0]

def run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
            region_map_path, cable_paths_list, cable_ea,
            cable_rest_scales=None, keep_stress=False):
    """Run FEM binary, return dict with scalars, verts, and optionally stress."""
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":    float(pressure),
        "motif":       int(motif),
        "cable_ea":    float(cable_ea),
        "cable_paths": cable_paths_list if cable_paths_list else [],
        "regions": [
            {"sf_wale":      float(sf_w[r]),
             "sf_course":    float(sf_c[r]),
             "knit_dir_deg": float(knit_dirs[r])}
            for r in range(9)
        ],
    }
    if cable_rest_scales is not None:
        params["cable_rest_scales"] = [float(x) for x in cable_rest_scales]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(OUT_DIR, f"run_{_call_count[0]:05d}")
    cmd = [BINARY, mesh_path, region_map_path, params_path, prefix]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  FEM error: {result.stderr[:200]}")
            return None

        scalars_path = prefix + "_scalars.csv"
        if not os.path.exists(scalars_path):
            return None

        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        out = {k: float(v) for k, v in row.items()}
        out["_prefix"] = prefix

        verts_path = prefix + "_verts.csv"
        if os.path.exists(verts_path):
            vdata = np.loadtxt(verts_path, delimiter=",", skiprows=1)
            out["verts"] = vdata[:, 1:]

        # Always read stress CSV (columns: face,S11,S22,S12,von_mises,p1,p2,T_wale,T_course)
        stress_path = prefix + "_stress.csv"
        if os.path.exists(stress_path):
            sdata = np.loadtxt(stress_path, delimiter=",", skiprows=1)
            out["T_wale_per_face"]   = sdata[:, 7]
            out["T_course_per_face"] = sdata[:, 8]
            out["T_wale_mean"]    = float(np.mean(sdata[:, 7]))
            out["T_course_mean"]  = float(np.mean(sdata[:, 8]))
            out["T_wale_max"]     = float(np.max(sdata[:, 7]))
            out["T_course_max"]   = float(np.max(sdata[:, 8]))
            if keep_stress:
                out["_stress_path"] = stress_path

        return out
    except Exception as e:
        print(f"  FEM exception: {e}")
        return None
    finally:
        try: os.unlink(params_path)
        except: pass

# ── Per-region tension summary ─────────────────────────────────────────────────
def compute_region_tension(out_fem, region_ids):
    """Return list of per-region tension dicts from a FEM result with stress data."""
    if out_fem is None or "T_wale_per_face" not in out_fem:
        return []
    T_w  = out_fem["T_wale_per_face"]
    T_c  = out_fem["T_course_per_face"]
    ids  = np.array(region_ids)
    result = []
    for r in range(9):
        mask = ids == r
        if not mask.any():
            result.append({"region_id": r,
                           "T_wale_mean": float("nan"), "T_course_mean": float("nan"),
                           "T_wale_max":  float("nan"), "T_course_max":  float("nan")})
        else:
            result.append({"region_id": r,
                           "T_wale_mean":   float(np.mean(T_w[mask])),
                           "T_course_mean": float(np.mean(T_c[mask])),
                           "T_wale_max":    float(np.max(T_w[mask])),
                           "T_course_max":  float(np.max(T_c[mask]))})
    return result

# ── Objective ─────────────────────────────────────────────────────────────────
def make_objective(mesh_path, V_target, interior_idx, region_map_path,
                   knit_dirs, pressure, motif, cable_paths_list, cable_ea):
    t_crown = float(V_target[:,2].max())
    history = []

    def objective(p):
        sf_w   = p[:9]
        sf_c   = p[9:18]
        scales = p[18:] if len(p) > 18 else None
        out = run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
                      region_map_path, cable_paths_list, cable_ea,
                      cable_rest_scales=scales)
        if out is None:
            return 1e6

        if "verts" in out:
            diff = out["verts"][interior_idx] - V_target[interior_idx]
            loss = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        else:
            loss = abs(out["crown_height"] - t_crown) / (t_crown + 1e-6)

        history.append({"call": _call_count[0], "loss": loss,
                        "crown_height": out.get("crown_height", 0.0)})
        if _call_count[0] % 10 == 0 or _call_count[0] == 1:
            print(f"    [{_call_count[0]:4d}]  rmse={loss:.4f} m  "
                  f"crown={out.get('crown_height',0):.4f} m  (target {t_crown:.4f})")
        return loss

    return objective, history

# ── Per-diameter optimisation ──────────────────────────────────────────────────
def optimise_one(diameter, motif, pressure, maxiter, cable_paths_dict):
    tag = diameter_tag(diameter)
    print(f"\n{'='*65}")
    print(f"  Diameter = {diameter} m  (tag={tag})")
    print(f"{'='*65}")

    mesh_path = ensure_scaled_mesh(diameter)
    V, F = load_off(mesh_path)
    fem_span  = V[:,0].max() - V[:,0].min()
    print(f"  Mesh: {len(V)}v / {len(F)}f  span={fem_span:.4f} m")

    V_target     = V.copy()
    bdry_mask    = np.abs(V_target[:,2]) < 0.01 * diameter
    interior_idx = np.where(~bdry_mask)[0]
    t_crown      = float(V_target[:,2].max())
    print(f"  Target crown={t_crown:.4f} m  interior={len(interior_idx)} verts")

    cable_names      = list(cable_paths_dict.keys())
    cable_paths_list = list(cable_paths_dict.values())
    n_cables         = len(cable_paths_list)
    cable_ea         = 157000.0
    print(f"  Cables: {n_cables} sections  EA={cable_ea:.0f} N  pressure={pressure} Pa")

    R = fem_span / 2
    region_ids = make_region_map(V, F, -R/3, R/3, -R/3, R/3)
    knit_dirs  = [KNIT_DIR[(r//3, r%3)] for r in range(9)]

    region_map_path = os.path.join(OUT_DIR, f"region_map_{tag}.json")
    with open(region_map_path, "w") as f:
        json.dump({"face_regions": region_ids}, f)
    counts = np.bincount(region_ids, minlength=9)
    print(f"  Regions: {counts.tolist()}")

    # Initial sanity check
    p0 = np.concatenate([np.full(18, 1.041), np.full(n_cables, 1.0)])
    print(f"\n  Sanity check (sf=1.041)...")
    out0 = run_fem(mesh_path, p0[:9], p0[9:18], knit_dirs, pressure, motif,
                   region_map_path, cable_paths_list, cable_ea,
                   cable_rest_scales=p0[18:])
    if out0 is None:
        print("  ERROR: FEM failed at initial point — skipping")
        return None
    diff0 = out0["verts"][interior_idx] - V_target[interior_idx]
    rmse0 = float(np.sqrt(np.mean(np.sum(diff0**2, axis=1))))
    print(f"  crown={out0['crown_height']:.4f} m  RMSE={rmse0:.4f} m  "
          f"T_wale={out0.get('T_wale_mean', float('nan')):.1f} N/m  "
          f"T_course={out0.get('T_course_mean', float('nan')):.1f} N/m")

    # L-BFGS-B optimisation
    bounds = [(0.9, 2.0)] * 18 + [(0.85, 1.05)] * n_cables
    print(f"\n  Optimising {18+n_cables} params, maxiter={maxiter}...")
    objective, history = make_objective(
        mesh_path, V_target, interior_idx, region_map_path,
        knit_dirs, pressure, motif, cable_paths_list, cable_ea)

    result = minimize(
        objective, p0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6, "eps": 0.05},
    )

    p_opt  = result.x
    sf_w   = p_opt[:9]
    sf_c   = p_opt[9:18]
    scales = p_opt[18:]

    print(f"\n  Converged: {result.success}  |  {result.message}")
    print(f"  Final RMSE: {result.fun:.4f} m   calls: {_call_count[0]}")
    print(f"  RMSE / diameter: {result.fun / diameter:.4f}")

    # Final FEM run: keep stress CSV for tension analysis
    print("  Running final FEM (with stress output)...")
    out_final = run_fem(mesh_path, sf_w, sf_c, knit_dirs, pressure, motif,
                        region_map_path, cable_paths_list, cable_ea,
                        cable_rest_scales=scales, keep_stress=True)
    final_crown = out_final.get("crown_height", float("nan")) if out_final else float("nan")

    # Persist stress CSV with a stable name
    final_stress_dest = os.path.join(OUT_DIR, f"B5_{tag}_final_stress.csv")
    if out_final and "_stress_path" in out_final and os.path.exists(out_final["_stress_path"]):
        shutil.copy(out_final["_stress_path"], final_stress_dest)
        print(f"  Stress saved: {os.path.basename(final_stress_dest)}")

    # Per-region tension
    tension_regions = compute_region_tension(out_final, region_ids)
    T_w_global = out_final.get("T_wale_mean",   float("nan")) if out_final else float("nan")
    T_c_global = out_final.get("T_course_mean", float("nan")) if out_final else float("nan")
    T_w_max    = out_final.get("T_wale_max",    float("nan")) if out_final else float("nan")
    T_c_max    = out_final.get("T_course_max",  float("nan")) if out_final else float("nan")

    print(f"  T_wale  mean={T_w_global:.1f}  max={T_w_max:.1f}  N/m")
    print(f"  T_course mean={T_c_global:.1f}  max={T_c_max:.1f}  N/m")

    result_dict = {
        "diameter_m":      diameter,
        "tag":             tag,
        "motif":           motif,
        "pressure_pa":     pressure,
        "fem_mesh":        mesh_path,
        "target_crown_m":  float(t_crown),
        "final_crown_m":   float(final_crown),
        "converged":       bool(result.success),
        "loss_rmse_m":     float(result.fun),
        "rmse_normalised": float(result.fun / diameter),
        "n_calls":         _call_count[0],
        "tension_global": {
            "T_wale_mean_Npm":    T_w_global,
            "T_course_mean_Npm":  T_c_global,
            "T_wale_max_Npm":     T_w_max,
            "T_course_max_Npm":   T_c_max,
            # normalise by p*D (Laplace scaling): constant if self-similar
            "T_wale_norm":   T_w_global / (pressure * diameter) if diameter else float("nan"),
            "T_course_norm": T_c_global / (pressure * diameter) if diameter else float("nan"),
        },
        "tension_regions": tension_regions,
        "regions": [
            {"region_id": r, "row": r//3, "col": r%3,
             "knit_dir_deg": knit_dirs[r],
             "sf_wale": float(sf_w[r]), "sf_course": float(sf_c[r])}
            for r in range(9)
        ],
        "cables": [
            {"name": cable_names[i], "rest_scale": float(scales[i])}
            for i in range(n_cables)
        ],
        "history": history[-30:],
    }

    out_json = os.path.join(OUT_DIR, f"B5_{tag}_optimised_params.json")
    with open(out_json, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"  Saved: {out_json}")
    return result_dict

# ── Post-process 1.2m result: add tension from its stress CSV ─────────────────
def postprocess_1p2m():
    """Patch B5_1p2m_optimised_params.json with tension data from its final stress CSV."""
    tag      = "1p2m"
    json_path   = os.path.join(OUT_DIR, f"B5_{tag}_optimised_params.json")
    stress_path = os.path.join(OUT_DIR, f"B5_{tag}_final_stress.csv")

    if not os.path.exists(json_path):
        print(f"Not found: {json_path}")
        return None

    with open(json_path) as f:
        d = json.load(f)

    if "tension_global" in d:
        print(f"  {json_path} already has tension data — skipping patch")
        return d

    # Re-run one FEM call with the saved optimal params to regenerate stress CSV
    if not os.path.exists(stress_path):
        print(f"  Stress CSV not found at {stress_path} — re-running final FEM...")
        mesh_path       = BASE_MESH
        cable_paths_dict = load_cable_paths(CABLE_PATHS_FILE)
        cable_paths_list = list(cable_paths_dict.values())

        V, F = load_off(mesh_path)
        fem_span = V[:,0].max() - V[:,0].min()
        R = fem_span / 2
        region_ids = make_region_map(V, F, -R/3, R/3, -R/3, R/3)
        region_map_path = os.path.join(OUT_DIR, f"region_map_{tag}.json")
        with open(region_map_path, "w") as f2:
            json.dump({"face_regions": region_ids}, f2)

        sf_w    = [reg["sf_wale"]   for reg in d["regions"]]
        sf_c    = [reg["sf_course"] for reg in d["regions"]]
        knit_dirs = [reg["knit_dir_deg"] for reg in d["regions"]]
        scales  = [c["rest_scale"]  for c in d["cables"]]

        out = run_fem(mesh_path, sf_w, sf_c, knit_dirs, d["pressure_pa"], d["motif"],
                      region_map_path, cable_paths_list, 157000.0,
                      cable_rest_scales=scales, keep_stress=True)

        if out and "_stress_path" in out:
            shutil.copy(out["_stress_path"], stress_path)
        else:
            print("  ERROR: could not regenerate stress data")
            return None

        # Recompute region_ids for tension
        sdata = np.loadtxt(stress_path, delimiter=",", skiprows=1)
        T_w = sdata[:, 7]
        T_c = sdata[:, 8]
    else:
        sdata = np.loadtxt(stress_path, delimiter=",", skiprows=1)
        T_w = sdata[:, 7]
        T_c = sdata[:, 8]
        mesh_path = BASE_MESH
        V, F = load_off(mesh_path)
        fem_span = V[:,0].max() - V[:,0].min()
        R = fem_span / 2
        region_ids = make_region_map(V, F, -R/3, R/3, -R/3, R/3)

    diameter = d["diameter_m"]
    pressure = d["pressure_pa"]
    d["tension_global"] = {
        "T_wale_mean_Npm":   float(np.mean(T_w)),
        "T_course_mean_Npm": float(np.mean(T_c)),
        "T_wale_max_Npm":    float(np.max(T_w)),
        "T_course_max_Npm":  float(np.max(T_c)),
        "T_wale_norm":   float(np.mean(T_w)) / (pressure * diameter),
        "T_course_norm": float(np.mean(T_c)) / (pressure * diameter),
    }
    d["tension_regions"] = []
    ids = np.array(region_ids)
    for r in range(9):
        mask = ids == r
        d["tension_regions"].append({
            "region_id": r,
            "T_wale_mean":   float(np.mean(T_w[mask])),
            "T_course_mean": float(np.mean(T_c[mask])),
            "T_wale_max":    float(np.max(T_w[mask])),
            "T_course_max":  float(np.max(T_c[mask])),
        })

    with open(json_path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Patched {json_path} with tension data")
    tw = d["tension_global"]
    print(f"  T_wale mean={tw['T_wale_mean_Npm']:.1f}  max={tw['T_wale_max_Npm']:.1f}  N/m")
    print(f"  T_course mean={tw['T_course_mean_Npm']:.1f}  max={tw['T_course_max_Npm']:.1f}  N/m")
    return d

# ── Comparison tables ──────────────────────────────────────────────────────────
def print_comparison(all_results):
    print(f"\n{'='*80}")
    print(f"  B5 SCALE COMPARISON  (pressure=1000 Pa, motif=1, cable EA=157000 N)")
    print(f"{'='*80}")

    # ── Shape / convergence ──
    print(f"\n  Shape fit:")
    print(f"  {'Diam':>6}  {'Conv':>5}  {'RMSE[m]':>9}  {'RMSE/D':>7}  "
          f"{'Crown_t[m]':>10}  {'Crown_f[m]':>10}  {'sf_w_mean':>9}  {'sf_c_mean':>9}")
    print("  " + "-"*78)
    for r in all_results:
        if r is None: continue
        sf_w_m = np.mean([reg["sf_wale"]   for reg in r["regions"]])
        sf_c_m = np.mean([reg["sf_course"] for reg in r["regions"]])
        print(f"  {r['diameter_m']:>5.1f}m  "
              f"{'Y' if r['converged'] else 'N':>5}  "
              f"{r['loss_rmse_m']:>9.4f}  "
              f"{r['rmse_normalised']:>7.4f}  "
              f"{r['target_crown_m']:>10.4f}  "
              f"{r['final_crown_m']:>10.4f}  "
              f"{sf_w_m:>9.4f}  "
              f"{sf_c_m:>9.4f}")

    # ── Knit tension ──
    print(f"\n  Knit tension at optimum (mean across all faces):")
    print(f"  {'Diam':>6}  {'T_wale[N/m]':>11}  {'T_course[N/m]':>13}  "
          f"{'T_w_max':>9}  {'T_c_max':>9}  {'T_w/(p·D)':>10}  {'T_c/(p·D)':>10}")
    print("  " + "-"*78)
    for r in all_results:
        if r is None: continue
        tg = r.get("tension_global", {})
        tw  = tg.get("T_wale_mean_Npm",  float("nan"))
        tc  = tg.get("T_course_mean_Npm", float("nan"))
        twx = tg.get("T_wale_max_Npm",   float("nan"))
        tcx = tg.get("T_course_max_Npm", float("nan"))
        twn = tg.get("T_wale_norm",   float("nan"))
        tcn = tg.get("T_course_norm", float("nan"))
        print(f"  {r['diameter_m']:>5.1f}m  "
              f"{tw:>11.1f}  {tc:>13.1f}  "
              f"{twx:>9.1f}  {tcx:>9.1f}  "
              f"{twn:>10.4f}  {tcn:>10.4f}")

    # ── Per-region sf ──
    print(f"\n  Per-region sf_wale / sf_course:")
    headers = "  ".join(f"{'d='+str(r['diameter_m'])+'m':>15}"
                        for r in all_results if r)
    print(f"  {'Region':>6}  {headers}")
    for reg_id in range(9):
        row_s = f"  {reg_id:>6}  "
        for r in all_results:
            if r is None: continue
            reg = r["regions"][reg_id]
            row_s += f"  w={reg['sf_wale']:.3f}/c={reg['sf_course']:.3f}  "
        print(row_s)

    # ── Per-region tension ──
    if any(r and r.get("tension_regions") for r in all_results):
        print(f"\n  Per-region mean T_wale [N/m] at optimum:")
        print(f"  {'Region':>6}  {headers}")
        for reg_id in range(9):
            row_s = f"  {reg_id:>6}  "
            for r in all_results:
                if r is None: continue
                treg = next((t for t in r.get("tension_regions",[])
                             if t["region_id"]==reg_id), {})
                row_s += f"  {'Tw='+f\"{treg.get('T_wale_mean',float('nan')):.0f}\"}" \
                         f"/Tc={treg.get('T_course_mean',float('nan')):.0f} N/m  "
            print(row_s)

    # ── Conclusion ──
    _print_conclusion(all_results)
    print(f"\n{'='*80}")

def _print_conclusion(all_results):
    valid = [r for r in all_results if r and r.get("tension_global")]
    if len(valid) < 2:
        return

    print(f"\n  ── Conclusion ──────────────────────────────────────────────────")

    diams = [r["diameter_m"] for r in valid]
    tw_means = [r["tension_global"]["T_wale_mean_Npm"]   for r in valid]
    tc_means = [r["tension_global"]["T_course_mean_Npm"] for r in valid]
    tw_norms = [r["tension_global"]["T_wale_norm"]   for r in valid]
    tc_norms = [r["tension_global"]["T_course_norm"] for r in valid]

    # Linear fit of T vs D
    tw_fit = np.polyfit(diams, tw_means, 1)
    tc_fit = np.polyfit(diams, tc_means, 1)

    print(f"  Scaling law: T_wale  ≈ {tw_fit[0]:+.1f}·D + {tw_fit[1]:+.1f}  N/m  "
          f"(R²={np.corrcoef(diams, tw_means)[0,1]**2:.3f})")
    print(f"               T_course ≈ {tc_fit[0]:+.1f}·D + {tc_fit[1]:+.1f}  N/m  "
          f"(R²={np.corrcoef(diams, tc_means)[0,1]**2:.3f})")

    tw_cv = np.std(tw_norms) / np.mean(tw_norms) if np.mean(tw_norms) else float("nan")
    tc_cv = np.std(tc_norms) / np.mean(tc_norms) if np.mean(tc_norms) else float("nan")
    print(f"  T/(p·D): wale CV={tw_cv:.3f}  course CV={tc_cv:.3f}  "
          f"({'self-similar' if max(tw_cv,tc_cv)<0.1 else 'not self-similar'})")

    sf_w_all = [[reg["sf_wale"]   for reg in r["regions"]] for r in valid]
    sf_c_all = [[reg["sf_course"] for reg in r["regions"]] for r in valid]
    sf_w_means = [np.mean(s) for s in sf_w_all]
    sf_c_means = [np.mean(s) for s in sf_c_all]
    print(f"  Mean sf_wale range: {min(sf_w_means):.3f} – {max(sf_w_means):.3f}  "
          f"(Δ={max(sf_w_means)-min(sf_w_means):.3f})")
    print(f"  Mean sf_course range: {min(sf_c_means):.3f} – {max(sf_c_means):.3f}  "
          f"(Δ={max(sf_c_means)-min(sf_c_means):.3f})")

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif",       type=int,   default=1)
    parser.add_argument("--pressure",    type=float, default=1000.0)
    parser.add_argument("--maxiter",     type=int,   default=200)
    parser.add_argument("--diameters",   type=float, nargs="+", default=ALL_DIAMETERS)
    parser.add_argument("--postprocess", action="store_true",
                        help="Patch 1.2m result with tension data and regenerate summary")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.postprocess:
        d = postprocess_1p2m()
        # Reload all results and reprint
        all_results = []
        for diam in sorted(ALL_DIAMETERS):
            tag = diameter_tag(diam)
            jp  = os.path.join(OUT_DIR, f"B5_{tag}_optimised_params.json")
            if os.path.exists(jp):
                with open(jp) as f:
                    all_results.append(json.load(f))
            else:
                all_results.append(None)
        if any(r for r in all_results):
            print_comparison(all_results)
            summary_path = os.path.join(OUT_DIR, "B5_multiscale_summary.json")
            with open(summary_path, "w") as f:
                json.dump([r for r in all_results if r], f, indent=2)
            print(f"\nSummary saved: {summary_path}")
        return

    if not os.path.exists(BINARY):
        print(f"FEM binary not found: {BINARY}")
        sys.exit(1)

    cable_paths_dict = load_cable_paths(CABLE_PATHS_FILE)
    if not cable_paths_dict:
        print("ERROR: no cable paths loaded")
        sys.exit(1)
    print(f"Loaded {len(cable_paths_dict)} cable sections from {CABLE_PATHS_FILE}")

    all_results = []
    for d in sorted(args.diameters):
        res = optimise_one(d, args.motif, args.pressure, args.maxiter, cable_paths_dict)
        all_results.append(res)

    if any(r is not None for r in all_results):
        print_comparison(all_results)

    summary = [r for r in all_results if r is not None]
    summary_path = os.path.join(OUT_DIR, "B5_multiscale_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

if __name__ == "__main__":
    main()
