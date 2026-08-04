"""
Knitting-direction sweep on a NON-axisymmetric boundary.

The circular-dome sweep (run_knit_dir_sweep.py) finds that theta_knit changes no
rotationally invariant scalar, because rotating the material axes on a circular
domain merely rotates the whole solution.  That argument fails the moment the
boundary has a preferred direction.  This script tests it: the same circular mesh
is scaled x2 in x to give an ellipse (semi-axes 1194 x 598 mm), and the same
sweep is run on it.

Now geometry and material each define a frame, and theta_knit is the angle
between them.  There is no theta -> 90-theta identity any more (that one came
from the circle's x<->y mirror symmetry), so the curves are genuinely asymmetric
about 45 degrees and must NOT be symmetrised when plotted.  theta in [0, 90]
still covers every distinct configuration: reflecting in the y axis maps the
ellipse to itself and the wale axis from theta to -theta = 180-theta, and an
orthotropic frame is invariant under a 180 degree turn.

Outputs (kept entirely separate from the circular-dome data):
    data/ellipse2x_flat.off          the mesh
    data/knit_dir_sweep_ellipse.csv  one row per run
    data/knit_dir_sweep_ellipse/     per-run vertex and stress files

Metrics recorded per run:
  global   crown_height, mean_stress, max_stress
  section  H_fit_*, H_binned_*, von_mises_*, r_*   (same estimators as figM)
  apex     k_wale, k_course, k_x, k_y, k_min, k_max, k_max_dir_deg
           (apex_curvature.py — the section estimators are nearly blind to
           directional anisotropy, see that module's docstring)

The apex fit neighbourhood is selected on the *unscaled circular* mesh, so the
ellipse and the circle are fitted from the identical 35 vertices.

Usage:
  python3 run_knit_dir_sweep_ellipse.py                 # 1 deg steps, both motifs
  python3 run_knit_dir_sweep_ellipse.py --step 3
  python3 run_knit_dir_sweep_ellipse.py --make-mesh     # (re)write the OFF only
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off, compute_curvatures
from fea_interface import run_fea, check_binary
from plot_section_profiles import _slice_plane
from plot_section_sensitivity import (_profile_roughness, _SECTION_TOL,
                                     ROUGHNESS_THRESHOLD, CROWN_MIN_M)
from section_curvature import profile_curvature_fit, profile_curvature_binned
from apex_curvature import apex_curvature

REPO_DATA   = os.path.join(os.path.dirname(os.path.dirname(
                  os.path.abspath(__file__))), "data")
ELLIPSE_OFF = os.path.join(REPO_DATA, "ellipse2x_flat.off")
SWEEP_CSV   = os.path.join(DATA_DIR, "knit_dir_sweep_ellipse.csv")
SWEEP_DIR   = os.path.join(DATA_DIR, "knit_dir_sweep_ellipse")

X_SCALE     = 2.0
THETA_RANGE = (0.0, 90.0)
SF          = 1.0
PRESSURE    = 1000.0
MOTIFS      = (1, 2)


def make_mesh(force=False):
    """Write the x-scaled ellipse, preserving topology and vertex order."""
    if os.path.exists(ELLIPSE_OFF) and not force:
        return ELLIPSE_OFF
    V, F = read_off(MESH_PATH)
    V = V.copy()
    V[:, 0] *= X_SCALE
    with open(ELLIPSE_OFF, "w") as fh:
        fh.write("OFF\n%d %d 0\n" % (len(V), len(F)))
        for v in V:
            fh.write("%.9g %.9g %.9g\n" % tuple(v))
        for t in F:
            fh.write("3 %d %d %d\n" % tuple(t))
    print(f"Wrote {ELLIPSE_OFF}  (semi-axes "
          f"{V[:, 0].max()*1000:.0f} x {V[:, 1].max()*1000:.0f} mm)")
    return ELLIPSE_OFF


def _metrics(verts_path, stress_path, knit_dir_deg, faces, ref_verts):
    """Section metrics (as in figM) plus apex principal curvatures."""
    out = {}
    V = pd.read_csv(verts_path).sort_values("vid")[["x", "y", "z"]].values
    H = compute_curvatures(V, faces)["H"]

    for key, axis in (("x0", 0), ("y0", 1)):
        pos, z_mm, _ = _slice_plane(V, faces, H, fixed_axis=axis)
        if len(pos) < 6:
            out[f"H_fit_{key}"] = out[f"H_binned_{key}"] = np.nan
            out[f"r_{key}"] = np.nan
            continue
        out[f"H_fit_{key}"]    = profile_curvature_fit(pos, z_mm)
        out[f"H_binned_{key}"] = profile_curvature_binned(pos, z_mm)
        out[f"r_{key}"]        = _profile_roughness(z_mm)
        out[f"span_{key}"]     = float(pos.max() - pos.min())

    ap = apex_curvature(V, knit_dir_deg, ref_verts=ref_verts)
    if ap is not None:
        out.update({f"apex_{k}": v for k, v in ap.items()})

    sdf = pd.read_csv(stress_path).sort_values("face")
    fid = sdf["face"].values.astype(int)
    cen = V[faces[fid]].mean(axis=1)
    for key, axis in (("x0", 0), ("y0", 1)):
        m = np.abs(cen[:, axis]) < _SECTION_TOL
        out[f"von_mises_{key}"] = (float(np.mean(sdf["von_mises"].values[m]))
                                   if m.any() else np.nan)
    m = np.abs(cen[:, 0]) < _SECTION_TOL
    out["T_wale_x0"] = float(np.mean(sdf["T_wale_Nm"].values[m])) if m.any() else np.nan
    m = np.abs(cen[:, 1]) < _SECTION_TOL
    out["T_course_y0"] = float(np.mean(sdf["T_course_Nm"].values[m])) if m.any() else np.nan
    return out


def run_sweep(step=1.0, sf=SF, pressure=PRESSURE, motifs=MOTIFS):
    check_binary()
    mesh = make_mesh()
    os.makedirs(SWEEP_DIR, exist_ok=True)

    _, faces = read_off(mesh)
    ref_verts, _ = read_off(MESH_PATH)      # unscaled circle: apex-fit neighbourhood

    n = int(round((THETA_RANGE[1] - THETA_RANGE[0]) / step)) + 1
    thetas = np.linspace(*THETA_RANGE, n)
    print(f"Ellipse sweep: {n} angles x {len(motifs)} motifs "
          f"(sf_wale = sf_course = {sf}, p = {pressure} Pa)")

    rows = []
    for motif in motifs:
        for i, th in enumerate(thetas):
            prefix = os.path.join(SWEEP_DIR, f"m{motif}_th{th:06.2f}")
            try:
                res = run_fea(sf, sf, th, pressure, motif, prefix,
                              timeout=600, mesh_path=mesh)
            except Exception as exc:
                print(f"  motif{motif} theta={th:5.1f}  FAILED: {exc}")
                rows.append({"motif": motif, "knit_dir": th, "sim_failed": True})
                continue
            row = {"motif": motif, "knit_dir": th, "sf_wale": sf,
                   "sf_course": sf, "pressure": pressure,
                   "crown_height": res["crown_height"],
                   "mean_stress":  res["mean_stress"],
                   "max_stress":   res["max_stress"],
                   "verts_path":   res["verts_path"],
                   "stress_path":  res["stress_path"]}
            row.update(_metrics(res["verts_path"], res["stress_path"],
                                th, faces, ref_verts))
            r_max = np.nanmax([row.get("r_x0", np.nan), row.get("r_y0", np.nan)])
            row["sim_failed"] = bool(
                not np.isfinite(row["crown_height"])
                or row["crown_height"] < CROWN_MIN_M
                or (np.isfinite(r_max) and r_max > ROUGHNESS_THRESHOLD))
            rows.append(row)
            if i % 10 == 0 or row["sim_failed"]:
                flag = "  FLAGGED" if row["sim_failed"] else ""
                print(f"  motif{motif} theta={th:5.1f}  "
                      f"crown={row['crown_height']*1000:6.1f} mm  "
                      f"k_wale={row.get('apex_k_wale', np.nan):.3f}  "
                      f"k_course={row.get('apex_k_course', np.nan):.3f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"\nSaved {SWEEP_CSV}  ({len(df)} rows, "
          f"{int(df['sim_failed'].sum())} flagged)")
    report(df)
    return df


def report(df):
    ok = df[~df["sim_failed"].astype(bool)]
    print("\n  Effect of theta_knit on the ellipse "
          "(circle values in brackets, from data/knit_dir_sweep.csv):")
    circ = None
    p = os.path.join(DATA_DIR, "knit_dir_sweep.csv")
    if os.path.exists(p):
        circ = pd.read_csv(p)
        circ = circ[~circ["sim_failed"].astype(bool)]
    for motif, sub in ok.groupby("motif"):
        print(f"\n    motif {motif}")
        for col, unit in (("crown_height", "mm"), ("mean_stress", "Pa"),
                          ("max_stress", "Pa")):
            v = sub[col].values * (1000 if col == "crown_height" else 1)
            swing = (v.max() - v.min()) / v.mean() * 100
            ref = ""
            if circ is not None:
                c = circ[circ.motif == motif][col].values * (
                    1000 if col == "crown_height" else 1)
                ref = f"   [circle {(c.max()-c.min())/c.mean()*100:.2f}%]"
            print(f"      {col:14s} {v.mean():8.1f} {unit:3s} "
                  f"varies {swing:6.2f}% over theta{ref}")
        for col in ("apex_k_wale", "apex_k_course", "apex_k_min", "apex_k_max",
                    "H_fit_x0", "H_fit_y0", "von_mises_x0", "von_mises_y0"):
            if col not in sub:
                continue
            v = sub[col].values
            v = v[np.isfinite(v)]
            if len(v) < 2:
                continue
            print(f"      {col:14s} {v.mean():8.3f}     "
                  f"varies {(v.max()-v.min())/v.mean()*100:6.2f}% over theta")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--make-mesh", action="store_true")
    ap.add_argument("--force-mesh", action="store_true")
    args = ap.parse_args()
    if args.make_mesh or args.force_mesh:
        make_mesh(force=args.force_mesh)
        if args.make_mesh:
            sys.exit(0)
    run_sweep(step=args.step)
