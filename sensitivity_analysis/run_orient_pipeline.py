"""
Pipeline for the cable-orientation sensitivity study.

Two new cable orientations per motif:
  cable_wale   — cable runs along the wale direction for each knit_dir
  cable_course — cable runs along the course direction for each knit_dir

L_rest is stored as a dimensionless ratio; actual rest length =
  L_rest_ratio × L_flat(knit_dir, cable_axis).

Steps
-----
  calib     — quick calibration sweep (knit_dir × axis grid, L_rest=0.92)
  generate  — 600 FEA runs (4 groups × 150 LHS samples)
  validate  — flag failed simulations
  train     — GP surrogate per group
  sobol     — Sobol sensitivity indices
  figures   — heatmap + regime map for the new groups

Usage
-----
  python3 run_orient_pipeline.py [--steps calib generate validate train sobol figures]
                                  [--jobs 4] [--no-cache]
"""

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MOTIFS, CABLE_AXES, PARAMS_CABLE_ORIENT,
    SCALAR_OUTPUTS, MESH_PATH, SOBOL_N_BASE, RANDOM_SEED,
)
from sampling import generate_orient_samples
from cable_path import compute_cable_indices
from fea_interface import run_fea, check_binary
from surrogate import ScalarSurrogate

os.makedirs(DATA_DIR, exist_ok=True)

ROUGHNESS_THRESHOLD = 0.10
CROWN_MIN_M         = 0.02
CABLE_TENSION_MIN_N = 1.0

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ORIENT_GROUPS = [f"motif{m}_cable_{ax}" for m in MOTIFS for ax in CABLE_AXES]

PARAM_LABELS = {
    "sf_wale":   r"$s_{wale}$",
    "sf_course": r"$s_{course}$",
    "knit_dir":  r"$\theta_{knit}$",
    "pressure":  r"$p$",
    "L_rest":    r"$L_{cable}$",
}

OUTPUT_LABELS = {
    "crown_height":          r"$h_{crown}$",
    "H_mean_x0":             r"$\bar{H}_{x0}$",
    "H_mean_y0":             r"$\bar{H}_{y0}$",
    "max_stress":            r"$\sigma_{max}$",
    "mean_stress":           r"$\bar{\sigma}$",
    "cable_tension":         r"$T_{cable}$",
    "boundary_reaction_mean": r"$\bar{R}_{bdry}$",
}

GROUP_LABELS = {
    "motif1_cable_wale":   r"Motif 1 — wale cable",
    "motif1_cable_course": r"Motif 1 — course cable",
    "motif2_cable_wale":   r"Motif 2 — wale cable",
    "motif2_cable_course": r"Motif 2 — course cable",
}

# ── Cable-path cache ──────────────────────────────────────────────────────────

_cable_cache: dict = {}

def _get_cable(knit_dir_deg: float, cable_axis: str):
    key = (round(knit_dir_deg, 4), cable_axis)
    if key not in _cable_cache:
        _cable_cache[key] = compute_cable_indices(knit_dir_deg, cable_axis)
    return _cable_cache[key]


# ── Step 0: Calibration sweep ─────────────────────────────────────────────────

def step_calib():
    """
    Coarse grid (knit_dir × motif × axis) with L_rest_ratio=0.92.
    Checks: crown_height ≥ 0.02 m, cable_tension ≥ 1 N.
    """
    check_binary()
    print("\n[Calib] Cable-orientation geometry check...")

    rows = []
    for motif in MOTIFS:
        for axis in CABLE_AXES:
            for kd in [0, 15, 30, 45, 60, 75, 90]:
                indices, L_flat = compute_cable_indices(kd, axis)
                L_rest = 0.92 * L_flat
                prefix = os.path.join(DATA_DIR, f"calib_orient_{motif}_{axis}_{kd:02d}")
                scalars_path = prefix + "_scalars.csv"

                if not os.path.exists(scalars_path):
                    try:
                        run_fea(
                            sf_wale=1.0, sf_course=1.0,
                            knit_dir_deg=float(kd), pressure=800.0,
                            motif=motif,
                            cable_indices=indices,
                            L_rest=L_rest,
                            output_prefix=prefix,
                        )
                    except Exception as e:
                        print(f"  FAILED motif{motif} {axis} kd={kd}: {e}")
                        continue

                with open(scalars_path) as f:
                    row = next(csv.DictReader(f))
                h = float(row["crown_height"])
                T = float(row["cable_tension"])
                ok = "OK" if h >= CROWN_MIN_M and T >= CABLE_TENSION_MIN_N else "FAIL"
                print(f"  [{ok}] motif{motif} {axis:6s} kd={kd:2d}°:  "
                      f"h={h*1000:.1f}mm  T={T:.1f}N  "
                      f"L_flat={L_flat:.4f}m  n_verts={len(indices)}")
                rows.append(dict(motif=motif, axis=axis, knit_dir=kd,
                                 crown_height=h, cable_tension=T,
                                 L_flat=L_flat, L_rest=L_rest, status=ok))

    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "calib_orient.csv")
    df.to_csv(path, index=False)
    print(f"\n  Calibration → {path}")
    n_fail = (df["status"] == "FAIL").sum()
    if n_fail:
        print(f"  WARNING: {n_fail} runs failed — check geometry before proceeding.")
    return df


# ── Step 1: Data generation ───────────────────────────────────────────────────

def _run_one_orient(sample: dict):
    sid    = sample["sample_id"]
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")

    if (os.path.exists(prefix + "_scalars.csv") and
            os.path.exists(prefix + "_verts.csv")):
        row = {"sample_id": sid}
        with open(prefix + "_scalars.csv") as f:
            for k, v in next(csv.DictReader(f)).items():
                try:    row[k] = float(v)
                except: row[k] = v
        return {**sample, **row, "L_flat": sample.get("L_flat", np.nan)}, True, "cached"

    try:
        cable_indices, L_flat = _get_cable(sample["knit_dir"], sample["cable_axis"])
        L_rest_actual = float(sample["L_rest"]) * L_flat

        result = run_fea(
            sf_wale      = sample["sf_wale"],
            sf_course    = sample["sf_course"],
            knit_dir_deg = sample["knit_dir"],
            pressure     = sample["pressure"],
            motif        = sample["motif"],
            cable_indices= cable_indices,
            L_rest       = L_rest_actual,
            output_prefix= prefix,
        )
        return {**sample, **result, "L_flat": L_flat}, True, "ok"
    except Exception as e:
        return ({**sample, "L_flat": np.nan}, False,
                f"FEA failed: {e}\n{traceback.format_exc()}")


def step_generate(jobs: int = 4, no_cache: bool = False):
    check_binary()
    samples = generate_orient_samples()
    print(f"\n[Step 1] Generating {len(samples)} orient FEA runs ({jobs} workers)...")

    if no_cache:
        for s in samples:
            for ext in ["_scalars.csv", "_verts.csv", "_stress.csv"]:
                p = os.path.join(DATA_DIR, f"{s['sample_id']:05d}{ext}")
                if os.path.exists(p):
                    os.unlink(p)

    results, n_ok, n_fail = [], 0, 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one_orient, s): s for s in samples}
        for fut in as_completed(futures):
            row, ok, info = fut.result()
            if ok:
                n_ok += 1
                results.append(row)
                if info != "cached":
                    print(f"  [{row['sample_id']:05d}] OK  "
                          f"h={row.get('crown_height',0)*1000:.1f}mm  "
                          f"T={row.get('cable_tension',0):.1f}N", flush=True)
            else:
                n_fail += 1
                print(f"  [{row['sample_id']:05d}] FAIL: {info[:120]}", flush=True)

    df = pd.DataFrame(results)
    path = os.path.join(DATA_DIR, "results_orient.csv")
    df.to_csv(path, index=False)
    print(f"\n  Done: {n_ok} OK, {n_fail} failed.  → {path}")
    return df


# ── Step 2: Validate ──────────────────────────────────────────────────────────

def _roughness(sid: int) -> float:
    vpath = os.path.join(DATA_DIR, f"{sid:05d}_verts.csv")
    if not os.path.exists(vpath):
        return np.nan
    verts = pd.read_csv(vpath).sort_values("vid")[["x","y","z"]].values
    from plot_section_profiles import _slice_plane
    from curvature import read_off
    _, faces = read_off(MESH_PATH)
    _, z_mm, _ = _slice_plane(verts, faces, np.zeros(len(faces)), fixed_axis=1)
    if len(z_mm) < 5 or z_mm.max() < 1e-3:
        return np.nan
    d2 = z_mm[2:] - 2*z_mm[1:-1] + z_mm[:-2]
    return float(np.sqrt(np.mean(d2**2)) / z_mm.max())


def step_validate(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        path = os.path.join(DATA_DIR, "results_orient.csv")
        if not os.path.exists(path):
            print("[Validate] results_orient.csv not found.")
            return None
        df = pd.read_csv(path)

    print(f"\n[Step 2] Validating {len(df)} orient samples...")
    df = df.copy()
    df["roughness"] = [_roughness(int(s)) for s in df["sample_id"]]
    df["sim_failed"] = (
        (df["crown_height"] < CROWN_MIN_M) |
        (df["roughness"].fillna(999) > ROUGHNESS_THRESHOLD) |
        (df["cable_tension"] < CABLE_TENSION_MIN_N)
    )
    n_bad = df["sim_failed"].sum()
    print(f"  Total failed: {n_bad}/{len(df)}")
    for grp in ORIENT_GROUPS:
        sub = df[df["group"] == grp]
        print(f"    {grp}: {(~sub['sim_failed']).sum()}/{len(sub)} valid")

    path = os.path.join(DATA_DIR, "results_orient.csv")
    df.to_csv(path, index=False)
    return df


# ── Step 3: Train surrogates ──────────────────────────────────────────────────

def step_train(df: pd.DataFrame = None) -> dict:
    if df is None:
        path = os.path.join(DATA_DIR, "results_orient.csv")
        if not os.path.exists(path):
            print("[Train] results_orient.csv not found.")
            return {}
        df = pd.read_csv(path)

    print(f"\n[Step 3] Training GP surrogates...")
    surrogates = {}
    for group in ORIENT_GROUPS:
        sub = df[df["group"] == group].copy()
        if "sim_failed" in sub.columns:
            n_before = len(sub)
            sub = sub[~sub["sim_failed"].fillna(False)]
            removed = n_before - len(sub)
            if removed:
                print(f"  {group}: removed {removed} failed samples")
        sub = sub[sub["crown_height"] >= CROWN_MIN_M]
        available = [c for c in SCALAR_OUTPUTS
                     if c in sub.columns and sub[c].notna().any()]
        sub = sub.dropna(subset=available)
        if len(sub) < 10:
            print(f"  {group}: only {len(sub)} valid — skipping")
            continue
        print(f"  {group}: {len(sub)} samples")

        # Use has_cable=True → input_keys = [sf_wale, sf_course, knit_dir, pressure, L_rest]
        surrogate = ScalarSurrogate(has_cable=True)
        metrics = surrogate.fit(sub)
        for col, m in metrics.items():
            print(f"    {col:30s}  R²={m['r2']:.3f}  RMSE={m['rmse']:.4f}")

        save_path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        surrogate.save(save_path)
        surrogates[group] = surrogate
        print(f"  Saved → {save_path}")
    return surrogates


# ── Step 4: Sobol analysis ────────────────────────────────────────────────────

def _sobol_for_orient_group(surrogate: ScalarSurrogate) -> dict:
    """Compute S1, ST for orient group using PARAMS_CABLE_ORIENT bounds."""
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze

    problem = {
        "num_vars": len(PARAMS_CABLE_ORIENT),
        "names":    list(PARAMS_CABLE_ORIENT.keys()),
        "bounds":   [list(v) for v in PARAMS_CABLE_ORIENT.values()],
    }
    X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=True)
    preds = surrogate.predict(X)

    results, s2_dict = {}, {}
    for col in SCALAR_OUTPUTS:
        if col not in preds:
            continue
        Y = preds[col]
        si = sobol_analyze.analyze(problem, Y, calc_second_order=True,
                                   print_to_console=False)
        results[col] = pd.DataFrame(
            {"S1": si["S1"], "ST": si["ST"],
             "S1_conf": si["S1_conf"], "ST_conf": si["ST_conf"]},
            index=problem["names"],
        )
        s2_dict[col] = si["S2"]
    return results, s2_dict, problem["names"]


def step_sobol(surrogates: dict = None) -> dict:
    if surrogates is None:
        surrogates = {}
        for grp in ORIENT_GROUPS:
            path = os.path.join(DATA_DIR, f"{grp}_scalar_surrogate.pkl")
            if os.path.exists(path):
                surrogates[grp] = ScalarSurrogate.load(path)

    if not surrogates:
        print("[Sobol] No surrogates found.")
        return {}

    print(f"\n[Step 4] Sobol analysis for orient groups...")
    all_results, all_s2 = {}, {}
    for group, surrogate in surrogates.items():
        print(f"  {group}...")
        results, s2_dict, param_names = _sobol_for_orient_group(surrogate)
        all_results[group] = results
        all_s2[group]      = s2_dict
        for col, df in results.items():
            st = df["ST"].dropna()
            if len(st):
                print(f"    {col}: top={st.idxmax()} (ST={st.max():.3f})")
        for col, df in results.items():
            p = os.path.join(DATA_DIR, f"sobol_{group}_{col}.csv")
            df.to_csv(p)
    return all_results, all_s2


# ── Step 5: Figures ───────────────────────────────────────────────────────────

_C_DIRECT      = np.array([0x21, 0x71, 0xb5]) / 255
_C_INTERACTION = np.array([0xe6, 0x55, 0x0d]) / 255
_C_WHITE       = np.array([1.0, 1.0, 1.0])
_PARTNER_MIN   = 0.15


def _regime_color(st, s1):
    st = max(0.0, min(1.0, float(st)))
    s1 = max(0.0, min(st, float(s1)))
    r_d = (s1 / st) if st > 1e-6 else 0.5
    hue = r_d * _C_DIRECT + (1 - r_d) * _C_INTERACTION
    sat = min(1.0, st ** 0.6)
    rgb = np.clip(sat * hue + (1 - sat) * _C_WHITE, 0.0, 1.0)
    lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return (*rgb, 1.0), ("white" if lum < 0.50 else "black")


plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})


def _sobol_load_or_compute(surrogates, all_s2=None):
    """Load saved Sobol results or recompute from surrogates."""
    results_all, s2_all = {}, {}
    for grp in ORIENT_GROUPS:
        group_results = {}
        for col in SCALAR_OUTPUTS:
            p = os.path.join(DATA_DIR, f"sobol_{grp}_{col}.csv")
            if os.path.exists(p):
                group_results[col] = pd.read_csv(p, index_col=0)
        if group_results:
            results_all[grp] = group_results
        if all_s2 and grp in all_s2:
            s2_all[grp] = all_s2[grp]
    return results_all, s2_all


def step_figures(sobol_data=None, save=True):
    if sobol_data is None:
        results_all, s2_all = _sobol_load_or_compute({})
    elif isinstance(sobol_data, tuple):
        results_all, s2_all = sobol_data
    else:
        results_all, s2_all = sobol_data, {}

    if not results_all:
        print("[Figures] No Sobol results found — run sobol step first.")
        return

    print(f"\n[Step 5] Generating orient figures...")
    _plot_sobol_heatmap(results_all, which="ST", save=save)
    _plot_sobol_heatmap(results_all, which="S1", save=save)
    _plot_regime_heatmap(results_all, s2_all, save=save)


def _plot_sobol_heatmap(results_all: dict, which: str = "ST", save: bool = True):
    groups = [g for g in ORIENT_GROUPS if g in results_all]
    if not groups:
        return

    param_names = list(PARAMS_CABLE_ORIENT.keys())
    n_groups    = len(groups)
    fig, axes   = plt.subplots(1, n_groups, figsize=(4.0*n_groups, 4.5),
                                constrained_layout=True)
    if n_groups == 1:
        axes = [axes]

    cmap   = plt.cm.YlOrRd
    im_ref = None
    for ax, grp in zip(axes, groups):
        sub = results_all[grp]
        out_names = [o for o in SCALAR_OUTPUTS if o in sub]
        mat = np.zeros((len(param_names), len(out_names)))
        for j, col in enumerate(out_names):
            for i, p in enumerate(param_names):
                if p in sub[col].index:
                    mat[i, j] = max(0, sub[col].loc[p, which])

        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        im_ref = im
        for i in range(len(param_names)):
            for j in range(len(out_names)):
                v  = mat[i, j]
                fg = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color=fg, fontweight="bold")

        ax.set_xticks(range(len(out_names)))
        ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                           rotation=35, ha="right")
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names])
        ax.set_title(GROUP_LABELS.get(grp, grp), pad=6)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(out_names)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(param_names)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    if im_ref is not None:
        lbl = (r"Total-order Sobol $S_T$" if which == "ST"
               else r"First-order Sobol $S_1$")
        fig.colorbar(im_ref, ax=axes, shrink=0.7, pad=0.02, label=lbl)

    tag = "ST" if which == "ST" else "S1"
    fig.suptitle(f"Cable-orientation study — {tag} Sobol indices", fontsize=11)
    if save:
        fname = f"figOrient_{tag}_heatmap"
        for ext in [".pdf", ".png"]:
            fig.savefig(os.path.join(FIG_DIR, fname + ext),
                        bbox_inches="tight", dpi=200 if ext == ".png" else None)
        print(f"  Saved: {fname}")
    return fig


def _plot_regime_heatmap(results_all: dict, s2_all: dict, save: bool = True):
    groups = [g for g in ORIENT_GROUPS if g in results_all]
    if not groups:
        return

    param_names = list(PARAMS_CABLE_ORIENT.keys())
    n_groups    = len(groups)
    fig, axes   = plt.subplots(1, n_groups, figsize=(4.5*n_groups, 4.5),
                                constrained_layout=True)
    if n_groups == 1:
        axes = [axes]

    for ax, grp in zip(axes, groups):
        sub = results_all[grp]
        s2  = s2_all.get(grp, {})
        out_names = [o for o in SCALAR_OUTPUTS if o in sub]
        n_p, n_o  = len(param_names), len(out_names)
        img       = np.ones((n_p, n_o, 4))

        for i, p in enumerate(param_names):
            for j, col in enumerate(out_names):
                if p not in sub[col].index:
                    continue
                st = max(0, sub[col].loc[p, "ST"])
                s1 = max(0, sub[col].loc[p, "S1"])
                rgba, fg = _regime_color(st, s1)
                img[i, j] = rgba

                txt = f"{st:.2f}"
                partner_label = ""
                if col in s2 and (st - s1) >= _PARTNER_MIN:
                    row_s2 = s2[col][i] if s2[col].ndim == 2 else s2[col][:, i]
                    row_s2 = np.where(np.arange(n_p) == i, -np.inf, row_s2)
                    best = int(np.nanargmax(row_s2))
                    if row_s2[best] > 0:
                        partner_label = PARAM_LABELS.get(param_names[best],
                                                         param_names[best])

                ax.text(j, i - (0.15 if partner_label else 0.0), txt,
                        ha="center", va="center", fontsize=9, color=fg,
                        fontweight="bold")
                if partner_label:
                    ax.text(j, i + 0.22,
                            r"$\mathit{" + partner_label.strip("$") + r"}$",
                            ha="center", va="center", fontsize=6.5, color=fg,
                            fontstyle="italic")

        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_o))
        ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                           rotation=35, ha="right")
        ax.set_yticks(range(n_p))
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names])
        ax.set_title(GROUP_LABELS.get(grp, grp), pad=6)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, n_o), minor=True)
        ax.set_yticks(np.arange(-0.5, n_p), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#2171b5", label=r"Direct  ($S_1$ large)"),
        Patch(color="#e6550d", label=r"Interaction  ($S_T-S_1$ large)"),
        Patch(color="#f7f7f7", label=r"Negligible  ($S_T \approx 0$)",
              edgecolor="grey", linewidth=0.5),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cable-orientation study — Sobol regime map\n"
                 r"(italic = dominant $S_2$ interaction partner)",
                 fontsize=11)
    if save:
        fname = "figOrient_regime"
        for ext in [".pdf", ".png"]:
            fig.savefig(os.path.join(FIG_DIR, fname + ext),
                        bbox_inches="tight", dpi=200 if ext == ".png" else None)
        print(f"  Saved: {fname}")
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cable-orientation pipeline")
    parser.add_argument(
        "--steps", nargs="+",
        default=["calib", "generate", "validate", "train", "sobol", "figures"],
        choices=["calib", "generate", "validate", "train", "sobol", "figures"])
    parser.add_argument("--jobs",     type=int,  default=4)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    df, surrogates, sobol_data = None, None, None

    if "calib" in args.steps:
        step_calib()

    if "generate" in args.steps:
        df = step_generate(jobs=args.jobs, no_cache=args.no_cache)

    if "validate" in args.steps:
        df = step_validate(df)

    if "train" in args.steps:
        surrogates = step_train(df)

    if "sobol" in args.steps:
        sobol_data = step_sobol(surrogates)

    if "figures" in args.steps:
        step_figures(sobol_data)

    print(f"\nPipeline completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
