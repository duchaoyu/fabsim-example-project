"""
Sobol sensitivity analysis for section-based outputs in the material study
(no-cable group only).

New outputs derived from per-sample verts/stress files:
  H_mean_x0       – mean profile curvature along x=0 section  (m⁻¹)
  H_mean_y0       – mean profile curvature along y=0 section  (m⁻¹)
  H_anisotropy    – (H_x0 - H_y0) / 0.5(H_x0 + H_y0)  (signed, dimensionless)
  vm_x0           – mean von Mises stress along x=0 section   (N/m)
  vm_y0           – mean von Mises stress along y=0 section   (N/m)

Trains a dedicated GP surrogate, runs Sobol on the full 7-D material parameter
space, and produces a heatmap in the style of the existing material figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, MESH_PATH, PARAMS_MATERIAL_NO_CABLE, SOBOL_N_BASE
from surrogate import ScalarSurrogate
from curvature import read_off, compute_curvatures
from plot_section_profiles import _slice_plane
from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SURROGATE_PATH = os.path.join(DATA_DIR, "material_nocable_section_surrogate.pkl")
SECTION_DATA_PATH = os.path.join(DATA_DIR, "material_nocable_section_metrics.csv")

_SECTION_TOL = 0.03   # m  (5% of 0.6 m radius)

_REST_VERTS, _FACES = read_off(MESH_PATH)

SECTION_OUTPUTS = [
    "H_mean_x0",
    "H_mean_y0",
    "H_anisotropy",
    "vm_x0",
    "vm_y0",
]

OUTPUT_LABELS = {
    "H_mean_x0":    r"$\bar{H}_{x=0}$",
    "H_mean_y0":    r"$\bar{H}_{y=0}$",
    "H_anisotropy": r"$\Delta H / \bar{H}$",
    "vm_x0":        r"$\bar{\sigma}_{x=0}$",
    "vm_y0":        r"$\bar{\sigma}_{y=0}$",
}

PARAM_LABELS = {
    "sf_wale":   r"$s_{wale}$",
    "sf_course": r"$s_{course}$",
    "knit_dir":  r"$\theta_{knit}$",
    "pressure":  r"$p$",
    "E1":        r"$E_1$ (N/m)",
    "r":         r"$r{=}E_1/E_2$",
    "nu":        r"$\nu_{12}$",
}

# Rows that belong to the material sub-space (highlighted in the plot)
MATERIAL_PARAMS = {"E1", "r", "nu"}


# ── Section metric computation ────────────────────────────────────────────────

def _profile_roughness(z: np.ndarray) -> float:
    if len(z) < 5 or z.max() < 1e-3:
        return np.nan
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    return float(np.sqrt(np.mean(d2 ** 2)) / z.max())


def _profile_curvature(pos_mm: np.ndarray, z_mm: np.ndarray) -> float:
    order = np.argsort(pos_mm)
    p, z = pos_mm[order], z_mm[order]
    span_mm = p[-1] - p[0]
    if span_mm < 1.0:
        return np.nan
    n_bins = max(10, int(span_mm / 5.0))
    bins = np.linspace(p[0], p[-1], n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    p_avg, z_avg = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            p_avg.append(p[mask].mean())
            z_avg.append(z[mask].mean())
    if len(p_avg) < 5:
        return np.nan
    p_avg = np.array(p_avg)
    z_avg = np.array(z_avg)
    ds  = np.sqrt(np.diff(p_avg) ** 2 + np.diff(z_avg) ** 2)
    s   = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 1.0:
        return np.nan
    dz  = np.gradient(z_avg, s)
    d2z = np.gradient(dz, s)
    kappa_mm = np.abs(d2z) / (1.0 + dz ** 2) ** 1.5
    inner = (s > 0.1 * s[-1]) & (s < 0.9 * s[-1])
    if inner.sum() == 0:
        return np.nan
    return float(np.mean(kappa_mm[inner])) * 1000.0   # mm⁻¹ → m⁻¹


def _section_metrics(sample_id: int) -> dict:
    nan_row = {k: np.nan for k in ["H_mean_x0", "H_mean_y0", "vm_x0", "vm_y0",
                                    "r_x0", "r_y0"]}
    vpath = os.path.join(DATA_DIR, f"{int(sample_id):05d}_verts.csv")
    spath = os.path.join(DATA_DIR, f"{int(sample_id):05d}_stress.csv")
    if not os.path.exists(vpath) or not os.path.exists(spath):
        return nan_row

    verts = pd.read_csv(vpath).sort_values("vid")[["x", "y", "z"]].values
    curv  = compute_curvatures(verts, _FACES)
    H     = curv["H"]

    metrics = {}
    for h_key, r_key, ax in [("H_mean_x0", "r_x0", 0), ("H_mean_y0", "r_y0", 1)]:
        pos, z_mm, _ = _slice_plane(verts, _FACES, H, fixed_axis=ax)
        if len(pos) < 5:
            metrics[h_key] = metrics[r_key] = np.nan
            continue
        metrics[h_key] = _profile_curvature(pos, z_mm)
        metrics[r_key] = _profile_roughness(z_mm)

    sdf      = pd.read_csv(spath).sort_values("face")
    face_ids = sdf["face"].values.astype(int)
    centroids = verts[_FACES[face_ids]].mean(axis=1)
    vm        = sdf["von_mises"].values

    mask_x0 = np.abs(centroids[:, 0]) < _SECTION_TOL
    mask_y0 = np.abs(centroids[:, 1]) < _SECTION_TOL
    metrics["vm_x0"] = float(np.mean(vm[mask_x0])) if mask_x0.any() else np.nan
    metrics["vm_y0"] = float(np.mean(vm[mask_y0])) if mask_y0.any() else np.nan

    return metrics


def build_section_df(force=False) -> pd.DataFrame:
    if not force and os.path.exists(SECTION_DATA_PATH):
        print(f"Loading cached section metrics from {SECTION_DATA_PATH}")
        return pd.read_csv(SECTION_DATA_PATH)

    df  = pd.read_csv(os.path.join(DATA_DIR, "material_results.csv"))
    sub = df[df["group"] == "material_nocable"].copy().reset_index(drop=True)
    print(f"Computing section metrics for {len(sub)} samples...")

    rows = []
    for i, row in sub.iterrows():
        m = _section_metrics(int(row["sample_id"]))
        rows.append(m)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sub)} done")

    metrics_df = pd.DataFrame(rows)

    # Curvature anisotropy index: signed normalised difference
    Hx = metrics_df["H_mean_x0"]
    Hy = metrics_df["H_mean_y0"]
    Hm = 0.5 * (Hx + Hy)
    metrics_df["H_anisotropy"] = np.where(Hm > 1e-6, (Hx - Hy) / Hm, np.nan)

    # Quality filter: reject rough profiles and very low crown heights
    r_max   = metrics_df[["r_x0", "r_y0"]].max(axis=1)
    failed  = (sub["crown_height"] < 0.02) | (r_max > 0.10)
    metrics_df.loc[failed, SECTION_OUTPUTS] = np.nan

    enriched = pd.concat([sub.reset_index(drop=True),
                           metrics_df.reset_index(drop=True)], axis=1)
    enriched.to_csv(SECTION_DATA_PATH, index=False)
    n_ok  = (~metrics_df["H_mean_x0"].isna()).sum()
    print(f"Saved section metrics → {SECTION_DATA_PATH}  ({n_ok}/{len(sub)} valid)")
    return enriched


# ── Surrogate ────────────────────────────────────────────────────────────────

def train_section_surrogate(df: pd.DataFrame, force=False) -> ScalarSurrogate:
    if not force and os.path.exists(SURROGATE_PATH):
        print(f"Loading cached section surrogate from {SURROGATE_PATH}")
        return ScalarSurrogate.load(SURROGATE_PATH)

    valid = df.dropna(subset=SECTION_OUTPUTS).copy()
    print(f"Training section surrogate on {len(valid)} valid samples...")

    sur = ScalarSurrogate(has_cable=False, bounds=PARAMS_MATERIAL_NO_CABLE)
    metrics = sur.fit(valid, output_cols=SECTION_OUTPUTS)
    for col, m in metrics.items():
        print(f"  {col}: R²={m['r2']:.3f}  RMSE={m['rmse']:.4g}")

    sur.save(SURROGATE_PATH)
    print(f"Saved section surrogate → {SURROGATE_PATH}")
    return sur


# ── Sobol ────────────────────────────────────────────────────────────────────

def run_section_sobol(sur: ScalarSurrogate) -> dict:
    problem = {
        "num_vars": len(PARAMS_MATERIAL_NO_CABLE),
        "names":    list(PARAMS_MATERIAL_NO_CABLE.keys()),
        "bounds":   [list(v) for v in PARAMS_MATERIAL_NO_CABLE.values()],
    }
    X = saltelli.sample(problem, SOBOL_N_BASE, calc_second_order=False)
    preds = sur.predict(X)

    results = {}
    for col in SECTION_OUTPUTS:
        if col not in preds:
            continue
        Y = preds[col]
        if np.std(Y) < 1e-10:
            continue
        si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                                   print_to_console=False)
        results[col] = pd.DataFrame(
            {"S1": si["S1"], "ST": si["ST"],
             "S1_conf": si["S1_conf"], "ST_conf": si["ST_conf"]},
            index=problem["names"],
        )
        top = results[col]["ST"].idxmax()
        print(f"  {col}: top={top} ST={results[col]['ST'].max():.3f}")
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_section_sobol_heatmap(results: dict, index="ST", save=True):
    assert index in ("ST", "S1")
    conf_col = "ST_conf" if index == "ST" else "S1_conf"

    param_names = list(PARAMS_MATERIAL_NO_CABLE.keys())
    out_names   = [o for o in SECTION_OUTPUTS if o in results]
    if not out_names:
        print("No section Sobol results to plot.")
        return

    mat      = np.zeros((len(param_names), len(out_names)))
    conf_mat = np.zeros_like(mat)
    for j, col in enumerate(out_names):
        for i, p in enumerate(param_names):
            mat[i, j]      = max(0, results[col].loc[p, index])
            conf_mat[i, j] = results[col].loc[p, conf_col]

    # Figure sizing matching existing material figures
    cell_w, cell_h = 0.90, 0.82
    pad_top, pad_bot, pad_left, pad_right = 1.4, 1.2, 1.6, 0.5
    n_p, n_o = len(param_names), len(out_names)
    fig_w = n_o * cell_w + pad_left + pad_right + 0.4
    fig_h = n_p * cell_h + pad_top  + pad_bot   + 0.4

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=pad_left/fig_w, right=1-pad_right/fig_w,
        bottom=pad_bot/fig_h,  top=1-pad_top/fig_h,
    )

    cmap = plt.cm.YlOrRd
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")

    # Highlight material-parameter rows with a background band
    for i, p in enumerate(param_names):
        if p in MATERIAL_PARAMS:
            ax.add_patch(plt.Rectangle(
                (-0.5, i - 0.5), n_o, 1,
                facecolor="none", edgecolor="#1155AA",
                linewidth=1.5, zorder=3,
            ))

    # Annotate cells
    for i in range(n_p):
        for j in range(n_o):
            v = mat[i, j]
            c = conf_mat[i, j]
            color = "white" if v > 0.6 else "black"
            ax.text(j, i - 0.20, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")
            ax.text(j, i + 0.18, f"±{c:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(n_o))
    ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_names],
                       rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names],
                       fontsize=11)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, n_o), minor=True)
    ax.set_yticks(np.arange(-0.5, n_p), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    label = (r"Total-order Sobol index $S_T$" if index == "ST"
             else r"First-order Sobol index $S_1$")

    # Horizontal colorbar
    plot_left  = pad_left / fig_w
    plot_right = 1 - pad_right / fig_w
    cbar_w   = (plot_right - plot_left) * 0.60
    cbar_left = plot_left + (plot_right - plot_left - cbar_w) / 2
    cbar_ax = fig.add_axes([cbar_left, 0.16/fig_h, cbar_w, 0.16/fig_h])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label=label)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    cbar.set_label(label, fontsize=13)

    idx_label = r"$S_T$" if index == "ST" else r"$S_1$"
    title = (f"Material sensitivity on section outputs — no cable\n"
             f"Blue border = material parameters ($E_1$, $r$, $\\nu_{{12}}$)"
             f";  index = {idx_label}")
    fig.suptitle(title, fontsize=12, y=1.01)

    if save:
        fname = "fig_material_section_sobol_ST" if index == "ST" else "fig_material_section_sobol_S1"
        path  = os.path.join(FIG_DIR, f"{fname}.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Material section Sobol analysis ===")
    df  = build_section_df()
    sur = train_section_surrogate(df)
    print("Running Sobol analysis...")
    results = run_section_sobol(sur)
    plot_section_sobol_heatmap(results, index="ST")
    plot_section_sobol_heatmap(results, index="S1")
    print("Done.")
