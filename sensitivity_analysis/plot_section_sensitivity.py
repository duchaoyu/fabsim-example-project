"""
Spearman correlation heatmap for section-based outputs (no-cable only).

Outputs:
  crown_height     – global scalar
  H_mean_x0        – mean curvature along x=0 section (wale cut)
  H_mean_y0        – mean curvature along y=0 section (course cut)
  T_wale_x0        – mean wale tension near x=0 section
  T_course_y0      – mean course tension near y=0 section
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH
from curvature import read_off, compute_curvatures
from plot_section_profiles import _slice_plane

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth":  0.8,
    "figure.dpi":      150,
})

_REST_VERTS, _FACES = read_off(MESH_PATH)

# 5 % of radius (0.6 m) as section proximity threshold
_SECTION_TOL = 0.03

OUTPUT_LABELS = {
    "crown_height": r"$h_{crown}$",
    "H_mean_x0":    r"$\bar{H}_{x=0}$",
    "H_mean_y0":    r"$\bar{H}_{y=0}$",
    "T_wale_x0":    r"$\bar{T}^{wale}_{x=0}$  (N/m)",
    "T_course_y0":  r"$\bar{T}^{course}_{y=0}$  (N/m)",
}

PARAM_LABELS = {
    "sf_wale":   r"$s_{wale}$",
    "sf_course": r"$s_{course}$",
    "knit_dir":  r"$\theta_{knit}$",
    "pressure":  r"$p$",
}

GROUP_LABELS = {
    "motif1_nocable": "Motif 1  (no cable)",
    "motif2_nocable": "Motif 2  (no cable)",
}


ROUGHNESS_THRESHOLD = 0.10   # r_max above this → simulation failed
CROWN_MIN_M         = 0.02   # crown height below this (m) → collapsed


def _profile_roughness(z: np.ndarray) -> float:
    """RMS of second differences normalised by peak height. Low = smooth dome."""
    if len(z) < 5 or z.max() < 1e-3:
        return np.nan
    d2 = z[2:] - 2 * z[1:-1] + z[:-2]
    return float(np.sqrt(np.mean(d2 ** 2)) / z.max())


def _profile_curvature(pos_mm: np.ndarray, z_mm: np.ndarray) -> float:
    """Mean arc-length curvature κ = |z''|/(1+z'²)^1.5 along a section profile.

    Both pos_mm and z_mm are in millimetres (as returned by _slice_plane).
    Multiple interpolated crossings at the same pos are averaged first.
    Returns curvature in m⁻¹. Interior 10% trimmed to avoid boundary artifacts.
    """
    # De-duplicate: bin pos into ~5 mm wide bins and average z
    order = np.argsort(pos_mm)
    p, z = pos_mm[order], z_mm[order]
    span_mm = p[-1] - p[0]
    if span_mm < 1.0:
        return np.nan
    n_bins = max(10, int(span_mm / 5.0))
    bins = np.linspace(p[0], p[-1], n_bins + 1)
    bin_idx = np.digitize(p, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
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
    # Arc-length parameterisation (units: mm)
    ds = np.sqrt(np.diff(p_avg) ** 2 + np.diff(z_avg) ** 2)
    s  = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 1.0:
        return np.nan
    dz  = np.gradient(z_avg, s)
    d2z = np.gradient(dz,    s)
    kappa_mm = np.abs(d2z) / (1.0 + dz ** 2) ** 1.5  # mm⁻¹
    inner = (s > 0.1 * s[-1]) & (s < 0.9 * s[-1])
    if inner.sum() == 0:
        return np.nan
    return float(np.mean(kappa_mm[inner])) * 1000.0  # convert to m⁻¹


def _section_metrics(sample_id: int) -> dict:
    nan_row = {k: np.nan for k in
               ["H_mean_x0", "H_mean_y0", "T_wale_x0", "T_course_y0",
                "von_mises_x0", "von_mises_y0", "r_x0", "r_y0"]}

    vpath = os.path.join(DATA_DIR, f"{sample_id:05d}_verts.csv")
    spath = os.path.join(DATA_DIR, f"{sample_id:05d}_stress.csv")
    if not os.path.exists(vpath) or not os.path.exists(spath):
        return nan_row

    verts = pd.read_csv(vpath).sort_values("vid")[["x", "y", "z"]].values

    # ── curvature sections ────────────────────────────────────────────────────
    curv = compute_curvatures(verts, _FACES)
    H = curv["H"]

    metrics = {}
    for h_key, r_key, fixed_axis in [("H_mean_x0", "r_x0", 0),
                                      ("H_mean_y0", "r_y0", 1)]:
        pos, z_mm, Hv = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
        if len(pos) < 5:
            metrics[h_key] = metrics[r_key] = np.nan
            continue
        # Profile curvature: arch κ in the (pos, z) plane — symmetric, smooth,
        # mesh-topology-independent. Replaces 3-D Laplacian mean curvature.
        metrics[h_key] = _profile_curvature(pos, z_mm)
        metrics[r_key] = _profile_roughness(z_mm)

    # ── stress sections: faces whose centroid is near each plane ─────────────
    sdf = pd.read_csv(spath).sort_values("face")
    face_ids = sdf["face"].values.astype(int)
    centroids = verts[_FACES[face_ids]].mean(axis=1)   # (n_faces, 3)

    mask_x0 = np.abs(centroids[:, 0]) < _SECTION_TOL
    mask_y0 = np.abs(centroids[:, 1]) < _SECTION_TOL

    T_wale     = sdf["T_wale_Nm"].values
    T_course   = sdf["T_course_Nm"].values
    von_mises  = sdf["von_mises"].values

    metrics["T_wale_x0"]      = float(np.mean(T_wale[mask_x0]))     if mask_x0.any() else np.nan
    metrics["T_course_y0"]    = float(np.mean(T_course[mask_y0]))   if mask_y0.any() else np.nan
    metrics["von_mises_x0"]   = float(np.mean(von_mises[mask_x0]))  if mask_x0.any() else np.nan
    metrics["von_mises_y0"]   = float(np.mean(von_mises[mask_y0]))  if mask_y0.any() else np.nan

    return metrics


def build_enriched(groups=("motif1_nocable", "motif2_nocable")) -> pd.DataFrame:
    """Load results.csv, append section-based metrics, and flag failed simulations."""
    cache = os.path.join(DATA_DIR, "results_with_sections.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        needed = {"H_mean_x0", "H_mean_y0", "T_wale_x0", "T_course_y0",
                  "von_mises_x0", "von_mises_y0", "r_x0", "r_y0"}
        if needed.issubset(df.columns):
            print(f"Loaded cached section metrics from {cache}")
            return df

    df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    sub = df[df["group"].isin(groups)].copy()

    rows = []
    n_total = len(sub)
    for idx, (_, row) in enumerate(sub.iterrows()):
        m = _section_metrics(int(row["sample_id"]))
        rows.append(m)
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{n_total} samples processed...")

    metrics_df = pd.DataFrame(rows, index=sub.index)
    enriched = pd.concat([sub, metrics_df], axis=1)

    # flag failed simulations
    r_max = enriched[["r_x0", "r_y0"]].max(axis=1)
    enriched["sim_failed"] = (
        (enriched["crown_height"] < CROWN_MIN_M) | (r_max > ROUGHNESS_THRESHOLD)
    )
    n_bad = enriched["sim_failed"].sum()
    print(f"  Flagged {n_bad}/{len(enriched)} samples as failed")

    enriched.to_csv(cache, index=False)
    print(f"Saved section metrics → {cache}")
    return enriched


def plot_section_correlation(save=True):
    df = build_enriched()

    valid = df[~df["sim_failed"]] if "sim_failed" in df.columns else df
    n_removed = len(df) - len(valid)
    if n_removed:
        print(f"  Removed {n_removed} failed simulations from plot")

    input_cols  = ["sf_wale", "sf_course", "knit_dir", "pressure"]
    output_cols = list(OUTPUT_LABELS.keys())
    groups      = ["motif1_nocable", "motif2_nocable"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "seagreen_div",
        ["#8B0000", "#ffffff", "#2E8B57"],   # dark-red → white → seagreen
    )
    im_ref = None

    for ax, group in zip(axes, groups):
        sub = valid[valid["group"] == group].copy()
        out_cols = [o for o in output_cols
                    if o in sub.columns and sub[o].std(skipna=True) > 1e-10]

        mat  = np.zeros((len(input_cols), len(out_cols)))
        pmat = np.ones_like(mat)

        for i, inp in enumerate(input_cols):
            for j, out in enumerate(out_cols):
                pair = sub[[inp, out]].dropna()
                if len(pair) < 5:
                    continue
                r, p = stats.spearmanr(pair[inp], pair[out])
                mat[i, j]  = r
                pmat[i, j] = p

        im = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        im_ref = im

        for i in range(len(input_cols)):
            for j in range(len(out_cols)):
                r, p = mat[i, j], pmat[i, j]
                sig   = "*" if p < 0.05 else ""
                color = "white" if abs(r) > 0.65 else "black"
                ax.text(j, i, f"{r:.2f}{sig}",
                        ha="center", va="center", fontsize=7.5, color=color)

        ax.set_xticks(range(len(out_cols)))
        ax.set_xticklabels([OUTPUT_LABELS.get(o, o) for o in out_cols],
                           rotation=35, ha="right")
        ax.set_yticks(range(len(input_cols)))
        ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in input_cols])
        ax.set_title(GROUP_LABELS.get(group, group), pad=6)
        ax.tick_params(length=0)

        ax.set_xticks(np.arange(-0.5, len(out_cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(input_cols)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        ax.tick_params(which="minor", bottom=False, left=False)

    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, shrink=0.7, pad=0.02,
                            label=r"Spearman $\rho$")
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle(
        r"Spearman correlations: parameters vs section-based outputs  (no cable)"
        "\n"
        r"(* $p{<}0.05$;  "
        r"$\bar{H}_{x=0/y=0}$ = mean curvature along section,  "
        r"$\bar{T}$ = mean tension near section)",
        fontsize=9,
    )

    if save:
        path = os.path.join(FIG_DIR, "figJ_section_sensitivity.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Computing section-based sensitivity...")
    plot_section_correlation()
    print("Done.")
