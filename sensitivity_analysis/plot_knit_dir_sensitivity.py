"""
Sensitivity analysis of knitting direction (theta_knit).

Fig M — Line sweep + anisotropy index
  Row 1: Crown height vs theta
  Row 2: Mean curvature H_x0 and H_y0 vs theta
  Row 3: Section stress sigma_x0 and sigma_y0 vs theta
  Row 4: Anisotropy indices  dH = (H_x0-H_y0)/(H_x0+H_y0)
                             ds = (s_x0-s_y0)/(s_x0+s_y0)
  Fixed: sf_wale = sf_course = 1.0, pressure = 1000 Pa.

Fig N — Section profile gallery at theta ≈ 0, 30, 45, 60, 90 degrees
  Shows z(s) and H(s) along x=0 and y=0 for representative samples.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MESH_PATH, PARAMS_NO_CABLE
from surrogate import ScalarSurrogate
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

GROUPS = ["motif1_nocable", "motif2_nocable"]
COLORS = {"motif1_nocable": "#2E8B57", "motif2_nocable": "#20B2AA"}
LABELS = {"motif1_nocable": "Motif 1", "motif2_nocable": "Motif 2"}

FIXED_SF   = 1.0
FIXED_P    = 1000.0
N_GRID     = 120
THETA_RANGE = (0.0, 90.0)

GALLERY_TARGETS = [0, 30, 45, 60, 90]
GALLERY_TOL     = 15.0   # deg


# ── symmetry-augmented GP training ───────────────────────────────────────────
#
# For a circular geometry the physical symmetry is:
#   (sf_w, sf_c, theta, p) ↔ (sf_c, sf_w, 90-theta, p)
# with H_mean_x0 ↔ H_mean_y0  and  von_mises_x0 ↔ von_mises_y0.
# Augmenting training data with the mirrored samples forces the GP to
# respect this symmetry, fixing the θ=0 / θ=90 inconsistency.

# Maps each output to its mirror partner under the 90° rotation
_MIRROR = {
    "H_mean_x0":    "H_mean_y0",
    "H_mean_y0":    "H_mean_x0",
    "von_mises_x0": "von_mises_y0",
    "von_mises_y0": "von_mises_x0",
    "crown_height": "crown_height",   # invariant under rotation
}


def _augment(df, col):
    """Return a symmetry-augmented DataFrame for training output `col`."""
    input_keys = list(PARAMS_NO_CABLE.keys())   # sf_wale, sf_course, knit_dir, pressure
    mirror_col  = _MIRROR[col]

    orig = df[input_keys + [col]].dropna().copy()
    orig = orig.rename(columns={col: "_y"})

    # mirror: swap sf_wale↔sf_course, flip knit_dir → 90-theta, use mirror output
    mirror = df[input_keys + [mirror_col]].dropna().copy()
    mirror = mirror.rename(columns={mirror_col: "_y"})
    mirror["sf_wale"]   = df["sf_course"]
    mirror["sf_course"] = df["sf_wale"]
    mirror["knit_dir"]  = 90.0 - df["knit_dir"]

    combined = pd.concat([orig, mirror], ignore_index=True).dropna()
    X = combined[input_keys].values
    y = combined["_y"].values
    return X, y


def _build_sym_gp(group, col, force=False):
    """Train (or load) a symmetry-augmented GP for `col`."""
    cache = os.path.join(DATA_DIR, f"{group}_{col}_sym_gp.pkl")
    if os.path.exists(cache) and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]
    df = df[df["group"] == group].copy()

    X, y = _augment(df, col)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   normalize_y=False, random_state=42)
    gp.fit(X_s, y_s)
    model = {"gp": gp, "scaler_X": scaler_X, "scaler_y": scaler_y}
    with open(cache, "wb") as f:
        pickle.dump(model, f)
    print(f"  Trained sym GP: {group} / {col}  (n={len(y)} incl. augmented)")
    return model


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_gp(group, col):
    """Load symmetry-augmented GP; fall back to plain GP if absent."""
    for suffix in ("_sym_gp", "_gp"):
        path = os.path.join(DATA_DIR, f"{group}_{col}{suffix}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


def _predict_gp(model, X):
    X_s = model["scaler_X"].transform(X)
    y_s = model["gp"].predict(X_s)
    return model["scaler_y"].inverse_transform(y_s.reshape(-1, 1)).ravel()


def _sweep_X(theta_grid):
    keys = list(PARAMS_NO_CABLE.keys())
    return np.column_stack([
        np.full(len(theta_grid), FIXED_SF)  if k == "sf_wale"   else
        np.full(len(theta_grid), FIXED_SF)  if k == "sf_course" else
        theta_grid                           if k == "knit_dir"  else
        np.full(len(theta_grid), FIXED_P)
        for k in keys
    ])


def _load_verts(sid):
    p = os.path.join(DATA_DIR, f"{sid:05d}_verts.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).sort_values("vid")[["x", "y", "z"]].values


def _best_sample(sub, target_deg):
    sub = sub.copy()
    sub["angle_dist"] = (sub["knit_dir"] - target_deg).abs()
    cands = sub[sub["angle_dist"] < GALLERY_TOL].copy()
    if cands.empty:
        return None
    cands["score"] = (
        ((cands["sf_wale"]   - FIXED_SF) / 0.3) ** 2 +
        ((cands["sf_course"] - FIXED_SF) / 0.3) ** 2 +
        ((cands["pressure"]  - FIXED_P)  / 500)  ** 2
    )
    return cands.nsmallest(1, "score").iloc[0]


# ── Fig M ─────────────────────────────────────────────────────────────────────

def _add_section_insets(ax_a, theta_targets=(0, 45, 90)):
    """
    Add small cross-section profile insets on the anisotropy panel
    at representative theta values using actual FEA samples.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]

    # Use motif1 for illustration; pick samples closest to sf~1, p~1000
    sub = df[df["group"] == "motif1_nocable"].copy()
    sub["sf_dist"] = ((sub["sf_wale"] - 1.0)**2 + (sub["sf_course"] - 1.0)**2 +
                      ((sub["pressure"] - 1000.0) / 500)**2)

    ax_xlim = ax_a.get_xlim()
    ax_ylim = ax_a.get_ylim()

    inset_w = 0.12   # axes fraction
    inset_h = 0.28
    # positions: (theta_target, x_anchor_in_axes_frac, va)
    positions = {
        0:  (0.03,  0.68),   # left
        45: (0.41,  0.12),   # centre-bottom
        90: (0.78,  0.68),   # right
    }

    for target in theta_targets:
        cands = sub[np.abs(sub["knit_dir"] - target) < 12].copy()
        if cands.empty:
            continue
        row = cands.nsmallest(1, "sf_dist").iloc[0]
        sid = int(row["sample_id"])
        verts = _load_verts(sid)
        if verts is None:
            continue

        curv  = compute_curvatures(verts, _FACES)
        H     = curv["H"]

        xf, yf = positions[target]
        ax_in = ax_a.inset_axes([xf, yf, inset_w, inset_h])
        ax_in.set_facecolor("white")

        for fixed_axis, ls, color in [(0, "-", "#2E8B57"), (1, "--", "#888")]:
            pos, z, _ = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
            if len(pos) < 3:
                continue
            s = (pos - pos.min()) / (pos.max() - pos.min())
            z_n = z / z.max() if z.max() > 1e-3 else z
            ax_in.plot(s, z_n, color=color, ls=ls, lw=1.2)

        ax_in.set_xlim(0, 1)
        ax_in.set_ylim(-0.05, 1.15)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title(f"θ={target}°", fontsize=6.5, pad=2)
        for sp in ax_in.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("0.5")

        # arrow from inset to the corresponding point on the dH=0 line at target theta
        if ax_xlim[1] > ax_xlim[0]:
            tx = (target - ax_xlim[0]) / (ax_xlim[1] - ax_xlim[0])
            ty = (0.0    - ax_ylim[0]) / (ax_ylim[1] - ax_ylim[0])
            # center-bottom of inset → data point
            ix = xf + inset_w / 2
            iy = yf if yf < 0.5 else yf
            ax_a.annotate("", xy=(tx, ty), xytext=(ix, iy if yf < 0.5 else yf),
                          xycoords="axes fraction", textcoords="axes fraction",
                          arrowprops=dict(arrowstyle="-", color="0.5",
                                          lw=0.7, connectionstyle="arc3,rad=0"))


def plot_sweep(save=True):
    theta = np.linspace(*THETA_RANGE, N_GRID)
    X = _sweep_X(theta)

    # build symmetry-augmented GPs (cached after first run)
    sym_cols = ["H_mean_x0", "H_mean_y0", "von_mises_x0", "von_mises_y0"]
    for group in GROUPS:
        for col in sym_cols:
            _build_sym_gp(group, col)

    fig, axes = plt.subplots(3, 1, figsize=(6, 9.5),
                             gridspec_kw={"hspace": 0.52}, sharex=True)
    ax_c, ax_s, ax_a = axes

    for group in GROUPS:
        color = COLORS[group]
        label = LABELS[group]

        # curvature sections — symmetry-augmented GP
        gp_hx = _load_gp(group, "H_mean_x0")
        gp_hy = _load_gp(group, "H_mean_y0")
        if gp_hx and gp_hy:
            hx = _predict_gp(gp_hx, X)
            hy = _predict_gp(gp_hy, X)
            ax_c.plot(theta, hx, color=color, lw=2, ls="-")
            ax_c.plot(theta, hy, color=color, lw=2, ls="--")
            denom = np.abs(hx) + np.abs(hy)
            dH = np.where(denom > 1e-9, (hx - hy) / denom, 0.0)
            ax_a.plot(theta, dH, color=color, lw=2, label=label)

        # section stress — symmetry-augmented GP
        gp_sx = _load_gp(group, "von_mises_x0")
        gp_sy = _load_gp(group, "von_mises_y0")
        if gp_sx and gp_sy:
            sx = _predict_gp(gp_sx, X)
            sy = _predict_gp(gp_sy, X)
            ax_s.plot(theta, sx, color=color, lw=2, ls="-")
            ax_s.plot(theta, sy, color=color, lw=2, ls="--")

    # ── formatting ────────────────────────────────────────────────────────────
    ax_c.set_ylabel(r"$\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  $\bar{H}$")

    ax_s.set_ylabel("Mean stress  (Pa)")
    ax_s.set_title("Section stress")

    ax_a.set_ylabel(r"$(H_{x=0} - H_{y=0})\,/\,(H_{x=0} + H_{y=0})$")
    ax_a.set_title("Curvature anisotropy index")
    ax_a.axhline(0, color="0.75", lw=0.8, ls=":")
    ax_a.legend(fontsize=8)
    ax_a.set_xlabel(r"Knitting direction  $\theta_{knit}$  (°)")

    for ax in axes:
        ax.set_xlim(*THETA_RANGE)
        ax.set_xticks([0, 15, 30, 45, 60, 75, 90])

    # combined legend: plane styles + motif colours
    plane_handles = [
        Line2D([0], [0], color="0.4", lw=1.5, ls="-",  label="$x=0$ section"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="$y=0$ section"),
    ]
    motif_handles = [
        Line2D([0], [0], color=COLORS[g], lw=2, label=LABELS[g])
        for g in GROUPS
    ]
    ax_c.legend(handles=motif_handles + plane_handles, fontsize=7.5, loc="best")
    ax_s.legend(handles=plane_handles, fontsize=7.5, loc="best")

    fig.suptitle(
        r"Effect of knitting direction $\theta_{knit}$"
        "\n"
        r"($s_{wale}=s_{course}=1.0$,  $p=1000$ Pa)",
        fontsize=10, y=1.02,
    )

    if save:
        path = os.path.join(FIG_DIR, "figM_knit_dir_sweep.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Fig N ─────────────────────────────────────────────────────────────────────
# Profiles are normalised to remove the sf/p effect:
#   shape row:     z_norm = z / z_max          (0 → 1)
#   curvature row: H_norm = H(s) / H_mean_interior  (relative distribution)
# This makes profiles comparable across samples with different (sf, p),
# focusing purely on the directional effect of theta.

def _normalise_profile(pos, vals, trim=0.10):
    """Return interior mask and normalised values (divided by interior mean)."""
    span = pos.max() - pos.min()
    mask = (pos > pos.min() + trim * span) & (pos < pos.max() - trim * span)
    interior = vals[mask]
    interior = interior[np.isfinite(interior)]
    mean_val = np.mean(interior) if len(interior) else 1.0
    if abs(mean_val) < 1e-9:
        mean_val = 1.0
    return vals / mean_val, mask


def plot_gallery(save=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]

    n_cols = len(GALLERY_TARGETS)
    n_rows = 2 * len(GROUPS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.8 * n_rows),
        gridspec_kw={"hspace": 0.60, "wspace": 0.35},
    )

    plane_colors = {"x=0": "#2E8B57", "y=0": "#e07b39"}

    for g_idx, group in enumerate(GROUPS):
        sub  = df[df["group"] == group]
        ax_z_row = 2 * g_idx
        ax_H_row = 2 * g_idx + 1

        for c_idx, target in enumerate(GALLERY_TARGETS):
            ax_z = axes[ax_z_row, c_idx]
            ax_H = axes[ax_H_row, c_idx]

            row = _best_sample(sub, target)
            if row is None:
                ax_z.set_visible(False); ax_H.set_visible(False); continue

            sid   = int(row["sample_id"])
            verts = _load_verts(sid)
            if verts is None:
                ax_z.set_visible(False); ax_H.set_visible(False); continue

            curv = compute_curvatures(verts, _FACES)
            H    = curv["H"]

            for plane, fixed_axis in [("x=0", 0), ("y=0", 1)]:
                pos, z, Hv = _slice_plane(verts, _FACES, H, fixed_axis=fixed_axis)
                if len(pos) < 3:
                    continue
                ls = "-" if fixed_axis == 0 else "--"
                c  = plane_colors[plane]

                # normalised shape: z / z_max  (0 → 1)
                z_norm = z / z.max() if z.max() > 1e-3 else z
                s_norm = pos / (pos.max() - pos.min()) * 2   # -1 … +1

                # normalised curvature: H(s) / H_mean_interior
                Hv_norm, _ = _normalise_profile(pos, Hv)
                # suppress boundary spikes in plot
                span = pos.max() - pos.min()
                interior = (pos > pos.min() + 0.1*span) & (pos < pos.max() - 0.1*span)

                ax_z.plot(s_norm, z_norm, color=c, ls=ls, lw=1.8, label=plane)
                ax_H.plot(s_norm[interior], Hv_norm[interior],
                          color=c, ls=ls, lw=1.8)

            ax_z.set_ylim(-0.05, 1.15)
            ax_z.axhline(0, color="0.8", lw=0.5, ls=":")
            ax_H.axhline(1, color="0.8", lw=0.5, ls=":")   # H/H_mean = 1 reference
            ax_z.set_xlim(-1.05, 1.05)
            ax_H.set_xlim(-1.05, 1.05)
            ax_z.tick_params(labelsize=7)
            ax_H.tick_params(labelsize=7)

            # column title: target angle + actual sample parameters
            col_title = (f"θ ≈ {target}°  (actual {row['knit_dir']:.0f}°)\n"
                         f"$s_w$={row['sf_wale']:.2f}  "
                         f"$s_c$={row['sf_course']:.2f}  "
                         f"$p$={row['pressure']:.0f} Pa")
            if g_idx == 0:
                ax_z.set_title(col_title, fontsize=7, pad=4)

            if c_idx == 0:
                ax_z.set_ylabel(f"{LABELS[group]}\n$z/z_{{max}}$", fontsize=8)
                ax_H.set_ylabel(r"$H\,/\,\bar{H}_{interior}$", fontsize=8)

            if g_idx == len(GROUPS) - 1:
                ax_H.set_xlabel(r"$s\,/\,2R$", fontsize=7)

            if c_idx == 0 and g_idx == 0:
                ax_z.legend(fontsize=7, loc="upper center",
                            handlelength=1.2, ncol=2, framealpha=0.7)

    fig.suptitle(
        "Normalised cross-section profiles at key knitting directions\n"
        r"(shape: $z/z_{max}$;  curvature: $H/\bar{H}$;  "
        r"solid=$x{=}0$,  dashed=$y{=}0$)",
        fontsize=9, y=1.01,
    )

    if save:
        path = os.path.join(FIG_DIR, "figN_knit_dir_gallery.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Fig M: knit direction sweep...")
    plot_sweep()
    print("Fig N: section profile gallery...")
    plot_gallery()
    print("Done.")
