"""
Effect of uniform stretch factor (sf_wale = sf_course = sf) on:
  - crown height
  - mean stress
  - mean curvature along x=0 and y=0 sections

Crown height and mean stress come from the scalar surrogate.
Section curvature uses a dedicated GP trained on log(H) — H spans more than a
decade, so a raw-space fit is dominated by the few high-curvature samples
(see surrogate.py::_LOG_OUTPUTS for the same convention).  The GP mean is drawn
with a +/-2 sigma band, and the quasi-isotropic raw samples
(|sf_wale - sf_course| < 0.10) are overlaid so the reader can judge support.
The sf region with no samples near the sf_wale = sf_course diagonal is shaded:
the sampling box is a square in (sf_wale, sf_course), so its corners — and hence
the ends of the diagonal — are only sparsely covered.

No-cable groups only; motif 1 vs motif 2 overlaid.
"""

import os
import sys
import hashlib
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_NO_CABLE
from surrogate import ScalarSurrogate

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "axes.linewidth":   0.8,
    "figure.dpi":       150,
})

# ── seagreen palette ──────────────────────────────────────────────────────────
COLORS = {
    "motif1_nocable": "#2E8B57",   # seagreen
    "motif2_nocable": "#20B2AA",   # lightseagreen
}
LABELS = {
    "motif1_nocable": "Motif 1",
    "motif2_nocable": "Motif 2",
}

SF_RANGE = (0.8, 1.4)
SF_ISO_TOL = 0.10          # |sf_wale - sf_course| < this → quasi-isotropic
N_BINS = 9
N_GRID = 80


def _fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """Short hash of the training set, so a changed CSV invalidates the cache."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return h.hexdigest()[:10]


def _build_curvature_gp(group: str, curv_col: str, df_all: pd.DataFrame) -> dict:
    """
    Train (or load cached) GP for a section curvature output, fitted in log space.

    The cache is keyed on a hash of the training data, so re-running after the
    results CSV changes (extra samples, outlier cleaning) retrains instead of
    silently reusing a model fitted to different data.  Note the private cache
    name: `{group}_{col}_gp.pkl` is also written by plot_sf_surface.py with
    different preprocessing, so it must not be shared.
    """
    input_keys = list(PARAMS_NO_CABLE.keys())
    sub = df_all[df_all["group"] == group].dropna(subset=input_keys + [curv_col])
    X = sub[input_keys].values
    y = sub[curv_col].values
    if (y <= 0).any():                      # log fit needs strictly positive H
        keep = y > 0
        X, y = X[keep], y[keep]

    fp = _fingerprint(X, y)
    cache = os.path.join(DATA_DIR, f"{group}_{curv_col}_loggp_{fp}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(np.log(y).reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=False, random_state=42,
    )
    gp.fit(X_s, y_s)

    model = {"gp": gp, "scaler_X": scaler_X, "scaler_y": scaler_y,
             "input_keys": input_keys, "log": True, "n_train": len(y)}
    with open(cache, "wb") as f:
        pickle.dump(model, f)
    print(f"  Trained curvature GP (log space): {group} / {curv_col}  (n={len(y)})")
    return model


def _predict_curv_gp(model: dict, X: np.ndarray, n_sigma: float = 0.0):
    """
    Predict H.  With n_sigma > 0 also return the (lower, upper) credible band,
    back-transformed from log space.
    """
    X_s = model["scaler_X"].transform(X)
    mu_s, sd_s = model["gp"].predict(X_s, return_std=True)
    inv = lambda v: model["scaler_y"].inverse_transform(v.reshape(-1, 1)).ravel()
    mu = inv(mu_s)
    lo, hi = inv(mu_s - n_sigma * sd_s), inv(mu_s + n_sigma * sd_s)
    if model.get("log"):
        mu, lo, hi = np.exp(mu), np.exp(lo), np.exp(hi)
    return (mu, lo, hi) if n_sigma > 0 else mu


def _diagonal_support(sub: pd.DataFrame, sf_grid: np.ndarray, radius=0.06):
    """
    Lowest sf on the grid that has at least one sample within `radius` of the
    sf_wale = sf_course diagonal.  Below this the sweep is extrapolation.
    """
    a = sub["sf_wale"].values
    b = sub["sf_course"].values
    cnt = np.array([np.sum((np.abs(a - v) < radius) & (np.abs(b - v) < radius))
                    for v in sf_grid])
    return sf_grid[cnt > 0].min() if (cnt > 0).any() else sf_grid[-1]


def _midpoint_params():
    defaults = {k: (lo + hi) / 2.0 for k, (lo, hi) in PARAMS_NO_CABLE.items()}
    defaults["knit_dir"] = 0.0
    defaults["pressure"] = 1000.0
    return defaults


def _surrogate_sweep(surrogate: ScalarSurrogate, output: str):
    """Return (sf_grid, y_pred) along the diagonal sf_wale = sf_course."""
    sf = np.linspace(*SF_RANGE, N_GRID)
    defaults = _midpoint_params()
    keys = list(PARAMS_NO_CABLE.keys())
    X = np.column_stack([
        sf if k in ("sf_wale", "sf_course") else
        np.full(N_GRID, defaults[k])
        for k in keys
    ])
    return sf, surrogate.predict(X)[output]


def _bin_curvature(df_iso, output_col, n_bins=N_BINS):
    """
    Bin quasi-isotropic samples by mean sf, return (bin_centres, means, stds).
    """
    df = df_iso.copy()
    df["sf_mean"] = (df["sf_wale"] + df["sf_course"]) / 2.0
    df = df.dropna(subset=[output_col])

    edges = np.linspace(*SF_RANGE, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(n_bins, np.nan)
    stds  = np.full(n_bins, np.nan)

    for k in range(n_bins):
        mask = (df["sf_mean"] >= edges[k]) & (df["sf_mean"] < edges[k + 1])
        vals = df.loc[mask, output_col].values
        if len(vals) >= 2:
            means[k] = np.mean(vals)
            stds[k]  = np.std(vals)

    return centres, means, stds


def plot_uniform_sf(save=True):
    # ── load data ─────────────────────────────────────────────────────────────
    df_all = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df_all.columns:
        df_all = df_all[~df_all["sim_failed"]]

    groups = ["motif1_nocable", "motif2_nocable"]
    sf = np.linspace(*SF_RANGE, N_GRID)

    # ── layout: 3 rows ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 8.5), constrained_layout=True,
                             sharex=True)
    ax_h, ax_s, ax_c = axes

    sf_support = 0.0   # left edge of the region backed by samples (all groups)

    for group in groups:
        color = COLORS[group]
        label = LABELS[group]
        sub   = df_all[df_all["group"] == group]
        sf_support = max(sf_support, _diagonal_support(sub, sf))

        # ── load surrogate ────────────────────────────────────────────────────
        path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
        if not os.path.exists(path):
            print(f"  No surrogate for {group}, skipping smooth curves")
            continue
        surrogate = ScalarSurrogate.load(path)

        # row 1: crown height
        _, y_h = _surrogate_sweep(surrogate, "crown_height")
        ax_h.plot(sf, y_h * 1000, color=color, lw=2, label=label)

        # row 2: mean stress
        _, y_s = _surrogate_sweep(surrogate, "mean_stress")
        ax_s.plot(sf, y_s, color=color, lw=2, label=label)

        # row 3: curvature via log-space GP sweep, with ±2σ band and raw samples
        keys = list(PARAMS_NO_CABLE.keys())
        defaults = _midpoint_params()
        X_sweep = np.column_stack([
            sf if k in ("sf_wale", "sf_course") else
            np.full(N_GRID, defaults[k])
            for k in keys
        ])
        iso = sub[(sub["sf_wale"] - sub["sf_course"]).abs() < SF_ISO_TOL]
        sf_iso = (iso["sf_wale"] + iso["sf_course"]).values / 2.0

        for h_col, ls, marker, curve_label in [
            ("H_mean_x0", "-",  "o", "x=0 section"),
            ("H_mean_y0", "--", "s", "y=0 section"),
        ]:
            gp_model = _build_curvature_gp(group, h_col, df_all)
            y_c, lo, hi = _predict_curv_gp(gp_model, X_sweep, n_sigma=2.0)
            ax_c.plot(sf, y_c, color=color, ls=ls, lw=2,
                      label=f"{label} ({curve_label})")
            ax_c.fill_between(sf, lo, hi, color=color, alpha=0.08, lw=0, zorder=0)
            ax_c.scatter(sf_iso, iso[h_col].values, s=9, marker=marker,
                         facecolors="none", edgecolors=color, lw=0.6,
                         alpha=0.55, zorder=1)

    # ── formatting ────────────────────────────────────────────────────────────
    ax_h.set_ylabel("Crown height  (mm)")
    ax_h.set_title(r"Crown height  vs uniform $s_f$")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    ax_s.set_ylabel(r"Mean stress  (Pa)")
    ax_s.set_title(r"Mean stress  vs uniform $s_f$")
    ax_s.legend(fontsize=8, loc="upper left")

    ax_c.set_ylabel(r"Mean curvature  $\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  vs uniform $s_f$"
                   "\n(log-space GP,  band = $\\pm2\\sigma$,  markers = quasi-isotropic samples)")
    ax_c.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax_c.set_xlabel(r"Uniform stretch factor  $s_f$  ($s_{wale} = s_{course}$)")
    ax_c.set_ylim(0, 2.4)     # ±2σ tails run far higher; clipped for readability

    for ax in axes:
        ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlim(*SF_RANGE)
        # sampling box corners are sparse → the diagonal sweep extrapolates here
        if sf_support > SF_RANGE[0]:
            ax.axvspan(SF_RANGE[0], sf_support, color="0.85", alpha=0.55, lw=0,
                       zorder=0)
    if sf_support > SF_RANGE[0]:
        ax_c.text((SF_RANGE[0] + sf_support) / 2, 0.03, "no samples\non diagonal",
                  transform=ax_c.get_xaxis_transform(), ha="center", va="bottom",
                  fontsize=6.5, color="0.35")

    fig.suptitle(
        r"Effect of uniform stretch factor on dome geometry and stress"
        "\n(other params fixed:  "
        r"$\theta_{knit}=0°$,  $p=1000$ Pa)",
        fontsize=9,
    )

    if save:
        path = os.path.join(FIG_DIR, "figK_uniform_sf.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting uniform stretch factor influence...")
    plot_uniform_sf()
    print("Done.")
