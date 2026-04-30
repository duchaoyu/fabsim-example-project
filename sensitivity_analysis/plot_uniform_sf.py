"""
Effect of uniform stretch factor (sf_wale = sf_course = sf) on:
  - crown height
  - mean stress
  - mean curvature along x=0 and y=0 sections

Surrogate is used for crown height and mean stress (smooth curves).
Raw data filtered to quasi-isotropic samples (|sf_wale - sf_course| < 0.10)
is used for curvature, binned and shown with mean ± std bands.

No-cable groups only; motif 1 vs motif 2 overlaid.
"""

import os
import sys
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


def _build_curvature_gp(group: str, curv_col: str, df_all: pd.DataFrame) -> dict:
    """Train (or load cached) GP surrogate for a section curvature output."""
    cache = os.path.join(DATA_DIR, f"{group}_{curv_col}_gp.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    input_keys = list(PARAMS_NO_CABLE.keys())
    sub = df_all[df_all["group"] == group].dropna(subset=input_keys + [curv_col])
    X = sub[input_keys].values
    y = sub[curv_col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-4)
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=False, random_state=42,
    )
    gp.fit(X_s, y_s)

    model = {"gp": gp, "scaler_X": scaler_X, "scaler_y": scaler_y,
             "input_keys": input_keys}
    with open(cache, "wb") as f:
        pickle.dump(model, f)
    print(f"  Trained curvature GP: {group} / {curv_col}  (n={len(sub)})")
    return model


def _predict_curv_gp(model: dict, X: np.ndarray) -> np.ndarray:
    X_s = model["scaler_X"].transform(X)
    y_s = model["gp"].predict(X_s)
    return model["scaler_y"].inverse_transform(y_s.reshape(-1, 1)).ravel()


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

    for group in groups:
        color = COLORS[group]
        label = LABELS[group]
        sub   = df_all[df_all["group"] == group]

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

        # row 3: curvature via GP surrogate sweep
        for h_col, ls, curve_label in [
            ("H_mean_x0", "-",  "x=0 section"),
            ("H_mean_y0", "--", "y=0 section"),
        ]:
            gp_model = _build_curvature_gp(group, h_col, df_all)
            keys = list(PARAMS_NO_CABLE.keys())
            defaults = _midpoint_params()
            X_sweep = np.column_stack([
                sf if k in ("sf_wale", "sf_course") else
                np.full(N_GRID, defaults[k])
                for k in keys
            ])
            y_c = _predict_curv_gp(gp_model, X_sweep)
            ax_c.plot(sf, y_c, color=color, ls=ls, lw=2,
                      label=f"{label} ({curve_label})")

    # ── formatting ────────────────────────────────────────────────────────────
    ax_h.set_ylabel("Crown height  (mm)")
    ax_h.set_title(r"Crown height  vs uniform $s_f$")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    ax_s.set_ylabel(r"Mean stress  (Pa)")
    ax_s.set_title(r"Mean stress  vs uniform $s_f$")
    ax_s.legend(fontsize=8, loc="upper left")

    ax_c.set_ylabel(r"Mean curvature  $\bar{H}$  (m$^{-1}$)")
    ax_c.set_title(r"Section curvature  vs uniform $s_f$  (GP surrogate)")
    ax_c.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax_c.set_xlabel(r"Uniform stretch factor  $s_f$  ($s_{wale} = s_{course}$)")

    for ax in axes:
        ax.axvline(1.0, color="0.7", lw=0.8, ls=":")
        ax.set_xlim(*SF_RANGE)

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
