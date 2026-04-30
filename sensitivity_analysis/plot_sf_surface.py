"""
Response surfaces: sf_wale × sf_course for crown height, mean stress,
and section curvatures (x=0 and y=0).

Fixed: knit_dir=0°, pressure=1000 Pa.
Layout: 4 rows (outputs) × 2 columns (motif 1 | motif 2).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_NO_CABLE
from surrogate import ScalarSurrogate

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

SF_RANGE  = (0.8, 1.4)
N_GRID    = 60
KNIT_DIR  = 0.0
PRESSURE  = 1000.0

GROUPS  = ["motif1_nocable", "motif2_nocable"]
COL_TITLES = ["Motif 1", "Motif 2"]

# (output_key, row_label, colorbar_unit, cmap, scale)
OUTPUTS = [
    ("crown_height", "Crown height",        "mm",          "viridis", 1000.0),
    ("von_mises_x0", r"Stress  $x{=}0$",   "Pa",          "plasma",  1.0),
    ("von_mises_y0", r"Stress  $y{=}0$",   "Pa",          "plasma",  1.0),
    ("H_mean_x0",    r"$\bar{H}$  $x{=}0$", r"m$^{-1}$", "cividis", 1.0),
    ("H_mean_y0",    r"$\bar{H}$  $y{=}0$", r"m$^{-1}$", "cividis", 1.0),
]


def _load_scalar_surrogate(group):
    path = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
    return ScalarSurrogate.load(path) if os.path.exists(path) else None


GP_COLS = ("H_mean_x0", "H_mean_y0", "von_mises_x0", "von_mises_y0")


def _load_or_train_gp(group, col):
    path = os.path.join(DATA_DIR, f"{group}_{col}_gp.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # train from enriched data
    df = pd.read_csv(os.path.join(DATA_DIR, "results_with_sections.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]
    input_keys = list(PARAMS_NO_CABLE.keys())
    sub = df[df["group"] == group].dropna(subset=input_keys + [col]).copy()

    # Remove IQR outliers in the target column to prevent kernel collapse
    q1, q3 = sub[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    sub = sub[sub[col] <= q3 + 3.0 * iqr]

    X = sub[input_keys].values
    y = sub[col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    kernel = (ConstantKernel(1.0) *
              Matern(nu=2.5, length_scale_bounds=(0.1, 10.0)) +
              WhiteKernel(1e-4))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                   normalize_y=False, random_state=42)
    gp.fit(X_s, y_s)
    model = {"gp": gp, "scaler_X": scaler_X, "scaler_y": scaler_y}
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Trained GP: {group} / {col}  (n={len(sub)})")
    return model


def _predict_curv_gp(model, X):
    X_s = model["scaler_X"].transform(X)
    y_s = model["gp"].predict(X_s)
    return model["scaler_y"].inverse_transform(y_s.reshape(-1, 1)).ravel()


def _make_grid():
    sf_w = np.linspace(*SF_RANGE, N_GRID)
    sf_c = np.linspace(*SF_RANGE, N_GRID)
    W, C = np.meshgrid(sf_w, sf_c)
    keys = list(PARAMS_NO_CABLE.keys())
    X = np.column_stack([
        W.ravel() if k == "sf_wale"   else
        C.ravel() if k == "sf_course" else
        np.full(N_GRID * N_GRID, KNIT_DIR)  if k == "knit_dir" else
        np.full(N_GRID * N_GRID, PRESSURE)
        for k in keys
    ])
    return sf_w, sf_c, W, C, X


def _predict(group, output, X, scalar_surr, gp_models):
    if output in GP_COLS:
        model = gp_models.get(output)
        if model is None:
            return None
        return _predict_curv_gp(model, X)
    else:
        if scalar_surr is None:
            return None
        preds = scalar_surr.predict(X)
        return preds.get(output)


def plot_sf_surface(save=True):
    sf_w, sf_c, W, C, X_grid = _make_grid()

    n_rows = len(OUTPUTS)
    n_cols = len(GROUPS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.0 * n_rows),
        constrained_layout=True,
    )

    for col_idx, group in enumerate(GROUPS):
        scalar_surr = _load_scalar_surrogate(group)
        gp_models = {col: _load_or_train_gp(group, col) for col in GP_COLS}

        for row_idx, (output, row_label, unit, cmap, scale) in enumerate(OUTPUTS):
            ax = axes[row_idx, col_idx]

            Z = _predict(group, output, X_grid, scalar_surr, gp_models)
            if Z is None:
                ax.set_visible(False)
                continue

            Z = Z.reshape(N_GRID, N_GRID) * scale

            cs = ax.contourf(W, C, Z, levels=20, cmap=cmap)
            ax.contour(W, C, Z, levels=10, colors="white",
                       linewidths=0.4, alpha=0.5)

            cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit, fontsize=8)
            cb.ax.tick_params(labelsize=7)

            # diagonal sf_wale = sf_course
            ax.plot([SF_RANGE[0], SF_RANGE[1]],
                    [SF_RANGE[0], SF_RANGE[1]],
                    color="white", lw=1.0, ls="--", alpha=0.7)

            ax.set_xlabel(r"$s_{wale}$", labelpad=2)
            ax.set_ylabel(r"$s_{course}$", labelpad=2)
            ax.set_xlim(*SF_RANGE)
            ax.set_ylim(*SF_RANGE)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            ax.set_title(f"{COL_TITLES[col_idx]}  —  {row_label}", pad=6)

    fig.suptitle(
        r"Response surfaces: $s_{wale}$ × $s_{course}$   "
        r"($\theta_{knit}=0°$,  $p=1000$ Pa;  dashed = uniform $s_f$)",
        fontsize=10,
    )

    if save:
        path = os.path.join(FIG_DIR, "figL_sf_surface.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting sf_wale × sf_course response surfaces...")
    plot_sf_surface()
    print("Done.")
