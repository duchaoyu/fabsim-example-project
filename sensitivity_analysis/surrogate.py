"""
GP surrogate models trained on FEA data.

Scalar outputs: one GP per output per group.
Field outputs (displacement, stress, curvature): PCA + one GP per PC per group.

Uses scikit-learn GaussianProcessRegressor with Matern(nu=2.5) kernel.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from config import (
    SCALAR_OUTPUTS, GP_PCA_VARIANCE, TRAIN_VAL_SPLIT, RANDOM_SEED,
    PARAMS_NO_CABLE, PARAMS_CABLE,
)

# Outputs that span a large dynamic range — fit log(y) instead of y
_LOG_OUTPUTS = {"H_mean_x0", "H_mean_y0", "cable_wale_tension", "cable_course_tension"}


def _make_kernel():
    return ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-4)


def _input_keys(has_cable: bool, bounds: dict = None) -> list:
    if bounds is not None:
        return list(bounds.keys())
    return list((PARAMS_CABLE if has_cable else PARAMS_NO_CABLE).keys())


class ScalarSurrogate:
    """One GP per scalar output for a single (motif, cable) group."""

    def __init__(self, has_cable: bool, bounds: dict = None):
        self.has_cable   = has_cable
        self.input_keys  = _input_keys(has_cable, bounds)
        self.scaler_X    = StandardScaler()
        self.scalers_y   = {}
        self.gps         = {}
        self.metrics     = {}

    def fit(self, df: pd.DataFrame, output_cols=None) -> dict:
        """
        Fit GPs. df must have columns = input_keys + output_cols.
        output_cols defaults to SCALAR_OUTPUTS from config.
        Returns dict of {output: {r2, rmse}}.
        """
        if output_cols is None:
            output_cols = SCALAR_OUTPUTS
        X = df[self.input_keys].values
        X_s = self.scaler_X.fit_transform(X)

        X_tr, X_val, idx_tr, idx_val = train_test_split(
            X_s, np.arange(len(df)),
            test_size=TRAIN_VAL_SPLIT,
            random_state=RANDOM_SEED,
        )

        for col in output_cols:
            if col not in df.columns:
                continue
            y = df[col].values
            if not np.isfinite(y).any():
                continue
            # Log-transform outputs with large dynamic range
            log_col = col in _LOG_OUTPUTS and (y > 0).all()
            self._log_cols = getattr(self, "_log_cols", set())
            if log_col:
                y = np.log(y)
                self._log_cols.add(col)
            sc = StandardScaler()
            y_s = sc.fit_transform(y.reshape(-1, 1)).ravel()
            self.scalers_y[col] = sc

            gp = GaussianProcessRegressor(
                kernel=_make_kernel(),
                n_restarts_optimizer=5,
                normalize_y=False,
                random_state=RANDOM_SEED,
            )
            gp.fit(X_tr, y_s[idx_tr])
            self.gps[col] = gp

            pred_s = gp.predict(X_val)
            pred_t = sc.inverse_transform(pred_s.reshape(-1, 1)).ravel()
            true_t = y[idx_val]
            if log_col:
                pred_t = np.exp(pred_t)
                true_t = np.exp(true_t)
            r2   = r2_score(true_t, pred_t)
            rmse = np.sqrt(mean_squared_error(true_t, pred_t))
            self.metrics[col] = {"r2": r2, "rmse": rmse}

        return self.metrics

    def predict(self, X):
        """X: (n, d) array of input parameters. Returns dict of output arrays."""
        X_s = self.scaler_X.transform(X)
        log_cols = getattr(self, "_log_cols", set())
        out = {}
        for col, gp in self.gps.items():
            pred_s = gp.predict(X_s)
            pred = self.scalers_y[col].inverse_transform(
                pred_s.reshape(-1, 1)
            ).ravel()
            if col in log_cols:
                pred = np.exp(pred)
            out[col] = pred
        return out

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ScalarSurrogate":
        with open(path, "rb") as f:
            return pickle.load(f)


class FieldSurrogate:
    """PCA + GP for one field output (displacement / stress / curvature)."""

    def __init__(self, field_name: str, has_cable: bool):
        self.field_name  = field_name
        self.has_cable   = has_cable
        self.input_keys  = _input_keys(has_cable)
        self.pca         = None
        self.scaler_X    = StandardScaler()
        self.gps         = []
        self.n_components= 0
        self.metrics     = {}

    def fit(self, X_params: np.ndarray, Y_field: np.ndarray) -> dict:
        """
        X_params: (n, d), Y_field: (n, n_nodes * n_dim).
        Returns metrics dict.
        """
        X_s = self.scaler_X.fit_transform(X_params)
        X_tr, X_val, Y_tr, Y_val = train_test_split(
            X_s, Y_field,
            test_size=TRAIN_VAL_SPLIT,
            random_state=RANDOM_SEED,
        )

        # PCA
        self.pca = PCA(n_components=GP_PCA_VARIANCE, svd_solver="full")
        Z_tr  = self.pca.fit_transform(Y_tr)
        Z_val = self.pca.transform(Y_val)
        self.n_components = Z_tr.shape[1]
        print(f"  {self.field_name}: {self.n_components} PCs "
              f"({GP_PCA_VARIANCE*100:.0f}% variance)")

        # One GP per PC
        self.gps = []
        r2s, rmses = [], []
        for k in range(self.n_components):
            gp = GaussianProcessRegressor(
                kernel=_make_kernel(),
                n_restarts_optimizer=3,
                normalize_y=True,
                random_state=RANDOM_SEED,
            )
            gp.fit(X_tr, Z_tr[:, k])
            self.gps.append(gp)

            pred = gp.predict(X_val)
            r2s.append(r2_score(Z_val[:, k], pred))
            rmses.append(np.sqrt(mean_squared_error(Z_val[:, k], pred)))

        self.metrics = {
            "mean_r2":  float(np.mean(r2s)),
            "mean_rmse": float(np.mean(rmses)),
            "n_components": self.n_components,
        }
        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (n, n_nodes*n_dim) field predictions."""
        X_s = self.scaler_X.transform(X)
        Z   = np.column_stack([gp.predict(X_s) for gp in self.gps])
        return self.pca.inverse_transform(Z)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FieldSurrogate":
        with open(path, "rb") as f:
            return pickle.load(f)
