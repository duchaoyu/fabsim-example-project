"""
Section-profile curvature estimators.

`profile_curvature_binned` is the estimator used to build
results_with_sections.csv (plot_section_sensitivity._profile_curvature): it bins
the plane crossings into ~5 mm bins, takes two numerical derivatives, and
averages over the interior 80% of arc length.

That estimator is unstable along a parameter sweep.  Both the bin count
(int(span/5)) and the interior window are discrete functions of the profile, so
as a parameter varies a crossing enters or leaves the window, or the bin count
changes by one, and the reported mean curvature steps.  Measured on the
knit-direction sweep the steps reach 10-25% of the mean, against a genuine
theta-signal of ~38% — the estimator noise is the same order as the effect being
plotted.

`profile_curvature_fit` replaces the binning and finite differences with a
least-squares polynomial fit to z(pos), evaluated analytically:

    kappa(s) = |z''| / (1 + z'^2)^1.5

on a dense grid over the interior of the span.  No binning, no finite
differences, and the interior window enters only through fixed fractions of the
span, so the result is a smooth function of the input parameters.

Both return curvature in m^-1 and take pos/z in millimetres, matching
_slice_plane's output.
"""

import numpy as np

TRIM = 0.10          # fraction of arc length dropped at each end
DEFAULT_DEGREE = 6
DENSE = 400


def _dedup(pos, z):
    """Sort by pos and average points sharing (nearly) the same pos."""
    order = np.argsort(pos)
    p, zz = np.asarray(pos)[order], np.asarray(z)[order]
    keep_p, keep_z = [], []
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and (p[j + 1] - p[i]) < 1e-9:
            j += 1
        keep_p.append(p[i:j + 1].mean())
        keep_z.append(zz[i:j + 1].mean())
        i = j + 1
    return np.array(keep_p), np.array(keep_z)


def profile_curvature_fit(pos_mm, z_mm, degree=DEFAULT_DEGREE,
                          return_fit=False):
    """
    Mean profile curvature (m^-1) from a polynomial fit to z(pos).

    Returns nan if the profile is too short or too sparse to fit.
    """
    p, z = _dedup(np.asarray(pos_mm, float), np.asarray(z_mm, float))
    if len(p) < 6:
        return (np.nan, None) if return_fit else np.nan
    span = p[-1] - p[0]
    if span < 1.0:
        return (np.nan, None) if return_fit else np.nan

    deg = int(min(degree, max(2, len(p) // 4)))
    # fit on a normalised abscissa for conditioning
    t = (p - p.mean()) / (span / 2.0)
    c = np.polyfit(t, z, deg)
    dz_dt = np.polyder(c, 1)
    d2z_dt2 = np.polyder(c, 2)
    scale = span / 2.0                      # dt/dpos = 1/scale

    tq = np.linspace(-1 + 2 * TRIM, 1 - 2 * TRIM, DENSE)
    dz = np.polyval(dz_dt, tq) / scale
    d2z = np.polyval(d2z_dt2, tq) / scale ** 2
    kappa_mm = np.abs(d2z) / (1.0 + dz ** 2) ** 1.5
    k = float(np.mean(kappa_mm)) * 1000.0   # mm^-1 -> m^-1

    if return_fit:
        resid = z - np.polyval(c, t)
        info = {"degree": deg, "n_points": len(p),
                "rms_resid_mm": float(np.sqrt(np.mean(resid ** 2))),
                "z_range_mm": float(z.max() - z.min())}
        return k, info
    return k


def profile_curvature_binned(pos_mm, z_mm, bin_mm=5.0):
    """The original estimator, kept for comparison (see module docstring)."""
    order = np.argsort(pos_mm)
    p, z = np.asarray(pos_mm)[order], np.asarray(z_mm)[order]
    span_mm = p[-1] - p[0]
    if span_mm < 1.0:
        return np.nan
    n_bins = max(10, int(span_mm / bin_mm))
    bins = np.linspace(p[0], p[-1], n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    p_avg, z_avg = [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() > 0:
            p_avg.append(p[m].mean())
            z_avg.append(z[m].mean())
    if len(p_avg) < 5:
        return np.nan
    p_avg, z_avg = np.array(p_avg), np.array(z_avg)
    ds = np.sqrt(np.diff(p_avg) ** 2 + np.diff(z_avg) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 1.0:
        return np.nan
    dz = np.gradient(z_avg, s)
    d2z = np.gradient(dz, s)
    kappa_mm = np.abs(d2z) / (1.0 + dz ** 2) ** 1.5
    inner = (s > TRIM * s[-1]) & (s < (1 - TRIM) * s[-1])
    if inner.sum() == 0:
        return np.nan
    return float(np.mean(kappa_mm[inner])) * 1000.0
