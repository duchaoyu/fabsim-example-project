"""
Pointwise principal curvatures at the crown, from a local quadric fit.

Why this exists
---------------
The section-curvature estimators in section_curvature.py average |kappa| along a
whole diameter.  That average is nearly blind to directional anisotropy: the two
cut planes share the apex and the clamped rim, so both profiles must turn through
almost the same total angle, and the mean of |kappa| over the profile therefore
comes out nearly equal even when the profiles differ by 7% of the rise in
between.  On the circular dome the section metric reports 0.9% anisotropy where
the true apex principal curvatures differ by 21%.

This module fits a quadric to the deformed vertices near the crown and returns
the curvature tensor there, so directional information survives.

Vertex selection
----------------
Vertices are chosen by their radius on a *reference* mesh (`ref_verts`), not by
distance on the deformed surface.  Passing the same reference mesh for two runs
guarantees they are fitted from the identical vertex set, which is what makes
runs comparable.  For the x-scaled ellipse study pass the unscaled circular mesh
as the reference so the ellipse and the circle sample the same 25 vertices.

Conventions
-----------
knit_dir_deg follows the C++ side (fem_batch_sensitivity.cpp:477):
    wale = (sin theta, cos theta),  course = (cos theta, -sin theta)
so at theta = 0 the wale axis is +y and the course axis is +x.

Curvatures are returned as unsigned magnitudes (m^-1) for a dome bulging in +z.
"""

import numpy as np

DEFAULT_R = 0.18      # m, radius on the reference mesh
MIN_POINTS = 12


def apex_vertex_mask(ref_verts, radius=DEFAULT_R):
    """Vertices within `radius` of the origin on the reference mesh."""
    return np.linalg.norm(np.asarray(ref_verts)[:, :2], axis=1) < radius


def apex_curvature(verts, knit_dir_deg, ref_verts=None, radius=DEFAULT_R):
    """
    Principal curvatures and directions at the crown.

    verts      : (n,3) deformed vertex positions, metres
    ref_verts  : (n,3) mesh used to pick the fit neighbourhood; defaults to
                 `verts` itself (then the neighbourhood is deformation-dependent
                 and runs are NOT strictly comparable)
    Returns a dict, or None if the neighbourhood is too small to fit.
    """
    V = np.asarray(verts, float)
    mask = apex_vertex_mask(V if ref_verts is None else ref_verts, radius)
    if mask.sum() < MIN_POINTS:
        return None

    apex = V[np.argmax(V[:, 2])]
    u = V[mask, 0] - apex[0]
    v = V[mask, 1] - apex[1]
    z = V[mask, 2] - apex[2]

    # z = a u^2 + b uv + c v^2 + p u + q v + r   (lab frame)
    A = np.column_stack([u * u, u * v, v * v, u, v, np.ones(mask.sum())])
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    a, b, c, p, q, _ = coef

    # first and second fundamental forms of the graph z = f(x,y) at the fit point
    g = 1.0 + p * p + q * q
    II = np.array([[2 * a, b], [b, 2 * c]]) / np.sqrt(g)
    I  = np.array([[1 + p * p, p * q], [p * q, 1 + q * q]])
    S  = np.linalg.solve(I, II)              # shape operator

    evals, evecs = np.linalg.eig(S)
    evals = np.real(evals)
    evecs = np.real(evecs)
    # dome bulges up -> both curvatures negative; report magnitudes
    order = np.argsort(np.abs(evals))        # k_min first
    k_min, k_max = np.abs(evals[order[0]]), np.abs(evals[order[1]])
    dir_max = evecs[:, order[1]]
    phi_max = np.degrees(np.arctan2(dir_max[1], dir_max[0])) % 180.0

    def normal_curv(w):
        w = np.asarray(w, float)
        return float(abs((w @ II @ w) / (w @ I @ w)))

    t = np.radians(knit_dir_deg)
    wale   = np.array([np.sin(t),  np.cos(t)])
    course = np.array([np.cos(t), -np.sin(t)])

    resid = z - A @ coef

    return {
        "k_min": k_min, "k_max": k_max,
        "k_ratio": k_max / k_min if k_min > 0 else np.nan,
        "k_max_dir_deg": phi_max,
        "k_x": normal_curv([1.0, 0.0]),
        "k_y": normal_curv([0.0, 1.0]),
        "k_wale": normal_curv(wale),
        "k_course": normal_curv(course),
        "H_apex": 0.5 * (k_min + k_max),
        "n_fit": int(mask.sum()),
        "fit_rms_mm": float(np.sqrt(np.mean(resid ** 2)) * 1000.0),
    }


def anisotropy_index(k_a, k_b):
    """Normalised difference, the same form as the section index in figM."""
    d = abs(k_a) + abs(k_b)
    return (k_a - k_b) / d if d > 1e-12 else 0.0
