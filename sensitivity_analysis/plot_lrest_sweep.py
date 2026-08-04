"""
figQ: Cable rest-length sweep — section metrics comparison.

For each L_rest (no-cable + 1.20, 1.30, 1.40 m and extended 0.90, 1.00, 1.10 m),
plot four panels:
  A. Crown height + cable tension vs L_rest
  B. Shape profiles z(s) along x=0 and y=0
  C. Mean-curvature profiles H(s) along x=0 and y=0
  D. Von Mises stress maps (heatmap on mesh) for selected cases

Motif 1 and Motif 2 shown in separate columns.
"""

import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MESH_PATH
from curvature import read_off
from plot_section_profiles import _slice_plane
from run_lrest_sweep import SF as SWEEP_SF, KNIT_DIR as SWEEP_KNIT, PRESSURE as SWEEP_P

FIG_DIR  = os.path.join(os.path.dirname(__file__), "figures")
SWEEP_DIR = os.path.join(os.path.dirname(__file__), "data", "lrest_sweep")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "figure.dpi": 150,
})

_VERTS_REST, _FACES = read_off(MESH_PATH)

# Weight given to the two clamped-rim points when fitting a section profile.
# They are a boundary condition (z = 0 exactly), not sample means, so they must
# outweigh a single bin; 10 pins the ends without overriding the interior data.
RIM_WEIGHT = 10.0

# Fixed y-range for the section-stress row (row 4).  The data spans 1287-2370 Pa
# over the plotted cases, so anchoring at 0 left the lower half of the panel empty
# and squeezed every curve into a band near the top.  Nothing is clipped: 287 Pa
# of margin below, 130 Pa above.
STRESS_YLIM = (1000.0, 2500.0)

# Fixed y-range for the section-curvature row (row 3).  Top is exactly 0: kappa
# <= 0 on physical grounds and, with the rim anchored in the fit, the least
# negative value anywhere in the sweep is -0.36, so nothing is clipped.
CURV_YLIM = (-4.0, 0.0)

# ── Gaussian curvature helpers ────────────────────────────────────────────────

def _find_boundary_verts(faces):
    cnt = defaultdict(int)
    for f in faces:
        for i in range(3):
            cnt[tuple(sorted([int(f[i]), int(f[(i+1)%3])]))] += 1
    bv = set()
    for e, c in cnt.items():
        if c == 1:
            bv.update(e)
    return bv

_BDR_VERTS = _find_boundary_verts(_FACES)

def _gaussian_curvature_vertices(verts):
    """Discrete angle-defect Gaussian curvature per vertex (NaN at boundary)."""
    n = len(verts)
    angle_sum = np.zeros(n)
    area_sum  = np.zeros(n)
    for f in _FACES:
        pts = [verts[f[i]] for i in range(3)]
        cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        face_area = 0.5 * np.linalg.norm(cross)
        if face_area < 1e-15:
            continue
        for i in range(3):
            a = pts[(i+1)%3] - pts[i]
            b = pts[(i+2)%3] - pts[i]
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if la < 1e-15 or lb < 1e-15:
                continue
            angle_sum[f[i]] += np.arccos(np.clip(np.dot(a, b) / (la * lb), -1, 1))
            area_sum[f[i]]  += face_area / 3.0
    K = np.full(n, np.nan)
    for i in range(n):
        if i not in _BDR_VERTS and area_sum[i] > 1e-15:
            K[i] = (2*np.pi - angle_sum[i]) / area_sum[i]
    return K


def _gauss_curv_section(verts, fixed_axis, band=0.12, n_bins=13, trim_outer=1):
    """Gaussian curvature K along a section, as a MEDIAN per position bin.

    K comes from the discrete angle defect (2*pi - sum theta)/A per vertex, so it
    needs no profile fit and no double differentiation — which is why it is a
    sounder basis for an argument than the fitted section curvature it replaced.

    It does, however, need robust averaging.  The raw per-vertex field is
    outlier-heavy wherever the cable distorts the mesh: on the taut runs it spans
    -121 to +1155 m^-2 while its interquartile range is only ~2, and within a
    single 250-350 mm radius ring it scatters -26.6..+10.7 about a median of 0.84.
    A mean, or a polynomial through the means, is captured by those few extremes;
    a median per bin is not.  Bins are wide (band 0.12R, 13 bins) for the same
    reason.

    trim_outer drops the outermost bin on each side: for the no-cable dome, which
    is essentially axisymmetric, the two sections disagree there by ~2x (3.64 vs
    1.75 m^-2), so that bin carries mesh bias rather than shape.
    """
    K = _gaussian_curvature_vertices(verts)
    R = 0.6
    near = (np.abs(verts[:, fixed_axis]) < band * R) & np.isfinite(K)
    if near.sum() < 12:
        return np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    s_mm = verts[near, pos_axis] * 1000.0
    k    = K[near]

    # mirror about the centre: the loading and geometry are symmetric, so this
    # doubles the sample count per bin without assuming anything new
    s_mm = np.concatenate([s_mm, -s_mm])
    k    = np.concatenate([k, k])

    S = R * 1000.0
    edges   = np.linspace(-S, S, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    med = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = (s_mm >= edges[i]) & (s_mm < edges[i + 1])
        if sel.sum() >= 3:
            med[i] = np.median(k[sel])

    if trim_outer > 0:
        med[:trim_outer] = np.nan
        med[len(med) - trim_outer:] = np.nan

    ok = np.isfinite(med)
    return centres[ok], med[ok]


# ── colour: one step of viridis per cable rest length ────────────────────────
# Colour encodes the cable REST LENGTH, not tension.  Tension was a poor carrier:
# it is an *output*, and t_max differs between the motifs (2.2 kN vs 4.2 kN), so
# the same colour meant a different tension in each column and could not be read
# back to a specific run.  L_rest is the swept input and takes seven values.
#
# L_rest is ordered, so the ramp is sequential (viridis, purple -> green -> yellow)
# rather than seven unordered hues: neighbouring lengths then sit next to each
# other in colour, and the progression itself carries "shorter cable = tighter".
# Ordinal gates on this ramp, against a white surface: min OKLCH dL between
# adjacent steps 0.081 (>= 0.06 required).  The lightest step (#9bd93c, L_rest
# 1.50 m) is 1.70:1 against white, under the 2.0 floor — kept deliberately so the
# ramp reaches yellow as intended, with the relief the floor requires: every
# length is named in the legend, so no curve is identified by colour alone.
VIRIDIS_SPAN = (0.0, 0.85)   # stop short of pure yellow, which vanishes on white

# The no-cable case is a reference, not one of the lengths, so it gets a hue from
# outside the ramp.  Orange is far from every viridis step to normal vision
# (worst dE 26.9, well over the 15 floor) at 3.20:1 on white.  Under simulated
# deuteranopia, though, it sits 6.8 from the 1.45 m green — inside the 6-8 warn
# band, not a clean pass.  That is legal only with secondary encoding, which is
# present: the no-cable curve is the only one drawn at lw 1.5 (the lengths are
# 1.0) and it is named in the legend, so it is never identified by hue alone.
# Grey was tried first and was worse: viridis' middle steps are low-chroma
# blue-teal, so every mid-grey landed 8.8-11.2 from a step, under the 15 floor
# for normal vision too.
NOCABLE_COLOR = "#eb6834"   # orange — reserved for "no cable"
TENSION_GREY  = "#52514e"   # row-1 tension axis ink + its bars

# Bound to the L_rest values themselves, so a colour follows the length rather
# than a position in the plotting loop.
_LEN_COLORS = {}

def _set_len_colors(l_vals):
    """Bind one viridis step per L_rest, ascending."""
    _LEN_COLORS.clear()
    vals = sorted({round(float(v), 2) for v in l_vals})
    lo, hi = VIRIDIS_SPAN
    n = max(len(vals) - 1, 1)
    for i, L in enumerate(vals):
        _LEN_COLORS[L] = mcolors.to_hex(cm.viridis(lo + (hi - lo) * i / n))

def _len_color(L_rest):
    return _LEN_COLORS.get(round(float(L_rest), 2), "0.5")


def _resolve_prefix(prefix):
    """Re-root a prefix from sweep_results.csv onto this checkout.

    The CSV stores absolute paths from the machine that ran the sweep (a macOS
    checkout, /Users/duch/...), so they resolve nowhere else.  _load_verts and
    _load_stress return None on a missing file, so every profile row of figQ was
    silently drawn empty rather than failing.  The run files themselves are in
    SWEEP_DIR under the same basenames.
    """
    return os.path.join(SWEEP_DIR, os.path.basename(str(prefix)))


def _load_verts(prefix):
    p = _resolve_prefix(prefix) + "_verts.csv"
    if not os.path.exists(p): return None
    return pd.read_csv(p).sort_values("vid")[["x","y","z"]].values


def _load_stress(prefix):
    p = _resolve_prefix(prefix) + "_stress.csv"
    if not os.path.exists(p): return None
    return pd.read_csv(p).sort_values("face")


def _smooth_section(verts, fixed_axis, n_eval=200, band=0.08, n_bins=40, trim=0.0,
                    z_crown_mm=None, piecewise=False):
    """
    Bin vertices near the section plane then fit a cubic spline — no symmetry
    enforced so creases or kinks at the centre are preserved.
    κ = z'' / (1+z'^2)^(3/2),  returned in m⁻¹.
    Returns (pos_mm, z_mm, kappa_per_m).
    """
    R = 0.6
    near = np.abs(verts[:, fixed_axis]) < band * R
    if near.sum() < 6:
        return np.array([]), np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    pos = verts[near, pos_axis] * 1000
    z   = verts[near, 2]        * 1000

    order = np.argsort(pos)
    pos, z = pos[order], z[order]
    span = pos.max() - pos.min()
    mask = (pos > pos.min() + trim*span) & (pos < pos.max() - trim*span)
    pos, z = pos[mask], z[mask]
    if len(pos) < 6:
        return np.array([]), np.array([]), np.array([])

    # For the piecewise (crease) case, mirror data about pos=0 before binning
    # so both halves are fit from symmetrized averages, giving a symmetric kink.
    if piecewise:
        pos_all = np.concatenate([pos, -pos])
        z_all   = np.concatenate([z,    z])
        order2  = np.argsort(pos_all)
        pos, z  = pos_all[order2], z_all[order2]

    # Save full data extent — used for evaluation range (bin centres fall short)
    p_lo, p_hi = pos.min(), pos.max()

    # Bin into means — removes per-vertex scatter while keeping real features
    edges   = np.linspace(p_lo, p_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means_z = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = (pos >= edges[k]) & (pos < edges[k+1])
        if sel.sum() >= 1:
            means_z[k] = np.mean(z[sel])
    ok = np.isfinite(means_z)
    if ok.sum() < 4:
        return np.array([]), np.array([]), np.array([])

    c_ok, z_ok = centres[ok], means_z[ok]

    # ── Anchor both ends at the clamped rim ──────────────────────────────────
    # The section runs to the rim, where the membrane is clamped: z = 0 there, and
    # that holds exactly (every boundary vertex reads z = -0.0000 mm in every run).
    # Without these two points the evaluated range was just the extent of whichever
    # vertices fell inside the sampling band, which is not symmetric — the x=0 cuts
    # ran e.g. -546.8..+595.9 mm, up to 49 mm lopsided, while the y=0 cuts looked
    # symmetric only because the piecewise branch mirrors its data.  Adding the rim
    # as weighted data makes every section span the same +/-R and stops the quartic
    # flaring free at the ends (which is what pushed kappa positive there).
    S = R * 1000.0
    c_ok = np.concatenate([[-S], c_ok, [S]])
    z_ok = np.concatenate([[0.0], z_ok, [0.0]])
    # The rim is a boundary condition, not one more noisy sample, so it carries
    # more weight than a single bin mean would.
    w_ok = np.concatenate([[RIM_WEIGHT], np.ones(len(c_ok) - 2), [RIM_WEIGHT]])
    p_lo, p_hi = -S, S

    def _single():
        """One polynomial over the whole section — smooth through s=0."""
        deg = min(4, len(c_ok) - 1)
        poly_z   = np.poly1d(np.polyfit(c_ok, z_ok, deg, w=w_ok))
        dpoly_z  = poly_z.deriv(1)
        ddpoly_z = poly_z.deriv(2)
        pos_s = np.linspace(p_lo, p_hi, n_eval)
        z_s   = poly_z(pos_s)
        if z_crown_mm is not None:
            z_s = z_s + (z_crown_mm - float(poly_z(0.0)))
        dz    = dpoly_z(pos_s)
        ddz   = ddpoly_z(pos_s)
        kappa = ddz / (1.0 + dz**2)**1.5 * 1000.0
        return pos_s, z_s, kappa, poly_z

    if not piecewise:
        pos_s, z_s, kappa, _ = _single()
        return pos_s, z_s, kappa

    # Fit two separate polynomials: pos<0 and pos>=0, so a crease at pos=0
    # is preserved as a discontinuity in slope rather than being smoothed away.
    def _half_fit(mask):
        if mask.sum() < 2:
            return None, None, None
        deg = min(4, mask.sum() - 1)
        c = np.polyfit(c_ok[mask], z_ok[mask], deg, w=w_ok[mask])
        p = np.poly1d(c)
        return p, p.deriv(1), p.deriv(2)

    neg_mask = c_ok < 0
    pos_mask = c_ok >= 0

    p_neg, dp_neg, ddp_neg = _half_fit(neg_mask)
    p_pos, dp_pos, ddp_pos = _half_fit(pos_mask)

    # ── Is there actually a crease? ──────────────────────────────────────────
    # Two independently fitted halves meet at s=0 at whatever angle their
    # polynomials happen to have, and because the data are mirrored about s=0
    # before binning those slopes are equal and opposite by construction.  So the
    # piecewise fit manufactures a V — and a large negative curvature spike — even
    # where the section is perfectly smooth.  Applied unconditionally it put a
    # spike on the no-cable dome (kappa_centre/kappa_flank 2.5x for motif 1, 4.6x
    # for motif 2) which reversed the true shape: read with one polynomial the
    # crown is *flatter* than the flanks, as a pressurised clamped membrane
    # should be.  It equally affected the slack-cable runs (1.45-1.50 m, tension
    # 0-112 N), where there is no crease either.
    #
    # So let the data decide: keep the two-piece fit only where it explains the
    # binned profile near the centre materially better than one polynomial does.
    # A real crease leaves a systematic residual that a single polynomial cannot
    # absorb; a smooth crown does not.
    # The test is run on the INTERIOR data only — the binned means, without the two
    # rim anchors.  With the anchors in, the comparison is rigged: the single fit
    # has to satisfy both of them while each half only has to satisfy one, so the
    # two-piece fit buys freedom in the centre and wins for reasons that have
    # nothing to do with a crease.  That regression put the manufactured spike back
    # on the no-cable dome.  Decide on the interior, then fit the chosen model with
    # the anchors for display.
    if p_neg is not None and p_pos is not None:
        c_int, z_int = c_ok[1:-1], z_ok[1:-1]
        span_c = np.max(np.abs(c_int)) if c_int.size else 1.0
        core   = np.abs(c_int) <= 0.30 * (span_c or 1.0)   # region a crease affects
        if core.sum() >= 4:
            def _fit_int(msk, deg_cap=4):
                if msk.sum() < 2:
                    return None
                return np.poly1d(np.polyfit(c_int[msk], z_int[msk],
                                            min(deg_cap, msk.sum() - 1)))
            f_all = _fit_int(np.ones(len(c_int), bool))
            f_neg = _fit_int(c_int < 0)
            f_pos = _fit_int(c_int >= 0)
            if f_all is not None and f_neg is not None and f_pos is not None:
                r_single = z_int[core] - f_all(c_int[core])
                r_pw = np.where(c_int[core] < 0,
                                z_int[core] - f_neg(c_int[core]),
                                z_int[core] - f_pos(c_int[core]))
                rms_single = float(np.sqrt(np.mean(r_single**2)))
                rms_pw     = float(np.sqrt(np.mean(r_pw**2)))
                # 0.75: the two-piece fit has more freedom, so it always fits at
                # least as well; it must win clearly before a crease is claimed.
                if not (rms_pw < 0.75 * rms_single):
                    pos_s, z_s, kappa, _ = _single()
                    return pos_s, z_s, kappa

    # Build piecewise evaluation arrays (NaN gap keeps the two halves separate)
    half_neg = n_eval // 2
    half_pos = n_eval - half_neg
    ps_neg = np.linspace(p_lo,   -1e-6, half_neg) if p_neg else np.array([])
    ps_pos = np.linspace(1e-6,    p_hi, half_pos) if p_pos else np.array([])

    pos_s = np.concatenate([ps_neg, [np.nan], ps_pos])
    z_s   = np.concatenate([
        p_neg(ps_neg) if p_neg is not None else np.array([]),
        [np.nan],
        p_pos(ps_pos) if p_pos is not None else np.array([])
    ])

    # Anchor: shift both halves so they meet at the shared crown z at pos=0
    if z_crown_mm is not None:
        if p_neg is not None:
            z_s[:half_neg]        += z_crown_mm - float(p_neg(0.0))
        if p_pos is not None:
            z_s[half_neg+1:]      += z_crown_mm - float(p_pos(0.0))

    # Curvature: computed per half (NaN at the gap)
    def _kappa(ps, dp, ddp):
        if dp is None or len(ps) == 0:
            return np.array([])
        dz  = dp(ps)
        ddz = ddp(ps)
        return ddz / (1.0 + dz**2)**1.5 * 1000.0

    kappa = np.concatenate([
        _kappa(ps_neg, dp_neg, ddp_neg),
        [np.nan],
        _kappa(ps_pos, dp_pos, ddp_pos)
    ])
    return pos_s, z_s, kappa


def _section_profiles(verts):
    """Return dict with x=0 and y=0 smooth profiles (pos, z, kappa)."""
    # Shared crown: vertex closest to (0,0) in xy — anchors both polynomial fits
    r2 = verts[:, 0]**2 + verts[:, 1]**2
    z_crown_mm = float(verts[np.argmin(r2), 2]) * 1000.0
    out = {}
    # x=0 section: along the cable — single smooth fit
    out["x0"] = _smooth_section(verts, 0, z_crown_mm=z_crown_mm, piecewise=False)
    # y=0 section: perpendicular to cable — piecewise fit reveals crease at x=0
    out["y0"] = _smooth_section(verts, 1, z_crown_mm=z_crown_mm, piecewise=True)
    return out


def _stress_section(verts, sdf, fixed_axis, n_bins=40):
    """Return (pos_mm, mean_von_mises) along the section plane (y=0), symmetrized."""
    SECTION_TOL = 0.04
    face_ids  = sdf["face"].values.astype(int)
    centroids = verts[_FACES[face_ids]].mean(axis=1)
    vm        = sdf["von_mises"].values

    mask = np.abs(centroids[:, fixed_axis]) < SECTION_TOL
    if not mask.any():
        return np.array([]), np.array([])

    pos_axis = 1 if fixed_axis == 0 else 0
    pts  = centroids[mask, pos_axis] * 1000
    vm_m = vm[mask]

    # Symmetrize before binning
    pts_sym = np.concatenate([pts,  -pts])
    vm_sym  = np.concatenate([vm_m,  vm_m])
    ord2    = np.argsort(pts_sym)
    pts, vm_m = pts_sym[ord2], vm_sym[ord2]

    p_lo, p_hi = pts.min(), pts.max()
    edges   = np.linspace(p_lo, p_hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    means   = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = (pts >= edges[k]) & (pts < edges[k+1])
        if sel.sum() >= 1:
            means[k] = np.mean(vm_m[sel])

    ok = np.isfinite(means)
    if ok.sum() < 4:
        return centres[ok], means[ok]

    c_ok, m_ok = centres[ok], means[ok]
    poly_vm = np.poly1d(np.polyfit(c_ok, m_ok, min(4, len(c_ok)-1)))
    pos_s   = np.linspace(p_lo, p_hi, 200)
    return pos_s, np.maximum(0, poly_vm(pos_s))


# ─────────────────────────────────────────────────────────────────────────────

# A cable carrying no tension exerts nothing, so its run says nothing about cable
# action — and at sf=1.1 the fully slack lengths return the *same* solution, not
# merely a similar one: 1.45 and 1.50 m both give T = 0.0 N in both motifs with
# identical crown heights (173.8 mm for motif 1, 202.7 mm for motif 2).  Plotting
# them adds duplicate curves.  Lengths are dropped only when every motif is slack,
# so both panels keep the same L_rest set.
#
# 1.40 m is retained because motif 2 still develops 33.7 N there.  Note motif 1 is
# already slack at 1.40 m (T = 0.0), so its 1.40 m curve is the slack solution.
SLACK_TENSION_N = 1.0


def _plotted_lengths(df):
    """L_rest values to draw: everything except those slack in every motif."""
    keep = []
    for L in sorted(df["L_rest_m"].dropna().unique()):
        t = df.loc[df["L_rest_m"] == L, "cable_tension"]
        if (t > SLACK_TENSION_N).any():
            keep.append(L)
    return keep


def plot_lrest_sweep(save=True):
    df = pd.read_csv(os.path.join(SWEEP_DIR, "sweep_results.csv"))

    # Build ordered label list (no_cable first, then increasing L_rest)
    L_VALS = sorted([v for v in df["L_rest_m"].dropna().unique()])
    LABELS  = ["no cable"] + [f"{v:.2f} m" for v in L_VALS]

    SECTION_TOL = 0.03  # faces within 30mm of plane

    motifs = [1, 2]
    n_rows, n_cols = 4, 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5*n_cols, 3.2*n_rows),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    twin_axes = []          # row-0 right-hand (cable tension) axes, one per column
    for col_idx, motif in enumerate(motifs):
        sub = df[df["motif"] == motif].copy()
        nocable_row  = sub[sub["L_rest_m"].isna()].iloc[0]
        cable_rows   = sub[sub["L_rest_m"].isin(_plotted_lengths(df))].sort_values("L_rest_m")

        # ── Row 0: crown height + tension summary ──────────────────────────────
        ax0 = axes[0, col_idx]
        ax0b = ax0.twinx()
        twin_axes.append(ax0b)
        # Layer order for this row: tension bars at the back, crown-height curve
        # over them, ticks and spines over everything.  twinx() puts the new axes
        # on top by construction, so the bars were covering the green curve; lift
        # ax0 above ax0b and drop its opaque background so the bars still show.
        ax0.set_zorder(ax0b.get_zorder() + 1)
        ax0.patch.set_visible(False)
        ax0.set_axisbelow(False)      # ticks/spines above the curve

        l_vals = cable_rows["L_rest_m"].values
        h_vals = cable_rows["crown_height"].values * 1000
        t_vals = cable_rows["cable_tension"].values

        # Bar width from the actual L_rest spacing, not a fixed number.  It was
        # 0.06 while the samples sit 0.05 apart, so every bar overlapped its
        # neighbours; deriving it keeps a gap if the sweep spacing ever changes.
        l_sorted = np.sort(l_vals)
        spacing  = np.min(np.diff(l_sorted)) if len(l_sorted) > 1 else 0.05
        bar_w    = 0.6 * spacing

        # This row carries two quantities on two axes, so "no cable" alone would
        # not say which one the reference line refers to.  It is a crown height,
        # and it belongs to the left axis.
        ax0.axhline(nocable_row["crown_height"]*1000, color=NOCABLE_COLOR,
                    lw=1.8, ls="--", label="crown height, no cable")
        ax0.plot(l_vals, h_vals, "o-", color="#2E8B57", lw=2, ms=6,
                 label="crown height, with cable")
        # Neutral grey: orange now means "no cable", and these bars are a
        # different quantity on their own axis, so they must not compete for it.
        ax0b.bar(l_vals, t_vals, width=bar_w, color="0.72", alpha=0.85,
                 label="cable tension")
        ax0b.axhline(0, color="0.7", lw=0.5)

        ax0.set_ylabel("Crown height  (mm)", color="#2E8B57")
        # Grey, matching the bars it labels — orange is reserved for nothing now
        # that no cable is grey too, and an axis label reads better as ink than as
        # a series colour.
        ax0b.set_ylabel("Cable tension  (N)", color=TENSION_GREY)
        ax0b.tick_params(axis="y", colors=TENSION_GREY, labelsize=8)
        ax0.set_xlabel(r"$L_{rest}$  (m)")
        ax0.set_title(f"Motif {motif} — crown height & cable tension")
        # Combined legend
        h1, l1 = ax0.get_legend_handles_labels()
        h2, l2 = ax0b.get_legend_handles_labels()
        ax0.legend(h1+h2, l1+l2, fontsize=7.5, loc="upper left")
        ax0.set_xlim(l_vals.min()-0.05, l_vals.max()+0.05)

        # ── Load section profiles for all cases ────────────────────────────────
        _set_len_colors(cable_rows["L_rest_m"].values)

        profiles = {}
        # no cable
        verts_nc = _load_verts(nocable_row["prefix"])
        if verts_nc is not None:
            profiles["no_cable"] = _section_profiles(verts_nc)

        for _, crow in cable_rows.iterrows():
            key = f"{crow['L_rest_m']:.2f}"
            verts_c = _load_verts(crow["prefix"])
            if verts_c is not None:
                profiles[key] = _section_profiles(verts_c)

        # ── Row 1: shape profiles z(s) ─────────────────────────────────────────
        ax1 = axes[1, col_idx]
        ax1.set_axisbelow(True)       # ticks/grid behind the profiles, not over them

        for key, prof in profiles.items():
            if key == "no_cable":
                color, lw, alpha = NOCABLE_COLOR, 1.5, 1.0
            else:
                color = _len_color(key)
                lw, alpha = 1.0, 0.95

            for plane, ls in [("x0", "-"), ("y0", "--")]:
                pos, z, _ = prof[plane]   # already smooth from poly fit
                if len(pos) < 3: continue
                ax1.plot(pos, z, color=color, ls=ls, lw=lw, alpha=alpha)

        ax1.set_xlabel("position along section  (mm)")
        ax1.set_ylabel("z  (mm)")
        ax1.set_title(f"Motif {motif} — shape profiles  (solid=x=0, dashed=y=0)")
        ax1.autoscale(axis="y", tight=False)
        y0, y1 = ax1.get_ylim()
        pad = 0.08 * (y1 - y0)
        ax1.set_ylim(y0 - pad, y1 + pad)
        # Legend.  No "low tension"/"high tension" swatches: they were drawn at
        # arbitrary fractions of t_max (0.3 and 0.9), so they named no actual run
        # and their colours meant nothing readable.  The colourbar already carries
        # the tension scale, so the legend only has to identify the no-cable case
        # and the two cut planes.
        legend_handles = [
            Line2D([0],[0], color=NOCABLE_COLOR, lw=2, label="no cable"),
            Line2D([0],[0], color="0.5", ls="-",  lw=1.2, label="x=0 plane"),
            Line2D([0],[0], color="0.5", ls="--", lw=1.2, label="y=0 plane"),
        ]
        # Bottom, not top: at the top the box sat over the crowns of the profiles,
        # which is where the curves separate and the figure is actually read.  The
        # lower centre is empty — every profile rises away from it.
        ax1.legend(handles=legend_handles, fontsize=7, loc="lower center",
                   ncol=3, framealpha=0.8)

        # ── Row 2: Gaussian curvature K across the cable ──────────────────────
        # K instead of the fitted section curvature: it is intrinsic, it comes
        # straight from the mesh by angle defect with no profile fit, and its sign
        # carries the argument — K > 0 is a synclastic dome, K < 0 an anticlastic
        # saddle, K = 0 developable.
        #
        # Only the y=0 cut (across the cable) is shown.  The x=0 cut runs directly
        # along the cable, over the vertices the cable distorts, and K there is not
        # recoverable at this mesh resolution: it returns -14 m^-2 in a rim bin at
        # 1.20 m and flips sign in a way that does not order with L_rest.
        ax2 = axes[2, col_idx]
        k_all = []

        for case_key, row_data in ([("no_cable", nocable_row)]
                                    + [(f"{r2['L_rest_m']:.2f}", r2)
                                       for _, r2 in cable_rows.iterrows()]):
            verts_g = _load_verts(row_data["prefix"])
            if verts_g is None:
                continue
            if case_key == "no_cable":
                color, ms, alpha = NOCABLE_COLOR, 3.6, 1.0
            else:
                color, ms, alpha = _len_color(case_key), 3.0, 0.95
            pos_g, K_g = _gauss_curv_section(verts_g, 1)
            if len(pos_g) < 3:
                continue
            k_all.extend(K_g[np.isfinite(K_g)])
            # Markers, not a curve: each point is the MEDIAN of the per-vertex K
            # in one position bin, so this row is a set of samples rather than a
            # continuous field.  A solid line implied a resolution the estimator
            # does not have — the depth of the central dip changes with bin count.
            # The thin line only guides the eye between samples of one case.
            ax2.plot(pos_g, K_g, color=color, lw=0.6, alpha=alpha,
                     marker="o", ms=ms, mfc=color, mec=color, mew=0.0)

        ax2.axhline(0, color="0.55", lw=0.8, ls="-")   # developable: K = 0
        ax2.set_xlabel("position along section  (mm)")
        ax2.set_ylabel(r"Gaussian curvature  $K$  (m$^{-2}$)")
        ax2.set_title(f"Motif {motif} — Gaussian curvature across the cable ($y{{=}}0$)")

        # ── Row 3: von Mises stress — binned along sections ────────────────────
        ax3 = axes[3, col_idx]

        for case_key, row_data in ([("no_cable", nocable_row)]
                                    + [(f"{r['L_rest_m']:.2f}", r)
                                       for _, r in cable_rows.iterrows()]):
            sdf    = _load_stress(row_data["prefix"])
            verts_d = _load_verts(row_data["prefix"])
            if sdf is None or verts_d is None: continue

            if case_key == "no_cable":
                color, lw, alpha = NOCABLE_COLOR, 1.5, 1.0
            else:
                color = _len_color(case_key)
                lw, alpha = 1.0, 0.95

            pts, vm_mean = _stress_section(verts_d, sdf, fixed_axis=1)
            if len(pts) < 2: continue
            ax3.plot(pts, vm_mean, color=color, lw=lw, alpha=alpha)

        ax3.set_xlabel("position along section  (mm)")
        ax3.set_ylabel("Von Mises stress  (Pa)")
        ax3.set_title(f"Motif {motif} — section stress  (y=0 plane)")
        ax3.set_ylim(bottom=0)


    # ── Match the two motif columns axis-for-axis ─────────────────────────────
    # Each row shows the same quantity for motif 1 and motif 2 and the columns are
    # read against each other, so both must carry the same scale.  Left to
    # autoscale they did not: crown height ran 160-260 mm against 140-280,
    # curvature -3.0 against -4.0 and stress 1750 against 1400 Pa, so equal
    # heights on the page meant different numbers.  Taking the union of each
    # row's limits and applying it to both columns makes the ticks identical as
    # well, since the locator is driven by the limits.
    def _match_row(row_axes):
        finite = [a for a in row_axes if all(np.isfinite(a.get_ylim()))]
        if len(finite) < 2:
            return
        lo = min(a.get_ylim()[0] for a in finite)
        hi = max(a.get_ylim()[1] for a in finite)
        xlo = min(a.get_xlim()[0] for a in finite)
        xhi = max(a.get_xlim()[1] for a in finite)
        for a in finite:
            a.set_ylim(lo, hi)
            a.set_xlim(xlo, xhi)

    for r in range(n_rows):
        _match_row([axes[r, c] for c in range(n_cols)])
    _match_row(twin_axes)      # row 0's cable-tension scale

    # Row 3 (Gaussian curvature) keeps the shared limits _match_row already gave
    # it: K is signed and both signs are meaningful, so it must not be clipped.

    # Row 4: fixed stress window, same in both columns, so the curves fill the
    # panel instead of hugging the top of a range that starts at zero.
    for c in range(n_cols):
        axes[3, c].set_ylim(STRESS_YLIM)

    # Shared legend for the rest lengths.  This replaces the cable-tension
    # colourbar: colour now identifies a discrete L_rest, so a continuous scale
    # would describe an encoding the figure no longer uses.
    len_handles = [Line2D([0], [0], color=NOCABLE_COLOR, lw=2.2, label="no cable")]
    len_handles += [Line2D([0], [0], color=_len_color(L), lw=2.2,
                           label=f"{L:.2f} m")
                    for L in sorted(_LEN_COLORS)]
    fig.legend(handles=len_handles, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title=r"$L_{rest}$", title_fontsize=8.5,
               framealpha=0.9, borderpad=0.7, labelspacing=0.6)

    fig.suptitle(
        # conditions taken from run_lrest_sweep, not typed in — the caption said
        # s_f=1.0 for a while after the sweep moved to 1.1
        rf"Frictionless sliding steel cable — rest-length sweep  "
        rf"($s_f$={SWEEP_SF}, $\theta$={SWEEP_KNIT:.0f}°, $p$={SWEEP_P:.0f} Pa)"
        "\n"
        r"Colour = cable rest length $L_{rest}$ (viridis, short$\to$long; orange = no cable)",
        fontsize=10, y=1.005)

    if save:
        path = os.path.join(FIG_DIR, "figQ_lrest_sweep.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting L_rest sweep metrics...")
    plot_lrest_sweep()
    print("Done.")
