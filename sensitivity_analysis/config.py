import os

# ── Mesh ──────────────────────────────────────────────────────────────────────
_SA_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SA_DIR)

MESH_PATH = os.environ.get(
    "CIRCULAR_FLAT_MESH",
    os.path.join(_REPO_DIR, "data", "circular_flat.off"),
)

FEM_BINARY = os.environ.get(
    "FEM_BINARY",
    os.path.join(_REPO_DIR, "build", "fem_batch_sensitivity"),
)

# ── Parameter bounds ──────────────────────────────────────────────────────────
PARAMS_NO_CABLE = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 1200.0),
}

# cable_wale_lrest / cable_course_lrest: absolute cable rest length in METRES.
# The cable is a diameter chord of the reference disc: span 1.197 m, arc length
# 1.29 m on the flat mesh.  So <1.29 pre-tensions the cable and >1.29 is slack
# until the dome rises; (1.2, 1.4) brackets that transition.
#
# These were a *fraction* of the cable's arc length until the cable geometry was
# corrected (see cable_path.py: the path was a zigzag 7-8x the chord span, and
# the wale cable was a 95 deg chord rather than a diameter).  Any cached cable
# data generated before that fix is invalid and must be re-run.
PARAMS_CABLE = {
    "sf_wale":            (0.8, 1.4),
    "sf_course":          (0.8, 1.4),
    "knit_dir":           (0.0, 90.0),
    "pressure":           (200.0, 1200.0),
    "cable_wale_lrest":   (1.2, 1.4),
    "cable_course_lrest": (1.2, 1.4),
}

# ── Discrete parameters ───────────────────────────────────────────────────────
MOTIFS     = [1, 2]
HAS_CABLE  = [False, True]
CABLE_AXES = ["wale", "course"]

# Samples per group (4 groups × N_SAMPLES = total FEA runs)
N_SAMPLES = 150

# Samples per group for cable orientation study
N_SAMPLES_ORIENT = 150

# Parameter bounds for the cable orientation study
# L_rest is an absolute rest length in metres (see the note above)
PARAMS_CABLE_ORIENT = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 1200.0),
    "L_rest":    (1.2, 1.4),
}

# Samples per group for the material sensitivity study (7D/9D parameter space)
N_SAMPLES_MATERIAL = 500

# ── Material parameters per motif ─────────────────────────────────────────────
# motif 1: course-stiff (E2/E1=2.50), motif 2: less course-stiff (E2/E1=1.60)
MOTIF_PARAMS = {
    1: {"E1": 5000.0, "E2": 12507.0, "nu": 0.198},
    2: {"E1": 5000.0, "E2": 8000.0,  "nu": 0.198},
}

# ── Cable ─────────────────────────────────────────────────────────────────────
# EA in N.  At E = 200 GPa this is A = EA / E, so 800 kN is A = 4.0 mm2 (~2.3 mm
# diameter).  It was 150 kN (A = 0.75 mm2), at which the tensions the study
# reports imply cable stresses of 1300-4000 MPa — past any steel.  The dome shape
# is almost insensitive to the choice, because the cable is effectively
# inextensible either way: over f = 0.93-0.99 at the validity-box centre, crown
# height differs by 2% and tension by 8% between 0.75 and 4.0 mm2
# (probe_cable_influence.py, figP).  What changes is whether the reported
# tensions describe a cable that could exist: 4.0 mm2 keeps them under ~130 MPa
# at the box centre, against 636 MPa for 0.75 mm2 with no margin for the corners.
CABLE_EA = 800000.0  # N — steel cable, A = 4.0 mm2 at E = 200 GPa

# ── Material sensitivity study parameter bounds ───────────────────────────────
# Wale-stiffer regime only: E1 > E2, r = E1/E2 ∈ (3, 5).
# E1 in N/m (2D membrane modulus = E_vol × thickness).
# E1=1000–8000 N/m ≈ 1–8 MPa for t=1 mm.
PARAMS_MATERIAL_NO_CABLE = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 1200.0),
    "E1":        (1000.0, 8000.0),
    "r":         (3.0, 5.0),
    "nu":        (0.45, 0.9),
}

PARAMS_MATERIAL_CABLE = {
    "sf_wale":            (0.8, 1.4),
    "sf_course":          (0.8, 1.4),
    "knit_dir":           (0.0, 90.0),
    "pressure":           (200.0, 1200.0),
    "cable_wale_lrest":   (1.2, 1.4),
    "cable_course_lrest": (1.2, 1.4),
    "E1":                 (1000.0, 8000.0),
    "r":                  (3.0, 5.0),
    "nu":                 (0.45, 0.9),
}

# ── Extended material parameter bounds (E1/E2 surface study) ──────────────────
# Uses E2 directly (not ratio r) to allow symmetric coverage of wale-stiffer
# (E1>E2) and course-stiffer (E2>E1) regimes, and the isotropic line E1=E2.
PARAMS_MATERIAL_EXT_NO_CABLE = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 1200.0),
    "E1":        (1000.0, 20000.0),
    "E2":        (1000.0, 20000.0),
    "nu":        (0.09, 0.3),
}
N_SAMPLES_MATERIAL_EXT = 500

# ── r-parameterised material study (r = E2/E1 as an explicit Sobol input) ──────
# IMPORTANT: here r is E2/E1 — the course/wale stiffness ratio, the paper's
# convention — so r >= 1 is the course-stiffer regime, consistent with
# MOTIF_PARAMS (motif 1: E1=5000, E2=12507).  The FEM binary consumes
# E2 = E1/r_bin, so the run scripts pass r_bin = 1/r; see
# sampling.generate_material_r_samples.  Getting that inversion wrong silently
# produces the wale-stiffer regime instead.
#
# Why a fresh design rather than reusing material_/material_ext_: Sobol needs a
# box design in the input variables it apportions.  material_ (r = E1/E2 ∈ 3–5)
# has E2/E1 ∈ 0.20–0.33, entirely outside r ≥ 1; material_ext_ samples E1 and E2
# independently, so r is not an input there, and its nu ≤ 0.3 / p ≤ 1200 leave
# half the nu range and 40% of the p range below unsampled.
#
# p up to 2000 Pa: probed and safe.  The corner (E1=1000, p=2000, r=5, nu=0.5)
# converges at h/R = 0.71 with max/mean stress 1.29.  Response is governed by
# pR/E1; raising p_max 1200 → 2000 moves the fraction of the box with
# pR/E1 > 0.3 from 2.6% to 6.6%, so ~93% stays in the already-sampled regime.
PARAMS_MATERIAL_R_NO_CABLE = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 2000.0),
    "E1":        (1000.0, 20000.0),
    "r":         (1.0, 5.0),
    "nu":        (0.1, 0.5),
}

# cable_wale_frac / cable_course_frac: cable rest length as a FRACTION of the
# cable-free section length, L_rest = frac * L_nocable, resolved per sample by one
# extra cable-free solve (fea_interface.nocable_section_lengths).
#
# This replaces the absolute rest length in metres, which could not be taut
# everywhere: L_nocable IS the slack threshold, and it runs from 1.2902 m on a
# barely-inflated dome to 1.3643 m on the tallest, so a fixed (1.2, 1.4) m range
# left 40% of cable runs slack.  A slack run is not a response — it is the
# no-cable model with an inert cable — and it cannot be filtered out afterwards
# either, because filtering on T > 0 conditions on the output instead of
# restricting the box, and Saltelli would then extrapolate into the deleted
# region.  frac < 1 is taut by construction and frac is a proper box.
#
# Upper bound 0.99 rather than 1.0 leaves the cable just engaged; the lower bound
# is set by the straight chord between the cable's pinned endpoints (1.1966 m),
# which is the hard geometric floor.  0.93 * 1.29 = 1.20 m clears it.
#
# NOTE the older studies below still carry cable_*_lrest in metres.  The names
# differ deliberately, so the two conventions cannot be confused — but all of that
# cable data predates the 2026-08-04 cable-path fix and is void regardless.
PARAMS_MATERIAL_R_CABLE = {
    "sf_wale":           (0.8, 1.4),
    "sf_course":         (0.8, 1.4),
    "knit_dir":          (0.0, 90.0),
    "pressure":          (200.0, 2000.0),
    "cable_wale_frac":   (0.93, 0.99),
    "cable_course_frac": (0.93, 0.99),
    "E1":                (1000.0, 20000.0),
    "r":                 (1.0, 5.0),
    "nu":                (0.1, 0.5),
}
N_SAMPLES_MATERIAL_R = 800

# ── Model-validity box for the r study ────────────────────────────────────────
# Same box, with the stretch factors cut at 0.95.  A membrane element carries no
# compression, so a run whose principal stress goes negative is outside the
# formulation, not merely noisy — and below s = 0.95 that is the norm rather than
# the exception (sampling 250 retained runs: 39% of faces in compression for
# s_course in 0.80-0.90 and 94% of runs affected, against 3% / 8% for s >= 1.0;
# s_wale behaves the same).  Those runs also carry most of the solver failures:
# convergence over the planned design is 83% on the full box and 96% with
# s_course >= 0.95.
#
# Pressure is deliberately left at 200 Pa.  The model is valid there; what
# degrades is the section-curvature estimator on a barely-inflated dome, and
# truncating an operating variable to compensate for a post-processing weakness
# would discard valid runs (a soft fabric at 200 Pa has pR/E1 = 0.12).
PARAMS_MATERIAL_R_VALID_NO_CABLE = {
    **PARAMS_MATERIAL_R_NO_CABLE,
    "sf_wale":   (0.95, 1.4),
    "sf_course": (0.95, 1.4),
}
PARAMS_MATERIAL_R_VALID_CABLE = {
    **PARAMS_MATERIAL_R_CABLE,
    "sf_wale":   (0.95, 1.4),
    "sf_course": (0.95, 1.4),
}

# ── Quality filter thresholds ─────────────────────────────────────────────────
# Applied during FEA data generation to reject bad simulations.
QUALITY_CROWN_MAX        = 2.0   # m  — above this → exploded
QUALITY_STRESS_RATIO_MAX = 10.0  # max_stress/mean_stress — above this → unsmooth/localised
# Below this → the membrane did not inflate (solver failed / collapsed).  This was
# 0.01 m, which was safe while crown heights ran 50-450 mm but silently rejected
# valid runs once the nu12 sweep reached 0.9: a stiff membrane (E1=20 kN/m,
# E2/E1=2.5, nu=0.9) legitimately rises only 7.6 mm at p=1000 Pa, with a smooth
# dome and max/mean stress of exactly 1.0.  Genuine failures return 0.0 exactly,
# and unsmooth ones are caught by QUALITY_STRESS_RATIO_MAX.
QUALITY_CROWN_MIN        = 0.001  # m

# ── Data output ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_SA_DIR, "data")

# ── Surrogate ─────────────────────────────────────────────────────────────────
GP_PCA_VARIANCE = 0.95
GP_KERNEL       = "matern_2.5"
TRAIN_VAL_SPLIT = 0.2
RANDOM_SEED     = 42

# ── Sobol ─────────────────────────────────────────────────────────────────────
SOBOL_N_BASE    = 1024    # SALib Saltelli: actual samples = N*(D+2)

# ── Scalar outputs tracked ────────────────────────────────────────────────────
SCALAR_OUTPUTS = [
    "crown_height",
    "H_mean_x0",
    "H_mean_y0",
    "max_stress",
    "mean_stress",
    "cable_wale_tension",
    "cable_course_tension",
    "boundary_reaction_mean",
]
