import os

# ── Mesh ──────────────────────────────────────────────────────────────────────
_SA_DIR   = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SA_DIR)

MESH_PATH = os.environ.get(
    "CIRCULAR_FLAT_MESH",
    "/Users/duch/Documents/PhD/knit/2024_prototypes/callibration/"
    "flat_no_shortrows/sensitivity_analysis/circular_flat.off",
)

FEM_BINARY = os.environ.get(
    "FEM_BINARY",
    os.path.join(_REPO_DIR, "build-mac", "fem_batch_sensitivity"),
)

# ── Parameter bounds ──────────────────────────────────────────────────────────
PARAMS_NO_CABLE = {
    "sf_wale":   (0.8, 1.4),
    "sf_course": (0.8, 1.4),
    "knit_dir":  (0.0, 90.0),
    "pressure":  (200.0, 1200.0),
}

# cable_wale_lrest / cable_course_lrest: rest-length as a fraction of the
# cable's geometric arc length on the reference mesh.
# 1.0 = slack (no tension), <1.0 = pre-tensioned.
# Range (0.90, 1.0) covers full regime: snap-through occurs near 0.92.
PARAMS_CABLE = {
    "sf_wale":            (0.8, 1.4),
    "sf_course":          (0.8, 1.4),
    "knit_dir":           (0.0, 90.0),
    "pressure":           (200.0, 1200.0),
    "cable_wale_lrest":   (0.90, 1.0),
    "cable_course_lrest": (0.90, 1.0),
}

# ── Discrete parameters ───────────────────────────────────────────────────────
MOTIFS    = [1, 2]
HAS_CABLE = [False, True]

# Samples per group (4 groups × N_SAMPLES = total FEA runs)
N_SAMPLES = 150

# Samples per group for the material sensitivity study (7D/9D parameter space)
N_SAMPLES_MATERIAL = 500

# ── Material parameters per motif ─────────────────────────────────────────────
# motif 1: course-stiff (E2/E1=2.50), motif 2: less course-stiff (E2/E1=1.60)
MOTIF_PARAMS = {
    1: {"E1": 5000.0, "E2": 12507.0, "nu": 0.198},
    2: {"E1": 5000.0, "E2": 8000.0,  "nu": 0.198},
}

# ── Cable ─────────────────────────────────────────────────────────────────────
CABLE_EA = 150000.0  # N — steel cable (~1 mm diameter, E=200 GPa)

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
    "cable_wale_lrest":   (0.90, 1.0),
    "cable_course_lrest": (0.90, 1.0),
    "E1":                 (1000.0, 8000.0),
    "r":                  (3.0, 5.0),
    "nu":                 (0.45, 0.9),
}

# ── Quality filter thresholds ─────────────────────────────────────────────────
# Applied during FEA data generation to reject bad simulations.
QUALITY_CROWN_MAX        = 2.0   # m  — above this → exploded
QUALITY_STRESS_RATIO_MAX = 10.0  # max_stress/mean_stress — above this → unsmooth/localised

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
