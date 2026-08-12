"""
Nominal working point and run tables for the geometric-imperfection study.

The question: if the as-built structure differs from the design, how far does the
equilibrium shape move?  Block A is the method check on the circular dome — a
one-at-a-time forward/backward perturbation of each factor at its tolerance
(tolerances.py), from which we read off both the raw response and the normalised
sensitivity, and check that the +/- responses are symmetric.  Symmetry licenses
linearising about the nominal point in every later block, and licenses rescaling
Block A when a measured tolerance replaces an estimated one.

The decisive comparison for Block A is not internal: the predicted spread in
crown height goes up against the deviation measured in §6.1.3.  If the prediction
is a fraction of the measurement, every later band is a loose lower bound and
§6.5.2 has to say so.

Reported outputs follow the §6.5.2 table:
    L_pos    RMS vertex deviation from the baseline equilibrium (m)
    h_crown  crown height (m)
    sigma_max maximum principal stress (N/m)
    H_apex   mean curvature at the crown (1/m), for Delta_H
"""
import os

from tolerances import TOLERANCES

HERE   = os.path.dirname(os.path.abspath(__file__))
REPO   = os.path.dirname(HERE)
SA_DIR = os.path.join(REPO, "sensitivity_analysis")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MESH  = os.environ.get("CIRCULAR_FLAT_MESH",
                            os.path.join(REPO, "data", "circular_flat.off"))
FEM_BINARY = os.environ.get("FEM_BINARY",
                            os.path.join(REPO, "build", "fem_batch_sensitivity"))
DATA_DIR   = os.path.join(HERE, "data")
MESH_DIR   = os.path.join(DATA_DIR, "meshes")
FIG_DIR    = os.path.join(HERE, "figures")

# ── Nominal working point ─────────────────────────────────────────────────────
# Inside the model-validity box of the Chapter 6 sensitivity study
# (sensitivity_analysis/config.py: sf >= 0.95 keeps the membrane in tension,
# p <= 2000 Pa).  Here the dome rises 164 mm on a 600 mm radius (h/R = 0.27) with
# a max/mean stress ratio of 1.11: a smooth, well-inflated equilibrium, neither
# nearly flat (where curvature estimators go noisy) nor near compression onset.
S_WALE_NOM   = 1.10
S_COURSE_NOM = 1.10
KNIT_DIR_NOM = 0.0       # deg. Convention (fem_batch_sensitivity.cpp): at
                         # theta = 0 the wale axis is +y, the course axis +x.
PRESSURE_NOM = 1000.0    # Pa
MOTIF        = 1         # only used if no material override is passed

# Material, stated explicitly so baseline and perturbed runs share one code path.
# E1 is the wale membrane modulus in N/m (= E_volumetric x thickness).
E1_NOM = 5000.0
# R_RATIO is the PAPER convention r = E2/E1, the course/wale stiffness ratio, so
# r > 1 is course-stiffer.  Motif 1 is E1 = 5000, E2 = 12507 N/m.
# The binary consumes r_bin = E1/E2 (it computes E2 = E1/r_bin), so every call
# site inverts via r_bin().  Getting this backwards silently swaps the material
# into the wale-stiffer regime with no error.
R_RATIO_NOM = 12507.0 / 5000.0    # = 2.5014
NU_NOM      = 0.198

# circular_flat.off is a flat disc of exactly this radius; A11/A12 rescale it.
R_BOUNDARY_NOM = 0.600

NOMINAL = {
    "s_wale":   S_WALE_NOM,
    "s_course": S_COURSE_NOM,
    "pressure": PRESSURE_NOM,
    "E1":       E1_NOM,
    "r":        R_RATIO_NOM,
    "nu":       NU_NOM,
    "R":        R_BOUNDARY_NOM,
}


def r_bin(r_ratio: float) -> float:
    """Paper ratio r = E2/E1  ->  the binary's r = E1/E2."""
    return 1.0 / r_ratio


def perturbed(factor: str, sign: int) -> float:
    """The perturbed value of `factor` at +/- one tolerance."""
    return TOLERANCES[factor].perturb(NOMINAL[factor], sign)


# ── Block A run table ─────────────────────────────────────────────────────────
# 13 runs: baseline plus +/- on six factors.  nu and rho are absent by design —
# nu enters from Block B, and the circular dome carries no cable.
BLOCK_A = [
    ("A0",  None,       0),
    ("A1",  "s_wale",   +1),
    ("A2",  "s_wale",   -1),
    ("A3",  "s_course", +1),
    ("A4",  "s_course", -1),
    ("A5",  "pressure", +1),
    ("A6",  "pressure", -1),
    ("A7",  "E1",       +1),
    ("A8",  "E1",       -1),
    ("A9",  "r",        +1),
    ("A10", "r",        -1),
    ("A11", "R",        +1),
    ("A12", "R",        -1),
]

BLOCK_A_FACTORS = ["s_wale", "s_course", "pressure", "E1", "r", "R"]

FACTOR_LABELS = {
    "s_wale":   r"$s_{\mathrm{wale}}$",
    "s_course": r"$s_{\mathrm{course}}$",
    "pressure": r"$p$",
    "E1":       r"$E_1$",
    "r":        r"$E_2/E_1$",
    "nu":       r"$\nu_{12}$",
    "rho":      r"$\rho$",
    "R":        r"$R$",
}

# Outputs carried through the analysis, in §6.5.2 order.
OUTPUTS = [
    "L_pos",
    "crown_height",
    "max_stress",
    "H_apex",
]
# Additional diagnostics recorded but not part of the headline table.
DIAGNOSTICS = [
    "h_over_R",
    "mean_stress",
    "stress_ratio",
    "boundary_reaction_mean",
    "L_pos_max",
    "L_pos_shape",
    "k_min",
    "k_max",
    "k_ratio",
    "fit_rms_mm",
]
