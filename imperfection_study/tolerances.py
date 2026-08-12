"""
The tolerance table.  Every run in this study is only as good as these numbers,
so they live in one place with their provenance and their error model attached.

Two things are recorded per parameter besides the magnitude:

`kind` — whether the error is SYSTEMATIC (one value applies to the whole
structure, because one panel is knitted from one fabric on one machine setting)
or INDEPENDENT (each region or each cable draws its own value, because each
turnbuckle is set by hand).  The stretch factors are both: the machine setting is
common to the panel, but region-to-region realisation varies.  This attribution
matters more than the magnitudes — it is what separates Block B from Block C, and
the systematic-versus-independent contrast is the finding that no other part of
the thesis provides.

`status` — MEASURED means the number came out of the named thesis section;
ESTIMATE means it is a stand-in until that section is run.  Anything still marked
ESTIMATE has to be declared as such in §6.5.2.

Updating a delta does NOT require re-running Block A.  Block A reports the
normalised sensitivity (the elasticity) alongside the raw response, and it checks
that the +/- responses are symmetric.  Symmetry is what licenses rescaling: if
delta_s from §6.2.3 comes back as 0.03 rather than the 0.05 assumed here, the
Block A response scales by 0.6.  Only if the symmetry check fails does the
magnitude have to be right before the run.
"""
from dataclasses import dataclass, field
from typing import Tuple

SYSTEMATIC  = "systematic"
INDEPENDENT = "independent"

MEASURED = "measured"
ESTIMATE = "estimate"


@dataclass(frozen=True)
class Tol:
    """One tolerance.

    value    : magnitude of one standard perturbation (see `relative`)
    relative : True  -> value is a fraction of the nominal
               False -> value is in the parameter's own units
    kind     : which error models this parameter participates in
    source   : where the number comes from
    status   : measured / estimate
    note     : anything a reader of §6.5.2 needs in order to trust the row
    """
    value: float
    relative: bool
    kind: Tuple[str, ...]
    source: str
    status: str
    note: str = ""

    def perturb(self, nominal: float, sign: int) -> float:
        """Apply +/- one standard perturbation to a nominal value."""
        if self.relative:
            return nominal * (1.0 + sign * self.value)
        return nominal + sign * self.value

    def absolute(self, nominal: float) -> float:
        """The perturbation in the parameter's own units, at this nominal."""
        return nominal * self.value if self.relative else self.value

    def rel_at(self, nominal: float) -> float:
        """The perturbation as a fraction of the nominal."""
        return self.value if self.relative else self.value / nominal


# ── The table ─────────────────────────────────────────────────────────────────
# NOTE on delta_s: this is the one tolerance with no measurement behind it, and
# on the evidence of the Chapter 6 sensitivity study it is also the one most
# likely to dominate the budget — the stretch factors carry the largest Sobol
# indices for crown height.  0.05 is used here because it is the step the inverse
# design's finite-difference gradient already takes (FDM/optimise_B5.py, eps=0.05),
# so the forward solver is known to resolve a perturbation of that size cleanly.
# That is a numerical justification, not a fabrication measurement.  §6.2.3
# replaces it.
TOLERANCES = {
    "s_wale": Tol(
        value=0.05, relative=False,
        kind=(SYSTEMATIC, INDEPENDENT),
        source="§6.2.3 commanded vs realised stretch factor",
        status=ESTIMATE,
        note="absolute on the stretch factor; 4.5% of the nominal 1.10. "
             "Systematic (machine setting, common to the panel) and independent "
             "(region-to-region realisation).",
    ),
    "s_course": Tol(
        value=0.05, relative=False,
        kind=(SYSTEMATIC, INDEPENDENT),
        source="§6.2.3 commanded vs realised stretch factor",
        status=ESTIMATE,
        note="as s_wale.",
    ),
    "pressure": Tol(
        value=0.05, relative=True,
        kind=(SYSTEMATIC,),
        source="§6.1.3 scatter in the sensor record while a state is held",
        status=ESTIMATE,
        note="50 Pa at the nominal 1000 Pa. Systematic: the whole structure "
             "sees one internal pressure at a given instant.",
    ),
    "E1": Tol(
        value=0.10, relative=True,
        kind=(SYSTEMATIC,),
        source="Chapter 4 biaxial tests, specimen-to-specimen scatter",
        status=ESTIMATE,
        note="Systematic only: one panel is one fabric, so a modulus error "
             "applies everywhere at once.",
    ),
    "r": Tol(
        value=0.10, relative=True,
        kind=(SYSTEMATIC,),
        source="Chapter 4 biaxial tests, specimen-to-specimen scatter",
        status=ESTIMATE,
        note="r is the paper ratio E2/E1. Systematic only, as E1.",
    ),
    "nu": Tol(
        value=0.10, relative=True,
        kind=(SYSTEMATIC,),
        source="Chapter 4 biaxial tests, specimen-to-specimen scatter",
        status=ESTIMATE,
        note="Systematic only. Not exercised in Block A (Poisson's ratio enters "
             "from Block B on).",
    ),
    "rho": Tol(
        value=0.001, relative=True,
        kind=(INDEPENDENT,),
        source="turnbuckle pitch",
        status=ESTIMATE,
        note="Rest-length scale. 1.25 mm of thread on a ~1.3 m cable is 0.1%. "
             "Independent per cable: each turnbuckle is set separately. "
             "Quantised rather than Gaussian in reality — a half-turn is the "
             "resolution — which the joint block should eventually respect. "
             "Not exercised in Block A (no cable on the circular dome).",
    ),
    "R": Tol(
        value=0.005, relative=False,
        kind=(SYSTEMATIC,),
        source="§6.1.2 screw spacing on the boundary ring",
        status=ESTIMATE,
        note="5 mm on a 600 mm radius, 0.83%. Systematic: a mis-set ring is "
             "round but the wrong size. An out-of-round boundary is a separate, "
             "non-uniform imperfection and is not this parameter.",
    ),
}


def summary_rows():
    """(name, magnitude string, kind, status, source) for printing."""
    rows = []
    for name, t in TOLERANCES.items():
        mag = f"{t.value:.4g}" + (" (rel)" if t.relative else " (abs)")
        rows.append((name, mag, "+".join(t.kind), t.status, t.source))
    return rows


def any_estimated() -> bool:
    return any(t.status == ESTIMATE for t in TOLERANCES.values())
