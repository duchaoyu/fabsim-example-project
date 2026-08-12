"""
The geometries Block A runs on, and what each factor means on each of them.

Two so far.  `disc` is the circular dome: a flat rest mesh that inflates into a
dome, with no design target of its own, so the reference for L_pos is its own
baseline equilibrium.  `2part` is the two-lobe case study: the rest mesh is
already the 3D shape the structure is meant to hold, so it IS the design target
and L_pos can be measured against it as well as against the baseline.

One factor needs a per-geometry definition.  "Boundary radius" is unambiguous on
a disc, but the 2part footprint is not a circle.  Both are implemented as a
uniform in-plane rescale by delta_R / R_char, where R_char is the mean in-plane
radius of the clamped boundary ring — on the disc this is exactly the radius, and
on the 2part it is the footprint scale.  That keeps the two geometries comparable,
which is the point of running Block A twice.

It is worth being explicit that on a 3D rest mesh this is not quite the same
physical error as a boundary set-out mistake.  Scaling the whole mesh in plane
rescales the rest metric of the entire panel, whereas mis-set screws move the
boundary while the panel stays the size it was knitted.  On the flat disc the two
coincide; on the 2part they do not, and the boundary-only variant (displace the
ring, blend to zero over the interior) is a separate perturbation worth adding
before Block B.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import mesh_tools

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


@dataclass(frozen=True)
class Geometry:
    name: str
    mesh: str
    s_wale: float
    s_course: float
    knit_dir_deg: float
    pressure: float
    E1: float
    r_ratio: float          # paper convention E2/E1
    nu: float
    target: Optional[str]   # design-target OFF, or None
    nominal_source: str     # where the working point comes from
    note: str = ""

    def r_char(self) -> float:
        """Mean in-plane radius of the clamped boundary ring, in metres."""
        V, _ = mesh_tools.load_off(self.mesh)
        return mesh_tools.boundary_radius(V)

    def params(self) -> dict:
        return {"sf_wale": self.s_wale, "sf_course": self.s_course,
                "knit_dir_deg": self.knit_dir_deg, "pressure": self.pressure,
                "E1": self.E1, "r_ratio": self.r_ratio, "nu": self.nu}


# Motif 1, shared by both geometries: E1 = 5000 N/m, E2 = 12507 N/m, nu = 0.198.
# r is the paper ratio E2/E1; fem_runner inverts it for the binary.
_MOTIF1 = dict(E1=5000.0, r_ratio=12507.0 / 5000.0, nu=0.198)

DISC = Geometry(
    name="disc",
    mesh=os.path.join(REPO, "data", "circular_flat.off"),
    s_wale=1.10, s_course=1.10, knit_dir_deg=0.0, pressure=1000.0,
    target=None,
    nominal_source="chosen inside the Chapter 6 model-validity box "
                   "(sf >= 0.95, p <= 2000 Pa)",
    note="Flat rest mesh, 600 mm design radius. No design target: the reference "
         "for L_pos is its own baseline equilibrium.",
    **_MOTIF1,
)

# The 2part rest mesh is the shape the structure is meant to hold, so the stretch
# factors are not free to be chosen — they are whatever makes the inflated
# equilibrium sit on that shape.  fit_nominal.py fits them and writes
# data/nominal_2part.json; the values below are the fallback if it has not been
# run.  1.043 is the value in src/membrane_orthotropic.cpp:141, which inflates
# 41 mm above the target, so it is a starting point rather than a working point.
TWOPART_FALLBACK_SF = 1.043

_2part_mesh = os.path.join(REPO, "data", "2part", "2part_opt_simu_m.off")


def _fitted_2part_sf():
    path = os.path.join(HERE, "data", "nominal_2part.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return d["sf_wale"], d["sf_course"], (
            f"fitted to the design target, L_target = "
            f"{d['L_target_mm']:.2f} mm ({d['n_calls']} FEM calls)")
    return (TWOPART_FALLBACK_SF, TWOPART_FALLBACK_SF,
            "src/membrane_orthotropic.cpp:141 (NOT fitted — run fit_nominal.py)")


def _make_2part():
    sw, sc, src = _fitted_2part_sf()
    return Geometry(
        name="2part", mesh=_2part_mesh,
        s_wale=sw, s_course=sc, knit_dir_deg=0.0, pressure=1000.0,
        target=_2part_mesh,
        nominal_source=src,
        note="Two-lobe case study, 341 v / 634 f. The rest mesh is the design "
             "target, so L_target measures how far the equilibrium sits from the "
             "shape the structure is meant to hold. knit_dir = 0 gives wale along "
             "+y, matching face_vectors (0,1,0) in membrane_orthotropic.cpp.",
        **_MOTIF1,
    )


def get(name):
    if name == "disc":
        return DISC
    if name == "2part":
        return _make_2part()
    raise ValueError(f"unknown geometry {name!r}; expected 'disc' or '2part'")


NAMES = ["disc", "2part"]
