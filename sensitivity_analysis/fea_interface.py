"""
Python wrapper around the fem_batch_sensitivity C++ binary.

Each call runs one FEM simulation and returns a dict of scalar outputs.
"""

import csv
import json
import os
import subprocess
import tempfile

from config import FEM_BINARY, MESH_PATH, CABLE_EA
from cable_path import (
    generate_cable_path, cable_path_length, load_off,
    WALE_CABLE_ANGLE, COURSE_CABLE_ANGLE,
)

_mesh_cache = {}

# Cable angle by slot name, so the two-pass helper and _make_cable_arg agree.
CABLE_ANGLES = {"wale": WALE_CABLE_ANGLE, "course": COURSE_CABLE_ANGLE}

def _get_mesh(mesh_path):
    if mesh_path not in _mesh_cache:
        _mesh_cache[mesh_path] = load_off(mesh_path)
    return _mesh_cache[mesh_path]


def _write_cable_json(indices, EA, L_rest):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"indices": [int(i) for i in indices], "EA": EA, "L_rest": L_rest}, tmp)
    tmp.close()
    return tmp.name


def _read_verts(path: str):
    """Deformed vertex positions from a *_verts.csv, ordered by vid.

    Returned as an (n, 3) array, not a list of tuples: cable_path_length
    subtracts rows of it.
    """
    import numpy as np
    with open(path) as f:
        rows = sorted(((int(r["vid"]), (float(r["x"]), float(r["y"]),
                                        float(r["z"])))
                       for r in csv.DictReader(f)), key=lambda t: t[0])
    return np.array([xyz for _, xyz in rows], dtype=float)


def nocable_section_lengths(sf_wale, sf_course, knit_dir_deg, pressure, motif,
                            output_prefix, E1=None, r=None, nu=None,
                            timeout=300, mesh_path=None, keep=False) -> dict:
    """Arc length of each cable path on the CABLE-FREE equilibrium, in metres.

    This is the quantity that decides whether a cable is slack: a cable longer
    than the free dome's section carries no load at all.  It is not a property of
    the mesh — it grows with inflation, from the flat arc (1.2902 m wale) on a
    barely-inflated dome to 1.3643 m on the tallest, so an absolute rest length
    that is taut for one parameter set is slack for another.  Normalising against
    it makes the rest length mean the same thing everywhere in the box.

    Costs one extra FEA solve per sample (~0.2 s).
    """
    pre = output_prefix + "__nocable"
    run_fea(sf_wale, sf_course, knit_dir_deg, pressure, motif, pre,
            cable_wale_lrest=-1.0, cable_course_lrest=-1.0,
            E1=E1, r=r, nu=nu, timeout=timeout, mesh_path=mesh_path)
    mesh = mesh_path or MESH_PATH
    V = _read_verts(pre + "_verts.csv")
    out = {name: cable_path_length(generate_cable_path(angle, mesh), V)
           for name, angle in CABLE_ANGLES.items()}
    if not keep:
        for suffix in ("_scalars.csv", "_verts.csv", "_stress.csv"):
            p = pre + suffix
            if os.path.exists(p):
                os.unlink(p)
    return out


def _write_material_json(E1: float, r: float, nu: float) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"E1": E1, "r": r, "nu": nu}, tmp)
    tmp.close()
    return tmp.name


def run_fea(
    sf_wale: float,
    sf_course: float,
    knit_dir_deg: float,
    pressure: float,
    motif: int,
    output_prefix: str,
    cable_wale_lrest: float = -1.0,
    cable_course_lrest: float = -1.0,
    cable_wale_frac: float = None,
    cable_course_frac: float = None,
    cable_EA: float = None,
    E1: float = None,
    r: float = None,
    nu: float = None,
    timeout: int = 300,
    mesh_path: str = None,
) -> dict:
    """
    Run fem_batch_sensitivity for one parameter set.

    The cable rest length can be given either way, but not both for the same
    slot:

    cable_wale_lrest / cable_course_lrest: absolute rest length in METRES.
      The reference chord spans ~1.20 m with an arc length of ~1.29 m on the flat
      mesh, so values below ~1.29 pre-tension the cable and values above it are
      slack until the dome rises. -1 means no cable in that direction.

    cable_wale_frac / cable_course_frac: rest length as a FRACTION of the
      cable-free section length, L_rest = frac * L_nocable.  This costs one extra
      no-cable solve per call (see nocable_section_lengths) and is the preferred
      convention: L_nocable is exactly the slack threshold, so frac < 1 is taut
      everywhere in the box, whereas any fixed length in metres is taut in one
      corner and slack in another.  Over (1.2, 1.4) m, 40% of runs came back
      slack and contributed a plateau that no transform can rescue.

      The floor is the straight chord between the cable's pinned endpoints
      (1.1966 m wale, 1.1970 m course) — below that the cable would have to be
      shorter than the straight line joining its own fixed ends.

    cable_EA: axial stiffness override in N; defaults to config.CABLE_EA.
      config.CABLE_EA = 150 kN is A = 0.75 mm2 at E = 200 GPa, so tensions of
      1000 N and up exceed what such a cable could carry.

    E1, r, nu: if all three are provided, override the motif material params.
      E2 is computed as E1/r inside the binary. E1 is in N/m (membrane modulus).

    mesh_path: run on a mesh other than config.MESH_PATH (used by the
      non-axisymmetric-boundary study, run_knit_dir_sweep_ellipse.py).  Note the
      cable paths are generated on this same mesh.

    Returns dict with keys: crown_height, max_stress, mean_stress,
      cable_wale_tension, cable_course_tension, boundary_reaction_mean.
    """
    mesh = mesh_path or MESH_PATH
    V, F = _get_mesh(mesh)
    tmpfiles = []
    EA = CABLE_EA if cable_EA is None else cable_EA

    # Resolve any fraction to metres before touching the binary, which only ever
    # speaks absolute rest lengths.
    fracs = {"wale": cable_wale_frac, "course": cable_course_frac}
    lrests = {"wale": cable_wale_lrest, "course": cable_course_lrest}
    for slot, frac in fracs.items():
        if frac is None:
            continue
        if lrests[slot] >= 0:
            raise ValueError(
                f"cable_{slot}_frac and cable_{slot}_lrest are two conventions "
                "for one quantity — pass exactly one")
        if not 0.5 < frac <= 1.0:
            raise ValueError(
                f"cable_{slot}_frac = {frac} is not a fraction of the "
                "cable-free section length; frac > 1 is slack by construction")
    if any(f is not None for f in fracs.values()):
        L_nc = nocable_section_lengths(
            sf_wale, sf_course, knit_dir_deg, pressure, motif, output_prefix,
            E1=E1, r=r, nu=nu, timeout=timeout, mesh_path=mesh)
        for slot, frac in fracs.items():
            if frac is not None:
                lrests[slot] = frac * L_nc[slot]

    def _make_cable_arg(lrest_m, angle_deg):
        if lrest_m < 0:
            return "none"
        indices = generate_cable_path(angle_deg, mesh)
        geo_len = cable_path_length(indices, V)
        chord = float(sum((V[indices[-1]][k] - V[indices[0]][k]) ** 2
                          for k in range(3)) ** 0.5)
        if lrest_m <= chord:
            raise ValueError(
                f"cable L_rest = {lrest_m:.4f} m is at or below the straight "
                f"chord between its pinned endpoints ({chord:.4f} m) — the cable "
                "cannot be shorter than the straight line joining its own ends")
        # L_rest is an absolute rest length in METRES — the convention
        # fem_batch_sensitivity.cpp and sampling.py already documented.  It used
        # to be a fraction of geo_len, which was meaningless while
        # generate_cable_path returned a zigzag 7-8x the chord span; both are
        # fixed together, so any cached cable data predating this is invalid.
        if not 0.4 * geo_len < lrest_m < 3.0 * geo_len:
            raise ValueError(
                f"cable L_rest = {lrest_m:.4f} m is implausible against a "
                f"geometric arc length of {geo_len:.4f} m — a fraction "
                f"(the old convention) rather than metres?")
        path = _write_cable_json(indices, EA, lrest_m)
        tmpfiles.append(path)
        return path

    cable_wale_arg   = _make_cable_arg(lrests["wale"],   WALE_CABLE_ANGLE)
    cable_course_arg = _make_cable_arg(lrests["course"], COURSE_CABLE_ANGLE)

    cmd = [
        FEM_BINARY,
        mesh,
        f"{sf_wale:.6f}",
        f"{sf_course:.6f}",
        f"{knit_dir_deg:.4f}",
        f"{pressure:.2f}",
        str(motif),
        cable_wale_arg,
        cable_course_arg,
        output_prefix,
    ]

    # Append material override if all three params are provided
    if E1 is not None and r is not None and nu is not None:
        mat_path = _write_material_json(E1, r, nu)
        tmpfiles.append(mat_path)
        cmd += ["auto", mat_path]  # fixed_vertices=auto, then material JSON

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"fem_batch_sensitivity failed (rc={result.returncode}):\n"
                f"{result.stderr}"
            )
    finally:
        for p in tmpfiles:
            if os.path.exists(p):
                os.unlink(p)

    scalars_path = output_prefix + "_scalars.csv"
    if not os.path.exists(scalars_path):
        raise FileNotFoundError(f"Expected output not found: {scalars_path}")

    with open(scalars_path) as f:
        row = next(csv.DictReader(f))
        scalars = {k: float(v) for k, v in row.items()}

    scalars["verts_path"]  = output_prefix + "_verts.csv"
    scalars["stress_path"] = output_prefix + "_stress.csv"
    # Record what the fractions resolved to, so a results CSV carries both the
    # sampled fraction and the metres the solver actually saw.  Without this the
    # run is not reproducible from the CSV alone: L_nocable depends on the whole
    # parameter set, so the fraction cannot be converted back after the fact.
    for slot in CABLE_ANGLES:
        if fracs[slot] is not None:
            scalars[f"cable_{slot}_lrest"]    = lrests[slot]
            scalars[f"cable_{slot}_L_nocable"] = L_nc[slot]
    return scalars


def check_binary():
    if not os.path.exists(FEM_BINARY):
        raise FileNotFoundError(
            f"fem_batch_sensitivity not found at {FEM_BINARY}.\n"
            f"Build it first: cmake --build build --target fem_batch_sensitivity"
        )
