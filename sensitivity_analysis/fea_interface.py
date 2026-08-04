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

def _get_mesh(mesh_path):
    if mesh_path not in _mesh_cache:
        _mesh_cache[mesh_path] = load_off(mesh_path)
    return _mesh_cache[mesh_path]


def _write_cable_json(indices, EA, L_rest):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"indices": [int(i) for i in indices], "EA": EA, "L_rest": L_rest}, tmp)
    tmp.close()
    return tmp.name


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
    E1: float = None,
    r: float = None,
    nu: float = None,
    timeout: int = 300,
    mesh_path: str = None,
) -> dict:
    """
    Run fem_batch_sensitivity for one parameter set.

    cable_wale_lrest / cable_course_lrest: absolute cable rest length in METRES.
    The reference chord spans ~1.20 m with an arc length of ~1.29 m on the flat
    mesh, so values below ~1.29 pre-tension the cable and values above it are
    slack until the dome rises. -1 means no cable in that direction.

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

    def _make_cable_arg(lrest_m, angle_deg):
        if lrest_m < 0:
            return "none"
        indices = generate_cable_path(angle_deg, mesh)
        geo_len = cable_path_length(indices, V)
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
        path = _write_cable_json(indices, CABLE_EA, lrest_m)
        tmpfiles.append(path)
        return path

    cable_wale_arg   = _make_cable_arg(cable_wale_lrest,   WALE_CABLE_ANGLE)
    cable_course_arg = _make_cable_arg(cable_course_lrest, COURSE_CABLE_ANGLE)

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
    return scalars


def check_binary():
    if not os.path.exists(FEM_BINARY):
        raise FileNotFoundError(
            f"fem_batch_sensitivity not found at {FEM_BINARY}.\n"
            f"Build it first: cmake --build build --target fem_batch_sensitivity"
        )
