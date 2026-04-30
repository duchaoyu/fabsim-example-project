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


def run_fea(
    sf_wale: float,
    sf_course: float,
    knit_dir_deg: float,
    pressure: float,
    motif: int,
    output_prefix: str,
    cable_wale_lrest: float = -1.0,
    cable_course_lrest: float = -1.0,
    timeout: int = 120,
) -> dict:
    """
    Run fem_batch_sensitivity for one parameter set.

    cable_wale_lrest / cable_course_lrest: rest-length as a fraction of the
    cable's geometric arc length on the reference mesh. Values < 1 pre-tension
    the cable; -1 means no cable in that direction.

    Returns dict with keys: crown_height, max_stress, mean_stress,
      cable_wale_tension, cable_course_tension, boundary_reaction_mean.
    """
    V, F = _get_mesh(MESH_PATH)
    tmpfiles = []

    def _make_cable_arg(lrest_frac, angle_deg):
        if lrest_frac < 0:
            return "none"
        indices = generate_cable_path(angle_deg, MESH_PATH)
        geo_len = cable_path_length(indices, V)
        L_rest  = lrest_frac * geo_len
        path = _write_cable_json(indices, CABLE_EA, L_rest)
        tmpfiles.append(path)
        return path

    cable_wale_arg   = _make_cable_arg(cable_wale_lrest,   WALE_CABLE_ANGLE)
    cable_course_arg = _make_cable_arg(cable_course_lrest, COURSE_CABLE_ANGLE)

    cmd = [
        FEM_BINARY,
        MESH_PATH,
        f"{sf_wale:.6f}",
        f"{sf_course:.6f}",
        f"{knit_dir_deg:.4f}",
        f"{pressure:.2f}",
        str(motif),
        cable_wale_arg,
        cable_course_arg,
        output_prefix,
    ]

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
