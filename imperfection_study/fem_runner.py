"""
Thin wrapper around fem_batch_sensitivity for the imperfection study.

Deliberately not sensitivity_analysis/fea_interface.py: that wrapper's job is the
cable study, and it generates cable paths on whichever mesh it is given.  The
imperfection blocks perturb the mesh itself, and Block A has no cable at all, so
going through the cable machinery would mean cable paths silently regenerated on a
rescaled disc.  Here the cable arguments are simply absent, and two options the
later blocks need — a per-face stretch-factor field, and stretch-factor
continuation — are exposed instead.

Binary CLI (fem_batch_sensitivity.cpp):
    <mesh> <sf_wale> <sf_course> <knit_dir_deg> <pressure> <motif>
    <cable_wale|none> <cable_course|none> <prefix>
    [fixed_vertices|auto] [material_json|none] [perface_sf_csv|none] [sf_ramp_steps]
"""
import csv
import json
import os
import subprocess
import tempfile

import numpy as np

import mesh_tools
from imperfection_config import FEM_BINARY, r_bin


def check_binary():
    if not os.path.exists(FEM_BINARY):
        raise FileNotFoundError(
            f"fem_batch_sensitivity not found at {FEM_BINARY}.\n"
            "Build it: cmake --build build --target fem_batch_sensitivity")


def read_verts(path):
    """Deformed positions as an (n,3) array ordered by vertex id."""
    with open(path) as f:
        rows = sorted(((int(r["vid"]),
                        (float(r["x"]), float(r["y"]), float(r["z"])))
                       for r in csv.DictReader(f)), key=lambda t: t[0])
    return np.array([xyz for _, xyz in rows], dtype=float)


def write_perface_sf(path, sf_wale_per_face, sf_course_per_face):
    """Per-face stretch-factor field, for the non-uniform imperfection blocks."""
    with open(path, "w") as f:
        f.write("fid,sf_wale,sf_course\n")
        for fid, (sw, sc) in enumerate(zip(sf_wale_per_face,
                                           sf_course_per_face)):
            f.write(f"{fid},{sw:.8f},{sc:.8f}\n")
    return path


def cable_length(mesh, indices):
    """Geometric length of the cable polyline on the rest mesh, in metres.

    This is what SlidingCable computes for itself when L_rest is not given, so
    passing it back explicitly reproduces the default; scaling it is how the
    rest-length tolerance is applied.
    """
    V, _ = mesh_tools.load_off(mesh)
    P = V[list(indices)]
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def run(mesh, prefix, sf_wale, sf_course, knit_dir_deg, pressure,
        E1, r_ratio, nu, motif=1, perface_sf_csv=None, sf_ramp_steps=1,
        cable=None, timeout=300):
    """One FEM solve.  Returns a dict of scalars plus the deformed vertices.

    r_ratio is the PAPER ratio E2/E1; the inversion to the binary's convention
    happens here, in one place.

    cable, if given, is {"indices": [...], "EA": N, "L_rest": m} and is passed
    as the binary's wale cable.  Omitting L_rest lets the solver take the
    geometric length; giving it is how a rest-length error is applied.
    """
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    mat = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump({"E1": float(E1), "r": float(r_bin(r_ratio)), "nu": float(nu)}, mat)
    mat.close()

    cable_arg = "none"
    if cable is not None:
        cf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"indices": [int(i) for i in cable["indices"]],
                   "EA": float(cable["EA"]),
                   **({"L_rest": float(cable["L_rest"])}
                      if cable.get("L_rest") else {})}, cf)
        cf.close()
        cable_arg = cf.name

    cmd = [FEM_BINARY, mesh,
           f"{sf_wale:.8f}", f"{sf_course:.8f}", f"{knit_dir_deg:.6f}",
           f"{pressure:.4f}", str(motif),
           cable_arg, "none", prefix,
           "auto", mat.name,
           perface_sf_csv if perface_sf_csv else "none",
           str(int(sf_ramp_steps))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"fem_batch_sensitivity failed (rc={proc.returncode}) for "
                f"{prefix}:\n{proc.stderr}")
    finally:
        os.unlink(mat.name)

    scalars_path = prefix + "_scalars.csv"
    if not os.path.exists(scalars_path):
        raise FileNotFoundError(f"expected output missing: {scalars_path}")
    with open(scalars_path) as f:
        out = {k: float(v) for k, v in next(csv.DictReader(f)).items()}

    out["verts"] = read_verts(prefix + "_verts.csv")
    return out
