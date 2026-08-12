"""
OFF mesh read/write and the geometric perturbations used by the imperfection
study.

Block A needs only a uniform in-plane rescale (boundary radius +/- delta).  The
non-uniform imperfections of the later blocks (an out-of-round boundary, a local
dimple, boundary out-of-plane waviness) belong here too, so they share one
tested reader and one convention: OFF, z up, disc centred on the origin.
"""
import os
import numpy as np


def load_off(path):
    """Return (V, F) as float (n,3) and int (m,3) arrays."""
    with open(path) as f:
        lines = [l for l in f.read().split("\n")]
    if not lines[0].strip().startswith("OFF"):
        raise ValueError(f"{path} is not an OFF file")
    nv, nf = (int(x) for x in lines[1].split()[:2])
    V = np.array([[float(x) for x in lines[2 + i].split()[:3]]
                  for i in range(nv)], dtype=float)
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]]
                  for i in range(nf)], dtype=int)
    return V, F


def save_off(path, V, F):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("OFF\n")
        f.write(f"{len(V)} {len(F)} 0\n")
        for v in V:
            f.write(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in F:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def boundary_radius(V):
    """Mean radius of the outermost ring of vertices, in metres.

    Measured rather than assumed, so a perturbed mesh reports the radius it
    actually has.  The reference disc is uniform enough that the outer ring is
    unambiguous: taking the vertices within 1% of the maximum radius picks up
    the boundary loop and nothing else.
    """
    rad = np.linalg.norm(V[:, :2], axis=1)
    ring = rad > 0.99 * rad.max()
    return float(rad[ring].mean())


def scale_radius(V, factor):
    """Uniform in-plane rescale about the origin.

    The disc is flat, so scaling x and y scales the whole rest geometry; element
    sizes scale with it, which is the intended meaning of "a disc of a different
    radius" rather than "the same disc remeshed".  z is left alone (it is zero on
    the reference mesh) so the routine stays correct if it is ever applied to a
    mesh with relief.
    """
    Vs = V.copy()
    Vs[:, 0] *= factor
    Vs[:, 1] *= factor
    return Vs


def radius_variant(base_mesh, factor, out_dir, tag):
    """Write a radius-scaled copy of base_mesh and return (path, radius_m)."""
    V, F = load_off(base_mesh)
    Vs = scale_radius(V, factor)
    path = os.path.join(out_dir, f"circular_flat_{tag}.off")
    save_off(path, Vs, F)
    return path, boundary_radius(Vs)
