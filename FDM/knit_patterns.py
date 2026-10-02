"""
Knitting patterns for the Section 7.4 case studies with compas_knit, timed.

For each case: the target mesh, its wale field (sign-consistent, per vertex) and the
optimised pre-strain per vertex go to compas_knit's three steps,

    1. trajectories  (stripes executable: stripe pattern -> trajectories -> stitches)
    2. pattern       (knitting order and stitch columns -> bitmap)
    3. knittable     (one yarn carrier, row after row)

and the wall-clock time of each is recorded.  The pre-strain varies by region in most
cases, so it is passed per vertex as columns 4-5 of the field file ("x y z sw sc"),
which the stripes executable reads as the local stretch along the wale and the course:
trajectory spacing 2 st_h / sw, stitch width st_w / sc.  Each vertex takes the stretch
of the FEM region nearest to it.

Convention.  The FEM rest shape is the target divided by sf (anisotropic_rest_shape.h,
s = 1/sf): sf > 1 means the knit is smaller than the target and is stretched onto it, so
on the target a stitch is sf * st_h tall.  compas_knit's generate_stripes divides the
spacing by its stretch factor, so it is given 1/sf (KNIT_CONVENTION=fem, the default).
KNIT_CONVENTION=compas_knit passes sf unchanged instead.

Stitch size: st_h 2.297 mm, st_w 3.54 mm (compas_knit defaults); bed 365 needles.

    COMPAS_KNIT=~/compas_knit ~/compas_knit/.venv/bin/python FDM/knit_patterns.py [case ...]

Outputs: FDM/knit_patterns/<case>/ (bitmaps, pixel data, trajectories) and
FDM/knit_patterns/summary.json.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KNIT = os.path.expanduser(os.environ.get("COMPAS_KNIT", "~/compas_knit"))
OPT = os.path.join(HERE, "optimisation")
OUT = os.path.join(HERE, "knit_patterns")
FABSIM = os.path.join(KNIT, "data", "fabsim")

from compas_knit.stripes import generate_stripes, read_trajectories, read_neighbours   # noqa: E402
from compas_knit.pattern import knitting_pattern, knittable, write_pattern             # noqa: E402

ST_H, ST_W, BED = 0.002297, 0.00354, 365
CONVENTION = os.environ.get("KNIT_CONVENTION", "fem")


# ── mesh I/O ──────────────────────────────────────────────────────────────────
def read_off(p):
    L = [l for l in open(p).read().split("\n") if l.strip() and not l.startswith("#")]
    nv, nf = map(int, L[1].split()[:2])
    V = np.array([list(map(float, L[2 + i].split()[:3])) for i in range(nv)])
    F = np.array([list(map(int, L[2 + nv + i].split()[1:4])) for i in range(nf)])
    return V, F


def read_obj_vertices(p):
    """Vertices in file order, duplicates kept, as the stripes executable reads them."""
    return np.array([list(map(float, l.split()[1:4])) for l in open(p) if l.startswith("v ")])


def write_obj(p, V, F):
    with open(p, "w") as f:
        for v in V:
            f.write("v %.10f %.10f %.10f\n" % tuple(v))
        for t in F:
            f.write("f %d %d %d\n" % tuple(np.asarray(t) + 1))


# ── per-vertex stretch from an FEM region map ─────────────────────────────────
def vertex_stretch(Vk, fem_V, fem_F, face_region, sf_of_region):
    """Each compas_knit vertex takes the stretch of the nearest FEM face's region."""
    tree = cKDTree(fem_V[fem_F].mean(1))
    _, idx = tree.query(Vk)
    reg = np.asarray(face_region)[idx]
    return np.array([sf_of_region[r] for r in reg])


def regions_json(name):
    d = json.load(open(os.path.join(OPT, name)))
    if "regions" in d:
        return d["regions"]
    return d["params"]["regions"]


def map_json(name):
    return json.load(open(os.path.join(OPT, name)))["face_regions"]


# ── the cases ─────────────────────────────────────────────────────────────────
def case_2part(tag, sf):
    """7.4.1: the 341-vertex 2-part mesh, wale along y, as best_fit_* used it."""
    V, F = read_off(os.path.join(ROOT, "data", "2part", "2part_opt_simu_m.off"))
    if isinstance(sf, tuple):
        S = np.tile(sf, (len(V), 1))
    else:                                    # E: per-face regions from the faces file
        reg = np.full(len(F), -1)
        cur = None
        for l in open(os.path.join(ROOT, "out", "sf_3region_adaptive_cable_faces.txt")):
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            if l.startswith("REGION_"):
                cur = int(l.split()[0][7:])
                continue
            for x in l.split():
                reg[int(x)] = cur
        S = vertex_stretch(V, V, F, reg, sf)
    field = np.tile([0.0, 1.0, 0.0], (len(V), 1))
    return V, F, field, S


def case_fabsim(model, fem_mesh, sf, fmap=None):
    """A compas_knit fabsim model (subdivided, sign-consistent field) with the FEM pre-strain."""
    mesh = os.path.join(FABSIM, model, model + ".obj")
    Vk = read_obj_vertices(mesh)
    field = np.loadtxt(os.path.join(FABSIM, model, model + "_vertex_directional_field.txt"))[:, :3]
    if fmap is None:
        S = np.tile(sf, (len(Vk), 1))
    else:
        fV, fF = read_off(fem_mesh)
        S = vertex_stretch(Vk, fV, fF, fmap, sf)
    return mesh, field, S


def sf_dict(regions):
    return {r["region_id"]: (r["sf_wale"], r["sf_course"]) for r in regions}


def build_cases():
    D5_FEM = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")
    C5_FEM = os.path.join(HERE, "data", "C5", "C5_remeshed_fem.off")
    e = {0: (1.01602, 1.00157), 1: (1.01602, 1.00157), 2: (1.05129, 1.00528)}
    cases = {
        "2part_A": ("7.4.1", "Middle crease, isotropic, no cable (A)", lambda: case_2part("A", (1.037220, 1.037220))),
        "2part_B": ("7.4.1", "Middle crease, isotropic, cable (B)", lambda: case_2part("B", (1.026573, 1.026573))),
        "2part_C": ("7.4.1", "Middle crease, anisotropic, no cable (C)", lambda: case_2part("C", (1.071567, 0.989013))),
        "2part_D": ("7.4.1", "Middle crease, anisotropic, cable (D)", lambda: case_2part("D", (1.041924, 1.012066))),
        "2part_E": ("7.4.1", "Middle crease, three regions, cable (E)", lambda: case_2part("E", e)),
    }
    g = json.load(open(os.path.join(OPT, "4part_g_pfix_result.json")))["slots"][0]
    cases["4part_G"] = ("7.4.2", "Crossing creases, global", lambda: case_fabsim(
        "4part", None, (g["sf_wale"], g["sf_course"])))

    def b5():
        V, F = read_off(os.path.join(ROOT, "data", "B5_remeshed_shared.off"))
        S = vertex_stretch(V, V, F, map_json("region_map_1p2m.json"),
                           sf_dict(regions_json("B5_1p2m_optimised_params.json")))
        return V, F, np.tile([1.0, 0.0, 0.0], (len(V), 1)), S
    cases["B5"] = ("7.4.3", "Free-form, 9 regions", b5)
    cases["C5"] = ("7.4.4", "Fluted dome", lambda: case_fabsim(
        "C5", C5_FEM, sf_dict(regions_json("C5_16region_pfix_optimised_sym.json")), map_json("C5_16region_map.json")))
    for key, title, res, rmap in [
            ("D5_1r", "Opening, 1 region", "d5_1r_pfix_optimised.json", "D5_region_map.json"),
            ("D5_4ra", "Opening, 4 adaptive regions", "d5_4ra_pfix_optimised.json", "d5_4ra_pfix_map.json"),
            ("D5_lap10", "Opening, 10 field-aligned regions", "d5_lap10_pfix_optimised.json", "d5_lap10_pfix_map.json"),
            ("D5_sym", "Opening, 10 symmetric, warm-started", "d5_sym_pfix_optimised.json", "d5_sym_pfix_region_map.json")]:
        cases[key] = ("7.4.5", title, (lambda res=res, rmap=rmap: case_fabsim(
            "D5", D5_FEM, sf_dict(regions_json(res)), map_json(rmap))))
    return cases


# ── run one ───────────────────────────────────────────────────────────────────
def run(key, section, title, make):
    out = os.path.join(OUT, key)
    os.makedirs(out, exist_ok=True)
    made = make()
    if len(made) == 4:                       # V, F, field, S: write the mesh
        V, F, field, S = made
        mesh = os.path.join(out, key + ".obj")
        write_obj(mesh, V, F)
    else:
        mesh, field, S = made
    name = os.path.splitext(os.path.basename(mesh))[0]
    fpath = os.path.join(out, key + "_vertex_field_stretch.txt")
    np.savetxt(fpath, np.c_[field, 1.0 / S if CONVENTION == "fem" else S], fmt="%.10g")
    print(f"\n== {key} ({section}) {title}: {len(S)} vertices, stretch wale "
          f"{S[:, 0].min():.4f}-{S[:, 0].max():.4f}, course {S[:, 1].min():.4f}-{S[:, 1].max():.4f}", flush=True)

    t0 = time.perf_counter()
    trajectories = generate_stripes(mesh, fpath, ST_H, ST_W, out_dir=out, check=False)
    t1 = time.perf_counter()
    stitches = read_trajectories(os.path.join(out, name + "_tri_path_recons.txt"))
    links = read_neighbours(os.path.join(out, name + "_neighbours.txt"))
    pattern = knitting_pattern(stitches, links, os.path.join(out, name + "_remesh.obj"),
                               os.path.join(out, name + "_remesh_vertex_directional_field.txt"))
    w, h = write_pattern(pattern["pixels"], os.path.join(out, key + "_pattern.bmp"),
                         os.path.join(out, key + "_pixel_data_dict.pkl"))
    t2 = time.perf_counter()
    pixels, report = knittable(pattern["pixels"])
    wk, hk = write_pattern(pixels, os.path.join(out, key + "_knittable.bmp"),
                           os.path.join(out, key + "_pixel_data_dict_knittable.pkl"))
    t3 = time.perf_counter()
    n_st = sum(len(s) for s in stitches)
    row = dict(case=key, section=section, title=title, convention=CONVENTION, trajectories=len(trajectories), stitches=n_st,
               pattern_w=w, pattern_h=h, knittable_w=wk, knittable_h=hk, fits_bed=bool(wk <= BED),
               t_trajectories_s=t1 - t0, t_pattern_s=t2 - t1, t_knittable_s=t3 - t2, t_total_s=t3 - t0,
               knittable_report=report,
               stretch_wale=[float(S[:, 0].min()), float(S[:, 0].max())],
               stretch_course=[float(S[:, 1].min()), float(S[:, 1].max())])
    print(f"   {len(trajectories)} trajectories, {n_st} stitches, pattern {w}x{h}, knittable {wk}x{hk}"
          f"{'' if wk <= BED else '  (WIDER THAN THE %d-NEEDLE BED)' % BED};  time: trajectories "
          f"{t1 - t0:.1f}s, pattern {t2 - t1:.1f}s, knittable {t3 - t2:.1f}s, total {t3 - t0:.1f}s", flush=True)
    return row


def main():
    cases = build_cases()
    keys = sys.argv[1:] or list(cases)
    path = os.path.join(OUT, "summary.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    rows = [r for r in rows if r["case"] not in keys]
    for k in keys:
        try:
            rows.append(run(k, *cases[k]))
        except Exception as e:                # keep going; report the failure
            print(f"   {k} FAILED: {e!r}", flush=True)
            rows.append(dict(case=k, section=cases[k][0], title=cases[k][1], error=repr(e)))
        os.makedirs(OUT, exist_ok=True)
        json.dump(rows, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
