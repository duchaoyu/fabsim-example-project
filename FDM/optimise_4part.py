"""
4-part (four-lobe) FEM inverse optimisation — knit + cable, D4-constrained.

Copied from FDM/optimise_C5_16region.py and adapted; the C5 script is NOT
modified.

Layout
------
The target surface is only APPROXIMATELY 4-fold symmetric (see fofin_4part.py:
rotating it by 90 deg moves the surface by up to ~31 mm, RMS ~5.5 mm), and its
triangulation is not symmetric at all.  D4 is therefore IMPOSED as a
parameter-sharing constraint, not detected from vertex correspondence:

  * the plan is cut into 4 quadrants (offset so the cuts fall between the
    lobes) and each quadrant into N_AZ azimuthal sub-wedges x N_RAD radial
    bands, giving N_WEDGE = N_AZ*N_RAD slots in the fundamental quarter;
  * there are 4*N_WEDGE physical regions, because knit_dir_deg is a GLOBAL
    in-plane angle and therefore differs by 90 deg between quadrants and cannot
    be shared;
  * but the sf parameters ARE shared: the design vector holds 2*N_WEDGE knit
    parameters, not 2*4*N_WEDGE.  Likewise the cables are grouped into D4
    orbits by azimuth and each orbit shares one rest-length scale.

knit_dir_deg is NOT an optimisation variable.  It is read, per face, from the
directional field in FDM/directional_field_4part.json and reduced to a
per-region circular mean (pi-periodic).

rest == target: the same OFF is passed as both.  There is no flat rest-shape
mesh in this pipeline, which is exactly why --min-disp-mm matters (see
_check_fem_valid).

Usage
-----
    .venv/bin/python FDM/optimise_4part.py [--n-az 2 --n-rad 3]
                                           [--maxiter 60] [--time-limit 3600]
"""
import argparse, csv, json, os, subprocess, sys, tempfile, time
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "4part")

MESH_PATH   = os.environ.get("FEM_MESH",   os.path.join(DATA, "4part_tri_m.off"))
TARGET_OFF  = os.environ.get("FEM_TARGET", os.path.join(DATA, "4part_tri_m.off"))
CABLE_PATHS_FILE = os.environ.get("FEM_CABLE_PATHS",
                                  os.path.join(DATA, "cable_paths_4part.json"))
FIELD_FILE  = os.path.join(DATA, "directional_field_4part.json")
BINARY      = os.environ.get("FEM_BINARY_NREGION",
                             os.path.join(HERE, "..", "build-linux", "fem_batch_nregion"))
OUT_DIR     = os.path.join(HERE, "optimisation")
REGION_MAP  = os.path.join(OUT_DIR, "4part_region_map.json")
CALL_LOG    = os.path.join(OUT_DIR, "4part_calls.jsonl")


# ── Mesh loader ───────────────────────────────────────────────────────────────
def load_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    V = np.array([[float(x) for x in lines[2 + i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in lines[2 + nv + i].split()[1:4]] for i in range(nf)])
    return V, F


def boundary_vertices(F):
    import collections
    cnt = collections.Counter()
    for f in F:
        for a, b in zip(f, np.roll(f, -1)):
            cnt[frozenset((int(a), int(b)))] += 1
    return {v for e, c in cnt.items() if c == 1 for v in e}


# ── Quadrant offset: put the cuts BETWEEN the lobes ──────────────────────────
def lobe_offset(V):
    """Azimuth (deg) of a lobe crest, from the 4-fold Fourier component of the
    azimuthal height profile.  Quadrant cuts are then placed at offset+45 deg."""
    az = np.degrees(np.arctan2(V[:, 1], V[:, 0])) % 360
    r  = np.hypot(V[:, 0], V[:, 1])
    m  = r > 0.15 * r.max()
    z  = V[:, 2]
    c  = np.sum(z[m] * np.cos(4 * np.deg2rad(az[m])))
    s  = np.sum(z[m] * np.sin(4 * np.deg2rad(az[m])))
    return float((np.degrees(np.arctan2(s, c)) / 4.0) % 90.0)


# ── D4 region map ─────────────────────────────────────────────────────────────
def build_region_map(V, F, n_az, n_rad, off, band_edges=None, crest_hw=None):
    """region id = slot*4 + quadrant, slot = sub_az*n_rad + band."""
    cen = V[F].mean(axis=1)
    r   = np.hypot(cen[:, 0], cen[:, 1])
    az  = (np.degrees(np.arctan2(cen[:, 1], cen[:, 0])) - off + 45.0) % 360.0
    quad = (az // 90.0).astype(int)
    loc  = az - quad * 90.0                    # 0..90 inside the quadrant
    sub  = np.minimum((loc / (90.0 / n_az)).astype(int), n_az - 1)
    if crest_hw is not None:
        # two mirror-symmetric sub-wedges: 0 = lobe crest (within crest_hw deg of
        # the crest at loc = 45), 1 = valley sectors either side of it
        assert n_az == 2, "--crest-hw needs --n-az 2"
        sub = (np.abs(loc - 45.0) > crest_hw).astype(int)
    # equal-area radial bands
    r_max = r.max()
    edges = r_max * np.sqrt(np.linspace(0, 1, n_rad + 1))[1:-1]
    if band_edges:
        assert len(band_edges) == n_rad - 1, "--band-edges needs n_rad-1 values"
        edges = r_max * np.asarray(sorted(band_edges))
    band  = np.searchsorted(edges, r)
    slot  = sub * n_rad + band
    return (slot * 4 + quad).astype(int).tolist(), slot.astype(int).tolist(), \
           quad.astype(int).tolist(), edges


def region_knit_dirs(face_region, n_regions, knit_face_deg):
    """Circular mean of the field's per-face wale angle, pi-periodic."""
    out = []
    kf = np.asarray(knit_face_deg)
    fr = np.asarray(face_region)
    for r in range(n_regions):
        a = np.deg2rad(kf[fr == r]) * 2.0        # doubled angle for pi-periodicity
        if len(a) == 0:
            out.append(0.0); continue
        out.append(float((np.degrees(np.arctan2(np.sin(a).sum(),
                                                np.cos(a).sum())) / 2.0) % 180.0))
    return out


# ── Cable D4 orbits ───────────────────────────────────────────────────────────
def cable_orbits(V, paths, tol=22.0):
    """Group cables into 90-deg orbits: two cables are in the same orbit if
    their (azimuth mod 90) agree within tol degrees and their radial extents
    match.  Each orbit shares one rest-length scale."""
    feats = []
    for p in paths:
        pts = V[p][:, :2]
        far = pts[np.argmax(np.hypot(pts[:, 0], pts[:, 1]))]
        feats.append((float(np.degrees(np.arctan2(far[1], far[0])) % 90.0),
                      float(np.hypot(*far))))
    orb, assigned = [], [-1] * len(paths)
    for i, (a, rr) in enumerate(feats):
        if assigned[i] >= 0:
            continue
        k = len(orb)
        members = [j for j in range(len(paths)) if assigned[j] < 0 and
                   min(abs(feats[j][0] - a), 90 - abs(feats[j][0] - a)) < tol and
                   abs(feats[j][1] - rr) < 0.15]
        for j in members:
            assigned[j] = k
        orb.append(members)
    return assigned, orb


# ── Hoop (ring) cable near the base ──────────────────────────────────────────
def ring_path(V, F, r_frac, n_way=16):
    """Closed vertex loop that follows r = r_frac * r_max as closely as the mesh
    allows: Dijkstra between n_way waypoints placed symmetrically in azimuth
    (so the loop is as x/y-mirror symmetric as the unsymmetric mesh permits),
    edge cost = length * (1 + 20 |r_mid - r0| / r_max).  First vertex repeated
    at the end, so the sliding cable closes on itself."""
    import heapq
    r = np.hypot(V[:, 0], V[:, 1]); rmax = r.max(); r0 = r_frac * rmax
    az = np.arctan2(V[:, 1], V[:, 0])
    adj = [dict() for _ in range(len(V))]
    for f in F:
        for a, b in zip(f, np.roll(f, -1)):
            a, b = int(a), int(b)
            rm = 0.5 * (r[a] + r[b])
            w = np.linalg.norm(V[a] - V[b]) * (1.0 + 20.0 * abs(rm - r0) / rmax)
            adj[a][b] = adj[b][a] = w
    way = []
    for k in range(n_way):
        t = 2 * np.pi * k / n_way
        cost = np.hypot(r * np.cos(az) - r0 * np.cos(t), r * np.sin(az) - r0 * np.sin(t))
        way.append(int(np.argmin(cost)))
    loop = [way[0]]
    for a, b in zip(way, way[1:] + way[:1]):
        dist = {a: 0.0}; prev = {}; pq = [(0.0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b: break
            if d > dist[u]: continue
            for v, w in adj[u].items():
                if d + w < dist.get(v, 1e18):
                    dist[v] = d + w; prev[v] = u; heapq.heappush(pq, (d + w, v))
        seg = [b]
        while seg[-1] != a: seg.append(prev[seg[-1]])
        loop += seg[::-1][1:]
    return loop


# ── FEM plumbing (from optimise_C5_16region.py) ───────────────────────────────
_call_count   = [0]
_out_prefix   = ["4part"]
_target_crown = [None]
_min_disp     = [0.0]
_t_start      = [None]
_time_limit   = [0.0]
_best         = [None]
_log_fh       = [None]
_call_dir     = [OUT_DIR]
_max_resid    = [1e-3]   # N, largest out-of-balance force accepted at the last stage


class _TimeUp(Exception):
    pass


def _track(loss, params):
    if _best[0] is None or loss < _best[0][0]:
        _best[0] = (float(loss), np.array(params, dtype=float).copy())
    if _time_limit[0] and _t_start[0] is not None:
        if time.time() - _t_start[0] > _time_limit[0]:
            raise _TimeUp()
    return loss


class _Result:
    def __init__(self, x, fun, nit, msg):
        self.x, self.fun, self.nit = np.asarray(x), float(fun), nit
        self.success, self.message = False, msg


def _run_minimize(fn, x0, **kw):
    _t_start[0] = time.time(); _best[0] = None
    try:
        res = minimize(fn, x0, **kw)
        if _best[0] is not None and _best[0][0] < res.fun - 1e-15:
            print(f"  note: scipy reported fun={res.fun:.7f} at its final x, but the "
                  f"best evaluation seen was {_best[0][0]:.7f}; reporting the best.")
            return _Result(_best[0][1], _best[0][0], getattr(res, "nit", -1),
                           "best evaluation (scipy res.fun disagreed)")
        return res
    except _TimeUp:
        el = time.time() - _t_start[0]
        print(f"\n  TIME LIMIT reached after {el:.1f} s - stopping with the best "
              f"point seen (loss {_best[0][0]:.7f}).")
        return _Result(_best[0][1], _best[0][0], -1,
                       f"stopped at time limit ({_time_limit[0]:.0f} s)")


def _check_fem_valid(verts, crown, V_rest):
    """Multi-check validity gate (verbatim rationale from optimise_C5_16region.py).

    The one that matters here is 2b: rest == target, so "barely deform" scores a
    near-perfect RMSE and the optimiser is drawn to a degenerate no-deformation
    optimum.  On C5 that produced a bogus 0.05 mm "fit".  KEEP the floor.
    """
    t_crown = _target_crown[0]
    if not np.all(np.isfinite(verts)):
        return False, "NaN/Inf in vertices"
    if V_rest is not None:
        max_disp = float(np.max(np.linalg.norm(verts - V_rest, axis=1)))
        if max_disp < 1e-8:
            return False, f"max_disp={max_disp:.2e} — rest shape returned"
        if _min_disp[0] > 0.0 and max_disp < _min_disp[0]:
            return False, (f"max_disp={max_disp*1000:.4f} mm < floor "
                           f"{_min_disp[0]*1000:.4f} mm — degenerate no-deformation fit")
    if t_crown is not None and (crown < 0.3 * t_crown or crown > 3.0 * t_crown):
        return False, f"crown={crown:.4f} outside physical range"
    min_z = float(verts[:, 2].min())
    if t_crown is not None and min_z < -0.01 * t_crown:
        return False, f"min z={min_z:.4f} — mesh folded"
    return True, "OK"


def run_fem(sf_wale, sf_course, knit_dirs, pressure, motif, region_map_path,
            cable_paths, cable_ea, cable_rest_scales, V_rest, n_regions,
            extra_log=None, cable_eas=None):
    """One FEM call.

    NOTE vs the C5 original: that version deleted its params temp file in a
    `finally`, so the per-call parameters were unrecoverable afterwards.  Here
    every call's parameters and outcome are appended to CALL_LOG (JSONL) before
    the temp file goes away.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    _call_count[0] += 1

    params = {
        "pressure":          float(pressure),
        "motif":             int(motif),
        "cable_ea":          float(cable_ea),
        "cable_paths":       cable_paths,
        "regions":           [{"sf_wale":      float(sf_wale[r]),
                               "sf_course":    float(sf_course[r]),
                               "knit_dir_deg": float(knit_dirs[r])}
                              for r in range(n_regions)],
        "cable_rest_scales": [float(s) for s in cable_rest_scales],
    }
    if cable_eas is not None:
        params["cable_eas"] = [float(e) for e in cable_eas]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                     delete=False, dir=OUT_DIR) as pf:
        json.dump(params, pf)
        params_path = pf.name

    prefix = os.path.join(_call_dir[0], f"{_out_prefix[0]}_{_call_count[0]:05d}")
    cmd    = [BINARY, MESH_PATH, region_map_path, params_path, prefix]

    rec = {"call": _call_count[0], "prefix": os.path.basename(prefix),
           "sf_wale": [float(x) for x in sf_wale],
           "sf_course": [float(x) for x in sf_course],
           "cable_rest_scales": [float(s) for s in cable_rest_scales],
           "knit_dir_deg": [float(d) for d in knit_dirs]}
    if extra_log:
        rec.update(extra_log)
    out = None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=float(os.environ.get("FEM_TIMEOUT", "120")))
        if res.returncode != 0:
            rec["status"] = f"rc={res.returncode}"
            print(f"  [{_call_count[0]:4d}] FEM error (rc={res.returncode}): {res.stderr[:200]}")
            return None
        # The binary exits 0 even when Newton fails and then writes (nearly) the
        # start state, i.e. the target itself, which scores a near-perfect fit.
        # Accept only a solve whose last load stage ends in equilibrium.
        resid = [float(l.split("max=")[1].split()[0]) for l in res.stderr.splitlines()
                 if l.startswith("SOLVER_RESIDUAL")]
        if resid and resid[-1] > _max_resid[0]:
            rec["status"] = f"not converged (final residual {resid[-1]:.3g})"
            print(f"  [{_call_count[0]:4d}] FEM NOT CONVERGED: final residual {resid[-1]:.3g}")
            return None
        scalars_path, verts_path = prefix + "_scalars.csv", prefix + "_verts.csv"
        if not os.path.exists(scalars_path):
            rec["status"] = "no scalars"
            print(f"  [{_call_count[0]:4d}] FEM error: no scalars output")
            return None
        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        out = {k: float(v) for k, v in row.items()}
        rec["scalars"] = {k: v for k, v in out.items()}
        if os.path.exists(verts_path):
            out["verts"] = np.loadtxt(verts_path, delimiter=",", skiprows=1)[:, 1:]
            rec["max_disp_mm"] = float(np.max(np.linalg.norm(
                out["verts"] - V_rest, axis=1)) * 1000.0) if V_rest is not None else None
            ok, reason = _check_fem_valid(out["verts"], out.get("crown_height", 0.0), V_rest)
            rec["status"] = "OK" if ok else reason
            if not ok:
                print(f"  [{_call_count[0]:4d}] FEM INVALID: {reason}")
                out = None
                return None
        else:
            rec["status"] = "no verts"
        return out
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
        print(f"  [{_call_count[0]:4d}] FEM TIMEOUT (no convergence)")
        return None
    except Exception as e:
        rec["status"] = f"exception: {e}"
        print(f"  [{_call_count[0]:4d}] FEM exception: {e}")
        return None
    finally:
        if _log_fh[0] is not None:
            _log_fh[0].write(json.dumps(rec) + "\n"); _log_fh[0].flush()
        try:
            os.unlink(params_path)
        except OSError:
            pass


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-az", type=int, default=2)
    ap.add_argument("--n-rad", type=int, default=3)
    ap.add_argument("--pressure", type=float, default=1000.0)
    ap.add_argument("--motif", type=int, default=1)
    ap.add_argument("--cable-ea", type=float, default=157000.0)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--min-disp-mm", type=float, default=5.0)
    ap.add_argument("--sf0-wale", type=float, default=1.05)
    ap.add_argument("--sf0-course", type=float, default=1.05)
    ap.add_argument("--scale0", type=float, default=0.97)
    ap.add_argument("--no-cables", action="store_true")
    ap.add_argument("--tag", type=str, default="4part")
    ap.add_argument("--p0-json", type=str, default=None,
                    help="warm start from the 'p' vector of a previous result JSON")
    ap.add_argument("--eps", type=float, default=2e-3,
                    help="L-BFGS-B finite-difference step")
    ap.add_argument("--sf-lo", type=float, default=1.001)
    ap.add_argument("--sf-hi", type=float, default=1.60)
    ap.add_argument("--face-knit", action="store_true",
                    help="write the field's PER-FACE knit angle into the region map "
                         "as face_knit_dirs_deg (fem_batch_nregion honours it and "
                         "then ignores the region-level knit_dir_deg); off by "
                         "default so the region map keeps the C5 format.")
    ap.add_argument("--band-edges", type=float, nargs="+", default=None,
                    help="radial band edges as fractions of r_max (n_rad-1 values); "
                         "default equal-area bands")
    ap.add_argument("--sym", choices=["d4", "d2"], default="d4",
                    help="d4: all four quadrants share a slot's parameters; d2: "
                         "mirror symmetry in x and in y only, so the lobes on the "
                         "x axis and those on the y axis get separate parameters")
    ap.add_argument("--ring", type=float, nargs="*", default=[],
                    help="add closed hoop cables at these r/r_max (each its own "
                         "rest-scale parameter)")
    ap.add_argument("--ring-ea", type=float, default=39250.0,
                    help="axial stiffness of the hoop cables (N); 39250 = 1 mm steel, "
                         "a quarter of the 2 mm main cables' 157000")
    ap.add_argument("--crest-hw", type=float, default=None,
                    help="with --n-az 2: split each quadrant into a lobe-crest sector "
                         "of this half-width (deg) and the valley sectors around it")
    ap.add_argument("--method", choices=["L-BFGS-B", "Powell"], default="L-BFGS-B",
                    help="Powell is derivative-free and copes with the solver noise "
                         "that stalls the finite-difference line search")
    ap.add_argument("--update-data", action="store_true",
                    help="also overwrite data/4part/4part_fem_best.{obj,npy}")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    _out_prefix[0] = args.tag
    _time_limit[0] = args.time_limit
    _min_disp[0]   = args.min_disp_mm / 1000.0
    _log_fh[0]     = open(CALL_LOG, "a")
    _call_dir[0]   = os.path.join(OUT_DIR, "4part_calls", args.tag)
    os.makedirs(_call_dir[0], exist_ok=True)
    region_map_file = os.path.join(OUT_DIR, f"{args.tag}_region_map.json")

    V, F = load_off(MESH_PATH)
    V_target, _ = load_off(TARGET_OFF)
    V_rest = V.copy()
    bd = boundary_vertices(F)
    interior_idx = np.array([i for i in range(len(V)) if i not in bd])
    _span = float(max(np.ptp(V[:, 0]), np.ptp(V[:, 1])))
    _target_crown[0] = float(V_target[:, 2].max())

    print(f"Mesh     : {MESH_PATH}")
    print(f"           {len(V)}v / {len(F)}f, span {_span:.4f} m, crown {_target_crown[0]:.4f} m")
    print(f"           {len(bd)} boundary / {len(interior_idx)} interior vertices")
    print("           rest == target (same OFF); no flat rest-shape mesh exists here.")
    print(f"  displacement floor: {args.min_disp_mm:.4f} mm "
          f"({args.min_disp_mm / (_span*1000.0)*100:.3f}% of span)"
          if _min_disp[0] > 0 else "  displacement floor: DISABLED")

    off = lobe_offset(V)
    face_region, slot, quad, r_edges = build_region_map(V, F, args.n_az, args.n_rad, off,
                                                        args.band_edges, args.crest_hw)
    n_wedge   = args.n_az * args.n_rad
    n_regions = 4 * n_wedge
    n_group   = 2 if args.sym == "d2" else 1     # parameter copies per slot
    n_par_kn  = n_wedge * n_group
    field = json.load(open(FIELD_FILE))
    rmap = {"face_regions": face_region}
    if args.face_knit:
        rmap["face_knit_dirs_deg"] = field["knit_dir_deg_face"]
    with open(region_map_file, "w") as f:
        json.dump(rmap, f)
    import collections
    cnt = collections.Counter(face_region)
    print(f"\nD4 layout: lobe crest azimuth {off:.1f} deg -> quadrant cuts at "
          f"{(off+45)%90:.1f} + k*90 deg")
    print(f"  {args.n_az} azimuthal sub-wedges x {args.n_rad} radial bands "
          f"(band edges r = {np.round(r_edges,4).tolist()} m)")
    print(f"  {n_wedge} slots in the fundamental quarter -> {n_regions} physical "
          f"regions, {2*n_par_kn} sf parameters (NOT {2*n_regions})")
    print(f"  faces per region: min {min(cnt.values())}, max {max(cnt.values())}")
    print(f"  region map -> {region_map_file}")

    knit_dirs = region_knit_dirs(face_region, n_regions, field["knit_dir_deg_face"])
    print(f"\nknit_dir_deg per region (FIXED from {os.path.basename(FIELD_FILE)}, "
          f"never optimised):")
    for s in range(n_wedge):
        print(f"  slot {s}: " + "  ".join(f"q{k}={knit_dirs[s*4+k]:6.1f}" for k in range(4)))

    cable_paths = list(json.load(open(CABLE_PATHS_FILE)).values()) \
        if (os.path.exists(CABLE_PATHS_FILE) and not args.no_cables) else []
    assigned, orbits = cable_orbits(V, cable_paths) if cable_paths else ([], [])
    cable_eas = [args.cable_ea] * len(cable_paths)
    for rf in args.ring:
        loop = ring_path(V, F, rf)
        rr = np.hypot(V[loop, 0], V[loop, 1]) / np.hypot(V[:, 0], V[:, 1]).max()
        print(f"Ring     : r/r_max {rf:.3f} -> {len(loop)-1} vertices, r/r_max "
              f"{rr.min():.3f}-{rr.max():.3f}, z {V[loop,2].min():.3f}-{V[loop,2].max():.3f} m, "
              f"EA {args.ring_ea:.0f} N")
        assigned.append(len(orbits)); orbits.append([len(cable_paths)])
        cable_paths.append(loop); cable_eas.append(args.ring_ea)
    n_orb = len(orbits)
    print(f"\nCables   : {len(cable_paths)} sections, lens {[len(p) for p in cable_paths]}")
    print(f"           {n_orb} D4 orbits (shared rest-scale): "
          f"{[len(o) for o in orbits]}")

    # design vector: [sf_wale(n_wedge), sf_course(n_wedge), scale(n_orb)]
    # d2: quadrants 0/2 (lobes on one axis) share, 1/3 share.  Within a quadrant
    # the sub-wedges are only mirror images of each other for n_az = 1, which is
    # what the d2 runs use.
    def expand(p):
        sw = np.empty(n_regions); sc = np.empty(n_regions)
        for s in range(n_wedge):
            for k in range(4):
                g = s * n_group + (k % n_group)
                sw[s*4+k] = p[g]; sc[s*4+k] = p[n_par_kn + g]
        scales = np.ones(len(cable_paths))
        for i, k in enumerate(assigned):
            scales[i] = p[2*n_par_kn + k]
        return sw, sc, scales

    p0 = np.r_[np.full(n_par_kn, args.sf0_wale), np.full(n_par_kn, args.sf0_course),
               np.full(n_orb, args.scale0)]
    if args.p0_json:
        prev = json.load(open(args.p0_json))["p"]
        if len(prev) == len(p0):
            p0 = np.array(prev, dtype=float)
            print(f"warm start from {args.p0_json}")
        else:
            print(f"WARNING: {args.p0_json} has {len(prev)} params, need {len(p0)} "
                  f"— ignoring the warm start")
    bounds = [(args.sf_lo, args.sf_hi)] * (2*n_par_kn) + [(0.75, 1.05)] * n_orb
    p0 = np.clip(p0, [b[0] for b in bounds], [b[1] for b in bounds])

    history = []

    def objective(p):
        sw, sc, scales = expand(p)
        out = run_fem(sw, sc, knit_dirs, args.pressure, args.motif, region_map_file,
                      cable_paths, args.cable_ea, scales, V_rest, n_regions,
                      extra_log={"p": [float(x) for x in p], "tag": args.tag},
                      cable_eas=cable_eas)
        if out is None or "verts" not in out:
            loss = 1e3
        else:
            d = out["verts"][interior_idx] - V_target[interior_idx]
            loss = float(np.sqrt(np.mean(np.sum(d**2, axis=1))))
        history.append(loss)
        if len(history) % 10 == 0:
            print(f"  [{_call_count[0]:4d}]  RMSE={loss*1000:.2f} mm")
        return _track(loss, p)

    print(f"\nBaseline at p0 (sf_wale={args.sf0_wale}, sf_course={args.sf0_course}, "
          f"scale={args.scale0}) ...")
    l0 = objective(p0)
    print(f"  baseline interior RMSE = {l0*1000:.3f} mm")

    print(f"\nOptimising {len(p0)} parameters ({args.method}, maxiter {args.maxiter}, "
          f"time limit {args.time_limit:.0f} s) ...")
    if args.method == "Powell":
        opts = {"maxiter": args.maxiter, "xtol": 1e-4, "ftol": 1e-6}
    else:
        opts = {"maxiter": args.maxiter, "eps": args.eps, "ftol": 1e-14}
    res = _run_minimize(objective, p0, method=args.method, bounds=bounds, options=opts)
    print(f"\n{res.message}   calls={_call_count[0]}")

    p_best = np.asarray(res.x)
    sw, sc, scales = expand(p_best)
    out = run_fem(sw, sc, knit_dirs, args.pressure, args.motif, region_map_file,
                  cable_paths, args.cable_ea, scales, V_rest, n_regions,
                  extra_log={"final": True, "tag": args.tag}, cable_eas=cable_eas)
    if out is None:
        print("FINAL RE-RUN INVALID — reporting the tracked best loss only.")
        rmse, maxdev, crown, maxdisp, verts = float(res.fun), None, None, None, None
    else:
        verts = out["verts"]
        d = np.linalg.norm(verts[interior_idx] - V_target[interior_idx], axis=1)
        rmse   = float(np.sqrt(np.mean(d**2)))
        maxdev = float(d.max())
        crown  = float(out.get("crown_height", verts[:, 2].max()))
        maxdisp = float(np.max(np.linalg.norm(verts - V_rest, axis=1)))
        print(f"\nBEST: interior RMSE {rmse*1000:.3f} mm, max deviation {maxdev*1000:.2f} mm, "
              f"crown {crown:.4f} m (target {_target_crown[0]:.4f} m)")
        print(f"      max displacement from rest {maxdisp*1000:.3f} mm  "
              f"(floor {args.min_disp_mm:.3f} mm)")
        if maxdisp < 1.25 * _min_disp[0]:
            print("      *** FLOOR-LIMITED: the best point sits against the "
                  "--min-disp-mm floor, i.e. the optimiser is still being pulled "
                  "toward the degenerate no-deformation optimum.  Treat this RMSE "
                  "as not physically meaningful. ***")

    resj = {
        "geometry": "4part", "tag": args.tag,
        "mesh": MESH_PATH, "target": TARGET_OFF, "rest_equals_target": True,
        "n_az": args.n_az, "n_rad": args.n_rad, "n_wedge_slots": n_wedge,
        "n_regions": n_regions, "n_params": int(len(p0)),
        "lobe_offset_deg": off, "radial_band_edges": [float(x) for x in r_edges],
        "pressure": args.pressure, "motif": args.motif, "cable_ea": args.cable_ea,
        "min_disp_mm": args.min_disp_mm,
        "baseline_rmse_mm": l0 * 1000.0,
        "rmse_mm": rmse * 1000.0,
        "max_dev_mm": None if maxdev is None else maxdev * 1000.0,
        "crown": crown, "target_crown": _target_crown[0],
        "max_disp_mm": None if maxdisp is None else maxdisp * 1000.0,
        "floor_limited": bool(maxdisp is not None and maxdisp < 1.25 * _min_disp[0]),
        "p": [float(x) for x in p_best],
        "sym": args.sym, "crest_hw": args.crest_hw, "method": args.method, "band_edges_frac": args.band_edges, "face_knit": args.face_knit,
        "rings_r_frac": args.ring, "ring_ea": args.ring_ea if args.ring else None,
        "cable_eas": [float(e) for e in cable_eas],
        "cable_paths": [[int(v) for v in c] for c in cable_paths],
        "slots": [{"slot": s, "group": g, "sf_wale": float(p_best[s*n_group + g]),
                   "sf_course": float(p_best[n_par_kn + s*n_group + g])}
                  for s in range(n_wedge) for g in range(n_group)],
        "regions": [{"region_id": r, "slot": r // 4, "quadrant": r % 4,
                     "sf_wale": float(sw[r]), "sf_course": float(sc[r]),
                     "knit_dir_deg": float(knit_dirs[r])} for r in range(n_regions)],
        "cable_orbits": [[int(i) for i in o] for o in orbits],
        "cable_rest_scales": [float(s) for s in scales],
        "calls": _call_count[0], "history": history,
        "message": str(res.message),
    }
    rj = os.path.join(OUT_DIR, f"{args.tag}_result.json")
    json.dump(resj, open(rj, "w"), indent=1)
    print(f"Saved: {rj}")
    print(f"Saved: {CALL_LOG}  (per-call parameters — the C5 script lost these)")

    if verts is not None:
        obj = os.path.join(OUT_DIR, f"{args.tag}_fem_best.obj")
        with open(obj, "w") as f:
            f.write(f"# 4part FEM knit+cable best fit ({args.tag})\n")
            f.write(f"# interior RMSE  : {rmse*1000:.3f} mm\n")
            f.write(f"# max deviation  : {maxdev*1000:.3f} mm\n")
            f.write(f"# crown          : {crown:.5f} m (target {_target_crown[0]:.5f} m)\n")
            f.write(f"# max disp/rest  : {maxdisp*1000:.3f} mm (floor {args.min_disp_mm} mm)\n")
            f.write(f"# regions        : {n_regions} ({n_wedge} D4 slots), "
                    f"{len(p0)} parameters\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for fc in F:
                f.write(f"f {fc[0]+1} {fc[1]+1} {fc[2]+1}\n")
        print(f"Saved: {obj}")
        if args.update_data:
            import shutil
            shutil.copy(obj, os.path.join(DATA, "4part_fem_best.obj"))
            np.save(os.path.join(DATA, "4part_fem_best_verts.npy"), verts)

    _log_fh[0].close()


if __name__ == "__main__":
    main()
