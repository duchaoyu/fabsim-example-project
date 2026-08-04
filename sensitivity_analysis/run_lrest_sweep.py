"""
Run FEM simulations for the cable rest-length sweep and save results (figQ).

Fixed: sf_wale = sf_course = 1.1, knit_dir = 0 deg, pressure = 1000 Pa,
       motifs 1 and 2, mesh = config.MESH_PATH.
Cable: the x=0 diametral cable from data/cable_indices.txt, passed in the WALE
       slot (it runs along y, which is the wale direction at knit_dir = 0).
L_rest values (m): no cable, 1.20 ... 1.50 in 0.05 steps.

Outputs to:  sensitivity_analysis/data/lrest_sweep/

sf was 1.0, which run_e1r_grid.py identifies as an unstable flat-membrane
bifurcation point and every other study in this project avoids by using 1.1.
figQ sat exactly on it, and the softer motif 2 fell off the branch in the taut
regime: re-running L_rest = 1.20 m gave T = 4870 N and a 170.5 mm crown against
the 4139 N / 140.6 mm that had been stored, and crown height ran non-monotonically
170.5 -> 128.3 -> 179.4 mm across 1.20/1.21/1.22 m.  At sf = 1.1 both motifs are
monotonic in L_rest and the compressive-face fraction roughly halves.

Three other things here were stale and would have stopped this script reproducing
its own output: the binary path pointed at a macOS build directory, the binary now
takes two cable slots (wale and course) rather than one, and it reports
cable_wale_tension / cable_course_tension rather than cable_tension.
"""
import json, os, subprocess, csv, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CABLE_EA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from config import FEM_BINARY
FEM_BIN      = FEM_BINARY
CABLE_IDX    = np.loadtxt(os.path.join(REPO, "data", "cable_indices.txt"), dtype=int)
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lrest_sweep")

# Use the original (non-snapped) mesh — no degenerate triangles
from config import MESH_PATH
CABLE_MESH = MESH_PATH

os.makedirs(OUT_DIR, exist_ok=True)

SF          = 1.1   # NOT 1.0 — see module docstring
KNIT_DIR    = 0.0
PRESSURE    = 1000.0
MOTIFS      = [1, 2]
# L_flat for this cable path ≈ 1.325 m; sweep from strongly pre-stressed to slack
L_REST_VALS = [None, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]  # None = no cable


def _run(sf, knit_dir, pressure, motif, l_rest_m):
    """Run one simulation; return dict of scalar outputs or None on failure."""
    tag = f"m{motif}_lr{'none' if l_rest_m is None else f'{l_rest_m:.2f}'}"
    prefix = os.path.join(OUT_DIR, tag)

    scalars_path = prefix + "_scalars.csv"
    verts_path   = prefix + "_verts.csv"
    if os.path.exists(scalars_path) and os.path.exists(verts_path):
        with open(scalars_path) as f:
            row = next(csv.DictReader(f))
        result = _normalise(row)
        result["prefix"] = prefix
        return result

    if l_rest_m is None:
        cable_arg = "none"
    else:
        cable_json = {"indices": [int(i) for i in CABLE_IDX],
                      "EA": float(CABLE_EA),
                      "L_rest": float(l_rest_m)}
        jpath = prefix + "_cable.json"
        with open(jpath, "w") as f:
            json.dump(cable_json, f)
        cable_arg = jpath

    # The binary takes a wale cable slot and a course cable slot; this study has
    # only the one diametral cable, which lies along y = wale at knit_dir = 0.
    cmd = [FEM_BIN, CABLE_MESH,
           f"{sf:.4f}", f"{sf:.4f}", f"{knit_dir:.2f}", f"{pressure:.1f}",
           str(motif), cable_arg, "none", prefix]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  FAILED {tag}: {r.stderr[:200]}")
        return None

    with open(scalars_path) as f:
        row = next(csv.DictReader(f))
    result = _normalise(row)
    result["prefix"] = prefix
    return result


def _normalise(row):
    """Scalar row as floats, with a single cable_tension column.

    The binary reports cable_wale_tension and cable_course_tension; only the wale
    slot is used here, and everything downstream (plot_lrest_sweep, the stored
    sweep_results.csv) reads cable_tension.
    """
    out = {k: float(v) for k, v in row.items()}
    if "cable_tension" not in out:
        out["cable_tension"] = out.get("cable_wale_tension", 0.0)
    return out


if __name__ == "__main__":
    rows = []
    for motif in MOTIFS:
        for l in L_REST_VALS:
            label = "no_cable" if l is None else f"{l:.2f}m"
            print(f"  motif={motif}  L_rest={label} ...", end=" ", flush=True)
            res = _run(SF, KNIT_DIR, PRESSURE, motif, l)
            if res:
                print(f"h={res['crown_height']*1000:.1f}mm  "
                      f"T={res['cable_tension']:.1f}N")
                rows.append({"motif": motif, "L_rest_m": l,
                             "label": label, **res})
            else:
                print("FAILED")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR, "sweep_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} rows → {out}")
