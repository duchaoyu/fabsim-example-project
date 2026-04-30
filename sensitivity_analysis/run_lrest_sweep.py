"""
Run FEM simulations for the cable rest-length sweep and save results.

Fixed: sf_wale=sf_course=1.0, knit_dir=0, pressure=800 Pa, motif=1 and 2
Cable: x=0 cable (cable_indices.txt), mesh=circular_flat_cable.off
L_rest values (m): no_cable, 0.95, 1.00, 1.10, 1.20, 1.30, 1.40

Outputs to:  sensitivity_analysis/data/lrest_sweep/
"""
import json, os, subprocess, csv, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CABLE_EA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEM_BIN      = os.path.join(REPO, "build-mac", "fem_batch_sensitivity")
CABLE_IDX    = np.loadtxt(os.path.join(REPO, "data", "cable_indices.txt"), dtype=int)
OUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lrest_sweep")

# Use the original (non-snapped) mesh — no degenerate triangles
from config import MESH_PATH
CABLE_MESH = MESH_PATH

os.makedirs(OUT_DIR, exist_ok=True)

SF          = 1.0
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
        result = {k: float(v) for k, v in row.items()}
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

    cmd = [FEM_BIN, CABLE_MESH,
           f"{sf:.4f}", f"{sf:.4f}", f"{knit_dir:.2f}", f"{pressure:.1f}",
           str(motif), cable_arg, prefix]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  FAILED {tag}: {r.stderr[:200]}")
        return None

    with open(scalars_path) as f:
        row = next(csv.DictReader(f))
    result = {k: float(v) for k, v in row.items()}
    result["prefix"] = prefix
    return result


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
