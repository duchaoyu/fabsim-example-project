"""
One-shot scaling: shrink all B5 geometry to a target diameter (default 1.2 m).

Scales:
  - data/B5_remeshed_shared.off, B5_remeshed_flat.off
  - data/B5_shared.off, B5_flat.off
  - data/B5_remeshed_target_verts.npy, B5_target_verts.npy
  - FDM/data/mesh_out_B5_20260501213116.json (FDM result vertex coords)

Cable-path JSON (vertex indices) and interior-idx .npy are unchanged.

Run once after each rescale; re-running with the same target is a no-op
(within 1 mm tolerance).
"""
import os, json, shutil
import numpy as np

TARGET_DIAMETER = 1.2  # m

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA     = os.path.join(ROOT, "data")
FDM_DATA = os.path.join(ROOT, "FDM", "data")

OFF_FILES = [
    os.path.join(DATA, "B5_remeshed_shared.off"),
    os.path.join(DATA, "B5_remeshed_flat.off"),
    os.path.join(DATA, "B5_shared.off"),
    os.path.join(DATA, "B5_flat.off"),
]
NPY_FILES = [
    os.path.join(DATA, "B5_remeshed_target_verts.npy"),
    os.path.join(DATA, "B5_target_verts.npy"),
]
JSON_FDM_RESULT = os.path.join(FDM_DATA, "mesh_out_B5_20260501213116.json")


def diameter_off(path):
    with open(path) as f:
        lines = f.readlines()
    nv = int(lines[1].split()[0])
    V = np.array([[float(x) for x in l.split()] for l in lines[2:2+nv]])
    return max(V[:,0].max() - V[:,0].min(), V[:,1].max() - V[:,1].min())


def scale_off(path, s):
    with open(path) as f:
        lines = f.readlines()
    nv, nf = int(lines[1].split()[0]), int(lines[1].split()[1])
    out = lines[:2]
    for i in range(2, 2+nv):
        x, y, z = (float(t) for t in lines[i].split())
        out.append(f"{x*s:.8f} {y*s:.8f} {z*s:.8f}\n")
    out.extend(lines[2+nv:2+nv+nf])
    with open(path, "w") as f:
        f.writelines(out)


def scale_npy(path, s):
    V = np.load(path)
    np.save(path, V * s)


def scale_fdm_json(path, s):
    with open(path) as f:
        d = json.load(f)
    for vkey, vdata in d["vertex"].items():
        vdata["x"] *= s
        vdata["y"] *= s
        vdata["z"] *= s
    # qpre is force-density (N/m). Scaling lengths by s scales lengths by s,
    # so to keep edge forces (qpre * length) unchanged we'd need qpre *= 1/s,
    # but for visualisation we only need coords. Leave qpre alone — relative
    # values still drive cable-path Dijkstra correctly.
    with open(path, "w") as f:
        json.dump(d, f)


def main():
    # Use the canonical FEM mesh diameter to derive the scale factor.
    ref = OFF_FILES[0]
    cur = diameter_off(ref)
    s   = TARGET_DIAMETER / cur
    print(f"Reference: {os.path.basename(ref)}  diameter={cur:.4f} m")
    print(f"Target:    {TARGET_DIAMETER:.4f} m   →  scale = {s:.6f}")

    if abs(cur - TARGET_DIAMETER) < 1e-3:
        print("Already at target scale — nothing to do.")
        return

    for p in OFF_FILES:
        if os.path.exists(p):
            d_before = diameter_off(p)
            scale_off(p, s)
            d_after = diameter_off(p)
            print(f"  {os.path.basename(p):35s}  {d_before:.4f} → {d_after:.4f} m")

    for p in NPY_FILES:
        if os.path.exists(p):
            V = np.load(p)
            d_before = max(V[:,0].max()-V[:,0].min(), V[:,1].max()-V[:,1].min())
            scale_npy(p, s)
            V2 = np.load(p)
            d_after = max(V2[:,0].max()-V2[:,0].min(), V2[:,1].max()-V2[:,1].min())
            print(f"  {os.path.basename(p):35s}  {d_before:.4f} → {d_after:.4f} m")

    if os.path.exists(JSON_FDM_RESULT):
        with open(JSON_FDM_RESULT) as f:
            d = json.load(f)
        xs = [v["x"] for v in d["vertex"].values()]
        ys = [v["y"] for v in d["vertex"].values()]
        d_before = max(max(xs)-min(xs), max(ys)-min(ys))
        scale_fdm_json(JSON_FDM_RESULT, s)
        with open(JSON_FDM_RESULT) as f:
            d2 = json.load(f)
        xs = [v["x"] for v in d2["vertex"].values()]
        ys = [v["y"] for v in d2["vertex"].values()]
        d_after = max(max(xs)-min(xs), max(ys)-min(ys))
        print(f"  {os.path.basename(JSON_FDM_RESULT):35s}  {d_before:.4f} → {d_after:.4f} m")

    print("Done.")


if __name__ == "__main__":
    main()
