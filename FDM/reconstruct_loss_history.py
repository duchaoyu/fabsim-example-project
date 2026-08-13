"""Rebuild the loss trajectory of a finished optimisation run from its output files.

The drivers stored only the last thirty evaluations of each run, but every
evaluation wrote <prefix>_NNNNN_verts.csv. Recomputing the objective from those
files recovers the full trajectory, and pairing it with the file mtimes puts the
loss on a wall-clock axis — which is what Section 6.5.1 needs and what an
iteration axis cannot show.

The objective is reproduced exactly as the D5 drivers define it
(optimise_D5_4region_adaptive.py:269, optimise_D5_laplacian.py:268): RMS distance
over the interior vertices, interior being everything inside 0.98 of the maximum
boundary radius.

The driver's validity gate (optimise_D5_4region_adaptive.py:183) has to be
reproduced too, and it is not optional bookkeeping. The forward solver writes its
output file before the driver inspects it, so the run directory contains
evaluations the driver rejected. Among them are evaluations that returned the rest
shape unchanged, and because the D5 rest mesh *is* the target, those score an
exact zero on the objective. Scoring the files without the gate therefore invents
a perfect optimum where the driver correctly saw a failed solve.

    python3 reconstruct_loss_history.py --out data/D5/loss_history.json
"""

import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIM_DIR = os.path.join(HERE, "optimisation")
TARGET = os.path.join(HERE, "data", "D5", "D5_remeshed_fem.off")

RUNS = [
    ("d5_cable2_opt", "1 region"),
    ("d5_4ra_v3",     "4 adaptive regions"),
    ("d5_10lap_v4",   "10 field-aligned regions"),
    ("d5_10lap_v3",   "10 field-aligned regions (short)"),
    ("d5_sym_v1",     "10 symmetric regions"),
]


def load_off(path):
    with open(path) as f:
        assert f.readline().strip().startswith("OFF")
        nv, nf, _ = map(int, f.readline().split())
        V = np.array([[float(x) for x in f.readline().split()[:3]]
                      for _ in range(nv)])
    return V


def interior_index(V):
    r = np.hypot(V[:, 0], V[:, 1])
    return np.where(~(r > r.max() * 0.98))[0]


PENALTY = 1e3   # what the driver's objective returns for a rejected evaluation


def check_valid(verts, V_rest, target_crown):
    """Verbatim port of _check_valid in the D5 drivers."""
    if not np.all(np.isfinite(verts)):
        return False, "NaN/Inf"
    if float(np.max(np.linalg.norm(verts - V_rest, axis=1))) < 1e-8:
        return False, "rest shape returned"
    crown = float(verts[:, 2].max())
    if crown < 0.3 * target_crown or crown > 3.0 * target_crown:
        return False, "crown out of range"
    if float(verts[:, 2].min()) < -0.01 * target_crown:
        return False, "mesh folded"
    return True, "OK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "D5",
                                                  "loss_history.json"))
    args = ap.parse_args()

    V_target = load_off(TARGET)
    idx = interior_index(V_target)
    tgt = V_target[idx]
    print(f"target: {len(V_target)} verts, {len(idx)} interior")

    out = {}
    for prefix, label in RUNS:
        pat = re.compile(rf"{re.escape(prefix)}_(\d{{5}})_verts\.csv$")
        found = {}
        for f in os.listdir(OPTIM_DIR):
            m = pat.match(f)
            if m:
                found[int(m.group(1))] = os.path.join(OPTIM_DIR, f)
        if not found:
            print(f"  {prefix}: no output files, skipped")
            continue

        calls, losses, stamps, valid = [], [], [], []
        rejected = {}
        for call in sorted(found):
            path = found[call]
            V = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3))
            if V.shape[0] != len(V_target):
                print(f"  {prefix}: vertex count {V.shape[0]} != target "
                      f"{len(V_target)}, skipped")
                calls = []
                break
            ok, why = check_valid(V, V_target, float(V_target[:, 2].max()))
            if ok:
                d = V[idx] - tgt
                losses.append(float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))))
            else:
                losses.append(PENALTY)
                rejected[why] = rejected.get(why, 0) + 1
            valid.append(ok)
            calls.append(call)
            stamps.append(os.path.getmtime(path))
        if not calls:
            continue

        t = np.array(stamps)
        t = t - t[0]
        best = np.minimum.accumulate(losses)
        if rejected:
            print(f"  {prefix:14s} rejected by the validity gate: "
                  + ", ".join(f"{n}x {w}" for w, n in rejected.items()))
        out[prefix] = dict(label=label, calls=calls, loss=losses,
                           elapsed=t.tolist(), best=best.tolist(),
                           n_evaluations=int(max(found)))
        # Where the improvement actually arrived.
        span = best[0] - best[-1]
        marks = {}
        for frac in (0.5, 0.9, 0.95, 0.99):
            if span > 0:
                k = int(np.argmax(best <= best[0] - frac * span))
                marks[frac] = dict(call=calls[k], elapsed=float(t[k]),
                                   loss_mm=1e3 * best[k])
        out[prefix]["milestones"] = marks
        print(f"  {prefix:14s} {len(calls):5d} solves  "
              f"{best[0]*1e3:6.2f} -> {best[-1]*1e3:6.2f} mm  "
              f"wall {t[-1]/60:6.1f} min")
        for frac, m in marks.items():
            print(f"      {100*frac:4.0f}% of the improvement by solve "
                  f"{m['call']:5d} ({100*m['elapsed']/t[-1]:5.1f}% of wall clock)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\n  saved: {args.out}")


if __name__ == "__main__":
    main()
