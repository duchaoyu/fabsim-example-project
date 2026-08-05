"""
Probe: the cable rest length as a fraction of the cable-free section length.

An absolute rest length in metres cannot be taut everywhere.  L_nocable — the
arc length of the cable path on the cable-free equilibrium, which is exactly the
slack threshold — runs from 1.2902 m on a barely-inflated dome to 1.3643 m on the
tallest.  A fixed 1.2-1.4 m range therefore left 40% of runs slack, and slack
runs are not a response: they are the no-cable model with an inert cable.  Under
L_rest = f * L_nocable, f < 1 is taut by construction, and f is a proper box so
Sobol sampling stays valid.

This sweeps f at the validity-box centre and reports the shape response, with a
no-cable reference.  It sweeps two cable areas at once, because config.CABLE_EA
= 150 kN is A = 0.75 mm2 at E = 200 GPa, at which the tensions the study already
reports (1000-3000 N) correspond to 1300-4000 MPa — past any steel.  The
comparison shows what the range costs in cable stress either way.

Usage:
    python3 probe_cable_influence.py [--n 7] [--jobs 8]
Products:
    data/probe_cable_frac.csv
    figures/figP_cable_frac_probe.{pdf,png}
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import DATA_DIR, MESH_PATH
from cable_path import (generate_cable_path, cable_path_length, load_off,
                        WALE_CABLE_ANGLE, COURSE_CABLE_ANGLE)
from fea_interface import run_fea
from plot_material_section_sobol import _section_metrics
from visualization import FIG_DIR

# Box centre of the validity box, so the probe sits where the study lives
BASE = {k: 0.5 * (lo + hi)
        for k, (lo, hi) in config.PARAMS_MATERIAL_R_VALID_CABLE.items()}

PROBE_ID0 = 90000     # far from any study's sample ids

FRAC_RANGE = (0.93, 0.99)
E_STEEL    = 200e9    # Pa, for reporting cable stress from EA
EA_CASES   = [("EA=150 kN (A=0.75 mm$^2$)", 150e3),
              ("EA=800 kN (A=4.0 mm$^2$)",  800e3)]


def one(sid, frac, EA):
    """One run at rest-length fraction `frac`; frac None means no cable."""
    prefix = os.path.join(DATA_DIR, f"{sid:05d}")
    kw = dict(cable_wale_lrest=-1.0, cable_course_lrest=-1.0)
    if frac is not None:
        kw = dict(cable_wale_frac=frac, cable_course_frac=frac, cable_EA=EA)
    out = run_fea(
        sf_wale=BASE["sf_wale"], sf_course=BASE["sf_course"],
        knit_dir_deg=BASE["knit_dir"], pressure=BASE["pressure"],
        motif=1, output_prefix=prefix,
        E1=BASE["E1"], r=BASE["r"], nu=BASE["nu"], timeout=600, **kw)
    out.update(_section_metrics(sid))
    # Normalised anisotropy, matching run_material_r_sobol.py.  This was a plain
    # difference here, so the probe and the study reported different quantities
    # under the same name.
    Hs = out.get("H_mean_x0", np.nan) + out.get("H_mean_y0", np.nan)
    out["H_anisotropy"] = ((out["H_mean_x0"] - out["H_mean_y0"]) / Hs
                           if Hs > 1e-6 else np.nan)
    return out


def main(n, jobs):
    V, _ = load_off(MESH_PATH)
    for nm, ang in (("wale", WALE_CABLE_ANGLE), ("course", COURSE_CABLE_ANGLE)):
        idx = generate_cable_path(ang, MESH_PATH)
        chord = float(np.linalg.norm(np.asarray(V[idx[-1]]) -
                                     np.asarray(V[idx[0]])))
        print(f"{nm:7s} cable: {len(idx)} nodes, flat arc "
              f"{cable_path_length(idx, V):.4f} m, straight chord {chord:.4f} m")
    lo, hi = FRAC_RANGE
    print(f"sweeping f = L_rest / L_nocable over ({lo}, {hi}) at the "
          f"validity-box centre, for {len(EA_CASES)} cable areas:")
    print({k: round(v, 4) for k, v in BASE.items() if "lrest" not in k})

    grid = np.linspace(lo, hi, n)
    jobs_list = [(PROBE_ID0, None, EA_CASES[0][1], "no cable")]
    sid = PROBE_ID0 + 1
    for label, EA in EA_CASES:
        for f in grid:
            jobs_list.append((sid, float(f), EA, label))
            sid += 1

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = {}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        fut = {ex.submit(one, s, f, EA): (s, f, EA, lab)
               for s, f, EA, lab in jobs_list}
        for fu in as_completed(fut):
            s, f, EA, lab = fut[fu]
            try:
                rows[s] = dict(fu.result(), frac=(f if f is not None else np.nan),
                               EA=EA, case=lab)
            except Exception as e:
                print(f"  sid {s} (f={f}) failed: {e}")

    df = pd.DataFrame([rows[k] for k in sorted(rows)])
    A = df["EA"] / E_STEEL
    df["wale_stress_MPa"] = df["cable_wale_tension"] / A / 1e6
    COLS = ["case", "frac", "cable_wale_lrest", "cable_wale_L_nocable",
            "crown_height", "H_anisotropy", "cable_wale_tension",
            "wale_stress_MPa", "max_stress"]
    COLS = [c for c in COLS if c in df.columns]
    df[COLS].to_csv(os.path.join(DATA_DIR, "probe_cable_frac.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\n" + df[COLS].to_string(index=False,
                                    float_format=lambda v: f"{v:.4g}"))

    ref = df[df["frac"].isna()]
    print("\nslack check: every cable row should carry tension > 0")
    swept = df[df["frac"].notna()]
    n_slack = (swept["cable_wale_tension"] <= 1e-9).sum()
    print(f"  {n_slack} of {len(swept)} cable runs slack "
          f"(expected 0 — f < 1 is taut by construction)")

    _plot(swept, ref)
    return df


def replot():
    """Redraw figP from data/probe_cable_frac.csv — no FEA.  The CSV drops the
    EA column, so recover it from the case label it was written with."""
    df = pd.read_csv(os.path.join(DATA_DIR, "probe_cable_frac.csv"))
    df["EA"] = df["case"].map(dict(EA_CASES))
    _plot(df[df["frac"].notna()], df[df["frac"].isna()])


def _plot(swept, ref):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 150,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.18, wspace=0.34)
    colours = ["#0077BB", "#EE7733"]

    panels = [("crown_height", "$h_\\mathrm{crown}$ (m)", False),
              ("cable_wale_tension", "$T_\\mathrm{wale}$ (N)", False),
              ("wale_stress_MPa", "cable stress (MPa)", True)]
    for ax, (col, ylab, is_stress), tag in zip(axes, panels, "abc"):
        for (label, EA), c in zip(EA_CASES, colours):
            s = swept[swept["EA"] == EA].sort_values("frac")
            ax.plot(s["frac"], s[col], "-o", color=c, markersize=3.4,
                    linewidth=1.3, label=label)
        if col in ref.columns and len(ref) and not is_stress:
            ax.axhline(ref[col].values[0], color="#555555", linestyle=":",
                       linewidth=0.9, label="no cable")
        if is_stress:
            ax.axhspan(800, ax.get_ylim()[1], color="#CC3311", alpha=0.07)
            ax.axhline(800, color="#CC3311", linestyle="--", linewidth=0.9,
                       label="800 MPa working limit")
        # Plain symbol on the axis; the tick values are a fraction, so the
        # normalisation has to be spelled out here or 0.93 reads as metres.
        ax.set_xlabel("$L_\\mathrm{rest}$   "
                      "(fraction of the cable-free section length)",
                      fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(f"({tag}) {ylab}", fontsize=9.5, pad=4)
        ax.legend(fontsize=6.4, frameon=False)
        ax.tick_params(labelsize=7)

    fig.suptitle("Cable rest length as a fraction of the cable-free section: "
                 "taut by construction, and what it costs in cable stress",
                 fontsize=11, y=0.97)
    base = os.path.join(FIG_DIR, "figP_cable_frac_probe")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=200)
    print(f"Saved: {base}.png / .pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7, help="f grid points")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure from the cached CSV, no FEA")
    args = ap.parse_args()
    if args.plot_only:
        replot()
    else:
        main(args.n, args.jobs)
