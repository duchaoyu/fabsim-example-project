"""
Figure 7.27: strategies A-E for the shell with middle crease, after the
2026-10-01 rerun (fixed pressure, stitch structure I).

Deviation = distance from each simulated vertex to the nearest point on the
target surface, the measure the Section 7.4 numbers use; mean and max are over
all vertices and quoted as % of the 1200 mm span.  One shared colour scale.

Inputs (written by the src/best_fit_* drivers):
    out/sf_iso_opt_result.off               A  isotropic
    out/sf_iso_cable_opt_result.off         B  isotropic + crease cable
    out/sf_opt_result.off                   C  anisotropic
    out/sf_cable_opt_result.off             D  anisotropic + crease cable
    out/sf_3region_adaptive_cable_result.off E  three regions + cable
    out/sf_3region_adaptive_cable_faces.txt     E region assignment

    .venv/bin/python FDM/figure_2part_strategies.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.tri import Triangulation

from deviation_tools import read_off, closest_dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
TARGET = os.path.join(ROOT, "data", "2part", "2part_opt_simu_m.off")
CABLE = os.path.join(ROOT, "imperfection_study", "data", "cable_2part_crease.json")
FIG = os.path.join(HERE, "figures", "2part_strategies")
SPAN_MM = 1200.0

# single-hue sequential ramp (blue 100 -> 700)
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
REGION_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2 = "#0b0b0b", "#52514e"

RUNS = [
    ("A", "Isotropic pre-strain, no cable", "sf_iso_opt_result.off", False),
    ("B", "Isotropic pre-strain, crease cable", "sf_iso_cable_opt_result.off", True),
    ("C", "Anisotropic pre-strain, no cable", "sf_opt_result.off", False),
    ("D", "Anisotropic pre-strain, crease cable", "sf_cable_opt_result.off", True),
    ("E", "Three regions, crease cable", "sf_3region_adaptive_cable_result.off", True),
]


def read_regions(path, n_faces):
    reg = np.full(n_faces, -1)
    cur = None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("REGION_"):
            cur = int(line.split()[0].split("_")[1])
        else:
            reg[int(line)] = cur
    return reg


def main():
    T, F = read_off(TARGET, faces=True)
    tri = Triangulation(T[:, 0], T[:, 1], F)
    cable = json.load(open(CABLE))["indices"]

    devs = {}
    for key, _, fname, _ in RUNS:
        X = read_off(os.path.join(OUT, fname))
        devs[key] = closest_dist(X, T, F) * 1000.0
    vmax = np.ceil(max(d.max() for d in devs.values()) / 5.0) * 5.0

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.edgecolor": INK2, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    fig.subplots_adjust(left=0.01, right=0.88, top=0.92, bottom=0.02, wspace=0.12, hspace=0.22)
    for ax, (key, label, _, has_cable) in zip(axes.flat, RUNS):
        d = devs[key]
        tp = ax.tripcolor(tri, d, cmap=SEQ, vmin=0, vmax=vmax, shading="gouraud")
        if has_cable:
            ax.plot(T[cable, 0], T[cable, 1], color="#e34948", lw=1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"({key}) {label}\nmean {d.mean():.2f} mm ({d.mean() / SPAN_MM * 100:.2f}%), "
                     f"max {d.max():.1f} mm ({d.max() / SPAN_MM * 100:.2f}%)",
                     fontsize=8, loc="left")
        print(f"{key}: mean {d.mean():.3f} mm  max {d.max():.3f} mm")

    # E region map
    ax = axes.flat[5]
    reg = read_regions(os.path.join(OUT, "sf_3region_adaptive_cable_faces.txt"), len(F))
    # drawn mirror-symmetric about the crease, as in Figure 7.31e (2part_span.png)
    from figure_2part_span import symmetric_regions
    reg = symmetric_regions(reg, T, F)
    ax.tripcolor(tri, facecolors=reg.astype(float), cmap=ListedColormap(REGION_COLOURS),
                 vmin=-0.5, vmax=2.5, edgecolors="#fcfcfb", linewidth=0.2)
    ax.plot(T[cable, 0], T[cable, 1], color="#e34948", lw=1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(E) regions: domes R0 and R1 share one pre-strain,\nR2 is the rest of the patch",
                 fontsize=8, loc="left")
    cent = T[F].mean(axis=1)
    R = np.abs(T[:, :2]).max()
    for r, name in enumerate(["R0", "R1", "R2"]):
        own = np.where(reg == r)[0]
        if r == 2:   # R2 is a ring: label it on the ring, not at its centroid
            k = own[np.argmin(np.linalg.norm(cent[own, :2] - [0.35 * R, -0.75 * R], axis=1))]
        else:
            m = cent[own].mean(axis=0)
            k = own[np.argmin(np.linalg.norm(cent[own] - m, axis=1))]
        ax.text(cent[k, 0], cent[k, 1], name, ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")

    cax = fig.add_axes([0.91, 0.2, 0.015, 0.6])
    cb = fig.colorbar(tp, cax=cax)
    cb.set_label("distance to target surface (mm)", color=INK2)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG + ".pdf", bbox_inches="tight")
    print("Saved", FIG + ".png")


if __name__ == "__main__":
    main()
