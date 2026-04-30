"""
Cable stiffener analysis — two figures:

figO_cable_comparison.pdf
  Left  col: Motif 1   Right col: Motif 2
  Row 0: Crown-height distributions — no-cable (violin) vs cable (split by taut/slack)
  Row 1: Surrogate sweep — crown height vs cable_angle at fixed (sf, p)
  Row 2: Stiffening scatter — crown_height vs cable_tension (taut samples only)

figP_cable_lrest.pdf
  Row 0: Cable tension vs rest-length factor α (analytical, taut samples)
         + threshold α★ distribution
  Row 1: Crown-height stiffening slope from data
         + practical rest-length guide table
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, PARAMS_NO_CABLE, PARAMS_CABLE, CABLE_EA
from surrogate import ScalarSurrogate

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth":  0.8,
    "figure.dpi":      150,
})

COLORS = {
    "motif1": "#2E8B57",
    "motif2": "#20B2AA",
    "taut":   "#E07B39",
    "slack":  "#AAAAAA",
    "nocable":"#4477AA",
}

GROUPS_NOCABLE = ["motif1_nocable", "motif2_nocable"]
GROUPS_CABLE   = ["motif1_cable",   "motif2_cable"]
MOTIF_LABELS   = {"motif1": "Motif 1", "motif2": "Motif 2"}

L_FLAT_MM = 1196.3   # flat rest length in mm (pre-computed)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_surrogate(group):
    p = os.path.join(DATA_DIR, f"{group}_scalar_surrogate.pkl")
    return ScalarSurrogate.load(p) if os.path.exists(p) else None


def _surrogate_angle_sweep(surr_cable, output="crown_height",
                            sf_wale=1.0, sf_course=1.0,
                            knit_dir=0.0, pressure=1000.0, n=80):
    """Sweep cable_angle 0–180° at fixed other params, return (angles, y)."""
    angles = np.linspace(0, 180, n)
    keys   = list(PARAMS_CABLE.keys())
    X = np.column_stack([
        np.full(n, sf_wale)   if k == "sf_wale"    else
        np.full(n, sf_course) if k == "sf_course"   else
        np.full(n, knit_dir)  if k == "knit_dir"    else
        np.full(n, pressure)  if k == "pressure"    else
        angles
        for k in keys
    ])
    return angles, surr_cable.predict(X)[output]


def _violin(ax, data, pos, color, width=0.35, label=None):
    """Draw a single slim violin at position pos."""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return
    parts = ax.violinplot([data], positions=[pos], widths=width,
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.75)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color(color)
        parts[part].set_linewidth(1.0)


# ── Figure O ──────────────────────────────────────────────────────────────────

def plot_cable_comparison(save=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    if "sim_failed" in df.columns:
        df = df[~df["sim_failed"]]

    motifs = ["motif1", "motif2"]
    n_rows, n_cols = 3, 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.2 * n_rows),
                             gridspec_kw={"hspace": 0.50, "wspace": 0.40})

    for col_idx, motif in enumerate(motifs):
        mc  = COLORS[motif]
        nc  = df[df["group"] == f"{motif}_nocable"]
        cab = df[df["group"] == f"{motif}_cable"]
        taut_mask  = cab["cable_tension"] > 0
        slack_mask = ~taut_mask

        # ── Row 0: distribution comparison ────────────────────────────────────
        ax = axes[0, col_idx]
        # Positions: 1=nocable, 2=slack cable, 3=taut cable
        _violin(ax, nc["crown_height"]  * 1000, 1, color=COLORS["nocable"])
        _violin(ax, cab.loc[slack_mask, "crown_height"] * 1000, 2,
                color=COLORS["slack"])
        _violin(ax, cab.loc[taut_mask,  "crown_height"] * 1000, 3,
                color=COLORS["taut"])

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["no cable", "cable\n(slack)", "cable\n(taut)"])
        ax.set_ylabel("Crown height  (mm)")
        ax.set_xlim(0.4, 3.6)
        n_taut = taut_mask.sum()
        ax.set_title(
            f"{MOTIF_LABELS[motif]} — height distribution\n"
            f"(no-cable n={len(nc)};  slack n={slack_mask.sum()};  taut n={n_taut})",
            pad=5)
        ax.legend(handles=[
            Patch(facecolor=COLORS["nocable"], label="no cable"),
            Patch(facecolor=COLORS["slack"],   label="cable slack"),
            Patch(facecolor=COLORS["taut"],    label="cable taut"),
        ], fontsize=7.5, loc="upper right")

        # ── Row 1: paired comparison — cable vs predicted no-cable ────────────
        ax = axes[1, col_idx]
        surr_nocable = _load_surrogate(f"{motif}_nocable")

        taut_m = cab[taut_mask].copy()
        if surr_nocable is not None and len(taut_m) > 0:
            # Predict no-cable height at same (sf_w, sf_c, knit_dir, pressure)
            X_nc = taut_m[list(PARAMS_NO_CABLE.keys())].values
            h_nc = surr_nocable.predict(X_nc)["crown_height"] * 1000
            h_cable = taut_m["crown_height"].values * 1000
            delta = h_cable - h_nc   # negative = cable suppresses dome

            sc2 = ax.scatter(h_nc, h_cable,
                             c=taut_m["cable_tension"].values,
                             cmap="plasma", s=35, alpha=0.85, zorder=4)
            lim = max(h_nc.max(), h_cable.max()) * 1.05
            ax.plot([0, lim], [0, lim], color="0.6", lw=1.0, ls="--",
                    label="no change")
            cb2 = fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.04)
            cb2.set_label("Cable tension (N)", fontsize=7.5)
            cb2.ax.tick_params(labelsize=7)
            ax.set_xlabel("Predicted height without cable  (mm)")
            ax.set_ylabel("Actual height with cable  (mm)")
            ax.set_title(f"{MOTIF_LABELS[motif]} — paired comparison (taut samples)\n"
                         "below diagonal = cable reduces dome height")
            ax.legend(fontsize=7.5)
            n_suppressed = (delta < -5).sum()
            ax.text(0.04, 0.97,
                    f"{n_suppressed}/{len(taut_m)} samples suppressed (>5 mm)",
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No taut samples / surrogate unavailable",
                    ha="center", va="center", transform=ax.transAxes)

        # ── Row 2: height reduction Δh vs cable tension ────────────────────────
        ax = axes[2, col_idx]
        taut_m = cab[taut_mask].copy()
        surr_nocable = _load_surrogate(f"{motif}_nocable")

        if surr_nocable is not None and len(taut_m) > 0:
            X_nc  = taut_m[list(PARAMS_NO_CABLE.keys())].values
            h_nc  = surr_nocable.predict(X_nc)["crown_height"] * 1000
            delta = taut_m["crown_height"].values * 1000 - h_nc   # <0 = suppressed

            sc3 = ax.scatter(taut_m["cable_tension"].values, delta,
                             c=taut_m["pressure"].values,
                             cmap="viridis", s=32, alpha=0.85, zorder=4)
            cb3 = fig.colorbar(sc3, ax=ax, fraction=0.046, pad=0.04)
            cb3.set_label("Pressure (Pa)", fontsize=7.5)
            cb3.ax.tick_params(labelsize=7)

            ax.axhline(0, color="0.5", lw=0.9, ls="--")
            if len(taut_m) >= 4:
                z = np.polyfit(taut_m["cable_tension"].values, delta, 1)
                t_r = np.linspace(0, taut_m["cable_tension"].max(), 80)
                ax.plot(t_r, np.polyval(z, t_r), color=COLORS["taut"],
                        lw=1.8, label=f"trend  {z[0]*1000:.1f} mm/kN")
            ax.set_xlabel("Cable tension  (N)")
            ax.set_ylabel(r"$\Delta h_{crown}$ = cable − no cable  (mm)")
            ax.set_title(f"{MOTIF_LABELS[motif]} — cable height reduction\n"
                         r"($\Delta h < 0$ = cable suppresses dome)")
            ax.legend(fontsize=7.5)
            ax.set_xlim(left=-5)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    va="center", transform=ax.transAxes)

    fig.suptitle(
        "Cable stiffener: influence on crown height\n"
        "(Row 1 = distributions;  Row 2 = angle sweep via surrogate;  "
        "Row 3 = tension–height scatter)",
        fontsize=9.5, y=1.005)

    if save:
        path = os.path.join(FIG_DIR, "figO_cable_comparison.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


# ── Figure P: rest length sensitivity ─────────────────────────────────────────

def plot_cable_lrest(save=True):
    df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))
    cable = df[df["has_cable"] == True].copy()
    taut  = cable[cable["cable_tension"] > 0].copy()

    # For each taut sample: L_current/L_rest = 1 + T/EA
    taut["L_ratio"] = 1.0 + taut["cable_tension"] / CABLE_EA

    fig, axes = plt.subplots(2, 2, figsize=(11, 8),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.42})

    # ── Panel (0,0): T(α) curves for representative taut samples ──────────────
    ax = axes[0, 0]
    alpha = np.linspace(0.4, 1.4, 300)

    # Pick 8 samples spanning the range of tension
    rep = taut.nlargest(4, "cable_tension")
    rep = pd.concat([rep, taut.nsmallest(4, "cable_tension")])

    for _, row in rep.iterrows():
        L_ratio = row["L_ratio"]
        T_new   = np.maximum(0.0, CABLE_EA * (L_ratio - alpha) / alpha)
        motif_k = f"motif{int(row['motif'])}"
        ax.plot(alpha, T_new, color=COLORS[motif_k], lw=1.3, alpha=0.7)

    # Mark α=1 (original)
    ax.axvline(1.0, color="0.5", lw=0.9, ls="--", label=r"$\alpha=1$  (original)")

    # Highlight common pre-stress levels
    for a, label in [(0.9, "−10%"), (0.8, "−20%"), (0.7, "−30%")]:
        ax.axvline(a, color="#E07B39", lw=0.7, ls=":", alpha=0.7)
        ax.text(a + 0.01, ax.get_ylim()[1] * 0.02, label, fontsize=6.5,
                color="#E07B39", va="bottom")

    ax.set_xlabel(r"Rest-length factor  $\alpha = L_{rest}/L_{flat}$")
    ax.set_ylabel("Cable tension  (N)")
    ax.set_xlim(0.4, 1.4)
    ax.set_ylim(bottom=0)
    ax.set_title(r"Cable tension vs rest-length factor $\alpha$"
                 "\n(each curve = one taut simulation sample)")
    ax.legend(handles=[
        Line2D([0],[0], color="0.5", lw=1.5, ls="--", label=r"$\alpha=1$  (original)"),
        Line2D([0],[0], color=COLORS["motif1"], lw=2, label="Motif 1"),
        Line2D([0],[0], color=COLORS["motif2"], lw=2, label="Motif 2"),
    ], fontsize=8, loc="upper right")

    # ── Panel (0,1): activation threshold α★ ──────────────────────────────────
    ax = axes[0, 1]
    # α★ = L_ratio = 1 + T/EA (the smallest α for which cable becomes taut)
    # For slack samples: α★ = L_current/L_flat = unknown, but must be ≤ 1
    # For taut samples: α★ > 1 (cable is already longer than flat rest)
    taut_m1 = taut[taut["motif"] == 1]
    taut_m2 = taut[taut["motif"] == 2]

    for sub, motif_k, label in [(taut_m1, "motif1", "Motif 1"),
                                 (taut_m2, "motif2", "Motif 2")]:
        if len(sub) < 3:
            continue
        ax.scatter(sub["pressure"], sub["L_ratio"],
                   color=COLORS[motif_k], s=35, alpha=0.8, label=label)

    ax.axhline(1.0, color="0.5", lw=0.9, ls="--", label=r"$\alpha^\star=1$  (flat rest)")
    ax.set_xlabel("Pressure  (Pa)")
    ax.set_ylabel(r"Activation threshold  $\alpha^\star = L_{def}/L_{flat}$")
    ax.set_title(r"Cable activation threshold $\alpha^\star$ vs pressure"
                 r"  (taut when $\alpha < \alpha^\star$)")
    ax.legend(fontsize=8)

    # ── Panel (1,0): crown height vs cable tension ─────────────────────────────
    ax = axes[1, 0]
    all_nocable = df[df["has_cable"] == False]

    for motif_k, label, m_int in [("motif1","Motif 1",1), ("motif2","Motif 2",2)]:
        nc_sub   = all_nocable[all_nocable["motif"] == m_int]
        taut_sub = taut[taut["motif"] == m_int]

        ax.scatter(np.zeros(len(nc_sub)), nc_sub["crown_height"] * 1000,
                   color=COLORS[motif_k], s=6, alpha=0.18, zorder=2)
        sc = ax.scatter(taut_sub["cable_tension"],
                        taut_sub["crown_height"] * 1000,
                        color=COLORS[motif_k], s=28, alpha=0.85, zorder=4,
                        label=f"{label} (taut)")

    ax.set_xlabel("Cable tension  (N)")
    ax.set_ylabel("Crown height  (mm)")
    ax.set_title("Crown height vs cable tension\n"
                 "(faint points at T=0 = no-cable reference)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=-5)

    # ── Panel (1,1): rest-length design guide ─────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")

    alphas   = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    L_rest_mm = [f"{a * L_FLAT_MM:.0f}" for a in alphas]
    # Expected tension at L_ratio = 1.05 (typical taut sample)
    L_ref = 1.05
    T_vals = [f"{max(0, CABLE_EA*(L_ref-a)/a):.0f}" for a in alphas]

    table_data = [
        [r"$\alpha$", r"$L_{rest}$ (mm)", r"$T$ @ $L_{def}=1.05\,L_{flat}$ (N)"]
    ] + [[str(a), lr, t] for a, lr, t in zip(alphas, L_rest_mm, T_vals)]

    tab = ax.table(cellText=table_data[1:],
                   colLabels=table_data[0],
                   loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1.0, 1.55)

    # Highlight rows where cable is definitely taut
    for row_idx in range(1, len(alphas) + 1):
        alpha_val = alphas[row_idx - 1]
        color = "#d5f0e0" if alpha_val < L_ref else "#f0f0f0"
        for col in range(3):
            tab[row_idx, col].set_facecolor(color)

    ax.set_title(r"Rest-length design guide  (EA=5000 N,  $L_{flat}$≈1196 mm)"
                 "\n(green rows = cable taut at typical inflation)",
                 fontsize=8.5, pad=8)

    fig.suptitle(
        r"Cable rest-length $L_{rest} = \alpha \cdot L_{flat}$: "
        "tension and stiffening sensitivity\n"
        r"($\alpha < 1$ = pre-stressed;  $\alpha = 1$ = flat reference)",
        fontsize=9.5, y=1.005)

    if save:
        path = os.path.join(FIG_DIR, "figP_cable_lrest.pdf")
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    return fig


if __name__ == "__main__":
    print("Plotting cable comparison (figO)...")
    plot_cable_comparison()
    print("Plotting rest-length sensitivity (figP)...")
    plot_cable_lrest()
    print("Done.")
