"""Figure: the force-density optimisation discovers the stiffener network by itself.

Reads the final crossvault result (mesh_out_cross_20240404130204.json, the last link
in the fofin_cross.py -> fofin_cross_session.py chain) and draws four panels:

  a  input      the hand-assigned force densities the run starts from
  b  converged  the optimised field
  c  change     dq = q_converged - q_input, i.e. what the optimiser decided on its own
  d  network    the edges the field itself singles out, in 3D

Colour follows the dataviz rules: one-hue blue ramp for magnitude, blue<->red with a
neutral grey midpoint for the signed change, categorical slots 1-2 for the two
stiffener families.

    python FDM/figure_stiffener_discovery.py
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "data", "crossvault", "mesh_out_cross_20240404130204.json")
TARGET = os.path.join(HERE, "data", "crossvault", "cross_vault_smooth.json")
OUT = os.path.join(HERE, "figures", "stiffener_discovery.png")

# the initial force densities, straight from fofin_cross.py:74,82-88
Q_UNIFORM = 1.0
Q_SEED = 3.0
LOOP_A = [146, 110, 153, 168, 94, 97, 70, 68, 156, 83, 82, 39, 40]
LOOP_B = [133, 123, 112, 14, 12, 17, 70, 69, 128, 7, 4, 10, 107]

Q_THRESHOLD = 2.0  # the 36 selected edges are identical anywhere in 2.0 <= q <= 2.2
Q_MAX = 5.0        # colour-ramp ceiling

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
SURFACE = "#fcfcfb"
GREY_MID = "#f0efec"
SERIES_1 = "#2a78d6"   # groin ribs
SERIES_2 = "#eb6834"   # boundary arches

# one-hue sequential blue, steps 100 -> 700 of the reference ramp
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
])


def _blend(c1, c2, t):
    a, b = np.array(to_rgb(c1)), np.array(to_rgb(c2))
    return tuple((1 - t) * a + t * b)


# diverging blue <-> red, neutral grey midpoint, three steps per arm. The red arm is
# derived from the palette's red by blending toward the midpoint / toward ink, so no
# step is eyeballed.
RED = "#e34948"
DIV_BR = LinearSegmentedColormap.from_list("div_br", [
    "#0d366b", "#2a78d6", "#9ec5f4", GREY_MID,
    _blend(GREY_MID, RED, 0.55), RED, _blend(RED, INK, 0.35),
])


def load_mesh(path):
    """compas-1 mesh JSON -> vertices, faces, edges with force densities."""
    d = json.load(open(path))
    V = {int(k): np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)])
         for k, v in d["vertex"].items()}
    F = [d["face"][k] for k in sorted(d["face"], key=int)]
    E = []
    for key, attr in d.get("edgedata", {}).items():
        u, w = (int(t) for t in key.strip("()").split(","))
        q = attr.get("qpre", Q_UNIFORM)
        E.append((u, w, float(q[0] if isinstance(q, list) else q)))
    return V, F, E


def initial_q(E):
    """The q field the optimisation was handed, reconstructed from fofin_cross.py."""
    seeded = set()
    for loop in (LOOP_A, LOOP_B):
        for a, b in zip(loop[:-1], loop[1:]):
            seeded.add(frozenset((a, b)))
    return np.array([Q_SEED if frozenset((u, w)) in seeded else Q_UNIFORM
                     for u, w, _ in E]), seeded


def components(edges):
    """Connected components of an edge subset, as (vertices, edges) pairs."""
    adj = collections.defaultdict(set)
    for u, w, _ in edges:
        adj[u].add(w)
        adj[w].add(u)
    seen, out = set(), []
    for n in adj:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            c = stack.pop()
            if c in comp:
                continue
            comp.add(c)
            seen.add(c)
            stack.extend(adj[c] - comp)
        out.append((comp, [e for e in edges if e[0] in comp and e[1] in comp]))
    return sorted(out, key=lambda ce: -len(ce[1]))


FIGW, FIGH = 11.6, 12.4
PLAN_LIM = (-0.075, 1.275)


def square(x, y, h):
    """Axes rect of height h whose data square comes out square on the page."""
    return [x, y, h * FIGH / FIGW, h]


def plan(ax, V, E, values, cmap, norm, lw=(0.5, 3.4)):
    segs = [[V[u][:2], V[w][:2]] for u, w, _ in E]
    mag = np.abs(values)
    width = lw[0] + (lw[1] - lw[0]) * (mag - mag.min()) / max(float(np.ptp(mag)), 1e-9)
    lc = LineCollection(segs, array=np.asarray(values), cmap=cmap, norm=norm,
                        linewidths=width, capstyle="round")
    ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.set_xlim(*PLAN_LIM)
    ax.set_ylim(*PLAN_LIM)
    ax.set_axis_off()
    return lc


def label(ax, text, xy, xytext, ha="center", va="center"):
    """Direct label with a surface halo so it stays legible over the network."""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8.5, color=INK_2, ha=ha, va=va,
                linespacing=1.4,
                bbox=dict(fc=SURFACE, ec="none", alpha=0.82, pad=1.6),
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=0.8,
                                shrinkA=1, shrinkB=2))


def colorbar(fig, rect, norm, cmap, text):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_label(text, fontsize=9, color=INK_2, labelpad=4)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=2, pad=2)
    return cb


def panel_title(fig, x, y, tag, title, subtitle):
    """Titles live in figure coordinates so every panel's caption block lines up,
    and multi-line subtitles grow downward instead of into the tag above."""
    fig.text(x, y, tag, fontsize=12, fontweight="bold", color=INK, va="baseline")
    fig.text(x + 0.018, y, title, fontsize=11.5, color=INK, va="baseline")
    fig.text(x, y - 0.011, subtitle, fontsize=9, color=INK_2, va="top",
             linespacing=1.55)


def main():
    V, F, E = load_mesh(RESULT)
    q = np.array([e[2] for e in E])
    q0, seeded = initial_q(E)
    dq = q - q0

    # fit of the converged shape to the target surface
    Vt, _, _ = load_mesh(TARGET)
    err = np.array([np.linalg.norm(V[k] - Vt[k]) for k in V])

    skeleton = [(u, w, qq) for (u, w, qq) in E if qq >= Q_THRESHOLD]
    comps = components(skeleton)
    ribs, arches = comps[0][1], [e for _, ce in comps[1:] for e in ce]
    rib_q = np.array([e[2] for e in ribs])
    arch_q = np.array([e[2] for e in arches])
    at_floor = int((q <= 0.1001).sum())

    print(f"edges {len(E)}  q {q.min():.2f}-{q.max():.2f}  at lower bound {at_floor}")
    print(f"skeleton q>={Q_THRESHOLD}: {len(skeleton)} edges, {len(comps)} components")
    print(f"  groin ribs      {len(ribs):3d} edges  q {rib_q.min():.2f}-{rib_q.max():.2f}")
    print(f"  boundary arches {len(arches):3d} edges  q {arch_q.min():.2f}-{arch_q.max():.2f}")
    print(f"  of these, hand-seeded: "
          f"{sum(1 for u, w, _ in skeleton if frozenset((u, w)) in seeded)}/{len(seeded)}")
    print(f"fit to target: mean {err.mean()*1000:.1f} mm  max {err.max()*1000:.1f} mm")
    print(f"dq on seeded edges: {dq[[frozenset((u,w)) in seeded for u,w,_ in E]].min():+.2f} "
          f"to {dq[[frozenset((u,w)) in seeded for u,w,_ in E]].max():+.2f}")

    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    norm_q = Normalize(0, Q_MAX)
    lim = float(np.abs(dq).max())
    XL, XR = 0.075, 0.545          # column origins
    ROW1, ROW2 = 0.545, 0.085      # panel bottoms
    T1, T2 = 0.885, 0.445          # title baselines

    # ---------------------------------------------------------------- a  input
    ax_a = fig.add_axes(square(XL, ROW1, 0.285))
    plan(ax_a, V, E, q0, SEQ_BLUE, norm_q)
    panel_title(fig, XL, T1, "a", "the force densities it starts from",
                f"q = {Q_UNIFORM:.1f} on all {len(E)} edges, with the two diagonals\n"
                f"set to {Q_SEED:.1f} by hand. Drawn on the converged geometry, as\n"
                f"in b, so that only the colours differ between the two.")
    ax_a.plot([0.35, 0.85], [-0.045, -0.045], color=INK_2, lw=1.4,
              solid_capstyle="butt", clip_on=False)
    ax_a.text(0.60, -0.075, "0.5 m", ha="center", va="top", fontsize=8.5, color=INK_2)

    # ------------------------------------------------------------ b  converged
    ax_b = fig.add_axes(square(XR, ROW1, 0.285))
    plan(ax_b, V, E, q, SEQ_BLUE, norm_q)
    panel_title(fig, XR, T1, "b", "the force densities it ends with",
                f"q = {q.min():.2f} – {q.max():.2f}; the shape now fits the target\n"
                f"surface to {err.mean()*1000:.0f} mm mean, {err.max()*1000:.0f} mm worst")

    def mid(edge):
        return 0.5 * (V[edge[0]][:2] + V[edge[1]][:2])

    def pick(edges, quadrant):
        """Highest-q edge of a family in a given quadrant, for a short leader."""
        sx, sy = quadrant
        cand = [e for e in edges if (mid(e)[0] - 0.6) * sx > 0
                and (mid(e)[1] - 0.6) * sy > 0]
        return max(cand or edges, key=lambda e: e[2])

    label(ax_b, f"crown ribs, q → {q.max():.1f}",
          mid(pick(ribs, (-1, 1))), (0.17, 0.80), ha="center")
    label(ax_b, f"the web beside each rib goes slack:\n{at_floor} edges end on the "
                f"bound q = 0.1",
          mid(pick([e for e in E if e[2] <= 0.1001], (1, 1))), (1.15, 1.10), ha="right")
    label(ax_b, f"boundary arches, q ≈ {arch_q.mean():.1f}",
          mid(min(arches, key=lambda e: mid(e)[1])), (0.60, 0.235), ha="center")

    colorbar(fig, [0.335, 0.500, 0.33, 0.011], norm_q, SEQ_BLUE,
             "force density q   (panels a, b)")

    # --------------------------------------------------------------- c  change
    ax_c = fig.add_axes(square(XL, ROW2, 0.285))
    plan(ax_c, V, E, dq, DIV_BR, Normalize(-lim, lim))
    panel_title(fig, XL, T2, "c", "what it decided by itself",
                "Δq = end − start. It stiffens the crown and the whole perimeter\n"
                "and releases everything between — including the outer thirds of\n"
                "the two diagonals it was handed, which it hands straight back.")
    colorbar(fig, [0.095, 0.040, 0.26, 0.011], Normalize(-lim, lim), DIV_BR,
             "change in force density Δq   (panel c)")

    # -------------------------------------------------------------- d  network
    ax_d = fig.add_axes([0.470, 0.060, 0.505, 0.335], projection="3d",
                        computed_zorder=False)
    ax_d.set_proj_type("ortho")
    ax_d.patch.set_visible(False)
    faces = [[V[i] for i in f] for f in F]
    ax_d.add_collection3d(Poly3DCollection(faces, facecolors=SURFACE, alpha=0.72,
                                           edgecolors="#cac8c2", linewidths=0.4,
                                           zorder=1))
    handles = []
    for family, colour, name in [(arches, SERIES_2, "boundary arches"),
                                 (ribs, SERIES_1, "groin ribs")]:
        segs = [[V[u], V[w]] for u, w, _ in family]
        qq = np.array([e[2] for e in family])
        ax_d.add_collection3d(Line3DCollection(
            segs, colors=colour, linewidths=1.6 + 2.0 * (qq - Q_THRESHOLD) / 2.9,
            capstyle="round", zorder=3))
        handles.append(plt.Line2D([], [], color=colour, lw=2.6,
                                  solid_capstyle="round", label=name))

    ax_d.set_xlim(0.02, 1.18)
    ax_d.set_ylim(0.02, 1.18)
    ax_d.set_zlim(0.0, 0.42)
    ax_d.set_box_aspect((1, 1, 0.40), zoom=1.45)
    ax_d.view_init(elev=24, azim=-58)
    ax_d.set_axis_off()
    panel_title(fig, XR, T2, "d", "the stiffener network that falls out",
                f"the {len(skeleton)} edges with q ≥ {Q_THRESHOLD:.1f} — the top "
                f"{100*len(skeleton)/len(E):.0f} %, and the same set anywhere\n"
                f"in 2.0 ≤ q ≤ 2.2. They separate into {len(comps)} disconnected lines, "
                f"4-fold symmetric,\nthough nothing in the run imposed that symmetry.")
    leg = ax_d.legend(handles=[handles[1], handles[0]],
                      labels=[f"groin ribs — {len(ribs)} edges, q {rib_q.min():.1f}"
                              f"–{rib_q.max():.1f}",
                              f"boundary arches — 4 × {len(arches)//4} edges, "
                              f"q {arch_q.min():.1f}–{arch_q.max():.1f}"],
                      loc="lower left", bbox_to_anchor=(0.06, 0.02), fontsize=9,
                      handlelength=1.6, borderpad=0.6, labelspacing=0.6,
                      frameon=True, facecolor=SURFACE, edgecolor="none",
                      framealpha=0.85)
    for text in leg.get_texts():
        text.set_color(INK_2)

    # ------------------------------------------------------------------ framing
    fig.text(XL, 0.962,
             "The optimisation finds the stiffeners without being told where they go",
             fontsize=15.5, color=INK, fontweight="semibold", va="baseline")
    fig.text(XL, 0.945,
             "Cross vault, 1.2 × 1.2 m, 169 nodes and 336 edges, pneumatic form-finding "
             "at p = 4.2 kPa. Fitting that target surface is the only\nobjective, and "
             "every force density is free — the two diagonals in a are an initial guess, "
             "not a constraint. Gradient descent\non the fit alone is enough to drive the "
             "field into two families of stiffening lines with a slack web in between.",
             fontsize=9.5, color=INK_2, va="top", linespacing=1.6)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
