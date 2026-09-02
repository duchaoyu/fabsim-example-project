"""Figure: how far the strategy-E shape moves when the region boundary is given
an axial stiffness.

Reads data/seam_stiffness_sweep.csv, written by ./build-headless/seam_imperfection.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
E2   = 12507.0          # N/m, course (stiff) membrane modulus
E1   = 5000.0           # N/m, wale
L_ED = 0.0759           # m, mean seam mesh edge — the band one edge stands for

# The membrane already spans the seam: the cable adds stiffness ON TOP of fabric
# that is already there.  So EA here is the seam's EXCESS over the fabric it
# replaces, not its total.  A seam mechanically identical to the fabric is EA = 0.
#   EA_excess = (k - 1) * E * w      k = seam stiffness ratio, w = seam width
# The natural reference is what the fabric itself carries across one mesh band:
FAB_COURSE = E2 * L_ED   # 949 N
FAB_WALE   = E1 * L_ED   # 379 N
PLAUSIBLE  = 400.0       # N — 20 mm seam at 4x the fabric's per-width stiffness
BASE_FIT_MEAN = 6.143    # mm, strategy E fit to target
BASE_CROWN    = 391.97   # mm

df = pd.read_csv(os.path.join(HERE, 'data', 'seam_stiffness_sweep.csv'))
for c, s in [('L_pos_rms', 1e3), ('L_pos_max', 1e3), ('fit_mean', 1e3),
             ('fit_max', 1e3), ('d_crown', 1e3)]:
    df[c] *= s

SEAMS = [('R01|R2', '#4C78A8', 'R0R1 | R2   material jump, 76 seg, 5.77 m'),
         ('R0|R1',  '#E45756', 'R0 | R1   no material jump, 17 seg, 1.31 m'),
         ('both',   '#54A24B', 'both seams')]

def read_off(p):
    L = open(p).read().split('\n')
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F

fig = plt.figure(figsize=(13.5, 8.6))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.30, wspace=0.28)

# ── (a) deviation from the seamless solution ─────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for name, col, lbl in SEAMS:
    for to, ls in [(1, '-'), (0, '--')]:
        d = df[(df.seam == name) & (df.tension_only == to)].sort_values('EA')
        ax.plot(d.EA, d.L_pos_rms, ls, color=col, lw=1.9 if to else 1.1,
                marker='o' if to else None, ms=3.2, alpha=1 if to else 0.55)
ax.axvspan(0.05, PLAUSIBLE, color='0.88', zorder=0)
ax.axhline(BASE_FIT_MEAN, color='k', lw=1, ls=':')
ax.text(0.13, BASE_FIT_MEAN*1.08, "strategy E's own fit error, 6.14 mm",
        fontsize=8, va='bottom')
ax.text(3, 19.5, 'plausible seam:\nexcess EA < 400 N', fontsize=8, color='0.3')
ax.set_xscale('log'); ax.set_xlabel('seam axial stiffness EA (N)')
ax.set_ylabel('deviation from the seamless\nshape, RMS interior (mm)')
ax.set_title('(a) how far the shape moves', fontsize=11, loc='left', pad=32)
sec = ax.secondary_xaxis('top', functions=(lambda x: x/FAB_COURSE, lambda r: r*FAB_COURSE))
sec.set_xlabel('excess stiffness / fabric stiffness over one 76 mm mesh band',
               fontsize=8.5, labelpad=2)

# ── (b) fit to the design target ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
for name, col, _ in SEAMS:
    d = df[(df.seam == name) & (df.tension_only == 1)].sort_values('EA')
    ax.plot(d.EA, d.fit_mean, '-', color=col, lw=1.9, marker='o', ms=3.2)
ax.axhline(BASE_FIT_MEAN, color='k', lw=1, ls=':')
ax.set_xscale('log'); ax.set_xlabel('seam axial stiffness EA (N)')
ax.set_ylabel('distance to the design target,\nmean (mm)')
ax.axvspan(0.05, PLAUSIBLE, color='0.88', zorder=0)
ax.set_title('(b) the seam makes the fit worse', fontsize=11, loc='left')

# ── (c) crown height ─────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for name, col, _ in SEAMS:
    d = df[(df.seam == name) & (df.tension_only == 1)].sort_values('EA')
    ax.plot(d.EA, d.d_crown, '-', color=col, lw=1.9, marker='o', ms=3.2)
ax.axhline(0, color='k', lw=1, ls=':')
ax.set_xscale('log'); ax.set_xlabel('seam axial stiffness EA (N)')
ax.set_ylabel('change in crown height (mm)')
ax.axvspan(0.05, PLAUSIBLE, color='0.88', zorder=0)
ax.set_title('(c) a stiff seam pulls the crown down', fontsize=11, loc='left')

# ── (d,e) where the deviation is, at the rigid-seam bound ────────────────────
Vb, F = read_off(os.path.join(REPO, 'out', 'seam_baseline.off'))
for k, (fn, ttl) in enumerate([('seam_rigid_R01_R2.off', 'rigid seam on R0R1 | R2'),
                               ('seam_rigid_R0_R1.off',  'rigid seam on R0 | R1')]):
    Vr, _ = read_off(os.path.join(REPO, 'out', fn))
    dev = np.linalg.norm(Vr - Vb, axis=1) * 1e3
    ax = fig.add_subplot(gs[1, k], projection='3d')
    fv = dev[F].mean(axis=1)
    pc = Poly3DCollection([[Vr[i] for i in f] for f in F], edgecolor='none')
    pc.set_array(fv); pc.set_cmap('magma'); pc.set_clim(0, 55)
    ax.add_collection3d(pc)
    ax.set_xlim(-.6,.6); ax.set_ylim(-.6,.6); ax.set_zlim(0,.55)
    ax.set_box_aspect((1.2,1.2,.6)); ax.view_init(elev=30, azim=-62); ax.set_axis_off()
    ax.set_title(f'({"de"[k]}) {ttl}\nmax {dev.max():.0f} mm', fontsize=10)
    if k == 1:
        cb = fig.colorbar(pc, ax=ax, shrink=0.55, pad=0.02)
        cb.set_label('displacement from the\nseamless shape (mm)', fontsize=8)

# ── legend ───────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=2, marker='o', ms=4, label=l) for _, c, l in SEAMS]
h += [Line2D([], [], color='0.35', lw=2, label='tension-only seam (fabric, buckles)'),
      Line2D([], [], color='0.35', lw=1.1, ls='--', label='two-sided seam (bonded tape)')]
ax.legend(handles=h, loc='upper left', frameon=False, fontsize=9)
ax.text(0, 0.46,
        'EA is the seam\'s EXCESS stiffness over the fabric it\n'
        'replaces: the membrane already spans the boundary,\n'
        'so a seam identical to the fabric is EA = 0.\n'
        '   EA = (k-1) x E x w,   k = stiffness ratio, w = width\n\n'
        'The fabric itself carries 949 N (course) / 379 N (wale)\n'
        'across one 76 mm mesh band.  A 20 mm seam at 4x the\n'
        'fabric stiffness is 750 N of excess in course, 300 N in\n'
        'wale — the shaded band.  Doubling the fabric over the\n'
        'whole band, EA ~ 1000 N, is the knee of the curve and\n'
        'is not a seam anyone would knit.\n\n'
        'Saturation is the rigid-seam bound: no model of the\n'
        'transition that only adds axial stiffness along the\n'
        'line can exceed it.',
        fontsize=8.2, va='top', transform=ax.transAxes, color='0.25')

out = os.path.join(HERE, 'figures', 'seam_stiffness.png')
plt.savefig(out, dpi=135, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', out)
