"""Figure: the other half of the seam question - a transition WEAKER than the
fabric, down to zero modulus.

The cable sweep (figure_seam_stiffness.py) can only add stiffness: EA >= 0 is
excess over the fabric the seam replaces.  Here the seam band's own moduli are
scaled by k, so k < 1 is a deficit and k -> 0 is a slit.

Reads data/seam_soft_modulus.csv and data/seam_stiffness_sweep.csv, both written
by ./build-headless/seam_imperfection.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
E2   = 12507.0          # N/m, course (stiff) membrane modulus
BASE_FIT_MEAN = 6.143   # mm, strategy E's own fit to target
SEAM_LEN = {'R01|R2': 5.3497, 'R0|R1': 1.2373, 'both': 5.3497 + 1.2373}   # m, rest
SEAMS = [('R01|R2', '#4C78A8', 'R0R1 | R2   material jump, 5.35 m'),
         ('R0|R1',  '#E45756', 'R0 | R1   no material jump, 1.24 m'),
         ('both',   '#54A24B', 'both seams')]

soft = pd.read_csv(os.path.join(HERE, 'data', 'seam_soft_modulus.csv'))
stiff = pd.read_csv(os.path.join(HERE, 'data', 'seam_stiffness_sweep.csv'))
stiff = stiff[stiff.tension_only == 1]
for d, cols in [(soft, ['L_pos_rms', 'L_pos_max', 'fit_mean', 'd_crown']),
                (stiff, ['L_pos_rms', 'fit_mean', 'd_crown'])]:
    for c in cols: d[c] *= 1e3

def read_off(p):
    L = open(p).read().split('\n')
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2+i].split()] for i in range(nv)])
    Fa = np.array([[int(x) for x in L[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, Fa

fig = plt.figure(figsize=(13.5, 8.4))
gs  = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], hspace=0.34, wspace=0.30)

# ── (a) deviation as the band is softened ───────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for name, col, _ in SEAMS:
    d = soft[soft.seam == name].sort_values('k', ascending=False)
    ax.plot(d.k, d.L_pos_rms, '-', color=col, lw=1.9, marker='o', ms=3.2)
    rigid = stiff[stiff.seam == name].L_pos_rms.max()      # EA -> inf saturation
    ax.axhline(rigid, color=col, lw=1, ls='--', alpha=0.7)
ax.axhline(BASE_FIT_MEAN, color='k', lw=1, ls=':')
ax.text(0.6, BASE_FIT_MEAN*1.1, "strategy E's own fit error, 6.14 mm", fontsize=8)
ax.text(0.02, 0.93, 'dashed: the rigid-seam bound\nfrom the stiff side', fontsize=8,
        color='0.3', transform=ax.transAxes, va='top')
ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
ax.set_xlabel('band modulus / fabric modulus  k   (softer $\\rightarrow$)')
ax.set_ylabel('deviation from the seamless\nshape, RMS interior (mm)')
ax.set_title('(a) softening is unbounded', fontsize=11, loc='left')

# ── (b) both sides on one signed per-length stiffness axis ──────────────────
# Soft side: the band loses (1-k) * E2 * w of course stiffness per unit length,
# w = band area / seam length.  Stiff side: the cable's EA is already an excess,
# spread over the same seam length it acts on - so both axes are N per metre of
# seam, divided by nothing, i.e. directly comparable as a total EA change.
ax = fig.add_subplot(gs[0, 1])
V0, F0 = read_off(os.path.join(REPO, 'data', '2part', '2part_opt_simu_m.off'))
tri = V0[F0]
area_tot = 0.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1).sum()
for name, col, _ in SEAMS:
    d = soft[soft.seam == name].sort_values('k', ascending=False)
    w = d.band_area_frac.iloc[0] * area_tot / SEAM_LEN[name]      # m, band width
    dEA = -(1.0 - d.k) * E2 * w                                   # N, deficit
    ax.plot(-dEA, d.L_pos_rms, '-', color=col, lw=1.9, marker='o', ms=3.2)
    s = stiff[stiff.seam == name].sort_values('EA')
    ax.plot(s.EA, s.L_pos_rms, '--', color=col, lw=1.4, marker='s', ms=2.6,
            alpha=0.75)
    print(f'{name}: band width {w*1000:.0f} mm, full deficit {-dEA.iloc[-1]:.0f} N')
ax.axhline(BASE_FIT_MEAN, color='k', lw=1, ls=':')
ax.axvspan(0.05, 400, color='0.88', zorder=0)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('magnitude of the seam’s stiffness change |ΔEA| (N)')
ax.set_ylabel('deviation from the seamless\nshape, RMS interior (mm)')
ax.set_title('(b) equal stiffness change, unequal effect', fontsize=11, loc='left')
ax.text(0.5, 300, 'solid: DEFICIT (soft band)\ndashed: EXCESS (stiff cable)\n'
        'shaded: plausible seam', fontsize=8, color='0.3')

# ── (c) how far the band itself opens ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
for name, col, _ in SEAMS:
    d = soft[soft.seam == name].sort_values('k', ascending=False)
    d = d[d.k < 1.0]                       # k = 1 is the baseline, growth is 0
    ax.plot(d.k, d.band_strain_max*100, '-', color=col, lw=1.9, marker='o', ms=3.2)
ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
ax.axhline(100, color='k', lw=1, ls=':')
ax.text(0.5, 130, 'band doubled in area', fontsize=8)
ax.set_xlabel('band modulus / fabric modulus  k')
ax.set_ylabel('largest area growth in the\nsoft band (%)')
ax.set_ylim(1, 1e8)
ax.set_title('(c) the slit opens without limit', fontsize=11, loc='left')

# ── (d,e) shape at a plausible weak seam, k = 0.1 ───────────────────────────
Vb, F = read_off(os.path.join(REPO, 'out', 'seam_baseline.off'))
for k, (fn, ttl) in enumerate([('seam_soft01_R01_R2.off', 'R0R1 | R2 band at k = 0.1'),
                               ('seam_soft01_R0_R1.off',  'R0 | R1 band at k = 0.1')]):
    Vr, _ = read_off(os.path.join(REPO, 'out', fn))
    dev = np.linalg.norm(Vr - Vb, axis=1) * 1e3
    ax = fig.add_subplot(gs[1, k], projection='3d')
    pc = Poly3DCollection([[Vr[i] for i in f] for f in F], edgecolor='none')
    pc.set_array(dev[F].mean(axis=1)); pc.set_cmap('magma'); pc.set_clim(0, 90)
    ax.add_collection3d(pc)
    ax.set_xlim(-.6,.6); ax.set_ylim(-.6,.6); ax.set_zlim(0,.55)
    ax.set_box_aspect((1.2,1.2,.6)); ax.view_init(elev=30, azim=-62); ax.set_axis_off()
    ax.set_title(f'({"de"[k]}) {ttl}\nmax {dev.max():.0f} mm', fontsize=10, y=0.97)
    if k == 1:
        cb = fig.colorbar(pc, ax=ax, shrink=0.55, pad=0.02)
        cb.set_label('displacement from the\nseamless shape (mm)', fontsize=8)

# ── legend and reading ──────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=2, marker='o', ms=4, label=l) for _, c, l in SEAMS]
ax.legend(handles=h, loc='upper left', frameon=False, fontsize=9)
ax.text(0, 0.60,
        'k scales BOTH moduli of every face on the seam ring, so\n'
        'the band gets weaker, not differently anisotropic. k = 0\n'
        'itself is not run: the element forms nu21 = nu E2/E1 and\n'
        'would divide by zero. k = 1e-6 is the slit in every sense\n'
        'that matters.\n\n'
        'The band is one face ring, 49 mm wide on a 5.35 m seam,\n'
        '17 % of the surface - wider than a knitted seam. So,\n'
        'like the rigid bound, it OVERSTATES the imperfection.\n\n'
        'The two sides are not symmetric. A rigid seam saturates:\n'
        '14.7 mm on the long seam, and no axial model can beat it.\n'
        'A weak seam does not saturate - the band takes the\n'
        'pressure it can no longer carry and balloons, so the\n'
        'answer runs off to metres. At an equal |dEA| of about\n'
        '400 N the deficit moves the shape 4.3x further than the\n'
        'excess: 17 mm against 3.9 mm (b).\n\n'
        'The short control seam is the exception: at 1.24 m and\n'
        '3.6 % of the surface it saturates near 40 mm, because\n'
        'the stiff fabric around it carries the load the band drops.\n'
        'Ballooning needs a long soft line, not just a soft one.',
        fontsize=8.2, va='top', transform=ax.transAxes, color='0.25')

out = os.path.join(HERE, 'figures', 'seam_soft.png')
plt.savefig(out, dpi=135, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', out)
