"""Is the region boundary a special place for the rest shape, and what do short
rows have to relieve there?

Panel (a): the incompatibility strain the anisotropic rest-shape least squares
carries per face, against distance to the seam and to the clamped ring.
Panel (b): the row-count mismatch across the seam, and what quantising the wale
stretch-factor ratio to 7/6 does to it.
"""
import os, collections
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

d = pd.read_csv(os.path.join(HERE, 'data', 'seam_rest_incompatibility.csv'))
d['dmin'] = d[['dist_to_seam_M', 'dist_to_seam_C']].min(axis=1)
d['amp']  = d[['strain_max', 'strain_min']].abs().max(axis=1) * 100

def read_off(p):
    L = open(p).read().split('\n')
    nv, nf = int(L[1].split()[0]), int(L[1].split()[1])
    V = np.array([[float(x) for x in L[2+i].split()] for i in range(nv)])
    F = np.array([[int(x) for x in L[2+nv+i].split()[1:4]] for i in range(nf)])
    return V, F
V, F = read_off(os.path.join(REPO, 'data', '2part', '2part_opt_simu_m.off'))
e2f = collections.defaultdict(list)
for fi, (a, b, c) in enumerate(F):
    for e in [(a,b), (b,c), (c,a)]: e2f[tuple(sorted(e))].append(fi)
bv = sorted({v for e, fs in e2f.items() if len(fs) == 1 for v in e})
C  = V[F].mean(axis=1)
d['dbdr'] = np.min(np.linalg.norm(C[:,None,:] - V[bv][None,:,:], axis=2), axis=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={'width_ratios':[1.15,0.8,1.25]})

# ── (a) strain vs distance to seam, coloured by distance to the clamped ring ─
ax = axes[0]
sc = ax.scatter(d.dmin*1000, d.amp, c=d.dbdr*1000, s=13, cmap='viridis', alpha=0.85)
ax.set_xlabel('distance to the nearest region seam (mm)')
ax.set_ylabel('rest-shape incompatibility strain (%)')
ax.set_title('(a) the seam is not a special place', fontsize=11, loc='left')
cb = fig.colorbar(sc, ax=ax); cb.set_label('distance to the clamped ring (mm)', fontsize=8.5)
ax.text(0.97, 0.96, 'r(strain, dist to seam)  = +0.34\nr(strain, dist to ring)  = -0.69',
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='0.25')

# ── (b) the controlled comparison ───────────────────────────────────────────
ax = axes[1]
far = d.dbdr > 0.08
vals = [d[far & (d.dmin < 0.05)].amp.mean(), d[far & (d.dmin > 0.12)].amp.mean()]
ax.bar(['within 50 mm\nof a seam', 'more than 120 mm\nfrom any seam'], vals,
       color=['#4C78A8', '#BAB0AC'], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)
ax.set_ylabel('mean incompatibility strain (%)')
ax.set_ylim(0, 22)
ax.set_title('(b) matched for ring distance:\n     ratio 0.94, no seam effect',
             fontsize=11, loc='left')

# ── (c) the short-row lattice ───────────────────────────────────────────────
ax = axes[2]
sf1_A, sf1_B = 1.25722, 1.47318          # wale stretch factors, R0R1 and R2
ratio = sf1_B / sf1_A                     # rest row-height ratio
period = 1.0 / (ratio - 1.0)              # courses of R0R1 per extra course in R2
n = np.linspace(0, 36, 2000)
for r, lbl, col in [(ratio, f'as fitted, {ratio:.4f}', '#E45756'),
                    (7/6,   'locked to 7/6 = 1.16667', '#54A24B')]:
    need  = n * (r - 1.0)                 # extra courses R2 needs after n courses
    resid = need - np.round(need)         # whole rows only
    ax.plot(n, resid, color=col, lw=1.6, label=lbl)
ax.axhline(0, color='k', lw=0.8, ls=':')
for y in (0.5, -0.5):
    ax.axhline(y, color='0.6', lw=0.8, ls='--')
for x in range(0, 37, 6):
    ax.axvline(x, color='0.9', lw=0.8, zorder=0)
ax.set_xlabel('courses along the seam')
ax.set_ylabel('residual mismatch (courses)')
ax.set_ylim(-0.85, 0.85)
ax.set_title('(c) short rows are quantised', fontsize=11, loc='left')
ax.legend(fontsize=8.5, frameon=False, loc='lower right', ncol=2)
ax.text(0.02, 0.97,
        f'R2 needs one extra course every {period:.1f} of R0R1.\n'
        'Whole rows only, so the residual sawtooths to\n'
        f'+/-0.5 course either way = +/-{50/period:.1f}% local wale strain.\n'
        'Locking the ratio to 7/6 does not shrink that,\n'
        'it makes it exactly periodic: the pattern returns\n'
        'to zero every 6 courses (grid lines), so one\n'
        '6-course block repeats along the whole seam.',
        transform=ax.transAxes, ha='left', va='top', fontsize=7.8, color='0.25')

plt.tight_layout()
out = os.path.join(HERE, 'figures', 'seam_shortrow.png')
plt.savefig(out, dpi=135, bbox_inches='tight')
plt.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
print('wrote', out)
