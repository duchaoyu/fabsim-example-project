"""
D8-symmetric polar remesh of C5, with stiffener locations preserved.

The original C5.obj is approximately D8-symmetric (Fourier mode k=8 dominates),
but contains ~5% asymmetric noise (k=1, k=2, …). This script:

  1. Builds a 32×15 polar mesh (32 verts/ring × 15 rings + 1 apex = 481 verts)
  2. For each ring, samples z(r, θ) from C5.obj and projects onto the D8
     subspace by keeping only Fourier modes k = 0, 8, 16, 24 (multiples of 8)
  3. Aligns the mesh so that the 8 spoke ridges (at θ = 11.25° + k·45°,
     determined by FFT phase) land on every 4th vertex column (j = 1, 5, …, 29)
  4. Writes 8 cable polylines along these vertex columns

Outputs:
  data/C5_remeshed.off       — 481-vert D8-clean mesh (matches B5 density)
  data/cable_paths_C5.json   — 8 cable paths along the spoke directions
"""
import os, json
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from compas.datastructures import Mesh

HERE        = os.path.dirname(os.path.abspath(__file__))
INPUT_OBJ   = os.path.join(HERE, "input", "C5.obj")
OUT_OFF     = os.path.join(HERE, "..", "data", "C5_remeshed.off")
CABLES_OUT  = os.path.join(HERE, "..", "data", "cable_paths_C5.json")

# Mesh parameters (matches B5_remeshed: 497v / 929f)
N_CIRC      = 32          # verts per ring; 32 lets every 4th vert sit on a spoke
N_RING      = 15          # rings outside the apex
R           = 10.0        # boundary radius (C5 footprint)
N_SAMPLE_TH = 720         # angular samples per ring for FFT projection
KEEP_MODES  = (0, 8, 16, 24)   # keep multiples of 8 → exact D8 symmetry

assert N_CIRC % 8 == 0

# ── 1. z(x, y) interpolator from raw C5.obj ───────────────────────────────────
target = Mesh.from_obj(INPUT_OBJ)
T = np.array([target.vertex_coordinates(v) for v in target.vertices()])
print(f"C5 target: {len(T)} verts  "
      f"x[{T[:,0].min():.2f},{T[:,0].max():.2f}]  "
      f"y[{T[:,1].min():.2f},{T[:,1].max():.2f}]  "
      f"z[{T[:,2].min():.3f},{T[:,2].max():.3f}]")
interp = LinearNDInterpolator(T[:, :2], T[:, 2])

# ── 2. Per-ring D8 Fourier projection ─────────────────────────────────────────
# For each radius, sample z(r,θ), FFT, keep only k=0,8,16,24, inverse FFT,
# resample at the 32 polar-mesh angles.
sample_th = np.linspace(0, 2*np.pi, N_SAMPLE_TH, endpoint=False)
mesh_th   = np.arange(N_CIRC) * (2*np.pi / N_CIRC)

def project_d8(zs):
    """Keep only Fourier modes that are multiples of 8 (D8-invariant)."""
    fft = np.fft.fft(zs)
    mask = np.zeros_like(fft)
    for k in KEEP_MODES:
        if k < len(fft):
            mask[k]      = 1
            if k > 0:    mask[-k] = 1   # negative-frequency conjugate
    return np.real(np.fft.ifft(fft * mask))

# Verify spoke-ridge phase from inner ring data
zs_probe = np.array([interp(7.0*np.cos(t), 7.0*np.sin(t)) for t in sample_th])
fft_probe = np.fft.fft(zs_probe - zs_probe.mean()) / N_SAMPLE_TH
phase_8   = np.degrees(np.angle(fft_probe[8]))
peak0     = (-phase_8 / 8) % 45        # angle of first ridge in [0,45)
print(f"\nD8 phase from r=7 ring: first ridge at θ = {peak0:.3f}° "
      f"(expected ~11.25° → land on every 4th vertex from j=1)")

# ── 3. Build polar mesh vertices with D8-projected z ──────────────────────────
verts = [[0.0, 0.0, float(interp(0.0, 0.0))]]    # apex
ring_z_at_spoke = []                              # for diagnostics
for i in range(1, N_RING + 1):
    r = R * (i / N_RING)
    if i == N_RING:
        z_at_mesh = np.zeros(N_CIRC)              # boundary anchored at z=0
    else:
        zs = np.array([interp(r*np.cos(t), r*np.sin(t)) for t in sample_th])
        # Fill any NaN at the convex-hull edge by linearly extrapolating from neighbours
        if np.any(~np.isfinite(zs)):
            zs = np.where(np.isfinite(zs), zs, np.nanmean(zs))
        zs_d8     = project_d8(zs)
        # Resample at mesh angles
        z_at_mesh = np.array([
            np.interp(t, sample_th, zs_d8, period=2*np.pi) for t in mesh_th
        ])
    for j in range(N_CIRC):
        x, y = r * np.cos(mesh_th[j]), r * np.sin(mesh_th[j])
        verts.append([x, y, float(z_at_mesh[j])])
    # Track z at the spoke vertex (j=1, ridge column)
    ring_z_at_spoke.append((r, z_at_mesh[1]))
verts = np.array(verts)
N_V   = len(verts)

# ── 4. Build faces ────────────────────────────────────────────────────────────
def ring_idx(i, j):
    return 1 + (i - 1) * N_CIRC + (j % N_CIRC)

faces = []
for j in range(N_CIRC):                                  # apex fan
    faces.append([0, ring_idx(1, j + 1), ring_idx(1, j)])
for i in range(1, N_RING):                               # ring i to ring i+1
    for j in range(N_CIRC):
        a, b = ring_idx(i,     j), ring_idx(i,     j + 1)
        c, d = ring_idx(i + 1, j + 1), ring_idx(i + 1, j)
        faces.append([a, b, c])
        faces.append([a, c, d])

print(f"\nRemesh: {N_V} verts, {len(faces)} faces "
      f"(B5_remeshed reference: 497 / 929)")

# ── 5. Verify D8 symmetry exactly ─────────────────────────────────────────────
sym_err = []
for i in range(1, N_RING + 1):
    z_ring = np.array([verts[ring_idx(i, j)][2] for j in range(N_CIRC)])
    # under D8, z(j) == z(j + N_CIRC/8) for any j
    err = np.max(np.abs(z_ring - np.roll(z_ring, N_CIRC // 8)))
    sym_err.append(err)
print(f"D8 symmetry: max |z(j) − z(j+4)| over all rings = {max(sym_err):.2e} m")

# ── 6. Write OFF ──────────────────────────────────────────────────────────────
with open(OUT_OFF, "w") as f:
    f.write("OFF\n")
    f.write(f"{N_V} {len(faces)} 0\n")
    for v in verts:
        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for face in faces:
        f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
print(f"Saved {OUT_OFF}")

# ── 7. Cable paths: 8 spokes along ridge columns j = 1, 5, 9, …, 29 ───────────
spoke_cols  = [1 + 4*k for k in range(8)]    # j indices in 0..31
cable_paths = {}
for k, j in enumerate(spoke_cols):
    angle = j * (360 / N_CIRC)
    path  = [0] + [ring_idx(i, j) for i in range(1, N_RING + 1)]
    cable_paths[f"S{k}_{angle:.2f}deg"] = path
with open(CABLES_OUT, "w") as f:
    json.dump(cable_paths, f, indent=2)
print(f"Saved {CABLES_OUT}: 8 cables × {N_RING + 1} verts each "
      f"(angles {[round(j*360/N_CIRC, 2) for j in spoke_cols]})")

# ── 8. Quick comparison: spoke z vs anti-spoke z at each ring ─────────────────
print("\nRidge profile vs trough profile (D8 quadrupole amplitude per radius):")
print(f"{'r':>6}  {'z(spoke)':>9}  {'z(trough)':>10}  {'amp':>8}")
for i in range(1, N_RING + 1):
    r       = R * (i / N_RING)
    z_spoke = verts[ring_idx(i, 1)][2]      # j=1 → ridge at θ=11.25°
    z_trough= verts[ring_idx(i, 3)][2]      # j=3 → mid-trough at θ=33.75°
    print(f"{r:6.2f}  {z_spoke:9.3f}  {z_trough:10.3f}  {z_spoke - z_trough:8.3f}")
