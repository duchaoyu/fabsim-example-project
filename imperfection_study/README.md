# Geometric imperfection study

How far does the equilibrium shape move when the as-built structure differs from
the design? Blocks A–E work up from a method check on the circular dome to a
joint tolerance budget on the fabricated case study.

## Layout

| File | Role |
|---|---|
| `tolerances.py` | **The δ table.** Magnitude, error model (systematic / independent), provenance and measured-vs-estimated status for each parameter. |
| `imperfection_config.py` | Nominal working point, run tables, output list. |
| `fem_runner.py` | Thin wrapper on `fem_batch_sensitivity`. No cable machinery; exposes the per-face stretch field and stretch continuation the later blocks need. |
| `mesh_tools.py` | OFF read/write and the geometric perturbations (Block A needs the uniform in-plane rescale; non-uniform imperfections go here too). |
| `geometry.py` | The geometries and what each factor means on each — including the per-geometry reading of "boundary radius". |
| `fit_nominal.py` | Fits the nominal stretch factors where the rest mesh *is* the design target, so Block A perturbs about a working point. |
| `run_block_A.py` | The 13 runs, plus a numerical-reproducibility probe. |
| `analyse_block_A.py` | Sensitivity table, linearity check, spread budget, figure. |
| `plot_shape.py` | Deviation fields and sections — where the tolerance shows up on the surface. |

```bash
python3 fit_nominal.py   --geometry 2part            # only for a geometry with a target
python3 run_block_A.py   --geometry disc|2part --keep-verts
python3 analyse_block_A.py --geometry disc|2part --check-overlap [--measured-deviation-mm X]
python3 plot_shape.py    --geometry 2part            # where the deviation is, not just how much
```

13 runs take 2.2 s on the 399-vertex disc and ~60 s on the 2part, so a block is
cheap enough to re-run whenever a δ changes.

All distances are RMS over **interior** vertices. The boundary is clamped, so its
deviation is zero by construction and averaging it in would deflate every loss by
a factor set by nothing but boundary discretisation.

## Geometries and nominal points

**`disc`** — `circular_flat.off`, motif 1: s_wale = s_course = 1.10, θ = 0°,
p = 1000 Pa, E₁ = 5000 N/m, E₂/E₁ = 2.5014, ν = 0.198, R = 600 mm. The dome rises
164.2 mm (h/R = 0.27) at a max/mean stress ratio of 1.11 — inside the
model-validity box of the Chapter 6 sensitivity study, clear of both the
nearly-flat regime where the curvature estimators go noisy and the
compression-onset corner. Flat rest mesh, no design target of its own, so 𝓛_pos is
referenced to its own baseline equilibrium.

**`2part`** — `data/2part/2part_opt_simu_m.off`, 341 v / 634 f, same motif-1
material and p = 1000 Pa. Here the rest mesh *is* the shape the structure is meant
to hold, so the stretch factors are not free to be chosen — they are whatever puts
the inflated equilibrium on that shape, and `fit_nominal.py` fits them. Two things
came out of that fit and both matter:

- The **anisotropic** two-parameter fit saturates at *both* bounds
  (s_wale → 1.400, s_course → 0.950). A uniform pre-strain cannot reach this
  target. A nominal on a bound also cannot be perturbed symmetrically, so it is
  unusable for Block A — the ± responses would differ because one side was
  clipped, not because the physics is asymmetric. Recorded in
  `data/fit_2part_anisotropic_saturated.json`, not installed.
- The **isotropic** fit is interior: s = 1.1171, and that is the Block A nominal.
  It leaves 𝓛_target = 24.6 mm standing, before any imperfection.

The 1.043 in `src/membrane_orthotropic.cpp:141` is 41 mm high on crown and 47.8 mm
in 𝓛_target, so it is a starting point rather than a working point.

## Status of the tolerances

**All six δ in Block A are currently estimates.** `tolerances.py` records the
source each one is waiting on. δ_s is the one with no measurement behind it and
the largest effect by a factor of 2.7, so §6.2.3 is the single most valuable input
to this study.

Replacing an estimate does **not** require re-running, *provided* the response is
linear over the tolerance — which the block checks. On the disc the worst asymmetry
is 13.4% (E₁), under the 15% threshold, so responses rescale: halve δ_s and
s_course's contribution halves. On the 2part, E₁ reaches 17.5% and E₂/E₁ 15.2%, so
those two rows do **not** rescale safely and want a smaller-δ re-run once the
Chapter 4 scatter is known. The stretch-factor rows, at 1.5–2.8%, are solidly
linear on both geometries — which is the important case, since δ_s is the estimate
most likely to be revised.

## Block A on the disc

Crown-height response at one tolerance, sorted by effect:

| factor | δ | Δh(+δ) | Δh(−δ) | half-range | elasticity | asym |
|---|---|---|---|---|---|---|
| s_course | 0.05 | −26.45 mm | +28.31 mm | −27.38 mm | −3.67 | 6.8% |
| E₁ | 500 N/m | −9.64 | +11.02 | −10.33 | −0.63 | 13.4% |
| s_wale | 0.05 | −10.64 | +9.61 | −10.12 | −1.36 | 10.2% |
| E₂/E₁ | 0.250 | −6.71 | +7.39 | −7.05 | −0.43 | 9.6% |
| p | 50 Pa | +5.06 | −5.23 | +5.14 | +0.63 | 3.3% |
| R | 5 mm | +2.24 | −2.23 | +2.23 | +1.63 | 0.4% |

Predicted spread with all six at one tolerance: **32.2 mm RSS** (19.6% of the
crown height), 62.2 mm if every error aligns. This is the number §6.1.3's measured
deviation goes against; pass it in with `--measured-deviation-mm` and the script
states the verdict and its consequence for §6.5.2.

Three findings beyond the table:

1. **s_course dominates**, at 2.7× the next factor, because the course direction
   is the stiff one (E₂ = 2.5 E₁) — pre-strain applied along it does the most work.
   Combined with δ_s being unmeasured, this is where the study's uncertainty sits.

2. **R has the second-largest elasticity (1.63) but nearly the smallest effect**,
   purely because its tolerance is tight (0.83% against 4.5–10% for the others).
   It is a factor worth controlling precisely rather than one that does not matter.

3. **The six factors are nearly degenerate on this geometry** — the pairwise
   cosine between their displacement fields is 0.996 median, and p/E₁ sit at
   −1.000. They excite one and the same axisymmetric inflation mode, differing
   only in sign and amplitude. Two consequences: Block A cannot attribute a
   measured deviation to a factor, only bound its size (attribution needs the
   multi-region case study, where the fields separate); and the joint L_pos of
   Block D will be broad and right-skewed with a mean below the RSS, because
   near-parallel contributions add and cancel algebraically. The crown-height RSS
   is unaffected by this — for a scalar output RSS is correct whenever the factor
   errors are independent, whatever the response shape.

The numerical floor is below 0.01 µm: re-solving the baseline along a different
stretch-factor continuation path reproduces it to the 1e-8 m output precision. Every
response above is physical by five orders of magnitude.

## Block A on the 2part

Same 13 runs, about the fitted isotropic nominal (s = 1.1171, h_crown = 386.2 mm,
𝓛_target = 24.6 mm).

| factor | δ | Δh(+δ) | Δh(−δ) | half-range | elasticity | asym | 𝓛_pos |
|---|---|---|---|---|---|---|---|
| s_course | 0.05 | −20.99 mm | +21.31 mm | −21.15 mm | −1.22 | 1.5% | 16.26 mm |
| s_wale | 0.05 | −14.97 | +15.40 | −15.19 | −0.88 | 2.8% | 10.64 |
| E₁ | 500 N/m | −5.44 | +6.48 | −5.96 | −0.15 | 17.5% | 4.51 |
| p | 50 Pa | +2.94 | −2.98 | +2.96 | +0.15 | 1.3% | 2.24 |
| E₂/E₁ | 0.250 | −2.27 | +2.64 | −2.46 | −0.06 | 15.2% | 2.35 |
| R | 5 mm | +0.94 | −0.93 | +0.93 | +0.28 | 1.5% | 3.74 |

Predicted spread: **27.0 mm RSS** in crown height — but only 7.0% of h here against
19.6% on the disc, because the 2part is more than twice as tall. The ordering is the
same as the disc's with one swap (s_wale overtakes E₁), so the *dominant tolerance
does not change with geometry* — which is the Block E question, answered in the
affirmative for these two.

The 2part adds what the disc could not show, because it has a design target:

**The standing mismatch dominates the tolerances, and they are geometrically
orthogonal.** 𝓛_target is 24.6 mm at the nominal, before any imperfection. The
largest single tolerance (s_course) adds 4.9 mm, 20% of that. And the two are not
merely different in size but in *kind*: the cosine between the mismatch field and
the s_course displacement field is **+0.006** — orthogonal to three decimal places.

`figures/blockA_2part_shape.png` shows why. The mismatch is a narrow stripe along
x = 0 reaching +90 mm: the design target has a sharp crease between its two lobes,
and a uniformly pre-strained membrane inflates into a single smooth dome instead —
the section through the lobes has the target dipping to 292 mm at the valley while
the equilibrium runs flat across at 380–385 mm. The tolerance perturbation, by
contrast, is a broad smooth mode that lifts or drops the whole cap by ~20 mm and
does nothing at the valley.

So for the 2part: **no tightening of these tolerances moves the shape toward its
target.** The uniform two-parameter model cannot form the crease — which is exactly
what the saturated anisotropic fit was reporting — and the fix is a richer
parameterisation (per-region factors, or a tie along the valley), not better
fabrication. This is a useful thing for §6.5.2 to be able to say, and it is only
visible because the geometry carries a target and the fields were compared rather
than just their magnitudes.

Also worth noting: 𝓛_target is *stationary* in the stretch factors — both ±δ make
it worse, by 4.84 and 4.89 mm — because the nominal is a fitted minimum. Cost at an
optimum is second order, so a central difference there is ~0 and meaningless; the
analysis reports the increase instead. In the unfitted directions (E₁, p, R) one
sign does improve 𝓛_target, since those were never optimised.

## Incidental finding: the reference disc is not round

`circular_flat.off` has its boundary vertices between 597.8 and 600.0 mm
(std 0.58 mm) — a 2.2 mm out-of-round span, 22% of δ_R. The nominal baseline
therefore already carries an out-of-round imperfection comparable to the tolerance
being tested. A dedicated out-of-round block has to be measured against this
mesh's existing scatter, not against a perfect circle, and δ_R as tested here is
a *uniform* radius error only — out-of-round is a separate, non-uniform
imperfection.

## Still to build

- **Block B** (17 runs, case study): as A plus ν₁₂ and cable rest-length scale ρ,
  at the optimum x*. Needs the n-region binary (`fem_batch_nregion`) rather than
  this one, since the case study is multi-region with cables.
- **Block C** (60 runs): independent per-region / per-cable draws, 20 shared seeds
  across the three sets. `fem_runner.write_perface_sf` is already in place for the
  per-region field.
- **Block D** (200 runs): joint, systematic and independent together. Report mean,
  σ, p5, p95 of L_pos; compare σ against the RSS of B and C; report the count of
  draws clipped at the [0.95, 1.4] stretch-factor bounds.
- **Block E** (17 runs): Block B on the second geometry. Block A already gives a
  partial answer — the dominant tolerance is s_course on both the disc and the
  2part — so Block E is testing whether that survives cables and regions.

Two things from the proposal that are worth settling before Block B, since they
change what gets measured rather than how:

- §6.4 currently gives one scan, so the comparison would be a distribution
  against a point. Two specimens from the same knit programme, or one specimen
  re-scanned after re-inflation, would make it two-sided.
- ρ is quantised in reality (a half-turn of the turnbuckle is the resolution),
  not Gaussian. Block C and D should sample it on that lattice.
