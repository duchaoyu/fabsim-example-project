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
| `run_block_A.py` | The 13 runs, plus a numerical-reproducibility probe. |
| `analyse_block_A.py` | Sensitivity table, linearity check, spread budget, figure. |

```bash
python3 run_block_A.py --keep-verts
python3 analyse_block_A.py --check-overlap [--measured-deviation-mm X]
```

13 runs take 2.2 s on the 399-vertex disc, so the whole block is cheap enough to
re-run whenever a δ changes.

## Nominal point

`circular_flat.off`, motif 1: s_wale = s_course = 1.10, θ = 0°, p = 1000 Pa,
E₁ = 5000 N/m, E₂/E₁ = 2.5014, ν = 0.198, R = 600 mm. The dome rises 164.2 mm
(h/R = 0.27) with a max/mean stress ratio of 1.11 — inside the model-validity box
of the Chapter 6 sensitivity study, well clear of both the nearly-flat regime
where the curvature estimators go noisy and the compression-onset corner.

## Status of the tolerances

**All six δ in Block A are currently estimates.** `tolerances.py` records the
source each one is waiting on. δ_s is the one with no measurement behind it and
the largest effect by a factor of 2.7, so §6.2.3 is the single most valuable input
to this study.

Replacing an estimate does **not** require re-running: Block A reports the
δ-free elasticity and confirms the response is linear over the tolerance
(worst asymmetry 13.4%, below the 15% threshold), so the responses rescale
linearly. Halving δ_s halves its contribution.

## Block A results

Crown-height response at one tolerance, sorted by effect:

| factor | δ | Δh(+δ) | Δh(−δ) | half-range | elasticity | asym |
|---|---|---|---|---|---|---|
| s_course | 0.05 | −26.45 mm | +28.31 mm | −27.38 mm | −3.67 | 6.8% |
| E₁ | 500 N/m | −9.64 | +11.02 | −10.33 | −0.63 | 13.4% |
| s_wale | 0.05 | −10.64 | +9.61 | −10.12 | −1.36 | 10.2% |
| E₂/E₁ | 0.250 | −6.71 | +7.39 | −7.05 | −0.43 | 9.6% |
| p | 50 Pa | +5.06 | −5.23 | +5.14 | +0.63 | 3.3% |
| R | 5 mm | +2.23 | −2.22 | +2.22 | +1.63 | 0.4% |

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
- **Block E** (17 runs): Block B on the second geometry.

Two things from the proposal that are worth settling before Block B, since they
change what gets measured rather than how:

- §6.4 currently gives one scan, so the comparison would be a distribution
  against a point. Two specimens from the same knit programme, or one specimen
  re-scanned after re-inflation, would make it two-sided.
- ρ is quantised in reality (a half-turn of the turnbuckle is the resolution),
  not Gaussian. Block C and D should sample it on that lattice.
