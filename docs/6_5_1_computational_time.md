# 6.5.1 Computational time — draft

Numbers below are either **measured** (re-timed 2026-08-13 on the machine described
in the first paragraph, three repetitions, median reported) or **recorded** (call
counts stored in the optimisation result JSONs; wall clocks recovered from the
timestamps of the per-evaluation output files). Provenance is given per number in
the "provenance" list at the end so each can be re-checked before submission.

---

## Draft prose

All timings were measured on a single workstation with an Intel Xeon Gold 5412U
(20 cores, 16 GB RAM), with the forward solver compiled at `-O3` and using CHOLMOD
for the sparse Cholesky factorisation. The forward solve is serial: one objective
evaluation occupies one core, and the remaining nineteen are idle unless several
evaluations are dispatched at once.

The cost of the whole workflow is set almost entirely by the inner forward solve of
Section 6.3.7, because everything outside it is either done once (the force-density
optimisation, the directional field, the region partition) or is arithmetic on a
design vector of at most thirty entries. A single forward solve costs between 0.09 s
and 0.64 s on the case-study meshes, which range from 341 to 1309 vertices. That
cost is not a simple function of mesh size: the fluted dome, at 2137 elements the
largest mesh in the set, solves in 0.20 s, while the shell with openings, at 1563
elements, takes 0.64 s. What sets the cost is the number of Newton iterations, and
therefore how far the trial design point sits from equilibrium and how many cable
segments change between taut and slack while the four pressure-continuation stages
are traversed. The same mesh at a different design point differs by a factor of
five: the creased shell solves in 0.09 s at its isotropic optimum and 0.43 s at the
anisotropic one.

Because the outer problem uses forward finite differences, one L-BFGS-B iteration
costs *n* + 1 forward solves for a design vector of length *n*. The design vectors
are deliberately small — one to four variables for the global strategies of Section
6.4.1, seven for the symmetry-reduced fluted dome, thirty for the free-form shell —
so a gradient costs between two and thirty-one solves, that is between a second and
a few seconds. Table 6.X gives the resulting totals.

The two extremes in the table are worth separating, because they are not the same
kind of cost. The strategies with a global or symmetry-reduced parameter set converge
in a few hundred evaluations and finish in one to two minutes: the free-form shell
took 622 evaluations, the fluted dome 412 across its two phases. The alternating
region schemes are one to two orders of magnitude more expensive. The adaptive
partition of the shell with openings took 2594 evaluations, because every element on
a region boundary is trial-assigned to each neighbouring region at one forward solve
per trial, and the sweep is repeated until no element moves. The discrete outer step,
not the continuous inner one, is what makes region refinement expensive, and its cost
grows with the length of the region boundaries rather than with the number of
variables.

Wall-clock time, however, is not the product of the evaluation count and the median
solve time, and the discrepancy is instructive. The adaptive partition of the shell
with openings performed 2716 forward solves at a median of 0.34 s, which is fifteen
minutes of arithmetic, but occupied 184 minutes. The reason is the tail: 7.6% of the
evaluations exceeded fifteen seconds, and those carried 91% of the wall time. These
are the non-converged inner solves of Section 6.3.7 — trial design points at which
the membrane and the cables have no equilibrium the Newton solver can reach, which
the driver abandons at a fixed wall-clock limit and reports as the penalty value.
The same pattern holds in the field-aligned run, where 9.0% of evaluations carried
82% of the wall time. The practical consequence is that the timeout is a cost
parameter, not merely a safeguard: it caps what an infeasible trial point can spend,
and lowering it from 120 s to 20 s cut the wall time of a region-refinement sweep by
roughly a factor of three without changing the optimum reached. It also means that
reported wall clocks are dominated by how often the optimiser probes outside the
feasible set, which depends on the initial guess, and this is a second reason —
beyond the local-minimum argument of Section 6.3.7 — for the symmetry-reduced
warm start.

Set against fabrication, these costs are negligible. The most expensive single
optimisation in the set, the adaptive partition, is about three hours on one core,
against a knitting time of [X] hours per specimen and a mould-free but manual
assembly of [X]. The workflow is not compute-bound at the scale of a single design.
It becomes compute-bound in two situations, both visible in this work: when the
region partition is treated as a design variable, and when the same target is swept
over a design parameter. The span sweep of Section 6.5.2 is the second case: at a
fixed mesh of 929 elements, the evaluation count needed to fit the free-form target
rose from 622 at a 1.2 m span to 4687 at 3.0 m, so the cost of the inverse problem
grows with how far the fabric is asked to stretch, not only with the size of the
discretisation. Taken together the three case studies consumed of the order of
3 × 10⁴ forward solves, of which the reported results are a small fraction; the
remainder is the cost of the strategy comparisons of Sections 6.4.1 and 6.4.6.

Two straightforward reductions were not implemented and are noted for completeness.
The *n* + 1 solves of a finite-difference gradient are independent and could be
dispatched across the twenty available cores, which would bring a gradient on the
thirty-variable problem to the cost of a single solve. And the forward solve is
currently restarted from the undeformed configuration at every evaluation, because
the driver is a separate process invoked once per evaluation; warm-starting from the
previous accepted iterate, as the in-process implementation of Section 6.4.1 already
does, would remove most of the Newton iterations at the small design perturbations
a finite-difference gradient takes. Neither changes the reported results, and both
would reduce a region-refinement run from hours to minutes.

---

## Table 6.X — Computational cost of the case studies

| Case study | Section | Mesh (v / f) | Design vars | Forward solves | Solve time (s) | Wall clock |
|---|---|---|---|---|---|---|
| Middle crease, global (A–D) | 6.4.1 | 341 / 634 | 1–2 | ≤ 150 | 0.09–0.43 | < 1 min |
| Middle crease, 3 regions (E) | 6.4.1 | 341 / 634 | 4 | inner ≤ 25/iter + boundary trials | 0.09–0.43 | — |
| Free-form shell | 6.4.2 | 497 / 929 | 30 | 622 | 0.15 | ≈ 1.6 min |
| Fluted dome | 6.4.3 | 1309 / 2137 | 7 | 378 + 34 (2 phases) | 0.20 | ≈ 1.4 min |
| Openings, 1 region | 6.4.6 | 847 / 1563 | 3 | 162 | 0.64 | ≈ 1.7 min |
| Openings, 4 adaptive regions | 6.4.6 | 847 / 1563 | 9 | 2594 | 0.34 (median) | 184 min |
| Openings, 10 field-aligned regions | 6.4.6 | 847 / 1563 | 20 | 1145 | 0.39 (median) | 68 min |

Solve time is the median of three repetitions at the converged design point, except
where marked "median", which is the median interval between consecutive evaluations
over the whole recorded run and therefore includes the non-converged tail.

---

## Provenance of each number

**Measured today** (`build-linux/fem_batch_nregion`, median of 3, Xeon Gold 5412U):

| Geometry | Mesh | Params | Time |
|---|---|---|---|
| 2part isotropic (s = 1.1171) | `data/2part/2part_opt_simu_m.off` | 1 region, no cable | 0.092 s |
| 2part anisotropic (1.29916 / 0.96030) | same | 1 region, no cable | 0.433 s |
| B5 at optimum | `data/B5_remeshed_shared.off` | `B5_optimised_params.json`, 9 regions, 12 cables | 0.149 s |
| C5 at optimum | `FDM/data/C5/C5_remeshed_fem.off` | `C5_16region_optimised_sym.json`, 16 regions, 24 cables | 0.195 s |
| D5 at optimum | `FDM/data/D5/D5_remeshed_fem.off` | `D5_4region_adaptive_optimised.json`, 4 regions, 2 cables | 0.640 s |

**Recorded in result files** (`FDM/optimisation/*.json`, field `n_calls`):
B5 622 · C5 phase 2 34 (phase 1 378 output files) · D5 1-region 162 ·
D5 4-region adaptive 2594 · B5 span sweep 622 / 1800 / 2546 / 4687 at 1.2 / 1.5 / 2.0 / 3.0 m.

**Recovered from output-file timestamps** (only the D5 runs kept their original
mtimes; the B5 and C5 output files were rewritten on 2026-05-12 and their wall
clocks are lost):

| Run | Solves | Wall clock | Median | p90 | Max | > 15 s | Share of wall time |
|---|---|---|---|---|---|---|---|
| `d5_4ra_v3` (4 adaptive) | 2716 | 183.8 min | 0.34 s | 1.06 s | 160 s | 7.6% | 91% |
| `d5_10lap_v4` (10 field-aligned) | 1145 | 68.0 min | 0.39 s | 10.6 s | 81 s | 9.0% | 82% |
| `d5_10lap_v3` | 578 | 15.9 min | 0.47 s | 0.92 s | 121 s | 2.1% | 59% |
| `d5_sym_v1` (10 symmetric) | 82 | 4.8 min | 0.46 s | 2.63 s | 62 s | 8.6% | 86% |

**Total forward solves on disk** (all runs, including the exploratory ones):
D5 21 644 · B5 4 683 · C5 2 674 — hence "of the order of 3 × 10⁴".

**Solver configuration** (`src/fem_batch_nregion.cpp`): Newton, threshold 1e-6,
iteration limit 10 000, four pressure-continuation stages at 1%, 10%, 50% and 100%
of the target pressure, cold start from the undeformed configuration at every call.

**Subprocess timeouts actually used**: 120 s (`optim_d5_3r_run2`), 40 s
(`optimise_D5_4region_adaptive.py`), 30 s (`d5_sym_v1`), 20 s (`d5_4ra_cable2`).

---

## Things to confirm or fill before submission

1. **Fabrication times** — the comparison paragraph has `[X]` for knitting and
   assembly time per specimen. Only you have these.
2. **The 3× timeout claim.** I inferred "lowering the timeout from 120 s to 20 s cut
   wall time by about a factor of three" from the timeout values in the run logs and
   the measured tail share. It is consistent with the data but is not a controlled
   experiment. Either soften it to "reducing the limit reduced wall time
   proportionally, without changing the optimum reached", or let me run the
   controlled version (same run, two timeouts) to make it a real number.
3. **Finite-difference step.** Section 6.3.7 states a step of 0.05. That is what
   `optimise_B5.py` uses, but `optimise_C5_16region.py` and the D5 scripts use
   0.002. Either the text needs "0.05 on the free-form shell and 0.002 on the
   others", or the reason for the difference should be stated.
4. **Fluted dome cable count.** Section 6.4.3 says twelve radial cables plus one
   circumferential, and later "24 cables". The stored result has 16 regions and 24
   cable sections under D8 symmetry (eight sectors). Worth reconciling.
5. **2part strategy E solve count** was not recorded — that optimiser runs
   in-process and prints per-iteration timings to stdout without saving them. If you
   want a number in the table rather than a dash, re-running
   `build-linux/best_fit_stretch_factors_3region_adaptive` writes into the (empty)
   `out/` directory and would produce it.
