#!/usr/bin/env bash
# Finish the fraction-convention re-run and regenerate everything downstream.
#
# Block 1 (ids 3000-4599) is already running when this starts, so wait it out
# before appending block 2 (ids 9000-12199) — step_generate reads the results CSV
# to append, and two writers would race.
#
# Cable samples cost two FEA solves each (the cable-free pass resolves
# L_rest = f * L_nocable); no-cable samples come from cache.
set -u
cd "$(dirname "$0")"
log() { echo "=== $(date '+%H:%M:%S') $*"; }

# Refuse to run twice.  Two instances raced once: both appended block 2, both
# wrote the same sample files, and 40 solver workers oversubscribed 20 cores.
# Nothing was corrupted that time, but only because the cache check let the
# second instance skip most of the first's work.
exec 9>/tmp/rerun_chain.lock
if ! flock -n 9; then
    echo "another rerun_chain.sh holds /tmp/rerun_chain.lock — refusing to start"
    exit 1
fi

while pgrep -f "run_material_r_sobol.py --steps generate" >/dev/null; do sleep 20; done
log "block 1 finished"
tail -3 rerun_frac_block1.log

# Abort on any stage failure.  The first run of this script continued past a
# failed surrogate fit and regenerated figW and the regime bars from the PREVIOUS
# surrogates — stale figures with fresh timestamps, which is worse than no
# figures at all.
run() { log "$*"; "$@" || { log "FAILED (rc=$?): $*"; exit 1; }; }

# The FEA and the section metrics are both current: results CSV has all 3912
# samples, and both section-metrics files were rebuilt under the fraction
# convention (cable 1924 rows / 1491 with curvature, no-cable 1988 / 1618).  So no
# --extend and no --force — this run only needs to refit the surrogates.
#
# Whenever the cable parameter naming changes, --force IS required: step_sections
# reuses a cached section-metrics CSV, and a cached file from the metre convention
# carries cable_*_lrest where the fit needs cable_*_frac, which kills the fit on a
# KeyError.
run python3 run_material_r_sobol.py --steps sections,sobol,plot --jobs 20

run python3 run_material_r_valid.py
run python3 plot_surrogate_validation.py

# run_sobol_robust WRITES the log1p tension tables that load_group prefers, so it
# has to precede every figure that reads them.  It was last here, so fig3 read
# yesterday's tables — still indexed by cable_*_lrest — and died on a KeyError.
run python3 run_sobol_robust.py
run python3 plot_material_r_sobol_combined.py
run python3 plot_sobol_regime_bars.py --cols 3
run python3 plot_sobol_regime_bars.py --cols 3 --split
run python3 plot_sobol_convergence.py
run python3 plot_material_r_regime.py
log "all done"

# Deliberately NOT re-run here, and why:
#   plot_lrest_sweep.py (figQ)  sweeps an ABSOLUTE L_rest in metres over
#       data/lrest_sweep/, generated with the old cable geometry and EA = 150 kN.
#       Superseded by probe_cable_influence.py / figP, which sweeps f instead.
#   plot_cable_analysis, plot_curvature_sensitivity, plot_section_*, plot_sf_*
#       read results.csv / material_results.csv — earlier studies whose cable data
#       predates the 2026-08-04 cable-path fix and is void regardless.
#   plot_knit_dir_*, plot_nu_surface, plot_sf_surface, plot_e1r_surface,
#       plot_uniform_sf read their own no-cable grids and are unaffected.
