#!/usr/bin/env bash
# Span study for Section 7.5.2: the middle-crease shell (7.4.1) strategies D and E
# re-optimised at spans 0.6-6.0 m.  Geometry and cable EA scale with the span;
# pressure (1000 Pa) and material (structure I) are fixed.  Every span runs in
# parallel; D is seconds, E minutes to an hour.  E's outer cap is raised from 6
# to 20 so each span can reach "no face swapped".
#
#     bash FDM/run_2part_span.sh            # then: python3 FDM/figure_2part_span.py
set -u
cd "$(dirname "$0")/.."
OUT=FDM/optimisation/2part_span
BIN=build-linux
mkdir -p "$OUT"
SPANS="0.6 0.9 1.2 1.5 1.8 2.4 3.0 4.2 6.0"
for D in $SPANS; do
  tag=$(printf "%.1f" "$D" | tr . p)
  s=$(python3 -c "print($D/1.2)")
  (
    "$BIN"/best_fit_stretch_factors_cable "$s" "$OUT/D_$tag" > "$OUT/D_$tag.log" 2>&1
    HEADLESS=1 "$BIN"/best_fit_stretch_factors_3region_adaptive_cable "$s" "$OUT/E_$tag" 20 \
      > "$OUT/E_$tag.log" 2>&1
  ) &
done
wait
echo "span study done"
