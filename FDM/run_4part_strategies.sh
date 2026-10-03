#!/usr/bin/env bash
# 4-part: two strategies against the largest deviation of the case studies
# (base band bulging outwards).  All runs: motif 5, 1000 Pa, per-face knit field,
# warm start at the global (G) optimum.  Writes optimisation/4part_s_*_result.json.
#     bash FDM/run_4part_strategies.sh
cd "$(dirname "$0")"
PY=../.venv/bin/python
C="--motif 5 --face-knit --sf0-wale 1.001 --sf0-course 1.044 --scale0 0.999 --maxiter 150 --time-limit 7200"
L=optimisation/4part_calls; mkdir -p $L
run() { tag=$1; shift; $PY optimise_4part.py $C --tag 4part_s_$tag "$@" > $L/4part_s_$tag.log 2>&1 & }
run G       --n-az 1 --n-rad 1                                  # baseline, per-face knit
run ring85  --n-az 1 --n-rad 1 --ring 0.85                      # 1: 1 mm hoop cable
run ring85t --n-az 1 --n-rad 1 --ring 0.85 --ring-ea 9810       # 1: 0.5 mm hoop cable
run ring80  --n-az 1 --n-rad 1 --ring 0.80
run ring90  --n-az 1 --n-rad 1 --ring 0.90
run band75  --n-az 1 --n-rad 2 --band-edges 0.75                # 2: base band region, D4
run band75d2 --n-az 1 --n-rad 2 --band-edges 0.75 --sym d2      # 2: same, x/y mirror only
run band80d2 --n-az 1 --n-rad 2 --band-edges 0.80 --sym d2
run band3d2 --n-az 1 --n-rad 3 --band-edges 0.55 0.80 --sym d2  # 2: three bands
run combo   --n-az 1 --n-rad 2 --band-edges 0.75 --sym d2 --ring 0.85   # 1 + 2
wait
echo done

# second batch, after ring 0.90 led the first: rings nearer the base, and combinations
run ring93   --n-az 1 --n-rad 1 --ring 0.93
run ring95   --n-az 1 --n-rad 1 --ring 0.95
run ring90t  --n-az 1 --n-rad 1 --ring 0.90 --ring-ea 9810
run combo90  --n-az 1 --n-rad 2 --band-edges 0.75 --sym d2 --ring 0.90
run ring2    --n-az 1 --n-rad 1 --ring 0.80 0.93
wait
echo done2

# third batch: derivative-free (Powell) runs, and crest/valley sector regions
run pw_G        --method Powell --n-az 1 --n-rad 1
run pw_ring90   --method Powell --n-az 1 --n-rad 1 --ring 0.90
run pw_crest    --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5
run pw_crest15  --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 15
run pw_crestd2  --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5 --sym d2
run pw_crest1   --method Powell --n-az 2 --n-rad 1 --crest-hw 22.5
run pw_crest_r90 --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5 --ring 0.90
wait
echo done3

# fourth batch: the d2 crest runs again, now that unconverged solves are rejected
# (the first pw_crestd2 "optimum" was a failed solve returning the target)
run pw_crestd2     --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5 --sym d2
run pw_crestd2_r90 --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5 --sym d2 --ring 0.90
wait
echo done4

# fifth: d2 crest + ring warm-started at the d4 crest + ring optimum (a special case of it)
python3 - <<'PY'
import json
d = json.load(open('optimisation/4part_s_pw_crest_r90_result.json'))['p']
p = [d[s] for s in range(4) for g in range(2)] + [d[4+s] for s in range(4) for g in range(2)] + d[8:]
json.dump({"p": p}, open('optimisation/4part_s_pw_crestd2_r90_warm_p0.json', 'w'))
PY
run pw_crestd2_r90w --method Powell --n-az 2 --n-rad 2 --band-edges 0.75 --crest-hw 22.5 --sym d2 --ring 0.90 \
    --p0-json optimisation/4part_s_pw_crestd2_r90_warm_p0.json
wait
