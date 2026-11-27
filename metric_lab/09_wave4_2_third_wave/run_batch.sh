#!/usr/bin/env bash
# usage: bash run_batch.sh <batch> [parallel=18] [panel=primary]   -- runs every "toy cid" job of a batch (resumable: finished jobs are skipped)
set -u
cd "$(dirname "$0")"
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
B=$1; N=${2:-18}; P=${3:-primary}
LOG=../../outputs/metric_lab/09_wave4_2_third_wave/W3r1/logs; mkdir -p "$LOG"
$PY run_search.py --batch "$B" --jobs --panel "$P" | tr -d '\r' > "$LOG/jobs_$B.txt"
echo "$(wc -l < "$LOG/jobs_$B.txt") jobs in $B"
cat "$LOG/jobs_$B.txt" | xargs -P "$N" -L 1 bash -c 'safe=$(echo "$1_$0" | tr "|@#" "__-"); '"$PY"' run_search.py --batch '"$B"' --toy "$0" --cid "$1" > "'"$LOG"'/'"$B"'_${safe}.log" 2>&1 || echo "FAILED $0 $1"'
echo "batch $B finished"
