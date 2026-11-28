#!/usr/bin/env bash
# Blind development pipeline: moment caches (per toy, parallel) then every registered search batch (parallel jobs).  Resumable.
cd "$(dirname "$0")"; PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
LOG=../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/logs; mkdir -p $LOG; N=${1:-14}; shift || true
BATCHES=${@:-"REP-1 REP-2 REP-2C REP-3 MG MS MIX RB-Q RB-MASK RB-SEC"}
TOYS="C3 C3L C3#r2 C1 C2 CR3 CR4 CR5 C1L C2L CR3L CR4L CR5L C1#r2 C2#r2 CR3#b5e-5 CR4#r2"
printf "%s\n" $TOYS | xargs -P $N -I{} bash -c "$PY run_replication.py --caches --toy '{}' > '$LOG/cache_{}.log' 2>&1 || echo 'CACHE FAIL {}'"
echo CACHES-DONE
for b in $BATCHES; do
  $PY run_replication.py --jobs $b | tr -d '\r' | xargs -P $N -L 1 bash -c "$PY run_replication.py --batch \$0 --toy \"\$1\" --cid \$2 --Q \$3 > \"$LOG/\$0_\$1_\$2_\$3.log\" 2>&1 || echo \"JOB FAIL \$0 \$1 \$2 \$3\""
  echo "BATCH-DONE $b"
done
echo ALL-DONE
