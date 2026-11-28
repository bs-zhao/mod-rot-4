#!/usr/bin/env bash
# CONF-A data generation (CPU, all toys) then the 24 NP trainings, N parallel GPU workers (resumable)
cd "$(dirname "$0")"; PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
LOG=../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/logs; N=${1:-3}
for t in C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L; do $PY run_confirmation.py --generate --toy $t > $LOG/conf_gen_$t.log 2>&1 || echo "GEN FAIL $t"; done
$PY run_confirmation.py --train-jobs | tr -d '\r' | xargs -P $N -L 1 bash -c 'PY='$PY'; $PY run_confirmation.py --train --toy $0 --train-seed $1 > '$LOG'/conf_train_$0_$1.log 2>&1 && echo "TRAINED $0 $1" || echo "TRAIN FAIL $0 $1"'
echo CONF-TRAIN-ALL-DONE
