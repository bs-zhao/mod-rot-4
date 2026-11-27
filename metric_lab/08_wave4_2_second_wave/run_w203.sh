#!/bin/bash
# W2-03 conditional HE: CH-A/CH-B/CH-C configurations at pilot seed 31 for the 12 primary toys
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; L=../../outputs/metric_lab/08_wave4_2_second_wave/W2r1/logs
FID=${FID:-pilot}; SEED=${SEED:-31}
echo ${TOYS:-C3 C3L CR3 CR3L CR5 CR5L C1 C2 CR4 C1L C2L CR4L} | tr ' ' '\n' | xargs -P ${NPAR:-6} -I{} bash -c "$PY run_conditional.py --toy {} --fid $FID --seed $SEED 2>&1 | grep -v COMPAT > $L/cond_${FID}_${SEED}_{}.log"
echo "W2-03 $FID $SEED done $(date)"
