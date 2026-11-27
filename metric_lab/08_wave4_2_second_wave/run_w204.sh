#!/bin/bash
# W2-04 inference batches IF-A, IF-B, IF-C on the 12 primary toys (finished toys are skipped)
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; L=../../outputs/metric_lab/08_wave4_2_second_wave/W2r1/logs
for b in ${BATCHES:-IF-A IF-B IF-C}; do
  echo ${TOYS:-C3 C3L CR3 CR3L CR5 CR5L CR4 CR4L C1 C2 C1L C2L} | tr ' ' '\n' | xargs -P ${NPAR:-6} -I{} bash -c "$PY run_inference.py --batch $b --toy {} 2>&1 | grep -v COMPAT > $L/inf_${b}_{}.log"
  echo "$b done $(date)"
done
