#!/bin/bash
# W2-01 blind searches (BS-A, BS-B, BS-C, BS-E, then BS-D).  Finished job files are skipped.
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; L=../../outputs/metric_lab/08_wave4_2_second_wave/W2r1/logs; mkdir -p $L
PRIMARY="C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L"; SECONDARY="C1#r2 C2#r2 C3#r2 CR3#b5e-5 CR4#r2"
for b in ${BATCHES:-BS-A BS-B BS-C BS-E}; do
  P=$PRIMARY; [ $b = BS-E ] && P=$SECONDARY
  echo $P | tr ' ' '\n' | xargs -P 12 -I{} bash -c "$PY run_blind.py --batch $b --toy '{}' --caches > '$L/${b}_caches_{}.log' 2>&1"
  $PY run_blind.py --batch $b --jobs | tr -d '\r' | xargs -P ${NPAR:-18} -L1 bash -c "$PY run_blind.py --batch $b --toy \"\$0\" --metric \"\$1\" > \"$L/${b}_\$0_\$1.log\" 2>&1"
  echo "batch $b done $(date)"
done
if [ -z "$SKIP_D" ]; then
  echo $PRIMARY | tr ' ' '\n' | xargs -P 12 -I{} bash -c "$PY run_blind.py --batch BS-D --toy '{}' > '$L/BS-D_{}.log' 2>&1"
  echo "batch BS-D done $(date)"
fi
