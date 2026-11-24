#!/bin/bash
# Independent validation of the PRE-REGISTERED candidate on replicate data/NPs (new subjects, new noise, newly trained NP). Nothing is tuned.
# candidate: inference i11_smooth03, metric zA(rowsp a=.75,q=.1) + zK(UXcol.a0.hoy), 4000 starts x 60 refinements x 3000 NM steps.   usage: ./validate_candidate.sh C1r2 C2r2 ...
export KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"
for M in "$@"; do
  $PY extract_z.py $M 2>&1 | grep -v COMPAT
  $PY rotation/infer.py $M 2>&1 | grep -E "Error|Traceback"
  (cd rotation && $PY c3_strong.py $M "i11_smooth03|rowsp.a0.75.q0.1+UXcol.a0.hoy" 2>&1 | grep -v COMPAT)
done
