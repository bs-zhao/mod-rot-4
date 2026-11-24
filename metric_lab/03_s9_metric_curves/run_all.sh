#!/bin/bash
# all saved NN databases (every outputs/nn/<MODEL>/<condition>/repeatNN with an S9 and/or S9_2 sweep); read-only on outputs/nn
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; ROOT=../../outputs/nn
for run in $(ls -d $ROOT/*/*/repeat*); do
  M=$(basename $(dirname $(dirname $run)))
  [ -d $run/s9_rotation_sweep ] || [ -d $run/s9_2_rotation_sweep ] || continue
  $PY run_database.py $M $run --n 4096 2>&1 | grep -v COMPAT
done
echo ALLDONE
