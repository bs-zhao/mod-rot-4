#!/bin/bash
# all saved NN databases (every outputs/nn/<MODEL>/<condition>/repeatNN with an S9 and/or S9_2 sweep): the 12 toys of
# toys.py (6 choice models, 6 <MODEL>L models) and the extra choice-model runs.  Read-only on outputs/nn.
# A run is skipped when every sweep it has (S9, S9_2) already has its output here; FORCE=1 recomputes everything.
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; ROOT=../../outputs/nn; OUT=../../outputs/metric_lab/03_s9_metric_curves
for run in $(ls -d $ROOT/*/*/repeat*); do
  M=$(basename $(dirname $(dirname $run))); rel=${run#$ROOT/}
  [ -d $run/s9_rotation_sweep ] || [ -d $run/s9_2_rotation_sweep ] || continue
  if [ -z "$FORCE" ]; then
    done_=1
    [ -d $run/s9_rotation_sweep ] && [ ! -f $OUT/$rel/s9/metrics_long.csv ] && done_=0
    [ -d $run/s9_2_rotation_sweep ] && [ ! -f $OUT/$rel/s9_2/metrics_long.csv ] && done_=0
    [ $done_ = 1 ] && { echo "skip $rel"; continue; }
  fi
  $PY run_database.py $M $run --n 4096 2>&1 | grep -v COMPAT
done
echo ALLDONE
