#!/bin/bash
# all saved NN databases that have an S9 and/or S9_2 sweep (read-only on outputs/nn); 3 in parallel
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/d/Programs/anaconda_new/envs/py11/python; cd "$(dirname "$0")"; mkdir -p ../../outputs/metric_lab/06_wave1_screen/logs
ls -d ../../outputs/nn/*/*/repeat* | while read run; do
  [ -d $run/s9_rotation_sweep ] || [ -d $run/s9_2_rotation_sweep ] || continue
  M=$(basename $(dirname $(dirname $run))); tag=$(echo $run | sed 's#.*/nn/##; s#/#_#g')
  echo "$M $run $tag"
done | xargs -P 3 -L 1 bash -c 'PY=/d/Programs/anaconda_new/envs/py11/python; $PY run_db.py $0 $1 2>&1 | grep -v COMPAT > ../../outputs/metric_lab/06_wave1_screen/logs/$2.log'
echo ALLDONE
