#!/bin/bash
# after training: extract Z -> 10 inference moments -> 132-metric grid (parallel) -> summary -> heat-maps.   usage: ./run_rotation.sh
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"
$PY extract_z.py C1 C2 C3 CR3 CR4 2>&1 | grep -v COMPAT
$PY rotation/infer.py C1 C2 C3 CR3 CR4 2>&1 | grep -E "done|Error|Traceback"
INFS=$($PY -c "import sys;sys.path.insert(0,'rotation');import grid;print(','.join(grid.INF))")
OUT=$PWD/rotation/grid_results.csv $PY rotation/grid.py $INFS C1,C2,C3,CR3,CR4 14 2>&1 | grep -E "tasks|ALLDONE|Error|Traceback"
$PY rotation/analyze.py 2>&1 | grep -v COMPAT > /dev/null
$PY rotation/headline.py C1 C2 C3 CR3 CR4 2>&1 | grep -v COMPAT
echo ROTATION_DONE $(date +%T)
