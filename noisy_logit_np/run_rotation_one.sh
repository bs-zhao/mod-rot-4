#!/bin/bash
# full rotation test (all 10 inferences x 132 metrics) for ONE already-trained model, CPU only.  usage: ./run_rotation_one.sh CR4 [workers]
export KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; M=$1; W=${2:-6}
$PY extract_z.py $M 2>&1 | grep -v COMPAT
$PY rotation/infer.py $M 2>&1 | grep -E "done|Error|Traceback"
INFS=$($PY -c "import sys;sys.path.insert(0,'rotation');import grid;print(','.join(grid.INF))")
OUT=$PWD/rotation/grid_results.csv $PY rotation/grid.py $INFS $M $W 2>&1 | grep -E "tasks|ALLDONE|Error|Traceback"
echo ONE_DONE $M $(date +%T)
