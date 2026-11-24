#!/bin/bash
# LIGHT rotation test on already-trained models (CPU only, 4 workers): Z extraction -> 4 inferences -> 132-metric grid -> summary
# usage: ./run_rotation_light.sh C1 C2 C3
export KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"
MODELS="$@"; CSV=$(echo $MODELS | tr ' ' ',')
$PY extract_z.py $MODELS 2>&1 | grep -v COMPAT
$PY rotation/infer.py $MODELS 2>&1 | grep -E "done|Error|Traceback"
OUT=$PWD/rotation/grid_results.csv $PY rotation/grid.py i01_AD,i03_gaussZ,i05_FD02,i07_poly3 $CSV 4 2>&1 | grep -E "tasks|ALLDONE|Error|Traceback"
echo LIGHT_DONE $(date +%T)
