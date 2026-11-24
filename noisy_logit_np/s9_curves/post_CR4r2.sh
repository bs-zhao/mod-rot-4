#!/bin/bash
# after CR4r2 (beta 5e-5) has finished training: extract Z, rebuild its sweeps, gate check, redo all analyses
export KMP_DUPLICATE_LIB_OK=TRUE CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")/.."
until grep -q "DONE best epoch" results/train_log_CR4r2_beta5e-5.txt; do sleep 20; done
$PY extract_z.py CR4r2 2>&1 | grep -v COMPAT
cd s9_curves; rm -rf results/raw/CR4/regNP/r2
$PY s9_regression.py CR4r2 2>&1 | grep -v COMPAT
$PY analyze_reg.py > results/analyze.log 2>&1; $PY angles_reg.py > results/angles.log 2>&1; $PY values_by_angle.py
echo POSTDONE
