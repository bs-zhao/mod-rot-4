#!/usr/bin/env bash
# CONF-A follow-up (resumable): payloads for every finished NP run, then the locked blind rule + controls (parallel), then evaluation once all
# registered runs have blind outputs (or with --partial for a declared sub-panel).  Requires a verified W4 lock (run_confirmation.py checks).
cd "$(dirname "$0")"; PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
LOG=../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/logs; N=${1:-4}
$PY run_confirmation.py --payload > $LOG/conf_payload.log 2>&1 || { echo PAYLOAD FAIL; exit 1; }
ls ../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/confirmation/payload/ | sed 's/\.npz$//' | tr -d '\r' | \
  xargs -P $N -I{} bash -c "$PY run_confirmation.py --blind --toy '{}' > '$LOG/conf_blind_{}.log' 2>&1 || echo 'BLIND FAIL {}'"
echo CONF-BLIND-DONE
