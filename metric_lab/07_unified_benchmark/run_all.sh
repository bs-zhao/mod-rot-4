#!/bin/bash
# Wave 4-2 first wave, full pipeline (W1-00 .. W1-05).  Toys / targets run in parallel; finished files are skipped (--resume).
set -e
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"; L=../../outputs/metric_lab/07_unified_benchmark/logs; mkdir -p $L
PRIMARY="C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L"; SECONDARY="C1#r2 C2#r2 C3#r2 CR3#b5e-5 CR4#r2"
$PY -m pytest -q tests
$PY validate.py                                                        # W1-00 gate (exit 1 on any failed check)
par() { echo $2 | tr ' ' '\n' | xargs -P ${3:-6} -I{} bash -c "$PY $1 --toys '{}' --resume 2>&1 | grep -v COMPAT > '$L/{}_$4.log'"; }
par "run_baseline.py --batch A" "$PRIMARY" 6 A                           # W1-01
[ -f configs/priority_panel.json ] || $PY analyze.py --freeze-priority  # frozen once, before any W1-02/03/04 result
par "run_measures.py --batch B" "$PRIMARY" 6 B
par "run_measures.py --batch C" "$PRIMARY" 6 C
par "run_measures.py --batch E" "$SECONDARY" 5 E
[ -f configs/escalation_D.json ] || $PY analyze.py --freeze-escalation
par "run_measures.py --batch D" "$PRIMARY" 4 D
par "run_bridge.py" "$PRIMARY" 4 bridge && $PY run_bridge.py --merge       # W1-03
OMP_NUM_THREADS=1 $PY run_global_oracle.py --list | xargs -P 4 -I{} bash -c "OMP_NUM_THREADS=1 $PY run_global_oracle.py --target {} > '$L/global_{}.log' 2>&1"
$PY run_global_oracle.py --merge                                       # W1-04
$PY analyze.py --panel primary && $PY synthesize.py                    # W1-05
