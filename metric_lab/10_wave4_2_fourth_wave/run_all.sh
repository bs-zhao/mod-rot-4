#!/usr/bin/env bash
# Full fourth-wave pipeline in execution order (resumable; every step skips finished outputs).  Windows py11; run from this folder.
set -eu
cd "$(dirname "$0")"
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
TOYS="C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L"
# W4-00 gate: unit tests + input manifest
$PY validate.py
# W4-05 registration (before any confirmation output exists) + fresh data / NP training (GPU, background; reads nothing from development)
$PY run_confirmation.py --register
bash conf_train.sh 2 &
# W4-01 / 02 / 04 blind: caches then every registered batch (configs/replication_protocol.json)
bash run_batch.sh 14
for t in $TOYS; do $PY run_replication.py --reeval --toy $t; done                     # REP-4 (blind)
# evaluation (locks first, then truth)
$PY run_evaluation.py --stage rep; $PY run_replay.py
$PY run_evaluation.py --stage measures; $PY run_evaluation.py --stage robustness
for t in $TOYS; do $PY run_measure_diagnostics.py --toy $t; done; $PY run_measure_diagnostics.py --merge
for t in $TOYS; do $PY run_gap_evaluation.py --otrue --toy $t; done
$PY run_gap_evaluation.py --gap1-jobs | tr -d '\r' | while read c t; do $PY run_gap_evaluation.py --gap1 --cid "$c" --target "$t"; done
for t in $TOYS; do $PY run_gap_evaluation.py --gap2 --toy $t; done; $PY run_gap_evaluation.py --merge
$PY analyze.py --panel development; $PY figures.py
# W4-05 lock (development selection, see README) and CONF-A
$PY freeze_algorithm.py --candidate W4-Q05-v1 --deployment D-ranked && $PY freeze_algorithm.py --verify
wait
$PY run_confirmation.py --payload
for r in $(ls ../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1/confirmation/payload/ | sed 's/\.npz$//'); do $PY run_confirmation.py --blind --toy "$r"; done
$PY run_confirmation.py --evaluate; $PY analyze.py; $PY figures.py
