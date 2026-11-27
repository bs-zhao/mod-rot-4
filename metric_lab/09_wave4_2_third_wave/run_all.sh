#!/usr/bin/env bash
# Full third-wave pipeline in execution order (resumable: every step skips finished outputs).  Windows py11; run from this folder.
set -eu
cd "$(dirname "$0")"
PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
TOYS="C1 C2 C3 CR3 CR4 CR5 C1L C2L C3L CR3L CR4L CR5L"
# blind preparation: product-grid gradient caches (W3-03)
for t in $TOYS; do $PY prepare.py --toy $t; done
# blind pilots
bash run_batch.sh MX-A 6
bash run_batch.sh RW-C 12
bash run_batch.sh ST-B 6
bash run_batch.sh MX-B 3
for t in $TOYS; do $PY run_search.py --batch MX-SEL --toy $t; done
for t in $TOYS; do $PY run_reduced.py --search --toy $t; done          # GPU
for t in $TOYS; do $PY run_context.py --encode --toy $t; done          # GPU
# W3-00 gate (needs MX-A for the search replay)
$PY validate.py --refit
# evaluation-only diagnostics and pilot evaluation (after locking)
$PY run_repair_diag.py
for t in $TOYS; do $PY run_local.py --toy $t; done; $PY run_local.py --merge
for t in $TOYS; do $PY run_reduced.py --local --toy $t; done; $PY run_reduced.py --merge
$PY run_evaluation.py --stage pilot --pareto
$PY run_oracle.py --jobs --cids E02,RW-ACT-b0.3,RW-ACT-b0.1,RW-GL-l0.75-a1.0,RW-GL-l0.5-a1.0,RW-POW-a0.75-c0.01,MX-F-eps0.1,MX-F-eps0.25,MX-F-w0.5-q9505 | tr -d '\r' | while read c t; do $PY run_oracle.py --cid "$c" --target "$t"; done
$PY run_oracle.py --merge
# preregistered escalation -> configs/finalists.json, then finalists (blind) and their evaluation
$PY escalate.py
bash run_batch.sh FIN 12
$PY run_fin_extra.py --jobs | tr -d '\r' | while read t c; do $PY run_fin_extra.py --reeval --toy $t --cid $c; $PY run_fin_extra.py --sens --toy $t --cid $c; done
$PY run_evaluation.py --stage finalists
# W3-04 disjoint-context searches / evaluation for E02 and the lead finalist(s)
for t in $TOYS; do for c in E02 $(cat results/context_cids.txt 2>/dev/null); do $PY run_context.py --search --toy $t --cid $c; done; $PY run_context.py --evaluate --toy $t; done
$PY run_context.py --merge
$PY analyze.py; $PY figures.py
# W3-06: freeze the development-selected algorithm (section 23 final rule; see README); confirmation (CONF-A) is not run in this wave
$PY freeze_algorithm.py --auto && $PY freeze_algorithm.py --verify
