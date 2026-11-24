#!/bin/bash
# trains the five REPLICATE regression NPs (new subjects, new noise) one after another; resumable (re-run the same script after a crash)
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/d/Programs/anaconda_new/envs/py11/python
cd "$(dirname "$0")"
for m in C1r2 C2r2 C3r2 CR3r2 CR4r2; do
  if [ -f results/${m}_train_summary.json ] && [ ! -f checkpoints/${m}_resume.pt ] && grep -q '"minutes"' results/${m}_train_summary.json && [ "$(/d/Programs/anaconda_new/envs/py11/python -c "import json;print(json.load(open('results/${m}_train_summary.json'))['epochs_run']>10)")" = "True" ]; then echo "[$m] already trained, skipping"; continue; fi
  $PY -u train.py $m 2>&1 | grep --line-buffered -v -E "UserWarning|detach|sm \+="
done
echo ALLDONE $(date +%T)
