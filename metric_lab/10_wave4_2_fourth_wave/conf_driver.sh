#!/usr/bin/env bash
# CONF-A driver: every 15 min, payloads + locked blind rule for newly trained NPs (conf_followup.sh).  When all registered runs have blind
# outputs: hash every confirmation blind file, THEN evaluate (run_confirmation.py --evaluate), analyze, figures; exit 0.
# Exit 2 if the training queue has stopped with runs missing (they stay failures in the denominator until resolved).
cd "$(dirname "$0")"; PY=${PY:-/d/Programs/anaconda_new/envs/py11/python}; export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
O=../../outputs/metric_lab/10_wave4_2_fourth_wave/W4r1; LOG=$O/logs
until grep -q CONF-BLIND-DONE $LOG/conf_followup_1.log 2>/dev/null; do sleep 30; done
i=2
while true; do
  bash conf_followup.sh 3 > $LOG/conf_followup_$i.log 2>&1; i=$((i+1))
  ntr=$(ls $O/confirmation/runs/*.json 2>/dev/null | wc -l); nbl=$(ls $O/blind/CONF-A/*/deployment.json 2>/dev/null | wc -l)
  echo "$(date +%T) trained=$ntr blind=$nbl"
  if [ "$nbl" -ge 24 ]; then
    $PY run_confirmation.py --evaluate > $LOG/conf_evaluate.log 2>&1 && $PY analyze.py > /dev/null 2>&1 && $PY figures.py > /dev/null 2>&1; echo CONF-EVALUATED; exit 0
  fi
  ntp=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_confirmation.py --train' } | Measure-Object).Count" | tr -d '\r')
  if [ "$ntp" = "0" ] && [ "$ntr" -lt 24 ]; then echo "STALLED: training queue idle with $ntr/24 runs trained"; exit 2; fi
  sleep 900
done
