"""EVALUATION-ONLY exact / practical Gap 1 global analytic search for W3 candidates (wave3/eval_only/oracle_global.py).
usage: python run_oracle.py --cid RW-ACT-b0.3 --target CR3:L:MASK-ALL      |  --jobs --cids a,b,c   |  --merge -> results/oracle_summary.csv"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES
import pandas as pd
from wave3 import jobs as JB
from wave3.eval_only import oracle_global as OGM

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cid"); ap.add_argument("--target"); ap.add_argument("--jobs", action="store_true"); ap.add_argument("--cids")
    ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.jobs:
        for c in a.cids.split(","):
            for t in OGM.TARGETS:
                print(f"{c} {t}")
    elif a.merge:
        pd.concat([pd.read_csv(f) for f in sorted(OGM.OG.glob("*.csv"))]).to_csv(RES / "oracle_summary.csv", index=False)
    else:
        import run_search as RS
        e = dict(spec=dict(type="fusion", fusion=RS.FUSIONS[a.cid]), surface="AD+RBF") if a.cid in RS.FUSIONS else JB.registry()[a.cid]
        if e["spec"]["type"] == "struct":
            raise SystemExit("struct candidates: analytic arm via run_local (grid) only")
        print(OGM.run(a.cid, e["spec"], a.target).T)
