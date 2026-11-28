"""W4-03 three-gap evaluation under labelled measures (EVALUATION-ONLY; wave4/eval_only/oracle.py).
usage: python run_gap_evaluation.py --otrue --toy CR4            O-true caches (C 31/73/101, D 211/613)
       python run_gap_evaluation.py --gap1 --cid <cid> --target CR4:L:FULL
       python run_gap_evaluation.py --gap1-jobs                   "cid target" lines
       python run_gap_evaluation.py --gap2 --toy CR4
       python run_gap_evaluation.py --merge                       -> results/oracle_global.csv, gap2_curves.csv.gz, gap2_summary.csv
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES
import pandas as pd
from wave4.eval_only import oracle as OR

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--otrue", action="store_true"); ap.add_argument("--gap1", action="store_true"); ap.add_argument("--gap1-jobs", action="store_true")
    ap.add_argument("--gap2", action="store_true"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--toy"); ap.add_argument("--cid"); ap.add_argument("--target"); a = ap.parse_args()
    if a.otrue:
        for fid, s in [("C", 31), ("C", 73), ("C", 101), ("D", 211), ("D", 613)]:
            OR.otrue(a.toy, fid, s); print(f"  otrue {a.toy} {fid} s{s}", flush=True)
    elif a.gap1_jobs:
        for c in OR.CIDS:
            for t in OR.TARGETS:
                print(c, t)
    elif a.gap1:
        print(OR.gap1(a.cid, a.target).T)
    elif a.gap2:
        print(len(OR.gap2(a.toy)), "gap2 rows")
    elif a.merge:
        fs = sorted(OR.OG.glob("*.csv"))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs]).to_csv(RES / "oracle_global.csv", index=False)
        fs = sorted((OR.OUT / "eval_only" / "gap2").glob("*.csv.gz"))
        if fs:
            df = pd.concat([pd.read_csv(f) for f in fs])
            bank = df.toy.isin(["CR3", "CR5", "CR3L", "CR5L"]); df["mask_role"] = "primary"
            df.loc[bank & (df["mask"] != "MASK-ALL"), "mask_role"] = "sensitivity"; df.to_csv(RES / "gap2_curves.csv.gz", index=False)
            g = df.groupby(["cid", "inferred", "seed", "mask_role", "mask", "variant"]).agg(n=("toy", "size"), O_perfect=("O_perfect", "sum"), O_marginal=("O_marginal", "sum"),
                                                                              I_perfect=("I_perfect", "sum"), I_marginal=("I_marginal", "sum"), both_marginal=("both_marginal", "sum"),
                                                                              pearson_med=("pearson", "median")).reset_index()
            g.to_csv(RES / "gap2_summary.csv", index=False)
        print("merged")
