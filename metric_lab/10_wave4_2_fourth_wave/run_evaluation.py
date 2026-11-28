"""EVALUATION-ONLY entry point.  Locks the job files of a stage, THEN opens T / R*.

usage: python run_evaluation.py --stage rep          REP-1/2/2C/3/4 -> results/rep_recovery.csv (+ rep_locks.csv, rotations, T-U matrices),
                                                       independent_reeval.csv (REP-4 flags + paired L_D(R_hat) - L_D(R*) on both D caches)
       python run_evaluation.py --stage measures     MG, MS, MIX -> results/measure_recovery.csv
       python run_evaluation.py --stage robustness   RB-Q, RB-MASK, RB-SEC -> results/robustness_recovery.csv
"""
import sys, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, PRIMARY, SECONDARY, CAND, prim_mask
import numpy as np, pandas as pd
from wave4.eval_only import recovery as RC
from wave4 import search as SE

STAGES = dict(rep=(("REP-1", "REP-2", "REP-2C", "REP-3", "REP-4"), "rep_recovery.csv", PRIMARY),
              measures=(("MG", "MS", "MIX"), "measure_recovery.csv", PRIMARY),
              robustness=(("RB-Q", "RB-MASK", "RB-SEC"), "robustness_recovery.csv", PRIMARY + SECONDARY))


def reeval_eval():
    from wave2.blind_inputs import load_payload
    from wave2.evaluate_recovery import EvalData
    fs = sorted(SE.BLIND.glob("REP-4/*/reeval.csv"))
    if not fs:
        return
    df = pd.concat([pd.read_csv(f) for f in fs]); out = []
    for toy, g in df.groupby("toy"):
        pl = load_payload(toy); ed = EvalData(toy); mask = prim_mask(pl)
        for v, gv in g.groupby("variant"):
            LD = {s: SE.objective(pl, CAND, "Q05", "D", s, mask, v)[0] for s in (211, 613)}; LC = {}
            for _, r in gv.iterrows():
                z = np.load(SE.job_path(r.batch, toy, CAND, "Q05")); R = z[f"{r.key}|R"]; Rs = ed.refs[v]["R_star"]
                if r.iseed not in LC:
                    LC[r.iseed] = SE.objective(pl, CAND, "Q05", "C", int(r.iseed), mask, v)[0]
                rr = r.to_dict(); rr["dL_C_vs_Rstar"] = LC[r.iseed](R) - LC[r.iseed](Rs)
                for s in LD:
                    rr[f"dL_D{s}_vs_Rstar"] = LD[s](R) - LD[s](Rs)
                out.append(rr)
    pd.DataFrame(out).to_csv(RES / "independent_reeval.csv", index=False); print("independent_reeval.csv", len(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); a = ap.parse_args()
    b, name, panel = STAGES[a.stage]; RC.evaluate(b, name, panel=panel)
    if a.stage == "rep":
        reeval_eval()
