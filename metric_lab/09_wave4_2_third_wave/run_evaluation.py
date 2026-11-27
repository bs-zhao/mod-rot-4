"""EVALUATION-ONLY entry point (W3-05).  Locks the job files of a stage, THEN opens T / R*.

usage: python run_evaluation.py --stage pilot        RW-C, MX-A, MX-B, MX-SEL, ST-B, ST-E -> results/pilot_recovery.csv (+ locks, trace, rotations)
       python run_evaluation.py --stage finalists    FIN, FIN-D (S1 / D searches), FIN-SENS -> finalist_recovery.csv, finalist_reeval.csv,
                                                     finalist_sensitivity_recovery.csv
       python run_evaluation.py --pareto             W3-02 section 9.1 Pareto front of the E / X component and fusion finalists (seed 31) -> pareto_front.csv
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, PRIMARY, prim_mask
import numpy as np, pandas as pd
from wave3.eval_only import blind_eval as BE
from wave3 import jobs as JB, fusion as FU

PILOT = ("RW-C", "MX-A", "MX-B", "MX-SEL", "ST-B", "ST-E")


def pareto():
    import run_search as RS
    from wave2.blind_inputs import load_payload
    from wave2.evaluate_recovery import EvalData, recovery, supervised
    rows = []
    srcs = [("MX-A", "MX-B08c03@AD"), ("MX-A", "MX-SL0UXUU@RBF")] + [("MX-B", c) for c in RS.FUSIONS]
    files = [JB.job_path(b, t, c) for t in PRIMARY for b, c in srcs]; BE.lock([f for f in files if f.is_file()], "pareto_locks.csv")
    for toy in PRIMARY:
        pl = load_payload(toy); ed = EvalData(toy); mask = prim_mask(pl)
        for v in ("s9", "s9_2"):
            comp = RS.components(pl, v, 31, mask); calE = FU.calibrate(comp["E"][0], pl.k); calX = FU.calibrate(comp["X"][0], pl.k); sup = supervised(pl, ed, v); pts = []
            for b, c in srcs:
                f = JB.job_path(b, toy, c)
                if not f.is_file():
                    continue
                z = np.load(f)
                for ss in (31, 73, 101):
                    key = f"{v}|Q01|{mask}|i31|s{ss}|S0"
                    for i, Q in enumerate(z[f"{key}|finalQ"]):
                        pts.append(dict(src=c, sseed=ss, idx=i, Q=Q, LE=comp["E"][0](Q), LX=comp["X"][0](Q)))
            P = np.array([[p["LE"], p["LX"]] for p in pts]); front = FU.pareto(P)
            for p, fr in zip(pts, front):
                rec, _ = recovery(pl, ed, v, p["Q"], sup)
                rows.append(dict(toy=toy, variant=v, src=p["src"], sseed=p["sseed"], idx=p["idx"], L_E=p["LE"], L_X=p["LX"], z_E=(p["LE"] - calE["center"]) / calE["scale"],
                                 z_X=(p["LX"] - calX["center"]) / calX["scale"], on_front=bool(fr), MCC=rec["MCC"], leak_max=rec["leak_max"], pass_lab05=rec["pass_lab05"],
                                 dist_to_Rstar=rec["dist_to_Rstar"]))
        print(f"  pareto {toy}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "pareto_front.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage"); ap.add_argument("--pareto", action="store_true"); a = ap.parse_args()
    if a.stage == "pilot":
        BE.evaluate(PILOT, "pilot_recovery.csv", trace=True, panel=PRIMARY)
    elif a.stage == "finalists":
        BE.evaluate(("FIN", "FIN-D"), "finalist_recovery.csv", trace=False, panel=PRIMARY)
        fs = sorted(JB.BLIND.glob("FIN-D/*/*.reeval.csv"))
        if fs:
            pd.concat([pd.read_csv(f) for f in fs]).to_csv(RES / "finalist_reeval.csv", index=False)
        if any(JB.BLIND.glob("FIN-SENS/*/*.npz")):
            BE.evaluate(("FIN-SENS",), "finalist_sensitivity_recovery.csv", trace=False, panel=PRIMARY)
    if a.pareto:
        pareto()
