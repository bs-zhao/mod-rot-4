"""Metrics that passed the S9 angle test (every sweep's preferred angle within 15 deg of the oracle: argmin in {0,15,75,90}) in AT LEAST ONE of
C1,C2,C3,CR4 (NN-side, S9 repeat01, choice NPs) -> blind search of R on each of the four NPs (i01_logit_AD, R*-free, T only for evaluation).
Search budget: C1/C2/CR4 400 Haar starts / 8 refinements (as the original grid); C3 (d=6) 2000 / 30.
Outputs results/blind_all.csv (+ pass flags), results/found_rotations.npz, results/figures/tu_<model>.png
"""
import os, sys, textwrap, pickle
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "work")); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "models", "src"))); os.chdir(os.path.join(HERE, "work"))
import numpy as np, pandas as pd, multiprocessing as mp, grid as GR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
OUT = os.path.join(HERE, "results"); os.makedirs(OUT + "/figures", exist_ok=True); INF = "i01_logit_AD"; MODELS = ["C1", "C2", "C3", "CR4"]; LIM = {"C1": .2, "C2": .2, "C3": .2, "CR4": .3}


def task(t):
    inf, model, name = t; st = GR.state(inf, model); fn = GR._M[name]
    def f(Q):
        UX, UU = GR.uxuu(st, Q)
        try: return fn(UX, UU, st)
        except Exception: return 1e9
    kw = dict(nh=2000, nst=30) if model == "C3" else {}
    v, Q = GR.search(st, f, **kw); return t, v, Q


def tu(st, Q):
    d = st["d"]; C = np.abs(np.corrcoef(np.c_[st["Zs"] @ Q.T, st["Ts"]].T)[:d, d:]); r, c = linear_sum_assignment(-C); order = np.empty(d, int); order[c] = r
    return C[order, :].T


if __name__ == "__main__":
    GR.state(INF, "C1"); avail = set(k for k in GR._M if not k.startswith("_"))
    sel = [m for m in pd.read_csv(os.path.join(HERE, "selected_metrics.csv")).metric if m in avail]; print(len(sel), "metrics;", [m for m in pd.read_csv(os.path.join(HERE, "selected_metrics.csv")).metric if m not in avail], "not available in this metric set", flush=True)
    tasks = [(INF, m, n) for m in ["C3", "CR4", "C2", "C1"] for n in sel]; tasks = [(a, b, c) for a, b, c in tasks]
    with mp.Pool(10) as pool: res = pool.map(task, tasks, chunksize=1)
    pickle.dump(res, open(OUT + "/search_raw.pkl", "wb")); ceil = GR.ceilings(tuple(MODELS)); rec = []; Qs = {}
    for (t, v, Q) in res:
        st = GR.state(INF, t[1]); mcc, mn, lm, lx = GR.evalQ(st, Q); Qs[f"{t[1]}|{t[2]}"] = Q
        rec.append(dict(model=t[1], metric=t[2], mcc=mcc, min_match=mn, leak_mean=lm, leak_max=lx, err_deg=GR.errdeg(Q, st["R"]), f=v, ceiling=ceil[t[1]][0], gap=ceil[t[1]][0] - mcc, success=(ceil[t[1]][0] - mcc <= .05) and lx < LIM[t[1]]))
    R = pd.DataFrame(rec); R.to_csv(OUT + "/blind_all.csv", index=False); np.savez_compressed(OUT + "/found_rotations.npz", **Qs); print("searched", flush=True)
