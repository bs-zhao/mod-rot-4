"""|corr(T,U)| heat maps of the BLIND-SEARCH solutions of every SUCCESSFUL cell (inference i01_logit_AD, binary-choice NPs).

Success = MCC >= supervised ceiling - 0.05 and max leakage < 0.2 (C series) / 0.3 (CR series), taken from the saved blind grids
(metric_lab/02 big_grid grid_results.csv = NP repeat01, grid_results_r2.csv = NP repeat02).  The searched rotations were never saved,
so each successful cell is re-searched with the identical deterministic procedure (Haar start seed 4, 400 starts, 8 refinements) and the
re-computed MCC / leakage / f are checked against the saved grid value.   U = Zs Q^T, T standardised; T used only for this evaluation.
usage: python run_heatmaps.py
"""
import os, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "work"); sys.path.insert(0, W); os.chdir(W)
import numpy as np, pandas as pd, multiprocessing as mp
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "models", "src")))
import grid as GR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
BIG = os.path.join(HERE, "..", "02_blind_nn_rotation", "results", "big_grid"); OUT = os.path.join(HERE, "results"); os.makedirs(OUT + "/figures", exist_ok=True)
INF = "i01_logit_AD"; LIM = {"C": .2, "CR": .3}


def ser(m): return "CR" if m.startswith("CR") else "C"


def task(t):
    inf, model, name = t; st = GR.state(inf, model); fn = GR._M[name]
    def f(Q):
        UX, UU = GR.uxuu(st, Q)
        try: return fn(UX, UU, st)
        except Exception: return 1e9
    v, Q = GR.search(st, f); return t, v, Q


def successes():
    names = ["inf", "model", "metric", "mcc", "mn", "lmean", "lmax", "err", "f"]; rows = []
    for fn, suf in (("grid_results.csv", ""), ("grid_results_r2.csv", "r2")):
        g = pd.read_csv(os.path.join(BIG, fn), header=None, names=names); g = g[g.inf == INF]
        g["model"] = g.model.where(g.model.str.endswith("r2") | (suf == ""), g.model + "r2") if suf else g.model
        rows.append(g)
    g = pd.concat(rows, ignore_index=True); ceil = GR.ceilings(tuple(sorted(g.model.unique())))
    g["ceil"] = g.model.map(lambda m: ceil[m][0]); g["lim"] = g.model.map(lambda m: LIM[ser(m.replace("r2", ""))])
    return g[(g.mcc >= g.ceil - .05) & (g.lmax < g.lim)].reset_index(drop=True), ceil


def tu(st, Q):
    d = st["d"]; U = st["Zs"] @ Q.T; C = np.abs(np.corrcoef(np.c_[U, st["Ts"]].T)[:d, d:])   # rows U, cols T
    r, c = linear_sum_assignment(-C); order = np.empty(d, int); order[c] = r              # U axis matched to each T column
    return C[order, :].T                                                                      # rows T, columns matched U (diagonal = matches)


if __name__ == "__main__":
    S, ceil = successes(); print(S.groupby("model").size().to_string(), flush=True)
    GR.state(INF, "C1")
    tasks = [(INF, r.model, r.metric) for r in S.itertuples()]
    with mp.Pool(10) as pool: res = pool.map(task, tasks, chunksize=1)
    import pickle; pickle.dump(res, open(OUT + "/search_raw.pkl", "wb"))
    bad = 0; recs = []; Qs = {}
    from modrot4.model_specs import get_spec
    for (t, v, Q), r in zip(res, S.itertuples()):
        st = GR.state(INF, r.model); mcc, mn, lm, lx = GR.evalQ(st, Q); ok = abs(mcc - r.mcc) < 1e-6 and abs(lx - r.lmax) < 1e-6
        bad += (not ok); recs.append(dict(model=r.model, metric=r.metric, mcc=mcc, mn=mn, leak_mean=lm, leak_max=lx, err_deg=GR.errdeg(Q, st["R"]), f=v, saved_mcc=r.mcc, saved_leak_max=r.lmax, ceiling=r.ceil, matches_saved=ok))
        Qs[f"{r.model}|{r.metric}"] = Q
    R = pd.DataFrame(recs); R.to_csv(OUT + "/successful_cells.csv", index=False); np.savez_compressed(OUT + "/found_rotations.npz", **Qs)
    print("re-search reproduces saved MCC/leak in", int(R.matches_saved.sum()), "of", len(R), "cells", flush=True)
    for model, grp in R.groupby("model"):
        base = model.replace("r2", ""); names = list(get_spec(base).parameter_names); st = GR.state(INF, model); n = len(grp); nc = min(8, n); nr = int(np.ceil(n / nc))
        grp = grp.sort_values("mcc", ascending=False)
        fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr), squeeze=False)
        for a_, r in zip(ax.ravel(), grp.itertuples()):
            H = tu(st, Qs[f"{model}|{r.metric}"]); a_.imshow(H, vmin=0, vmax=1, cmap="viridis")
            for i in range(H.shape[0]):
                for j in range(H.shape[1]): a_.text(j, i, f"{H[i,j]:.1f}".lstrip("0") if H[i, j] < .995 else "1", ha="center", va="center", fontsize=5.5, color="w" if H[i, j] < .6 else "k")
            a_.set_xticks([]); a_.set_yticks(range(len(names))); a_.set_yticklabels(names, fontsize=5); a_.set_title(f"{r.metric}\nMCC {r.mcc:.3f} leak {r.leak_mean:.2f}/{r.leak_max:.2f}", fontsize=6)
        for a_ in ax.ravel()[n:]: a_.axis("off")
        fig.suptitle(f"{model}: |corr(T, U)| of the blind-search solution, every successful metric ({INF}); ceiling MCC {ceil[model][0]:.3f}, rows = true T, columns = matched U axes (diagonal = matches)", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(f"{OUT}/figures/tu_heatmaps_{model}.png", dpi=120); plt.close(fig)
    print("DONE", flush=True)
