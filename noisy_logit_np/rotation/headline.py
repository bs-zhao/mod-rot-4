"""Headline metrics per model with heat-maps of |corr(U,T)| after rotation (rows = U axes reordered to match T; diagonal = matched).
usage: python headline.py MODEL [MODEL ...]   (uses inference i01_AD; panels: supervised ceiling, A rowsp, B global UX 1-Hoyer, F1=A+B, F2=zA+zB, F3=zA+zB+zC)
"""
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import grid
from common import RESULTS
grid._M = grid.MET.build()
PANELS = [("A rowsp (a=.5,q=.1)", "rowsp.a0.5.q0.1"), ("B global UX 1-Hoyer", "UX.hoy"), ("column entropy UXcol.a0.ent", "UXcol.a0.ent"), ("F3 = zA+zB+zC", "Fz.A+B+C")]
INF = sys.argv[1] if False else "i01_AD"
for model in sys.argv[1:]:
    st = grid.state(INF, model); d = st["d"]; R = st["R"]; mt = np.load(Path(grid.HERE) / f"meta_{model}.npz")
    def corrmat(Q):
        U = st["Zs"] @ Q.T; C = np.abs(np.corrcoef(np.c_[U, st["Ts"]].T)[:d, d:]); r, c = linear_sum_assignment(-C)
        return C[[r[list(c).index(j)] for j in range(d)]]
    res = [("supervised ceiling (R*)", R), ("UNROTATED (Q = I)", np.eye(d))]
    for title, name in PANELS:
        fn = grid._M[name]
        def f(Q, fn=fn):
            UX, UU = grid.uxuu(st, Q)
            try: return fn(UX, UU, st)
            except Exception: return 1e9
        v, Q = grid.search(st, f, nh=800, nst=16, maxiter=1000); res.append((title, Q))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, (title, Q) in zip(axs.ravel(), res):
        M = corrmat(Q); mcc, mn, lm, lx = grid.evalQ(st, Q); im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        for i in range(d):
            for j in range(d): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i, j] < .6 else "k", fontsize=9)
        ax.set_xticks(range(d)); ax.set_xticklabels([f"T{j+1}" for j in range(d)]); ax.set_yticks(range(d)); ax.set_yticklabels([f"U{i+1}" for i in range(d)])
        ax.set_title(f"{title}\nMCC {mcc:.3f} leak {lm:.3f}/{lx:.3f}", fontsize=10)
    fig.suptitle(f"{model} (noisy-logit NP, MSE): |corr(U,T)| after rotation, inference {INF}", fontsize=12); fig.colorbar(im, ax=axs, shrink=.7)
    fig.savefig(RESULTS / f"heatmap_{model}.png", dpi=105, bbox_inches="tight"); plt.close(fig); print("saved", model, flush=True)
