"""Redraw the |corr(T,U)| figures from results/found_rotations.npz (no search).  usage: python replot.py"""
import os, sys, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "work")); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "models", "src"))); os.chdir(os.path.join(HERE, "work"))
import numpy as np, pandas as pd, grid as GR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run_heatmaps import tu, INF, OUT
from modrot4.model_specs import get_spec
R = pd.read_csv(OUT + "/successful_cells.csv"); Q = np.load(OUT + "/found_rotations.npz")
for model, grp in R.groupby("model"):
    names = list(get_spec(model.replace("r2", "")).parameter_names); st = GR.state(INF, model); grp = grp.sort_values("mcc", ascending=False); n = len(grp); nc = max(3, min(8, n)); nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(2.3 * nc, 2.5 * nr + .5), squeeze=False)
    for a_, r in zip(ax.ravel(), grp.itertuples()):
        H = tu(st, Q[f"{model}|{r.metric}"]); a_.imshow(H, vmin=0, vmax=1, cmap="viridis")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]): a_.text(j, i, f"{H[i,j]:.2f}"[1:] if H[i, j] < .995 else "1", ha="center", va="center", fontsize=5.5, color="w" if H[i, j] < .6 else "k")
        a_.set_xticks([]); a_.set_yticks(range(len(names))); a_.set_yticklabels(names, fontsize=5); a_.set_title(f"{r.metric}\nMCC {r.mcc:.3f} leak {r.leak_mean:.2f}/{r.leak_max:.2f}", fontsize=6)
    for a_ in ax.ravel()[n:]: a_.axis("off")
    ttl = f"{model}: |corr(T,U)| of the blind-search solution, all {n} successful metrics ({INF}). Supervised ceiling MCC {R[R.model==model].ceiling.iloc[0]:.3f}. Rows = true T, columns = matched U axes (diagonal = matched pairs), titles: MCC, leak mean/max"
    fig.suptitle("\n".join(textwrap.wrap(ttl, 17 * nc)), fontsize=9); fig.tight_layout(rect=(0, 0, 1, .95 if nr > 1 else .88)); fig.savefig(f"{OUT}/figures/tu_heatmaps_{model}.png", dpi=120); plt.close(fig)
print("ok")
