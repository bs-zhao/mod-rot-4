"""UT |corr| heat maps + summary tables from results/blind_all.csv and found_rotations.npz.  usage: python plot.py"""
import os, sys, textwrap
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "work")); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "models", "src"))); os.chdir(os.path.join(HERE, "work"))
import numpy as np, pandas as pd, grid as GR
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run import tu, INF, OUT
from modrot4.model_specs import get_spec
R = pd.read_csv(OUT + "/blind_all.csv"); Q = np.load(OUT + "/found_rotations.npz")
for model, grp in R.groupby("model"):
    names = list(get_spec(model).parameter_names); st = GR.state(INF, model); grp = grp.sort_values("mcc", ascending=False); n = len(grp); nc = 10; nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.45 * nr + .6), squeeze=False)
    for a_, r in zip(ax.ravel(), grp.itertuples()):
        H = tu(st, Q[f"{model}|{r.metric}"]); a_.imshow(H, vmin=0, vmax=1, cmap="viridis")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]): a_.text(j, i, f"{H[i,j]:.1f}"[1:] if H[i, j] < .95 else "1", ha="center", va="center", fontsize=5, color="w" if H[i, j] < .6 else "k")
        a_.set_xticks([]); a_.set_yticks(range(len(names))); a_.set_yticklabels(names, fontsize=4.5)
        a_.set_title(f"{r.metric}\nMCC {r.mcc:.3f} leak {r.leak_mean:.2f}/{r.leak_max:.2f}", fontsize=5.5, color="green" if r.success else "black", fontweight="bold" if r.success else "normal")
        if r.success:
            for s in a_.spines.values(): s.set_edgecolor("lime"); s.set_linewidth(2.5)
    for a_ in ax.ravel()[n:]: a_.axis("off")
    ttl = f"{model}: |corr(T,U)| of the blind-search solution for the {n} metrics that passed the S9 angle test in >=1 database (sorted by MCC; green = blind-search success: MCC within .05 of ceiling {R[R.model==model].ceiling.iloc[0]:.3f} and leak max < {0.3 if model=='CR4' else 0.2}). Rows = true T, columns = matched U axes."
    fig.suptitle("\n".join(textwrap.wrap(ttl, 22 * nc)), fontsize=9); fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(f"{OUT}/figures/tu_{model}.png", dpi=110); plt.close(fig)
W = R.pivot(index="metric", columns="model", values=["mcc", "leak_max", "leak_mean", "success"]); W.columns = [f"{a}_{b}" for a, b in W.columns]
W["n_models_success"] = R.groupby("metric").success.sum(); W.sort_values(["n_models_success", "mcc_C3"], ascending=False).round(3).to_csv(OUT + "/summary_by_metric.csv"); print("ok")
