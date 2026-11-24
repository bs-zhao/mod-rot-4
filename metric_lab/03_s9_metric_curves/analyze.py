"""Curves + oracle(analytic)-vs-NN correlations for every saved database.   usage: python analyze.py

Reads outputs/metric_lab/03_s9_metric_curves/<MODEL>/<condition>/<repeat>/<variant>/{rotations,metrics_long}.csv.
Writes   results/correlations_long.csv   (database x metric: pooled Pearson/Spearman, mean per-curve Pearson)
         results/correlation_summary.csv (per metric over databases), results/key_metrics_table.md, results/figures/*
Sample point = one saved rotation (baseline R* once + every sweep rotation with angle > 0; the 0-degree sweep entries are
duplicates of R*).  "curve" correlation = Pearson over the 7 angles (0..90) of one axis subset, averaged over subsets.
"""
import sys, csv
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "metric_lab" / "03_s9_metric_curves"; RES = HERE / "results"; FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
KEY = ["rowsp.a0.75.q0.1", "UXcol.a0.hoy", "CAND.rowsp.75+UXcol.hoy", "UX.hoy", "UX.ent", "UXcol.a0.ent", "POOL.hoy", "UXrow.a0.5.ent",
       "S.udshare", "UUdiag.hoy", "Fz.A+B+C", "SL0.UX.e0.1"]


def short(cond): return cond.replace("beta", "b").replace("_lambda0_ctx0p25to0p8", "")[:22]


def load():
    dbs = {}
    for var in ("s9", "s9_2"):
        for f in sorted(OUT.glob(f"*/*/*/{var}/metrics_long.csv")):
            model, cond, rep = f.parents[3].name, f.parents[2].name, f.parents[1].name
            rot = pd.read_csv(f.parent / "rotations.csv"); met = pd.read_csv(f)
            key = f"{model}|{short(cond)}|{rep}|{var}"
            o = met[met.source == "oracle"].set_index("rotation").loc[rot.rotation].drop(columns="source"); n = met[met.source == "nn"].set_index("rotation").loc[rot.rotation].drop(columns="source")
            dbs[key] = dict(rot=rot, oracle=o.reset_index(drop=True), nn=n.reset_index(drop=True), model=model, var=var)
    return dbs


def pr(a, b):
    if np.std(a) < 1e-14 or np.std(b) < 1e-14 or not np.isfinite(a).all() or not np.isfinite(b).all(): return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def analyse(dbs):
    rows = []
    for key, d in dbs.items():
        rot = d["rot"]; uniq = ((rot.candidate_type == "baseline_unswept") | (rot.angle_deg > 0)).values
        for m in d["oracle"].columns:
            o, n = d["oracle"][m].values, d["nn"][m].values
            sp = spearmanr(o[uniq], n[uniq])[0] if (np.std(o[uniq]) > 1e-14 and np.std(n[uniq]) > 1e-14) else np.nan
            cs = []
            for ax, g in rot[rot.candidate_type == "sweep"].groupby("axes_1based"):
                g = g.sort_values("angle_deg"); cs.append(pr(o[g.index], n[g.index]))
            rows.append(dict(db=key, model=d["model"], variant=d["var"], metric=m, n=int(uniq.sum()), pearson=pr(o[uniq], n[uniq]), spearman=sp,
                             curve_pearson_mean=float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan, curve_pearson_min=float(np.nanmin(cs)) if np.isfinite(cs).any() else np.nan))
    return pd.DataFrame(rows)


def curves_fig(key, d, metrics, path):
    rot = d["rot"]; sw = rot[rot.candidate_type == "sweep"]; ks = sorted(sw.k.unique()); cmap = plt.cm.viridis(np.linspace(0, .85, len(ks)))
    nc = 4; nr = int(np.ceil(len(metrics) / nc)); fig, ax = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.2 * nr), squeeze=False)
    for a_, m in zip(ax.ravel(), metrics):
        zs = {}
        for s in ("oracle", "nn"):
            v = d[s][m].values; zs[s] = (v - v.mean()) / (v.std() + 1e-14)
        for k, c in zip(ks, cmap):
            gk = sw[sw.k == k]; angs = sorted(gk.angle_deg.unique())
            for s, ls in (("oracle", "-"), ("nn", "--")):
                for _, g in gk.groupby("axes_1based"): a_.plot(g.angle_deg, zs[s][g.index], ls, color=c, alpha=.18, lw=.8)
                mean = [zs[s][gk[gk.angle_deg == t].index].mean() for t in angs]
                a_.plot(angs, mean, ls, color=c, lw=2, label=f"k={k} {'analytic' if s=='oracle' else 'NN'}")
        pe = d["_corr"].get(m, (np.nan, np.nan)); a_.set_title(f"{m}\npooled r={pe[0]:.2f}  curve r={pe[1]:.2f}", fontsize=9); a_.set_xlabel("angle (deg)"); a_.set_xticks(range(0, 91, 30)); a_.grid(alpha=.3)
    for a_ in ax.ravel()[len(metrics):]: a_.axis("off")
    h, l = ax.ravel()[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=min(6, len(l)), fontsize=8)
    fig.suptitle(f"{key}: metric vs rotation angle (z-scored per source; thick = mean over subsets of the same size, thin = each subset; solid analytic, dashed NN)", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .96)); fig.savefig(path, dpi=110); plt.close(fig)


def scatter_fig(key, d, metrics, path):
    rot = d["rot"]; nc = 4; nr = int(np.ceil(len(metrics) / nc)); fig, ax = plt.subplots(nr, nc, figsize=(4 * nc, 3.6 * nr), squeeze=False)
    for a_, m in zip(ax.ravel(), metrics):
        sc = a_.scatter(d["oracle"][m], d["nn"][m], c=rot.angle_deg, cmap="plasma", s=14); pe = d["_corr"].get(m, (np.nan, np.nan))
        a_.set_title(f"{m}  r={pe[0]:.2f}", fontsize=9); a_.set_xlabel("analytic"); a_.set_ylabel("NN"); a_.grid(alpha=.3)
    for a_ in ax.ravel()[len(metrics):]: a_.axis("off")
    fig.colorbar(sc, ax=ax, label="angle (deg)", shrink=.6); fig.suptitle(f"{key}: analytic vs NN metric, one point per saved rotation", fontsize=10)
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    dbs = load(); print(len(dbs), "databases:", *dbs, sep="\n  ")
    df = analyse(dbs); df.to_csv(RES / "correlations_long.csv", index=False)
    summ = df.groupby("metric").agg(pearson_median=("pearson", "median"), pearson_min=("pearson", "min"), pearson_mean=("pearson", "mean"), spearman_median=("spearman", "median"),
                                     curve_median=("curve_pearson_mean", "median"), n_db=("pearson", "count"), n_nan=("pearson", lambda s: int(s.isna().sum()))).sort_values("pearson_median", ascending=False)
    summ.to_csv(RES / "correlation_summary.csv")
    for key, d in dbs.items():
        sub = df[df.db == key].set_index("metric"); d["_corr"] = {m: (sub.loc[m, "pearson"], sub.loc[m, "curve_pearson_mean"]) for m in sub.index}
        safe = key.replace("|", "__"); curves_fig(key, d, [m for m in KEY if m in d["oracle"].columns], FIG / f"curves__{safe}.png")
        scatter_fig(key, d, [m for m in KEY if m in d["oracle"].columns], FIG / f"scatter__{safe}.png")
    # heat map of pooled Pearson: key metrics x databases
    piv = df[df.metric.isin(KEY)].pivot(index="metric", columns="db", values="pearson").loc[[m for m in KEY if m in set(df.metric)]]
    fig, ax = plt.subplots(figsize=(1 + .55 * piv.shape[1], 1 + .45 * piv.shape[0])); im = ax.imshow(piv.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([c.replace("|", "\n") for c in piv.columns], fontsize=6, rotation=90); ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]): ax.text(j, i, "" if np.isnan(piv.values[i, j]) else f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=.7, label="pooled Pearson r (analytic vs NN)"); fig.tight_layout(); fig.savefig(RES / "figures" / "heatmap_key_metrics.png", dpi=130); plt.close(fig)
    piv.round(3).to_csv(RES / "key_metrics_pearson.csv")
    print(summ.head(25).round(3).to_string()); print(summ.tail(10).round(3).to_string())
