"""Angle of the minimum + pass classification for the regression-NP sweeps.   usage: python angles_reg.py   (after s9_regression.py and analyze_reg.py)

For every database (model x NP x variant), source (analytic / NN), metric (lower = better) and axis-subset sweep: the angle(s) in 0,15,..,90 that
minimise the metric (ties within 1e-9 relative all counted; constant curves = undefined, never a pass).
  PERFECT: every sweep of the database has its minimum at 0 or 90;   MARGINAL: every sweep has it in {0,15,75,90}.
Outputs results/argmin_long.csv (one row per db,source,metric,sweep), argmin_summary.csv (per db,source,metric: share of sweeps at 0/90, mean
distance of the minimum from the nearest aligned angle, mean/median minimum angle, histogram) and angle_pass_summary.csv, figures/argmin_*.png
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_reg import load, RES, FIG, KEY

ALIGN, NEAR = {0, 90}, {0, 15, 75, 90}
dbs = load(); rows = []
for key, d in dbs.items():
    rot = d["rot"]; sw = rot[rot.candidate_type == "sweep"]; groups = [(ax, g.sort_values("angle_deg")) for ax, g in sw.groupby("axes_1based")]
    for src in ("oracle", "nn"):
        for m in d[src].columns:
            v = d[src][m].values
            for ax, g in groups:
                x = v[g.index]; tol = 1e-9 * (abs(x).max() + 1e-12)
                if not np.isfinite(x).all() or x.max() - x.min() <= tol: continue
                best = sorted(int(a) for a, xx in zip(g.angle_deg, x) if xx <= x.min() + tol)
                a = min(best, key=lambda t: min(t, 90 - t))                               # on a tie report the angle closest to aligned
                rows.append(dict(db=key, model=d["model"], variant=d["var"], source=src, metric=m, axes=ax, k=len(ax.split("-")), argmin=a, ties=",".join(map(str, best)),
                                 aligned=bool(set(best) & ALIGN), near=bool(set(best) & NEAR), dist_to_aligned=min(a, 90 - a)))
A = pd.DataFrame(rows); A.to_csv(RES / "argmin_long.csv", index=False)
nsweep = {k: len(g) for k, g in ((k, dd["rot"][dd["rot"].candidate_type == "sweep"].groupby("axes_1based")) for k, dd in dbs.items())}
S = A.groupby(["db", "model", "variant", "source", "metric"]).agg(n_def=("argmin", "count"), frac_aligned=("aligned", "mean"), frac_near=("near", "mean"), mean_dist=("dist_to_aligned", "mean"),
                                                                   mean_argmin=("argmin", "mean"), median_argmin=("argmin", "median")).reset_index()
S["n_sweeps"] = S.db.map(nsweep); S["perfect"] = (S.n_def == S.n_sweeps) & (S.frac_aligned == 1); S["marginal"] = (S.n_def == S.n_sweeps) & (S.frac_near == 1)
S.to_csv(RES / "argmin_summary.csv", index=False)
P = S.groupby(["db", "source"]).agg(n_perfect=("perfect", "sum"), n_marginal=("marginal", "sum")).reset_index(); P.to_csv(RES / "angle_pass_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(P.pivot(index="db", columns="source", values=["n_perfect", "n_marginal"]).to_string())
# figure: where is the minimum?  key metrics x sources, strip of argmin angles over all sweeps, one row per model (nn1 + r2 pooled)
keym = [m for m in KEY if m in set(A.metric)]
for var in ("s9", "s9_2"):
    models = sorted(A[A.variant == var].model.unique(), key=lambda m: ["C1", "C2", "C3", "CR3", "CR4"].index(m)); fig, ax = plt.subplots(len(models), len(keym), figsize=(2.1 * len(keym), 1.9 * len(models)), squeeze=False, sharex=True)
    for i, mo in enumerate(models):
        for j, mt in enumerate(keym):
            a_ = ax[i, j]
            for src, c, off in (("oracle", "tab:blue", -3), ("nn", "tab:red", 3)):
                q = A[(A.variant == var) & (A.model == mo) & (A.metric == mt) & (A.source == src)].argmin
                h = np.array([(q == t).mean() if len(q) else np.nan for t in range(0, 91, 15)]); a_.bar(np.arange(0, 91, 15) + off, h, width=5, color=c, alpha=.8, label="analytic" if src == "oracle" else "NN")
            a_.set_ylim(0, 1); a_.set_xticks([0, 45, 90]); a_.tick_params(labelsize=6)
            if i == 0: a_.set_title(mt, fontsize=6)
            if j == 0: a_.set_ylabel(mo, fontsize=8)
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=8); fig.suptitle(f"{var}: share of axis-subset sweeps whose minimum is at each angle (all NPs of the model pooled)", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96)); fig.savefig(FIG / f"argmin_hist_{var}.png", dpi=120); plt.close(fig)
