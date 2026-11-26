"""Which angle does each metric prefer?  For every database, source (analytic / NN), metric and axis-subset sweep, find the
angle (0,15,...,90) with the smallest metric value (all metrics: lower = better; ties within 1e-9 relative count for all tied angles).
  PERFECT pass : in EVERY sweep of the database the preferred angle is 0 or 90   (0 and 90 are the oracle R* up to signed permutation)
  MARGINAL pass: in EVERY sweep the preferred angle is in {0, 15, 75, 90}
usage: python angle_preference.py   -> results/angle_pref_long.csv, angle_pref_db_summary.csv, angle_pref_metric_summary.csv
"""
import numpy as np, pandas as pd
from analyze import load, RES
PERF, MARG = {0, 90}, {0, 15, 75, 90}
dbs = load(); rows = []
for key, d in dbs.items():
    rot = d["rot"]; sw = rot[rot.candidate_type == "sweep"]; groups = [g.sort_values("angle_deg") for _, g in sw.groupby("axes_1based")]
    for src in ("oracle", "nn"):
        V = d[src]
        for m in V.columns:
            v = V[m].values; okP = okM = 0; bad = []
            for g in groups:
                x = v[g.index]; tol = 1e-9 * (abs(x).max() + 1e-12)
                if not np.isfinite(x).all() or x.max() - x.min() <= tol: continue      # constant / undefined curve (e.g. UU blocks of the linear analytic C models) never counts as a pass
                best = {int(a) for a, xx in zip(g.angle_deg, x) if xx <= x.min() + tol}
                okP += bool(best & PERF); okM += bool(best & MARG)
                if not (best & MARG): bad.append(sorted(best)[0])
            rows.append(dict(db=key, model=d["model"], toy=d["toy"], cls=d["cls"], variant=d["var"], source=src, metric=m, n_sweeps=len(groups), n_perfect=okP, n_marginal=okM,
                             perfect=okP == len(groups), marginal=okM == len(groups)))
L = pd.DataFrame(rows); L.to_csv(RES / "angle_pref_long.csv", index=False)
S = L.groupby(["db", "source"]).agg(metrics=("metric", "count"), n_perfect=("perfect", "sum"), n_marginal=("marginal", "sum")).reset_index()
S.to_csv(RES / "angle_pref_db_summary.csv", index=False)
M = L.groupby(["metric", "source"]).agg(dbs_perfect=("perfect", "sum"), dbs_marginal=("marginal", "sum"), n_db=("db", "count")).reset_index()
M.to_csv(RES / "angle_pref_metric_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
S.to_csv(RES / 'angle_pref_db_summary.csv', index=False)
print(S.pivot(index="db", columns="source", values=["n_perfect", "n_marginal"]).to_string())
nall = L.db.nunique()
for src in ("nn", "oracle"):
    q = M[M.source == src]; print(src, f"perfect in all {nall}:", int((q.dbs_perfect == nall).sum()), f" marginal in all {nall}:", int((q.dbs_marginal == nall).sum()))
# the 12 toys x 2 variants (designated runs only): per toy and per class
P = L[L.toy.notna()]
T = P.groupby(["cls", "toy", "variant", "source"]).agg(n_perfect=("perfect", "sum"), n_marginal=("marginal", "sum")).reset_index()
T.to_csv(RES / "angle_pref_toys.csv", index=False)
C = P.groupby(["metric", "source", "cls"]).agg(dbs_perfect=("perfect", "sum"), dbs_marginal=("marginal", "sum"), n_db=("db", "count")).reset_index()
C.to_csv(RES / "angle_pref_metric_by_class.csv", index=False)
print(T.pivot_table(index=["cls", "toy", "variant"], columns="source", values=["n_perfect", "n_marginal"]).to_string())
for cls in ("B", "C"):
    for src in ("nn", "oracle"):
        q = C[(C.cls == cls) & (C.source == src)]; n = int(q.n_db.max()) if len(q) else 0
        print(f"class {cls} {src}: perfect in all {n} databases: {int((q.dbs_perfect == n).sum())}, marginal in all {n}: {int((q.dbs_marginal == n).sum())}")
