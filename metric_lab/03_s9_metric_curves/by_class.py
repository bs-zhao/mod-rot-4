"""Separate report for each class of toys: B (choice models C1-CR5) and C (noisy-logit models C1L-CR5L).

Uses only the designated run of each toy (toys.py), with its S9 and S9_2 sweeps: 12 databases per class.
Reads results/correlations_long.csv and results/angle_pref_long.csv (run analyze.py and angle_preference.py first).
Writes results/class_B/ and results/class_C/:
  eval1_pearson_toys.csv      metric x (toy, variant): pooled Pearson analytic vs NN
  eval1_family_median.csv     metric family x toy: median pooled Pearson
  eval1_heatmap_key.png       key metrics x databases
  eval2_counts.csv            per toy and variant: number of metrics passing (NN, analytic, both), perfect / marginal
  eval2_metric_pass.csv       per metric: in how many databases it passes (both sources), for C-family, CR-family, all
  eval2_pass_lists.md         the metrics that pass in every database of each scope
usage: python by_class.py
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import RES, KEY
from toys import TOYS

NAMES = {"B": "class B - binary choice (C1-CR5)", "C": "class C - continuous noisy logit (C1L-CR5L)"}
corr = pd.read_csv(RES / "correlations_long.csv"); corr = corr[corr.toy.notna()]
ang = pd.read_csv(RES / "angle_pref_long.csv"); ang = ang[ang.toy.notna()]

for cls in ("B", "C"):
    out = RES / f"class_{cls}"; out.mkdir(exist_ok=True)
    toys = [t for t, (c, _) in TOYS.items() if c == cls]; cols = [(t, v) for t in toys for v in ("s9", "s9_2")]
    # ---------- evaluation 1 ----------
    c = corr[corr.cls == cls]
    piv = c.pivot_table(index="metric", columns=["toy", "variant"], values="pearson").reindex(columns=cols)
    piv.round(4).to_csv(out / "eval1_pearson_toys.csv")
    fam = c.assign(family=c.metric.str.split(".").str[0]).groupby(["family", "toy"]).pearson.median().unstack("toy").reindex(columns=toys)
    fam["all_C_family"] = c[~c.model.str.startswith("CR")].assign(family=lambda d: d.metric.str.split(".").str[0]).groupby("family").pearson.median()
    fam["all_CR_family"] = c[c.model.str.startswith("CR")].assign(family=lambda d: d.metric.str.split(".").str[0]).groupby("family").pearson.median()
    fam.round(3).sort_values("all_CR_family", ascending=False).to_csv(out / "eval1_family_median.csv")
    k = piv.loc[[m for m in KEY if m in piv.index]]
    fig, ax = plt.subplots(figsize=(1.5 + .62 * k.shape[1], 1.2 + .45 * k.shape[0]))
    im = ax.imshow(k.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(k.shape[1])); ax.set_xticklabels([f"{t}\n{v}" for t, v in k.columns], fontsize=7)
    ax.set_yticks(range(k.shape[0])); ax.set_yticklabels(k.index, fontsize=8)
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            if np.isfinite(k.values[i, j]): ax.text(j, i, f"{k.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=.7, label="pooled Pearson r (analytic vs NN)"); ax.set_title(NAMES[cls], fontsize=10)
    fig.tight_layout(); fig.savefig(out / "eval1_heatmap_key.png", dpi=130); plt.close(fig)
    # ---------- evaluation 2 ----------
    a = ang[ang.cls == cls]
    w = a.pivot_table(index=["toy", "variant", "metric"], columns="source", values=["perfect", "marginal"], aggfunc="first")
    w.columns = [f"{x}_{s}" for x, s in w.columns]; w = w.astype(bool)
    w["perfect_both"] = w.perfect_nn & w.perfect_oracle; w["marginal_both"] = w.marginal_nn & w.marginal_oracle
    cnt = w.groupby(["toy", "variant"]).sum().reindex(cols)[["perfect_nn", "perfect_oracle", "perfect_both", "marginal_nn", "marginal_oracle", "marginal_both"]]
    cnt.to_csv(out / "eval2_counts.csv")
    w = w.reset_index(); w["family"] = np.where(w.toy.str.startswith("CR"), "CR", "C")
    rows = []
    for m, g in w.groupby("metric"):
        r = dict(metric=m)
        for scope, gg in (("C_family", g[g.family == "C"]), ("CR_family", g[g.family == "CR"]), ("all", g)):
            r[f"{scope}_n_db"] = len(gg); r[f"{scope}_perfect"] = int(gg.perfect_both.sum()); r[f"{scope}_marginal"] = int(gg.marginal_both.sum())
        rows.append(r)
    mp = pd.DataFrame(rows).sort_values(["all_marginal", "all_perfect"], ascending=False); mp.to_csv(out / "eval2_metric_pass.csv", index=False)
    lines = [f"# {NAMES[cls]}: metrics that pass evaluation 2 in every database of a scope", "",
             "A metric passes a database when the preferred angle of every sweep is 0/90 (perfect) or in {0,15,75,90} (marginal),",
             "for the analytic AND the NN values (note 04.2). Databases: the designated run of each toy, S9 and S9_2.", ""]
    for scope, lab in (("C_family", "C1-C3 family (6 databases)"), ("CR_family", "CR3-CR5 family (6 databases)"), ("all", "all 6 models (12 databases)")):
        n = int(mp[f"{scope}_n_db"].max())
        P = mp[mp[f"{scope}_perfect"] == n].metric.tolist(); M = mp[mp[f"{scope}_marginal"] == n].metric.tolist()
        lines += [f"## {lab}", f"- perfect in all {n}: {len(P)}" + (": " + ", ".join(f"`{x}`" for x in P) if P else ""),
                  f"- marginal in all {n}: {len(M)}" + (": " + ", ".join(f"`{x}`" for x in M) if M else ""), ""]
    (out / "eval2_pass_lists.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"===== {NAMES[cls]}"); print(cnt.to_string())
    print(fam[["all_C_family", "all_CR_family"]].round(2).sort_values("all_CR_family", ascending=False).to_string())
    print("\n".join(lines[5:]))
