"""W2-05 synthesis: coverage tables of every module from the saved result tables (no chart-only or hand-typed numbers).

usage: python analyze.py      -> results/*.csv, results/report_tables.md
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import RES, OUT, LAB07, PRIMARY, SECONDARY
import numpy as np, pandas as pd

A1 = set(pd.read_csv(LAB07 / "results" / "next_wave_candidates.csv").query("lane == 'A1'").id)


def md(df, ff=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.{ff}f}")
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values) + "\n"


# ---------------------------------------------------------------- W2-01
def blind_tables(out):
    d = pd.read_csv(RES / "blind_recovery.csv"); d = d[d.batch != "baseline"].copy(); b = pd.read_csv(RES / "blind_recovery.csv").query("batch == 'baseline'")
    d["cell"] = d.Q.astype(str) + "/" + d["mask"].astype(str)
    # per toy x variant x cell: pass in all seeds / fraction
    g = d.groupby(["batch", "cid", "cell", "toy", "cls", "fam_group", "variant"]).agg(n=("pass_lab05", "size"), n_pass=("pass_lab05", "sum"), MCC_med=("MCC", "median"),
          sup=("supervised_MCC", "first"), leak_med=("leak_max", "median"), leak_max=("leak_max", "max"), dist_Rstar_med=("dist_to_Rstar", "median"),
          cat=("category", lambda s: s.value_counts().index[0])).reset_index()
    g["all_seeds"] = g.n_pass == g.n; g["majority"] = g.n_pass * 2 > g.n
    g.to_csv(RES / "blind_toy_variant.csv", index=False)
    rows = []
    for (batch, cid, cell, v), gg in g.groupby(["batch", "cid", "cell", "variant"]):
        r = dict(batch=batch, cid=cid, cell=cell, variant=v, n_toys=len(gg), toys_all_seeds=int(gg.all_seeds.sum()), toys_majority=int(gg.majority.sum()))
        for cls in ("B", "C"):
            for fg in ("C", "CR"):
                s = gg[(gg.cls == cls) & (gg.fam_group == fg)]; r[f"{cls}-{fg}_all_seeds"] = f"{int(s.all_seeds.sum())}/{len(s)}"
        rows.append(r)
    cov = pd.DataFrame(rows)
    lib = g.groupby(["batch", "cid", "cell"]).agg(libraries=("all_seeds", "size"), libs_all_seeds=("all_seeds", "sum"), libs_majority=("majority", "sum")).reset_index()
    cov.to_csv(RES / "blind_coverage.csv", index=False); lib.to_csv(RES / "blind_library_coverage.csv", index=False)
    cat = d.groupby(["batch", "cid", "fam_group", "category"]).size().unstack(fill_value=0).reset_index(); cat.to_csv(RES / "blind_categories.csv", index=False)
    # seed stability of the selected rotations (blind diagnostic) and the TEST-truth agreement across seeds
    from wave2.optimizers import rel_dist
    R = np.load(RES / "found_rotations.npz"); st = []
    for (batch, cid, cell, toy, v), gg in d.groupby(["batch", "cid", "cell", "toy", "variant"]):
        Rs = [R[i] for i in gg.row_id if i in R.files]
        if len(Rs) > 1:
            dd = [rel_dist(Rs[i], Rs[j]) for i in range(len(Rs)) for j in range(i + 1, len(Rs))]
            st.append(dict(batch=batch, cid=cid, cell=cell, toy=toy, variant=v, n=len(Rs), max_pair_dist=max(dd), median_pair_dist=float(np.median(dd)),
                           pass_frac=float(gg.pass_lab05.mean())))
    pd.DataFrame(st).to_csv(RES / "blind_seed_stability.csv", index=False)
    out.append("### W2-01 blind search: toys whose selected R passes the Lab 05 criterion in EVERY search (x integration) seed\n\n" +
               md(cov[cov.cid.isin(["E01", "E02", "C01", "C02", "C03", "C04", "N03"])].sort_values(["batch", "cid", "cell", "variant"])))
    out.append("### W2-01 24-library coverage (toy x variant passing in all seeds / in the majority of seeds)\n\n" + md(lib))
    out.append("### W2-01 post-lock diagnosis of the selected rotations (count of searches)\n\n" + md(cat))
    e = g[(g.batch == "BS-B") & g.cid.isin(["E01", "E02"])][["cid", "toy", "variant", "n_pass", "n", "MCC_med", "sup", "leak_med", "leak_max", "dist_Rstar_med", "cat"]]
    out.append("### W2-01 BS-B (split-safe, principal): E01 / E02 per toy x variant (9 = 3 integration x 3 search seeds)\n\n" + md(e))
    bb = b.groupby(["cid", "fam_group"])[["MCC", "leak_max", "pass_lab05"]].mean().reset_index()
    out.append("### W2-01 no-search baselines (mean over toys x variants)\n\n" + md(bb))
    # BS-D
    fs = sorted((OUT / "blind" / "BS-D").glob("*/*.reeval.csv"))
    if fs:
        rv = pd.concat([pd.read_csv(f) for f in fs]); rv.to_csv(RES / "blind_bsd_reevaluation.csv", index=False)
        t = rv.groupby(["cid"]).agg(cells=("key", "size"), D_best_is_selected=("D_best_is_selected", "mean"), not_converged=("not_converged", "sum"),
                                    separated_near_minima=("separated_near_minima", "sum"), seed_inconsistent=("seed_inconsistent", "sum"), D_prefers_other=("D_prefers_other", "sum"),
                                    escalated=("escalate", "sum")).reset_index()
        out.append("### W2-01 BS-D independent 1024 x 128 re-evaluation of the BS-B E01 / E02 finalists (blind diagnostics)\n\n" + md(t))
    return d, g


# ---------------------------------------------------------------- W2-02
def oracle_tables(out):
    f = RES / "oracle_recovery.csv"
    if not f.is_file():
        return
    r = pd.read_csv(f); a = r[r.solution == "best_any"].copy()
    tv = [c for c in a.columns if c.startswith("transfer_") and c.endswith("_good") and c != "transfer_good_all_variants"]
    t = a.groupby(["fam_group", "exact", "outcome"]).size().rename("metric_x_target_x_toy").reset_index()
    out.append("### W2-02 outcome categories of the best analytic solution found (93 priority metrics x 8 logit targets x 2 toys)\n\n" + md(t))
    s = a[a.exact == "displaced minimum"]
    for th in (.05, .1, .15, .2, 1.0):
        pass
    disp = s.assign(dbin=pd.cut(s.d_sp, [0, .02, .05, .1, .15, .2, 1])).groupby(["fam_group", "dbin"], observed=True).agg(n=("ideal_MCC", "size"),
           ideal_good=("ideal_good", "mean"), transfer_good=("transfer_good_all_variants", "mean"), ideal_MCC_min=("ideal_MCC", "min"),
           ideal_leak_max=("ideal_leak_max", "max")).reset_index()
    out.append("### W2-02 displaced analytic minima: practical recovery vs signed-permutation distance\n\n" + md(disp))
    per = a.groupby("metric").agg(targets=("target", "nunique"), rows=("ideal_good", "size"), ideal_all=("ideal_good_minimum", "all"), transfer_all=("transfer_good_minimum", "all"),
                                  inconclusive=("exact", lambda s: int((s == "inconclusive").sum())), displaced=("exact", lambda s: int((s == "displaced minimum").sum())),
                                  tied=("exact", lambda s: int((s == "flat / tied non-permutations").sum())), worst_ideal_MCC=("ideal_MCC", "min"),
                                  worst_transfer_MCC_s9=("transfer_s9_MCC", "min")).reset_index()
    per["A1"] = per.metric.isin(A1); per.to_csv(RES / "oracle_metric_practical.csv", index=False)
    good = per[per.ideal_all & per.transfer_all]
    out.append(f"Metrics whose best analytic solution satisfies ideal AND transferred recovery on all 8 logit targets x both toys x both variants: "
               f"{len(good)} of {len(per)} (A1: {int(good.A1.sum())} of {int(per.A1.sum())}); with no inconclusive target: "
               f"{int((good.inconclusive == 0).sum())}.\n\n" + md(good.sort_values("metric")[["metric", "A1", "displaced", "tied", "inconclusive", "worst_ideal_MCC", "worst_transfer_MCC_s9"]]))
    rp = pd.read_csv(RES / "oracle_replay_audit.csv")
    out.append(f"Replay identity: {len(rp)} metric x target searches, max |L_final replay - Lab 07| = {rp.max_abs_L_diff.max():.2e}, "
               f"max |d_SP diff| = {rp.max_abs_dsp_diff.max():.2e}; persisted suspicious Q reproduced exactly in {int((rp.savedQ_rel_diff == 0).sum())} / {int(rp.savedQ_rel_diff.notna().sum())}.\n")
    fs = pd.read_csv(RES / "oracle_fine_summary.csv")
    fsm = fs.merge(a[["target", "metric", "fam_group"]].drop_duplicates(["target", "metric"]), on=["target", "metric"])
    out.append("### W2-02 fine-angle diagnostics around I (planes and random directions, |theta| <= 15 deg)\n\n" +
               md(fsm.groupby("fam_group").agg(cells=("metric", "size"), identity_not_local_min=("identity_local_min_fine", lambda s: int((~s).sum())),
                                               median_best_lower_theta=("best_lower_theta", lambda s: float(np.nanmedian(np.abs(s))) if s.notna().any() else np.nan)).reset_index()))
    lf = RES / "oracle_local_escape.csv"
    if lf.is_file():
        le = pd.read_csv(lf)
        out.append("### W2-02 local refinements from I and small perturbations (prioritised metrics)\n\n" +
                   md(le.groupby("start_deg").agg(n=("metric", "size"), returned_to_identity=("returned_to_identity", "mean"), lower_than_identity=("lower_than_identity", "mean")).reset_index()))
        ux = le[le.metric == "L03|UX.lq.1"].groupby(["target", "start_deg"]).agg(returned=("returned_to_identity", "mean"), lower=("lower_than_identity", "mean")).reset_index()
        out.append("UX.lq.1 local escape / basin coverage:\n\n" + md(ux))
    return a


# ---------------------------------------------------------------- W2-03
def conditional_tables(out):
    f = RES / "conditional_pass_summary.csv.gz"
    if not f.is_file():
        return
    c = pd.read_csv(f); c = c[c["mask"].isin(["FULL", "MASK-ALL"])]
    piv = c.groupby(["fid", "seed", "metric", "config"]).agg(libs=("O_perfect", "size"), O_perfect=("O_perfect", "sum"), I_marginal=("I_marginal", "sum"),
                                                           both_marginal=("both_marginal", "sum"), both_perfect=("both_perfect", "sum")).reset_index()
    for fg in ("C", "CR"):
        s = c[c.fam_group == fg].groupby(["fid", "seed", "metric", "config"]).agg(**{f"{fg}_O_perfect": ("O_perfect", "sum"), f"{fg}_I_marginal": ("I_marginal", "sum"),
                                                                                    f"{fg}_both_marginal": ("both_marginal", "sum")}).reset_index()
        piv = piv.merge(s, on=["fid", "seed", "metric", "config"], how="left")
    piv.to_csv(RES / "conditional_coverage.csv", index=False)
    p = piv[(piv.fid == "pilot") & (piv.seed == 31)]
    main = p[p.config.isin(["P0", "P1", "P2", "P4K8", "P4K8eq", "P4K4", "P4K16"])].pivot_table(index="metric", columns="config", values=["CR_O_perfect", "CR_I_marginal", "C_I_marginal"]).reset_index()
    main.columns = [" ".join(map(str, c_)).strip() for c_ in main.columns]
    out.append("### W2-03 pooled vs conditional (pilot, seed 31, FULL / MASK-ALL; counts over the 12 CR / 12 C libraries)\n\n" + md(main, 0))
    best = p.copy(); base = best[best.config == "P0"].set_index("metric")
    best["dO_CR"] = best.CR_O_perfect - best.metric.map(base.CR_O_perfect); best["dI_CR"] = best.CR_I_marginal - best.metric.map(base.CR_I_marginal)
    best["dI_C"] = best.C_I_marginal - best.metric.map(base.C_I_marginal)
    imp = best[(best.config != "P0") & ((best.dO_CR > 0) | (best.dI_CR > 0))].sort_values(["dI_CR", "dO_CR"], ascending=False)
    out.append("### W2-03 configurations that improve CR analytic-perfect or CR inferred-marginal coverage over P0 (same points)\n\n" +
               md(imp[["metric", "config", "O_perfect", "I_marginal", "both_marginal", "CR_O_perfect", "dO_CR", "CR_I_marginal", "dI_CR", "C_I_marginal", "dI_C"]].head(40), 0))
    allpass = p[(p.libs == 24) & ((p.O_perfect == 24) | (p.I_marginal == 24) | (p.both_marginal == 24))]
    out.append("### W2-03 configurations passing on all 24 libraries (analytic perfect, inferred marginal or joint marginal)\n\n" +
               md(allpass[["metric", "config", "O_perfect", "I_marginal", "both_marginal", "both_perfect"]], 0))
    mc = pd.read_csv(f); mc = mc[mc["mask"] == "MASK-COL"]
    if len(mc):
        t = mc[mc.config.isin(["P0", "P1", "P4K8"])].groupby(["metric", "config"])[["O_perfect", "I_marginal"]].sum().unstack().reset_index()
        t.columns = [" ".join(map(str, c_)).strip() for c_ in t.columns]
        out.append("### W2-03 MASK-COL (fixed banks, 8 libraries): valid-subspace estimand\n\n" + md(t, 0))
    return piv


# ---------------------------------------------------------------- W2-04
def inference_tables(out):
    rows = []
    for b in ("IF-A", "IF-B"):
        f = RES / f"inference_{b}_summary.csv.gz"
        if f.is_file():
            rows.append(pd.read_csv(f))
    if not rows:
        return
    d = pd.concat(rows, ignore_index=True); d = d[d["mask"].isin(["FULL", "MASK-ALL"])].copy()
    d["selector"] = d.get("selector", pd.Series(index=d.index, dtype=object)).fillna("-"); d["fam_group"] = np.where(d.toy.str.startswith("CR"), "CR", "C")
    d["A1"] = d.metric.isin(A1)
    g = d[d.A1].groupby(["batch", "estimator", "source", "selector", "setting" if False else "batch"]).size()
    key = ["batch", "estimator", "source", "selector"]
    t = d[d.A1].groupby(key + ["fam_group"]).agg(libs=("toy", lambda s: s.nunique()), I_perfect=("I_perfect", "mean"), I_marginal=("I_marginal", "mean"),
                                                 both_perfect=("both_perfect", "mean"), pearson=("pearson", "median"), UX_err=("UX_rel_err", "median"),
                                                 UU_err=("UU_rel_err", "median"), zero_false=("oracle_zero_false_energy", "median")).reset_index()
    t = t.pivot_table(index=key, columns="fam_group", values=["I_perfect", "I_marginal", "pearson", "UX_err", "UU_err"]).reset_index()
    t.columns = [" ".join(map(str, c_)).strip() for c_ in t.columns]; t.to_csv(RES / "inference_estimator_comparison.csv", index=False)
    out.append("### W2-04 A1 metrics (33): fraction of library x metric cells with inferred perfect / marginal passes, median Eval1 Pearson and matrix errors\n\n" + md(t))
    # metric-level: CR coverage of A1 metrics per estimator
    m = d[d.A1 & (d.fam_group == "CR")].groupby(key + ["metric"]).I_marginal.sum().reset_index()
    best = m.groupby(key).I_marginal.agg(["max", "mean"]).reset_index().rename(columns={"max": "best_A1_metric_CR_marginal_libs", "mean": "mean_CR_marginal_libs"})
    out.append("### W2-04 best A1 metric per estimator: CR libraries (of 12) with an inferred marginal pass\n\n" + md(best))
    f = RES / "inference_IF-C_summary.csv.gz"
    if f.is_file():
        c = pd.read_csv(f); c = c[c["mask"].isin(["FULL", "MASK-ALL"])]; c["fam_group"] = np.where(c.toy.str.startswith("CR"), "CR", "C"); c["A1"] = c.metric.isin(A1)
        tc = c[c.A1].groupby(["energy", "fam_group"]).agg(I_perfect=("I_perfect", "mean"), I_marginal=("I_marginal", "mean"), UX_err=("UX_rel_err", "median"),
                                                          UU_err=("UU_rel_err", "median"), zero_false=("oracle_zero_false_energy", "median"),
                                                          neg_eig=("frac_negative_J_eigen", "mean")).reset_index()
        out.append("### W2-04 IF-C cross-fitting of the observed-data surrogate (A1 metrics, Q01 medium, 3 splits x 3 seeds)\n\n" + md(tc))
    return d


def synthesis(out):
    """W2-05: next-wave registry and three-gap evidence, every number computed from the tables above"""
    g = pd.read_csv(RES / "blind_toy_variant.csv"); rows = []
    for batch in ("BS-B", "BS-SL"):
        gb = g[(g.batch == batch) & (g.cell == "Q01/" + g.cell.str.split("/").str[1])]
        for (cid, v), gg in gb.groupby(["cid", "variant"]):
            if cid in ("N03",):
                continue
            ok = gg.all_seeds; fails = ",".join(sorted(gg[~ok].toy))
            rows.append(dict(entity="algorithm", id=f"{batch}:{cid}:{v}", metric=gg.iloc[0].cid, evidence=f"{int(ok.sum())}/{len(gg)} toys pass the Lab 05 criterion in every seed ({batch}, Q01 primary mask)",
                             fails=fails, lane="independent confirmation (fresh data / NP seeds / families)" if ok.sum() == len(gg) else
                             ("objective repair on the failing toys, then confirmation" if ok.sum() >= len(gg) - 2 else "not advanced as an algorithm")))
    pr = RES / "oracle_metric_practical.csv"
    if pr.is_file():
        p = pd.read_csv(pr)
        for r in p[p.ideal_all & p.transfer_all].itertuples():
            rows.append(dict(entity="analytic metric", id=r.metric, metric=r.metric, evidence=f"best analytic solution recovers T (ideal and transferred) on all 8 logit targets; displaced on {r.displaced}, tied on {r.tied}, inconclusive on {r.inconclusive}",
                             fails="", lane="Gap 2 inference target (practical Gap 1 adequate)" + (" [A1]" if r.A1 else "")))
    nw = pd.DataFrame(rows); nw.to_csv(RES / "next_wave_candidates.csv", index=False)
    out.append("### W2-05 next-wave registry (algorithms)\n\n" + md(nw[nw.entity == "algorithm"]))
    return nw


def main():
    out = ["# Lab 08 report tables (generated by analyze.py from results/)\n"]
    vc = pd.read_csv(RES / "validation_checks.csv")
    out.append(f"### W2-00\n\n{len(vc)} checks, {int((~vc.passed).sum())} failed.  " +
               "; ".join(f"{r.check}: {r.value}" for r in vc[vc.check.str.startswith(("claim", "info", "status", "unit"))].itertuples()) + "\n")
    blind_tables(out); oracle_tables(out); conditional_tables(out); inference_tables(out); synthesis(out)
    (RES / "report_tables.md").write_text("\n".join(out), encoding="utf-8"); print("\n".join(out)[:20000])


if __name__ == "__main__":
    main()
