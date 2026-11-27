"""Third-wave synthesis (03_third_wave section 27): every table in results/report_tables.md is generated here from saved numeric outputs.

usage: python analyze.py          (after validate / run_repair_diag / run_local / pilot + finalist evaluation / context / reduced-axis merges)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import RES, PRIMARY, leak_threshold
import numpy as np, pandas as pd

OUT_MD = RES / "report_tables.md"
LINES = []


def md(df, title, note=""):
    LINES.append(f"\n## {title}\n")
    if note:
        LINES.append(note + "\n")
    LINES.append(df.to_markdown(index=False, floatfmt=".4g") if len(df) else "_(no rows)_")


def read(name, **kw):
    f = RES / name
    return pd.read_csv(f, **kw) if f.is_file() else pd.DataFrame()


def family(cid):
    for p, f in (("RW-", "RW"), ("MX-", "MX"), ("ST-E", "RA"), ("ST-", "GD")):
        if cid.startswith(p):
            return f
    return "CTRL"


def toy_cov(br):
    br = br[br.panel == "primary"].copy(); br["thr"] = br.toy.map(leak_threshold); br["lex"] = br.leak_max - br.thr
    g = br.groupby(["batch", "cid", "variant", "toy"]).agg(ok=("pass_lab05", "all"), n=("pass_lab05", "size"), gap=("MCC_gap", "max"), lex=("lex", "max"),
                                                         obj=("category", lambda s: (s == "objective / estimand issue").sum())).reset_index()
    return g


def coverage_table(g):
    piv = g.pivot_table(index=["batch", "cid"], columns="variant", values="ok", aggfunc="sum").rename(columns={"s9": "S9 toys", "s9_2": "S9_2 toys"})
    lib = g.groupby(["batch", "cid"]).ok.sum().rename("libraries /24"); n = g.groupby(["batch", "cid"]).n.min().rename("runs per toy")
    fail = g[~g.ok].groupby(["batch", "cid", "variant"]).toy.apply(lambda s: ",".join(sorted(s))).unstack().rename(columns={"s9": "S9 failures", "s9_2": "S9_2 failures"})
    worst = g.groupby(["batch", "cid", "variant"]).gap.max().unstack().rename(columns={"s9": "worst gap S9", "s9_2": "worst gap S9_2"})
    out = pd.concat([piv, lib, n, worst, fail], axis=1).reset_index(); out["family"] = out.cid.map(family)
    return out.sort_values(["batch", "S9_2 toys", "S9 toys"], ascending=[True, False, False])


def main():
    val = read("validation_checks.csv")
    if len(val):
        md(pd.DataFrame([dict(checks=len(val), passed=int(val.passed.sum()), failed=int((~val.passed).sum()),
                              failed_names="; ".join(val[~val.passed].check.astype(str) + " " + val[~val.passed].toy.astype(str)))]), "W3-00 validation gate")
    # ---- W3-01 RW-A
    dec = read("repair_decomposition.csv.gz")
    if len(dec):
        t = dec[dec.axis == "total"]
        s = t.groupby(["toy", "variant"]).agg(cells=("dL", "size"), pass_rate=("pass_lab05", "mean"), dL=("dL", "mean"), score_term=("score_term", "mean"),
                                               weight_term=("weight_term", "mean"), weight_dominates=("weight_term", lambda w: float((w < 0).mean())),
                                               max_residual=("residual", lambda r: float(r.abs().max())), archived_err=("archived_rel_err", "max")).reset_index()
        md(s, "W3-01 RW-A: exact decomposition L(R_hat) - L(R*) of E02 on the frozen TRAIN caches (9 cells per row)",
           "score_term = sum wbar ds (row sparsity change), weight_term = sum sbar dw (energy re-weighting); weight_dominates = share of cells with a negative weight term.")
        ax = dec[dec.axis != "total"].copy(); ax["axis"] = ax.axis.astype(int)
        a = ax[ax.toy.isin(["CR3", "CR5", "CR3L", "CR5L"])].groupby(["toy", "variant", "axis"]).agg(w_ref=("w_ref", "mean"), w_blind=("w_blind", "mean"),
                                                                                                s_ref=("s_ref", "mean"), s_blind=("s_blind", "mean")).reset_index()
        md(a, "W3-01 RW-A: aligned rows of the E02 blind solution vs R* (TRAIN cache; mean over 9 cells; axis = matched parameter index, evaluation-only)")
    ovr = read("repair_old_vs_ref.csv.gz")
    if len(ovr):
        f = ovr[ovr.toy.isin(["CR3", "CR5"])]
        o = ovr.groupby("cid").agg(cells=("old_beats_ref", "size"), old_beats_ref_all=("old_beats_ref", "mean")).join(
            f.groupby("cid").old_beats_ref.mean().rename("old_beats_ref_CR3_CR5")).reset_index().sort_values("old_beats_ref_CR3_CR5")
        md(o, "W3-01: does the old E02 wrong solution still beat R* under each recipe? (share of the 216 / 36 cells)")
    # ---- local flags
    loc = read("local_flags.csv.gz")
    if len(loc):
        p = loc[(loc.fidelity == "pilot") & loc["mask"].isin(["FULL", "MASK-ALL", "ALLROWS"])]
        l = p.groupby("cid").agg(libs=("toy", "size"), O_perfect=("O_perfect", "sum"), O_marginal=("O_marginal", "sum"), I_marginal=("I_marginal", "sum"),
                                  joint_marginal=("both_marginal", "sum"), I_strict_marginal=("I_strict_marginal", "sum"), median_pearson=("pearson", "median"),
                                  worst_O_margin=("O_min_margin", "min")).reset_index()
        l["CR_I_marginal"] = l.cid.map(p[p.toy.str.startswith("CR")].groupby("cid").I_marginal.sum()); l["family"] = l.cid.map(family)
        md(l.sort_values(["joint_marginal", "O_perfect"], ascending=False), "Local Gap 1 / Gap 2 (Q01 pilot / TEST grid, 24 libraries; RW-B and ST-A)")
        c = loc[(loc.fidelity == "C") & loc["mask"].isin(["FULL", "MASK-ALL"])]
        if len(c):
            cc = c.groupby(["cid", "seed"]).agg(O_perfect=("O_perfect", "sum"), I_marginal=("I_marginal", "sum"), joint=("both_marginal", "sum")).reset_index()
            cc = cc.groupby("cid").agg(O_perfect_min=("O_perfect", "min"), I_marginal_min=("I_marginal", "min"), joint_min=("joint", "min")).reset_index()
            md(cc.sort_values("joint_min", ascending=False).head(25), "Local flags at fidelity C (seeds 31/73/101; minimum over seeds, 24 libraries)")
    # ---- blind pilots / finalists
    br = read("pilot_recovery.csv")
    if len(br):
        g = toy_cov(br); ct = coverage_table(g); ct.to_csv(RES / "coverage.csv", index=False)
        md(ct[ct.batch.isin(["RW-C"])], "W3-01 RW-C blind pilot (TRAIN Q01 C seed 31, search seeds 31/73/101; toy = all 3 runs pass)")
        md(ct[ct.batch.isin(["MX-A", "MX-B", "MX-SEL"])], "W3-02 MX-A factorial, MX-B same-frame fusions, MX-SEL blind selector (seed 31 caches)")
        md(ct[ct.batch.isin(["ST-B", "ST-E"])], "W3-03 ST-B gradient / contrast and ST-E reduced-axis blind pilots")
        cat = br.groupby(["batch", "category"]).size().unstack(fill_value=0).reset_index(); md(cat, "Pilot failure categories (all runs)")
        sel = br[br.batch == "MX-SEL"]
        if len(sel):
            md(sel.groupby(["variant"]).agg(runs=("toy", "size"), pass_rate=("pass_lab05", "mean"), abstain_flag_rate=("abstain_flag", "mean"),
                                           pass_when_flagged=("pass_lab05", lambda s: float(s[sel.loc[s.index, "abstain_flag"] == True].mean()) if (sel.loc[s.index, "abstain_flag"] == True).any() else np.nan),
                                           pass_when_not_flagged=("pass_lab05", lambda s: float(s[sel.loc[s.index, "abstain_flag"] != True].mean()))).reset_index(),
               "MX-SEL: disagreement flag (E vs X minimiser distance > .10) vs recovery (reported, never used for selection)")
    par = read("pareto_front.csv")
    if len(par):
        md(par.groupby(["toy", "variant"]).agg(points=("on_front", "size"), front=("on_front", "sum"), front_pass=("pass_lab05", lambda s: int(s[par.loc[s.index, "on_front"]].sum())),
                                               any_pass=("pass_lab05", "sum")).reset_index(), "W3-02 Pareto diagnostics (union of component and fusion finalists, seed 31; T colouring post hoc)")
    fin = read("finalist_recovery.csv")
    if len(fin):
        g = toy_cov(fin); ct = coverage_table(g); ct.to_csv(RES / "finalist_coverage.csv", index=False)
        md(ct, "W3-05 finalists: 3 integration x 3 search seeds (a toy passes only if all 9 runs pass)")
    fd = read("finalist_reeval.csv")
    if len(fd):
        md(fd.groupby(["cid", "variant"]).agg(cells=("key", "size"), D_best_is_selected=("D_best_is_selected", "mean"), escalated=("escalate", "mean"),
                                              seed_spread=("seed_spread", "max")).reset_index(), "W3-05 finalists: independent high-fidelity re-evaluation (blind)")
    fs = read("finalist_sensitivity_recovery.csv")
    if len(fs):
        g = fs.groupby(["cid", "Q", "mask", "variant", "toy"]).pass_lab05.all().reset_index()
        md(g.groupby(["cid", "Q", "mask", "variant"]).pass_lab05.agg(["sum", "size"]).reset_index().rename(columns={"sum": "toys pass", "size": "toys"}),
           "W3-05 finalists: measure / mask sensitivity (seed-31 caches, 3 search seeds)")
    og = read("oracle_summary.csv")
    if len(og):
        md(og[["cid", "target", "category", "best_dL", "best_dsp", "ideal_MCC", "ideal_leak_max", "transfer_all", "practical_adequate"]],
           "W3-05 exact / practical Gap 1 (analytic global search, 3 search seeds; D-cache confirmation)")
        md(og.groupby("cid").agg(targets=("target", "size"), empirical_support=("category", lambda s: int((s == "empirical support").sum())),
                                 counterexamples=("category", lambda s: int((s == "global counterexample").sum())), practical=("practical_adequate", "sum")).reset_index(),
           "Gap 1 summary per candidate (8 analytic targets)")
    # ---- ST-C full vs interior
    if len(br):
        bank = br[br.toy.isin(["CR3", "CR5", "CR3L", "CR5L"]) & br.cid.str.startswith("ST-")]
        if len(bank):
            sc = bank[bank.cid.isin(["ST-DU-glob", "ST-DU-glob-INT", "ST-DU-col", "ST-DU-col-INT", "ST-DA-glob", "ST-DA-glob-INT", "ST-DA-col", "ST-DA-col-INT",
                                      "ST-HE-B08c03", "ST-HE-B01c03"])]
            md(sc.groupby(["cid", "variant"]).agg(runs=("toy", "size"), pass_rate=("pass_lab05", "mean"), MCC=("MCC", "mean"), leak=("leak_max", "mean")).reset_index(),
               "W3-03 ST-C bank designs: full observed rows vs interior-only contrast vs pooled HE on the identical product grid")
    ra = read("reduced_axis_summary.csv.gz")
    if len(ra):
        md(ra, "W3-03 ST-E reduced-axis local sweeps (decoder arm; analytic arm only for unbounded C families)")
    # ---- W3-04
    cd = read("context_diagnostics.csv")
    if len(cd):
        keep = [c for c in ["toy", "rep", "ood_frac", "n_context_median", "R2_mean", "sup_MCC_s9_2", "raw_MCC_s9_2", "query_nll", "query_acc", "query_r2", "repeat_corr_mean",
                            "eff_rank", "cov_cond"] if c in cd.columns]
        md(cd[keep], "W3-04 DC-N / DC-R Stage A diagnostics (TEST; same-context supervised reference, TRAIN-fit frames)")
    cr = read("context_recovery.csv")
    if len(cr):
        g = cr.groupby(["cid", "rep", "variant", "toy"]).pass_same_ctx.all().reset_index()
        md(g.groupby(["cid", "rep", "variant"]).pass_same_ctx.agg(["sum", "size"]).reset_index().rename(columns={"sum": "toys pass", "size": "toys"}),
           "W3-04 Stage B: blind recovery per representation (3 search seeds; criterion against the same-context ceiling)")
    cl = read("context_local.csv")
    if len(cl):
        md(cl.groupby(["cid", "rep", "variant"]).agg(O_perfect=("O_perfect", "sum"), I_marginal=("I_marginal", "sum"), pearson=("pearson", "median")).reset_index(),
           "W3-04 local Gap 2 per representation (identical relative sweeps around the same-context R*)")
    OUT_MD.write_text("# Wave 4-2 third wave: report tables (generated by analyze.py)\n" + "\n".join(LINES), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
