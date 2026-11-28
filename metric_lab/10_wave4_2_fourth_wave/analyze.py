"""Analyzer-generated tables (results/report_tables.md, coverage.csv, next_wave_candidates.csv).  Reads only results/*.csv (post-lock).
usage: python analyze.py [--panel development|confirmation|all]"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import RES, LAB09, PRIMARY, SECONDARY, CAND, E02
import numpy as np, pandas as pd

OUTMD = []


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "_(no rows)_"


def h(title, body):
    OUTMD.append(f"\n## {title}\n\n{body}\n")


def read(name):
    f = RES / name
    return pd.read_csv(f) if f.is_file() else None


def coverage(df, by, panel=PRIMARY):
    """toys passing ALL runs of the cell / toys with any run; seed-level pass counts"""
    g = df.groupby(by + ["toy"]).agg(ok=("pass_lab05", "all"), n=("pass_lab05", "size"), k=("pass_lab05", "sum")).reset_index()
    r = g.groupby(by).agg(toys_all_pass=("ok", "sum"), toys=("toy", "nunique"), runs_pass=("k", "sum"), runs=("n", "sum")).reset_index()
    fail = g[~g.ok].groupby(by).apply(lambda x: " ".join(f"{t}({int(k)}/{int(n)})" for t, k, n in zip(x.toy, x.k, x.n)), include_groups=False).rename("failing (pass/runs)")
    return r.merge(fail.reset_index(), how="left", on=by).fillna({"failing (pass/runs)": "-"})


def lab09_controls():
    f = pd.read_csv(LAB09 / "results" / "finalist_recovery.csv"); f = f[(f.batch == "FIN") & (f.panel == "primary") & f.cid.isin([CAND, E02])].copy()
    f["Q"] = "Q01"; f["fidelity"] = "C"; f["batch"] = "W3-FIN (Lab 09)"; return f


def development():
    rep = read("rep_recovery.csv")
    if rep is not None:
        rep = rep[rep.panel == "primary"]; rep["Qf"] = rep.Q + "@" + rep.fidelity
        ctl = lab09_controls(); ctl["Qf"] = "Q01@C"
        allr = pd.concat([rep, ctl], ignore_index=True)
        cv = coverage(allr[allr.budget != "S1"], ["batch", "cid", "Qf", "variant"]); cv.to_csv(RES / "coverage.csv", index=False)
        h("W4-01 replication coverage (toys passing ALL runs of a cell; search budget S0)", md(cv))
        main = rep[(rep.cid == CAND) & (rep.Q == "Q05") & (rep.budget == "S0")]
        inter = main.groupby(["variant", "toy"]).agg(ok=("pass_lab05", "all"), n=("pass_lab05", "size")).reset_index()
        h("Q05 RW-GL: intersection over REP-1 + REP-2 + REP-3 (every C and D search)", md(inter.groupby("variant").agg(toys_all=("ok", "sum"), runs=("n", "sum")).reset_index()) +
          "\n\nper toy:\n\n" + md(inter.pivot(index="toy", columns="variant", values="ok").reset_index()))
        fails = main[~main.pass_lab05.astype(bool)]
        h("Q05 RW-GL failures: post-lock classification on the identical objective", md(fails[["batch", "toy", "variant", "key", "MCC", "MCC_gap", "leak_max", "L_hat", "L_Rstar",
                                                                                                 "dL_vs_Rstar", "tau", "category"]].sort_values(["variant", "toy"])))
        st = main.groupby(["variant", "toy"]).agg(MCC_med=("MCC", "median"), MCC_min=("MCC", "min"), gap_max=("MCC_gap", "max"), leak_med=("leak_max", "median"),
                                                 leak_max=("leak_max", "max"), sup_MCC=("supervised_MCC", "first"), abs_good=("abs_good", "mean")).reset_index()
        h("Q05 RW-GL distributions over all REP runs", md(st))
        s1 = rep[rep.budget == "S1"]
        if len(s1):
            h("REP-4 blind S1 escalations (reported separately; selections not replaced)", md(s1[["toy", "variant", "key", "L_hat", "MCC", "leak_max", "pass_lab05", "category"]]))
    ie = read("independent_reeval.csv")
    if ie is not None:
        g = ie.groupby(["batch", "variant"]).agg(cells=("key", "size"), escalated=("escalate", "sum"), D211_best_sel=("D211_best_is_selected", "mean"), D613_best_sel=("D613_best_is_selected", "mean"),
                                               not_conv=("not_converged", "sum"), sep_minima=("separated_near_minima", "sum"), seed_incons=("seed_inconsistent", "sum"),
                                               C_below_star=("dL_C_vs_Rstar", lambda x: int((x < 0).sum())), D211_below_star=("dL_D211_vs_Rstar", lambda x: int((x < 0).sum())),
                                               D613_below_star=("dL_D613_vs_Rstar", lambda x: int((x < 0).sum()))).reset_index()
        h("REP-4 independent D re-evaluation of every REP-1/2 cell (rank stability; dL vs R* on identical / independent caches)", md(g))
        t = ie.groupby(["toy", "variant"]).agg(dC=("dL_C_vs_Rstar", "median"), dD211=("dL_D211_vs_Rstar", "median"), dD613=("dL_D613_vs_Rstar", "median"),
                                               dC_min=("dL_C_vs_Rstar", "min"), dC_max=("dL_C_vs_Rstar", "max")).reset_index()
        h("Paired L(R_hat) - L(R*) by toy (median over the 18 REP-1/2 cells; negative = found rotation preferred to R*)", md(t, ".2e"))
    if ie is not None and rep is not None:
        r1 = rep[(rep.batch == "REP-1") & (rep.cid == CAND) & (rep.Q == "Q05") & (rep.budget == "S0")].merge(ie[ie.batch == "REP-1"][["toy", "key", "L_D211_selected"]], on=["toy", "key"])
        out = []
        for (t, v), g in r1.groupby(["toy", "variant"]):
            dr = g.sort_values(["L_D211_selected", "key"]).iloc[0]; fx = g[(g.iseed == 31) & (g.sseed == 31)].iloc[0]
            out.append(dict(toy=t, variant=v, D_ranked_key=dr.key.split("|")[3] + "|" + dr.key.split("|")[4], D_ranked_pass=bool(dr.pass_lab05), fixed_pair_pass=bool(fx.pass_lab05),
                            all_run_pass=int(g.pass_lab05.sum())))
        o = pd.DataFrame(out); o.to_csv(RES / "deployment_readout_dev.csv", index=False)
        h("Single-deployment rules on the development REP-1 cells (fixed pair i31/s31 is the post-hoc FIN-SENS cell itself, hence biased)",
          md(o) + "\n\n" + md(o.groupby("variant")[["D_ranked_pass", "fixed_pair_pass"]].sum().reset_index()))
    mr = read("measure_recovery.csv")
    if mr is not None and rep is not None:
        mg = pd.concat([mr, rep[(rep.cid == CAND) & (rep.Q == "Q05") & (rep.batch == "REP-1")].assign(Q="M-D (=Q05)"), lab09_controls().query("cid == @CAND").assign(Q="M-P (=Q01)")])
        mg = mg[mg.Q.isin(["M-E", "M-J", "M-I", "M-D (=Q05)", "M-P (=Q01)"])]
        h("W4-02 controlled measure grid (3 integration x 3 search seeds, all 12 toys)", md(coverage(mg, ["Q", "variant"])))
        ms = mr[mr.batch.isin(["MS", "MIX"])]
        h("W4-02 covariance / scale / truncation / subpool / mixture sensitivity (s9)", md(coverage(ms, ["batch", "Q"])))
    rb = read("robustness_recovery.csv")
    if rb is not None:
        h("W4-04 robustness: other measures, masks (primary panel)", md(coverage(rb[rb.panel == "primary"], ["batch", "Q", "mask", "variant"])))
        sec = rb[rb.panel == "secondary"]
        if len(sec):
            h("W4-04 secondary saved NPs (5 toys; alternative beta is not a training-seed replicate)", md(coverage(sec, ["cid", "Q", "variant"], SECONDARY)))
    for name, title in (("support_diagnostics.csv", "Support diagnostics (seed 31 C TRAIN laws vs empirical Z)"), ("tail_energy.csv", "Tail energy share of the base-frame UX energy"),
                        ("predictive_by_tail.csv", "Decoder prediction on actual VALIDATION rows by latent tail stratum")):
        d = read(name)
        if d is not None:
            h(title, md(d))
    dec = read("loss_decomposition.csv")
    if dec is not None:
        d = dec[(dec.a == "R_star") & dec.b.isin(["R_Q01", "R_Q05"])]
        h("Exact GL loss differences dL = L(b) - L(R*) = score term + weight term (x .75) + global term (x .25); b_preferred = dL < 0", md(d[["toy", "variant", "law", "b", "dL", "score_term",
                                                                                                                                                  "weight_term", "d_global", "b_preferred"]], ".4f"))
    bb = read("block_bootstrap.csv")
    if bb is not None:
        h("Latent-block bootstrap of loss differences (400 resamples)", md(bb, ".4f"))
    og = read("oracle_global.csv")
    if og is not None:
        h("W4-03 Gap 1 under O-true (analytic, global search)", md(og[["cid", "target", "category", "n_lower", "n_confirmed", "best_dL", "best_dsp", "ideal_MCC", "ideal_leak_max",
                                                                     "transfer_all", "practical_adequate"]], ".4f"))
    g2 = read("gap2_summary.csv")
    g2c = read("gap2_curves.csv.gz")
    if g2c is not None:
        x = g2c[(g2c.mask_role == "primary") & ~g2c.both_marginal.astype(bool)]
        h("W4-03 Gap 2: libraries failing both-marginal (primary mask)", md(x.groupby(["cid", "inferred", "toy", "variant"]).size().rename("n_seeds").reset_index()))
    if g2 is not None:
        p = g2[g2.mask_role == "primary"]
        h("W4-03 Gap 2: both-marginal / perfect counts over the 24 primary-mask libraries (12 toys x 2 variants)", md(p.groupby(["cid", "inferred", "seed", "variant"]).agg(
            libs=("n", "sum"), O_perfect=("O_perfect", "sum"), I_perfect=("I_perfect", "sum"), O_marginal=("O_marginal", "sum"), I_marginal=("I_marginal", "sum"),
            both_marginal=("both_marginal", "sum"), pearson_med=("pearson_med", "median")).reset_index()))


def confirmation():
    c = read("confirmation_recovery.csv"); r = read("confirmation_runs.csv")
    if r is not None:
        h("CONF-A runs (every registered run; Stage A descriptive)", md(r))
    if c is None:
        h("CONF-A", "_not evaluated_"); return
    c["arm"] = c.cid + "|" + c.Q + "|" + c.batch
    h("CONF-A all-run coverage by arm and variant (unit: toy x NP training run; 9 seeds per cell)", md(coverage(c.assign(toy=c.toy_id), ["arm", "variant"])))
    d = c[c.deploy_D | c.deploy_fixed]
    for rule in ("deploy_D", "deploy_fixed"):
        x = c[c[rule]]
        g = x.groupby(["variant", "toy"]).agg(runs_pass=("pass_lab05", "sum"), runs=("pass_lab05", "size")).reset_index()
        h(f"CONF-A single deployment ({rule}): pass per toy over the 2 NP training runs", md(g) + f"\n\ntoys with both NPs passing: " +
          ", ".join(f"{v}: {int((gg.runs_pass == gg.runs).sum())}/{len(gg)}" for v, gg in g.groupby("variant")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--panel", default="all"); a = ap.parse_args()
    OUTMD.append("# Lab 10 report tables (analyze.py; generated from results/*.csv)\n")
    if a.panel in ("development", "all"):
        development()
    if a.panel in ("confirmation", "all"):
        confirmation()
    (RES / "report_tables.md").write_text("\n".join(OUTMD), encoding="utf-8"); print("report_tables.md", len(OUTMD), "sections")
