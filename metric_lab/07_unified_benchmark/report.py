"""Markdown tables for the README, generated only from the saved result tables (no chart-only or hand-typed numbers).
usage: python report.py  -> results/report_tables.md"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2.common import RES, CONFIGS
import numpy as np, pandas as pd

TOYS_B = ["C1", "C2", "C3", "CR3", "CR4", "CR5"]; TOYS_C = [t + "L" for t in TOYS_B]


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values) + "\n"


def main():
    out = []
    ch = pd.read_csv(RES / "validation_checks.csv"); fd = pd.read_csv(RES / "fd_check.csv")
    out.append(f"### W1-00\n\n{len(ch)} checks, {int((~ch.passed).sum())} failed.\n\n" +
               md(fd.groupby("method")[["max_abs_rel_err", "ux_moment_rel_err", "frac_points_crossing_x_boundary"]].max().reset_index(), 4))
    cc = pd.read_csv(RES / "class_pass_counts.csv")
    q = cc[(cc.Q == "Q01") & (cc.fidelity == "pilot")]
    cols = {"n_libraries": "libs", "n_metrics_all_O_perfect": "analytic perfect", "n_metrics_all_I_perfect": "inferred perfect", "n_metrics_all_I_marginal": "inferred marginal",
            "n_metrics_all_both_perfect": "both perfect", "n_metrics_all_both_marginal": "both marginal", "n_metrics_all_both_strict_perfect": "both strict-perfect",
            "n_metrics_all_both_perfect_legacy": "both perfect (legacy tol.)"}
    for cls in ("B", "C"):
        t = q[q.cls == cls][["channel_group", "mask_policy", "scope"] + list(cols)].rename(columns=cols)
        out.append(f"### W1-01 class {cls}: metrics (of 415) passing on EVERY library of the scope (Q01, pilot, seed 31)\n\n" + md(t))
    fw = pd.read_csv(RES / "fourway.csv"); f = fw[(fw.Q == "Q01") & (fw.fidelity == "pilot") & (fw["mask"].isin(["FULL", "MASK-ALL"]))]
    f = f.groupby(["cls", "fam_group", "channel"])[["cells", "O_pass_I_pass", "O_pass_I_fail", "O_fail_I_pass", "O_fail_I_fail", "O_fail_constant", "O_fail_wrong_min", "I_fail_constant", "I_fail_wrong_min"]].sum().reset_index()
    out.append("### W1-01 four-way table (library x metric cells, marginal flags, Q01 pilot, FULL / MASK-ALL)\n\n" + md(f))
    ps = pd.read_csv(RES / "pass_summary.csv.gz", low_memory=False); pan = json.loads((CONFIGS / "priority_panel.json").read_text())
    a = ps[(ps.panel == "primary") & (ps.Q == "Q01") & (ps.fidelity == "pilot") & (ps.analytic_source == "oracle") & ps["mask"].isin(["FULL", "MASK-ALL"])]
    e1 = a.groupby(["toy", "channel"]).agg(pearson_all=("pearson", "median"), pearson_priority=("pearson", lambda s: s[a.loc[s.index, "metric"].isin(pan["ids"])].median())).reset_index()
    cr = pd.read_csv(RES / "correlations.csv.gz"); cr = cr[(cr.Q == "Q01") & (cr.fidelity == "pilot") & (cr.analytic_source == "oracle") & cr["mask"].isin(["FULL", "MASK-ALL"])]
    e1 = e1.merge(cr.groupby(["toy", "channel"])[["UX_rel_err", "UU_rel_err", "oracle_zero_false_energy"]].mean().reset_index(), on=["toy", "channel"])
    e1["order"] = e1.toy.map({t: i for i, t in enumerate(TOYS_B + TOYS_C)}); e1 = e1.sort_values(["order", "channel"]).drop(columns="order")
    out.append("### W1-01 evaluation 1 (median pooled Pearson analytic vs decoder over metrics; matrix errors averaged over rotations; S9 and S9_2 pooled)\n\n" + md(e1))
    # W1-02 measures
    m = cc[(cc.fidelity == "pilot") & (cc.channel_group == "L/C") & (cc.mask_policy == "MASK-ALL") & (cc.scope != "all")]
    m = m.pivot_table(index=["cls", "scope"], columns="Q", values="n_metrics_all_both_marginal").reset_index()
    out.append("### W1-02 measure sensitivity: metrics both-marginal on every library of the scope (pilot, L/C, MASK-ALL)\n\n" + md(m, 0))
    if (RES / "integration_diagnostics.csv.gz").is_file():
        ig = pd.read_csv(RES / "integration_diagnostics.csv.gz")
        rows = []
        for fid in ("C", "D"):
            if f"{fid}_n_seeds" not in ig:
                continue
            g = ig[ig[f"{fid}_n_seeds"] == 3]
            for cls, gg in g.groupby("cls"):
                st = gg[f"{fid}_both_marginal_seeds"].isin([0, 3]) & gg[f"{fid}_O_marginal_seeds"].isin([0, 3]) & gg[f"{fid}_I_marginal_seeds"].isin([0, 3])
                rows.append(dict(fidelity=fid, cls=cls, cells=len(gg), frac_same_status_3_seeds=float(st.mean()), frac_both_marginal_all_3=float((gg[f"{fid}_both_marginal_seeds"] == 3).mean())))
        if "stable_last_two_fidelities" in ig:
            for cls, gg in ig.dropna(subset=["stable_last_two_fidelities"]).groupby("cls"):
                rows.append(dict(fidelity="C+D", cls=cls, cells=len(gg), frac_same_status_3_seeds=float(gg.stable_last_two_fidelities.astype(bool).mean())))
        out.append("### W1-02 seed / fidelity stability of the priority panel (cells = library x measure x channel x mask x metric)\n\n" + md(pd.DataFrame(rows)))
    # W1-03
    if (RES / "representation_diagnostics.csv").is_file():
        rp = pd.read_csv(RES / "representation_diagnostics.csv")
        rp = rp[["toy", "variant", "supervised_MCC", "supervised_min", "supervised_leak_max", "unrotated_MCC", "unrotated_leak_max", "linear_R2_min", "cov_condition", "map_condition"]]
        out.append("### W1-03 representation (TEST subjects; evaluation only)\n\n" + md(rp))
        ft = pd.read_csv(RES / "failure_table.csv")
        ft = ft[ft["mask"].isin(["FULL", "MASK-ALL"])][["toy", "variant", "channel", "analytic_passes", "frac_kept_BO", "frac_kept_BA", "frac_kept_D", "frac_kept_S", "coverage", "pattern"]]
        out.append("### W1-03 bridge: fraction of the analytic passes (priority panel, marginal) kept by each surface, same points\n\n" + md(ft, 2))
        bc = pd.read_csv(RES / "bridge_curves.csv.gz"); bc = bc[bc["mask"].isin(["FULL", "MASK-ALL"])]
        O = bc[bc.arm == "O"][["toy", "variant", "channel", "metric", "arm_marginal"]].rename(columns={"arm_marginal": "O"})
        rows = []
        for br in ("BO", "BA"):
            for a_, arm in ((0.0, br), (0.25, f"{br}:0.25"), (0.5, f"{br}:0.5"), (0.75, f"{br}:0.75"), (1.0, "D")):
                g = bc[bc.arm == arm].merge(O, on=["toy", "variant", "channel", "metric"]); g = g[g.O]
                for fg, gg in g.assign(fg=g.toy.str.startswith("CR").map({True: "CR", False: "C"}), cls=g.toy.str.endswith("L").map({True: "C", False: "B"})).groupby(["cls", "fg"]):
                    rows.append(dict(bridge=br, a=a_, cls=fg[0], family_group=fg[1], frac_kept=float(gg.arm_marginal.mean())))
        it = pd.DataFrame(rows).pivot_table(index=["bridge", "cls", "family_group"], columns="a", values="frac_kept").reset_index()
        out.append("### W1-03 interpolation (1-a) bridge + a decoder: fraction of analytic passes kept\n\n" + md(it, 2))
        fdg = pd.read_csv(RES / "fit_diagnostics.csv")
        out.append("### W1-03 T-free surrogate S (selection on VALIDATION loss)\n\n" + md(fdg[["toy", "selected_degree", "selected_lambda", "half_refit_UX_rel_diff", "S_vs_D_eta_corr_Q01", "D_vs_O_eta_corr_Q01(eval-only)", "S_vs_O_eta_corr_Q01(eval-only)"]], 3))
    if (RES / "global_oracle_summary.csv").is_file():
        gl = pd.read_csv(RES / "global_oracle_summary.csv")
        for col in ("category", "category_dsp>0.05", "category_dsp>0.15"):
            t = gl.assign(cat=gl[col].str.replace(r" \(.*", "", regex=True)).pivot_table(index=["family", "channel", "mask"], columns="cat", values="metric", aggfunc="count", fill_value=0).reset_index()
            out.append(f"### W1-04 global analytic categories per target (priority-panel metrics; distance threshold {col.split('>')[-1] if '>' in col else '1e-3'})\n\n" + md(t, 0))
        L = gl[gl.channel == "L"]
        sup = L.groupby("metric")["category_dsp>0.05"].apply(lambda s: s.str.startswith("empirical global support").all())
        out.append("Metrics with empirical global support (d_SP > 0.05) on every logit/continuous target: " + ", ".join(f"`{m}`" for m in sorted(sup[sup].index)) + "\n")
    if (RES / "next_wave_candidates.csv").is_file():
        nw = pd.read_csv(RES / "next_wave_candidates.csv")
        out.append("### W1-05 lanes\n\n" + md(nw.groupby(["lane", "entity"]).size().rename("n").reset_index(), 0))
    (RES / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out)[:6000])


if __name__ == "__main__":
    main()
