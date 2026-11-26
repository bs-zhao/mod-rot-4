"""W1-05: synthesis and next-wave hand-off (01_first_wave_README section 7.2).  Deterministic rules, written before reading the tables.

usage: python synthesize.py      (needs results/pass_summary.csv.gz, integration_diagnostics.csv, global_oracle_summary.csv,
                                  bridge_curves.csv.gz, representation_diagnostics.csv)
writes results/metric_evidence.csv (one row per metric, all 415), next_wave_candidates.csv, failure_table.csv, lane_rules.json
Lanes (they can overlap; within a lane rank by coverage, strict margin, seed stability, then name; ties at a cutoff are all kept):
  A theory candidate  A1 : analytic PERFECT on all 24 logit/continuous libraries (12 toys x S9/S9_2, Q01) under at least one mask policy,
                            analytic MARGINAL on all valid libraries for every measure Q01-Q05 (pilot), and no reproducible global
                            counterexample farther than d_SP/sqrt(2k) > 0.05 on any family (global support or inconclusive is allowed)
                     A2 : JOINT (analytic and inferred) marginal pass on all 6 libraries of one class x family-group scope (Q01, MASK-ALL policy),
                            and the analytic arm marginal on every library of that scope for every measure; global result reported
  B metric repair         : analytic constant or wrong-minimum curve on >= 1 primary library, or a reproducible global counterexample (> 0.05)
  C empirical candidate   : inferred MARGINAL on every library of a class x family-group scope (Q01 pilot) while the analytic arm fails on
                            at least one of them (inferred-only); for priority metrics also stable in all three fidelity-C seeds
  D representation-limited: (library level) the supervised bridge B-orth loses >= 20% of >= 10 analytic passes of the priority panel,
                            or the supervised ceiling is low (S8 MCC < 0.9 or leak max >= 0.3)
  E estimand / measurement: pass status that depends on the mask policy (fixed banks), changes across seeds / fidelities, or libraries with
                            mask coverage < 0.5
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2.common import RES, CONFIGS
import numpy as np, pandas as pd

RULES = __doc__.split("Lanes")[1]


def d_scope_ok(P, m, cls, fg):
    d = P[(P.metric == m) & (P.cls == cls) & (P.fam_group == fg) & ((P["mask"] == "FULL") | (P["mask"] == "MASK-ALL"))]
    return bool(len(d)) and bool(d.O_marginal.all())


def main():
    ps = pd.read_csv(RES / "pass_summary.csv.gz", low_memory=False)
    ps = ps[(ps.panel == "primary") & (ps.analytic_source == "oracle")]
    pan = json.loads((CONFIGS / "priority_panel.json").read_text()); prio = set(pan["ids"])
    A = ps[(ps.Q == "Q01") & (ps.fidelity == "pilot")]; LC = A[A.channel != "YP"]
    ev = []
    for m, g in LC.groupby("metric"):
        r = dict(metric=m, priority=m in prio, role=pan["roles"].get(m, ""))
        for pol in ("MASK-ALL", "MASK-COL"):
            d = g[(g["mask"] == "FULL") | (g["mask"] == pol)]
            r[f"O_perfect_libs_{pol}"] = int(d.O_perfect.sum()); r[f"O_marginal_libs_{pol}"] = int(d.O_marginal.sum())
            for cls in ("B", "C"):
                for fg in ("C", "CR"):
                    dd = d[(d.cls == cls) & (d.fam_group == fg)]
                    r[f"I_marginal_{cls}_{fg}_{pol}"] = int(dd.I_marginal.sum()); r[f"both_marginal_{cls}_{fg}_{pol}"] = int(dd.both_marginal.sum())
                    r[f"both_perfect_{cls}_{fg}_{pol}"] = int(dd.both_perfect.sum())
            r[f"min_O_margin_{pol}"] = float(d.O_min_margin.min()); r[f"O_fail_constant_{pol}"] = int((d.O_fail == "constant").sum())
            r[f"O_fail_wrong_min_{pol}"] = int((d.O_fail == "wrong_minimum").sum())
        r["mask_dependent"] = bool(g[g["mask"] == "MASK-ALL"].set_index(["toy", "variant"])[["O_marginal", "I_marginal"]].ne(
            g[g["mask"] == "MASK-COL"].set_index(["toy", "variant"])[["O_marginal", "I_marginal"]]).any().any()) if (g["mask"] == "MASK-ALL").any() else False
        ev.append(r)
    ev = pd.DataFrame(ev).set_index("metric")
    # measure stability of the analytic arm (pilot, all measures, every valid library)
    P = ps[(ps.fidelity == "pilot") & (ps.channel != "YP")]
    for pol in ("MASK-ALL", "MASK-COL"):
        d = P[(P["mask"] == "FULL") | (P["mask"] == pol)]
        ok = d.groupby(["metric", "Q"]).O_marginal.all().unstack()
        ev[f"O_marginal_all_measures_{pol}"] = ok.all(1).reindex(ev.index).fillna(False)
        ev[f"O_measures_ok_{pol}"] = ok.sum(1).reindex(ev.index)
    # seed stability (fidelity C, priority panel)
    integ = pd.read_csv(RES / "integration_diagnostics.csv.gz") if (RES / "integration_diagnostics.csv.gz").is_file() else None
    if integ is not None:
        q1 = integ[(integ.Q == "Q01") & (integ.channel != "YP")]
        if "C_n_seeds" in q1:
            st = q1.assign(O_stable=lambda x: x.C_O_marginal_seeds.isin([0, 3]), I_stable=lambda x: x.C_I_marginal_seeds.isin([0, 3]))
            ok = st.O_stable & st.I_stable
            ev["seed_stable_frac_C"] = ok.groupby(st.metric).mean().reindex(ev.index)
            ev["seed_status_changes_C"] = (~ok).groupby(st.metric).sum().reindex(ev.index)
    # global analytic categories (L targets)
    gl = pd.read_csv(RES / "global_oracle_summary.csv") if (RES / "global_oracle_summary.csv").is_file() else None
    if gl is not None:
        L = gl[gl.channel == "L"]
        for col, tag in (("category", "1e-3"), ("category_dsp>0.05", "0.05"), ("category_dsp>0.15", "0.15")):
            if col in L:
                ev[f"global_counterexamples_{tag}"] = L.groupby("metric")[col].apply(lambda s: int((s == "global counterexample").sum())).reindex(ev.index)
                ev[f"global_targets_{tag}"] = L.groupby("metric")[col].size().reindex(ev.index)
        ev["global_families_counterexample_0.05"] = L[L["category_dsp>0.05"] == "global counterexample"].groupby("metric").family.apply(lambda s: ",".join(sorted(set(s)))).reindex(ev.index)
    # bridge (priority)
    br = pd.read_csv(RES / "bridge_curves.csv.gz") if (RES / "bridge_curves.csv.gz").is_file() else None
    if br is not None:
        b = br[(br.channel != "YP") & ((br["mask"] == "FULL") | (br["mask"] == "MASK-ALL"))]
        for arm in ("D", "S", "BO", "BA"):
            ev[f"bridge_{arm}_marginal_libs"] = b[b.arm == arm].groupby("metric").arm_marginal.sum().reindex(ev.index)
        ev["bridge_O_marginal_libs"] = b[b.arm == "O"].groupby("metric").arm_marginal.sum().reindex(ev.index)
    ev.to_csv(RES / "metric_evidence.csv")
    # -------------------------------------------------------------- lanes
    cand = []
    for m, r in ev.iterrows():
        perf24 = max(r["O_perfect_libs_MASK-ALL"], r["O_perfect_libs_MASK-COL"]) == 24
        meas = bool(r["O_marginal_all_measures_MASK-ALL"] or r["O_marginal_all_measures_MASK-COL"])
        gce = r.get("global_counterexamples_0.05", np.nan); gassessed = not pd.isna(gce)
        ctrl = "control" in str(r["role"]) or m.startswith("L06|CTRL")
        if perf24 and meas and (not gassessed or gce == 0) and not ctrl:
            cand.append(dict(entity="metric", id=m, lane="A1", reason="analytic perfect on all 24 L/C libraries, measure-stable" + ("" if gassessed else "; global diagnostic not run (not in priority panel)"),
                             evidence=f"O perfect {max(r['O_perfect_libs_MASK-ALL'], r['O_perfect_libs_MASK-COL'])}/24; inferred marginal B C/CR {r['I_marginal_B_C_MASK-ALL']}/6,{r['I_marginal_B_CR_MASK-ALL']}/6; C C/CR {r['I_marginal_C_C_MASK-ALL']}/6,{r['I_marginal_C_CR_MASK-ALL']}/6; global counterexamples(>0.05) {gce}",
                             rank_key=(-(r['I_marginal_B_CR_MASK-ALL'] + r['I_marginal_C_CR_MASK-ALL']), -r["min_O_margin_MASK-ALL"])))
        fails = r["O_fail_constant_MASK-ALL"] + r["O_fail_wrong_min_MASK-ALL"]
        if (fails > 0 or (gassessed and gce > 0)) and not ctrl:
            why = []
            if r["O_fail_constant_MASK-ALL"]: why.append(f"analytic constant on {r['O_fail_constant_MASK-ALL']} libs")
            if r["O_fail_wrong_min_MASK-ALL"]: why.append(f"analytic wrong minimum on {r['O_fail_wrong_min_MASK-ALL']} libs")
            if gassessed and gce > 0: why.append(f"global counterexample (>0.05) on {r.get('global_families_counterexample_0.05')}")
            if m in prio:
                cand.append(dict(entity="metric", id=m, lane="B", reason="; ".join(why), evidence=f"O perfect {r['O_perfect_libs_MASK-ALL']}/24 (MASK-ALL)", rank_key=(-r["O_perfect_libs_MASK-ALL"],)))
        for cls in ("B", "C"):
            for fg in ("C", "CR"):
                n = r[f"I_marginal_{cls}_{fg}_MASK-ALL"]; o = r[f"both_marginal_{cls}_{fg}_MASK-ALL"]
                stable = r.get("seed_status_changes_C", np.nan)
                if n == 6 and o == 6 and not ctrl and not perf24:
                    sc = d_scope_ok(P, m, cls, fg)
                    if sc:
                        cand.append(dict(entity="metric", id=m, lane="A2", reason=f"joint marginal pass on all 6 class-{cls} {fg}-family libraries; analytic measure-stable in that scope",
                                         evidence=f"both perfect {r[f'both_perfect_{cls}_{fg}_MASK-ALL']}/6; global counterexamples(>0.05) {gce} ({r.get('global_families_counterexample_0.05')})", scope=f"{cls}:{fg}", rank_key=(-r[f'both_perfect_{cls}_{fg}_MASK-ALL'], m)))
                if n == 6 and o < 6 and (m not in prio or stable == 0) and not ctrl:
                    cand.append(dict(entity="metric", id=m, lane="C", reason=f"inferred marginal on all 6 class-{cls} {fg}-family libraries" + ("" if m in prio else " (pilot only; not seed-checked)"),
                                     evidence=f"analytic also passes on {o}/6; seed status changes (C) {stable}", scope=f"{cls}:{fg}", rank_key=(-o, m)))
        if r["mask_dependent"] or (not pd.isna(r.get("seed_status_changes_C", np.nan)) and r.get("seed_status_changes_C", 0) > 0):
            if m in prio:
                cand.append(dict(entity="metric", id=m, lane="E", reason=("mask-policy dependent pass status; " if r["mask_dependent"] else "") +
                                 (f"{int(r.get('seed_status_changes_C', 0))} (library x measure x channel) cells change status across seeds" if r.get("seed_status_changes_C", 0) else ""),
                                 evidence="", rank_key=(m,)))
    # library-level lanes D / E and the failure table (priority panel, Q01 pilot, bridge points)
    fail = []
    rep = pd.read_csv(RES / "representation_diagnostics.csv") if (RES / "representation_diagnostics.csv").is_file() else None
    if br is not None:
        for (toy, v, ch, mk), g in br[br.metric.isin(prio)].groupby(["toy", "variant", "channel", "mask"]):
            O = g[g.arm == "O"].set_index("metric").arm_marginal
            frac = {a: float(g[g.arm == a].set_index("metric").arm_marginal[O[O].index].mean()) if O.any() else np.nan for a in ("D", "S", "BO", "BA")}
            pattern = ("no analytic passes on this library (nothing to keep)" if not O.any() else
                       "O good, bridges good, D bad -> decoder / conditional surface" if frac["BO"] >= .8 and frac["BA"] >= .8 and frac["D"] < .5 else
                       "O good, B-orth bad -> alignment / geometry" if frac["BO"] < .8 else
                       "O good, B-aff bad -> affine map / residual / support" if frac["BA"] < .8 else "decoder keeps most analytic passes")
            row = dict(toy=toy, variant=v, channel=ch, mask=mk, analytic_passes=int(O.sum()), **{f"frac_kept_{a}": frac[a] for a in frac},
                       S_better_than_D=bool(frac["S"] > frac["D"]), pattern=pattern, bridge_valid=bool(g.bridge_valid.iloc[0]), coverage=float(g.coverage.iloc[0]))
            fail.append(row)
            if O.sum() >= 10 and frac["BO"] < .8:                       # at least 10 analytic passes to judge a loss of 20%
                cand.append(dict(entity="library", id=f"{toy}|{v}|{ch}|{mk}", lane="D", reason="B-orth loses >= 20% of the analytic passes", evidence=f"kept by BO {frac['BO']:.2f}, BA {frac['BA']:.2f}, D {frac['D']:.2f}", rank_key=(frac["BO"],)))
    if rep is not None:
        for _, r in rep.iterrows():
            lim = 0.3 if str(r.family).startswith("CR") else 0.2
            if r.supervised_MCC < .9 or r.supervised_leak_max >= lim:
                cand.append(dict(entity="library", id=f"{r.toy}|{r.variant}", lane="D", reason="low supervised ceiling", evidence=f"S8 MCC {r.supervised_MCC:.3f}, leak max {r.supervised_leak_max:.3f}", rank_key=(r.supervised_MCC,)))
    cov = pd.read_csv(RES / "coverage.csv") if (RES / "coverage.csv").is_file() else None
    if cov is not None:
        low = cov[(cov.coverage_min < .5) & (cov.analytic_source == "oracle")].groupby(["toy", "mask"]).coverage_min.min()
        for (toy, mk), c in low.items():
            cand.append(dict(entity="library", id=f"{toy}|{mk}", lane="E", reason="mask coverage < 0.5 (derivative estimand on a small interior subset)", evidence=f"min coverage {c:.2f}", rank_key=(c,)))
    cd = pd.DataFrame(cand)
    cd["rank_key"] = cd.rank_key.astype(str)
    cd = cd.sort_values(["lane", "entity", "rank_key", "id"]); cd["rank_in_lane"] = cd.groupby(["lane", "scope"] if "scope" in cd else ["lane"], dropna=False).cumcount() + 1
    cd["limitations"] = "development selection on the current 12 toys (synthetic truth); needs frozen independent confirmation (Step 6C)"
    cd.drop(columns="rank_key").to_csv(RES / "next_wave_candidates.csv", index=False)
    pd.DataFrame(fail).to_csv(RES / "failure_table.csv", index=False)
    (RES / "lane_rules.json").write_text(json.dumps(dict(rules=RULES.strip()), indent=1))
    print(cd.groupby(["lane", "entity"]).size().to_string())


if __name__ == "__main__":
    main()
