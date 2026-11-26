"""W1-01/02/05 analysis: evaluations 1 and 2 from the saved loss files, class tables, frozen selection rules.

usage: python analyze.py --panel primary            (all available batches -> results/*.csv)
       python analyze.py --freeze-priority          (after batch A: deterministic W1-01 rule -> configs/priority_panel.json)
       python analyze.py --freeze-escalation        (after batch C: Batch-D escalation rule -> configs/escalation_D.json)
Outputs (results/): pass_summary.csv.gz, correlations.csv.gz, class_pass_counts.csv, fourway.csv, coverage.csv, integration_diagnostics.csv,
                    priority_selection.csv;  large: outputs/.../W1/analysis/sweep_minima.csv.gz, metrics_long_priority.csv.gz
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2 import common
from wave4_2.common import RES, CONFIGS, OUT, libraries
import numpy as np, pandas as pd
from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1, unique_rows, matrix_errors

RUN = OUT / "W1"; ANA = RUN / "analysis"
FLAGS = ("perfect", "marginal", "strict_perfect", "strict_marginal")
HIST17 = ["POOL.lq.1", "POOL.lq.25", "POOLrms.ent", "POOLrms.lq.5", "SL0.UX.e0.01", "SL0.UX.e0.03", "SL0.UX.e0.05", "SL0.UXUU.e0.01", "SL0.UXUU.e0.03",
          "SL0.UXUU.e0.05", "SL0m.UX", "SL0m.UX.wide", "SL0m.UXUU", "SL0m.UXUU.wide", "UX.lq.1", "UXcol.a0.lq.25", "UXrms.lq.5"]
CR_CONT6 = ["Fz.A+B+C", "Fz.A+K+C", "UXrow.a0.5.lq.25", "rowsp.a0.5.q0.1", "rowsp.a0.5.q0.5", "rowsp.a0.75.q0.1"]
FIXED_PANEL = ([f"L03|{m}" for m in HIST17] + ["L03|Fz.A+B+C+G"] + [f"L03|{m}" for m in CR_CONT6] + ["L03|rowsp.a0.25.q0.1"] +
               ["L03|rowsp.a1.0.q0.1", "L03|UXrow.a1.lq.25", "L03|CAND.rowsp.75+UXcol.hoy"] +
               ["L06|B01.M03.c0.01", "L06|B01.M03.c0.03", "L06|B01.M04.c0.01", "L06|B08.M03.c0.01", "L06|B08.M03.c0.03", "L06|B09.w75.M03.c0.01"] +
               ["L06|CTRL.EA", "L06|CTRL.EB", "L03|UUdiag.hoy", "L06|B02.M03.c0.01"])
FIXED_ROLE = {**{i: "hist17_analytic_perfect" for i in FIXED_PANEL[:17]}, "L03|Fz.A+B+C+G": "hist_extra_marginal",
              **{f"L03|{m}": "cr_continuous_joint_perfect" for m in CR_CONT6}, "L03|rowsp.a0.25.q0.1": "cr_continuous_joint_marginal",
              "L03|rowsp.a1.0.q0.1": "inferred_leader", "L03|UXrow.a1.lq.25": "inferred_leader", "L03|CAND.rowsp.75+UXcol.hoy": "inferred_leader",
              **{i: "lab06_candidate" for i in FIXED_PANEL[28:34]}, "L06|CTRL.EA": "control_invariant_energy", "L06|CTRL.EB": "control_invariant_energy",
              "L03|UUdiag.hoy": "control_undefined_or_weak_UU", "L06|B02.M03.c0.01": "control_weak_UU"}


def lib_index():
    idx = {}
    for panel in ("primary", "secondary"):
        for l in libraries(panel):
            idx[l["run"]] = l
    return idx


def iter_loss_files():
    for f in sorted(RUN.glob("*/*/*/losses/*/*/*.npz")):
        rel = f.relative_to(RUN).parts           # model, cond, rep, 'losses', variant, tag, file
        run = "/".join(rel[:3]); Q, fid, seed = rel[5].split("_"); ch, mask = f.stem.split("_", 1)
        yield f, run, rel[4], Q, fid, int(seed[1:]), ch, mask


def analyse_file(f, lib, variant, Q, fid, seed, ch, mask, want_sweeps=False):
    z = np.load(f, allow_pickle=False); ids = [str(i) for i in z["ids"]]; man = json.loads(str(z["manifest"]))
    sw = sweeps(man); n_exp = 2 ** lib["k"] - lib["k"] - 1; reg = np.array([i.split("|")[0] for i in ids])
    rows, swrows, corr = [], [], []
    Li = z["nn|L"]; csI = curve_stats(Li, sw, "harmonised"); csIl = curve_stats(Li, sw, "legacy", reg); fI = library_flags(csI, n_exp); fIl = library_flags(csIl, n_exp)
    mats = z["nn|A"].reshape(len(man), -1); keep, ded = unique_rows(man, np.concatenate([mats, z["nn|B"].reshape(len(man), -1)], 1))
    for osrc in [s for s in ("oracle", "oracle_leg") if f"{s}|L" in z.files]:
        Lo = z[f"{osrc}|L"]; csO = curve_stats(Lo, sw, "harmonised"); csOl = curve_stats(Lo, sw, "legacy", reg)
        fO = library_flags(csO, n_exp); fOl = library_flags(csOl, n_exp)
        e1 = eval1(Lo, Li, man, sw); e1d = eval1(Lo, Li, man, sw, rows=ded)
        info_o = json.loads(str(z[f"{osrc}|info"])); info_i = json.loads(str(z["nn|info"]))
        base = dict(toy=lib["toy"], cls=lib["cls"], family=lib["family"], fam_group="CR" if lib["family"].startswith("CR") else "C", panel=lib["panel"],
                    variant=variant, Q=Q, fidelity=fid, seed=seed, channel=ch, mask=mask, analytic_source=osrc,
                    valid_oracle=info_o["valid"], valid_nn=info_i["valid"], n_sweeps=fO["n_sweeps"], complete=bool(fO["complete"]))
        for j, mid in enumerate(ids):
            r = dict(base, metric=mid)
            for fl in FLAGS:
                r[f"O_{fl}"] = bool(fO[fl][j]); r[f"I_{fl}"] = bool(fI[fl][j]); r[f"both_{fl}"] = bool(fO[fl][j] and fI[fl][j])
            r["O_perfect_legacy"] = bool(fOl["perfect"][j]); r["O_marginal_legacy"] = bool(fOl["marginal"][j])
            r["I_perfect_legacy"] = bool(fIl["perfect"][j]); r["I_marginal_legacy"] = bool(fIl["marginal"][j])
            r["both_perfect_legacy"] = r["O_perfect_legacy"] and r["I_perfect_legacy"]; r["both_marginal_legacy"] = r["O_marginal_legacy"] and r["I_marginal_legacy"]
            for side, fl_, cs in (("O", fO, csO), ("I", fI, csI)):
                r[f"{side}_n_marginal_sweeps"] = int(fl_["n_marginal"][j]); r[f"{side}_min_margin"] = float(fl_["min_margin"][j])
                r[f"{side}_constant"] = bool(fl_["any_constant"][j]); r[f"{side}_nonfinite"] = bool(fl_["any_nonfinite"][j])
                r[f"{side}_n_interior_tie"] = int(fl_["n_interior_tie"][j])
                r[f"{side}_fail"] = ("pass" if fl_["marginal"][j] else "nonfinite" if fl_["any_nonfinite"][j] else "constant" if fl_["any_constant"][j] else "wrong_minimum")
            r.update(pearson=float(e1["pearson"][j]), spearman=float(e1["spearman"][j]), curve_pearson_mean=float(e1["curve_mean"][j]),
                     curve_pearson_min=float(e1["curve_min"][j]), pearson_dedup=float(e1d["pearson"][j]), n_unique=e1["n"], n_unique_dedup=e1d["n"],
                     range_O=float(e1["range_o"][j]), range_I=float(e1["range_i"][j]))
            rows.append(r)
        # matrix-level errors (rotation-averaged), evaluation only
        errs = np.array([matrix_errors(z[f"{osrc}|A"][i].astype(float), z[f"{osrc}|B"][i].astype(float), z["nn|A"][i].astype(float), z["nn|B"][i].astype(float))
                         for i in range(len(man))])
        corr.append(dict({k: v for k, v in base.items()}, UU_rel_err=float(np.nanmean(errs[:, 0])) if np.isfinite(errs[:, 0]).any() else np.nan,
                         UX_rel_err=float(np.nanmean(errs[:, 1])), oracle_zero_false_energy=float(np.nanmean(errs[:, 2])),
                         oracle_zero_A=info_o["zero_A"], oracle_zero_B=info_o["zero_B"], nn_zero_A=info_i["zero_A"], coverage_min=float(np.min(info_o["coverage"])),
                         n_points_oracle=info_o["n_points"], n_points_nn=info_i["n_points"], nonfinite_oracle=info_o["nonfinite"], nonfinite_nn=info_i["nonfinite"],
                         pearson_median=float(np.nanmedian(e1["pearson"])) if np.isfinite(e1["pearson"]).any() else np.nan))
        if want_sweeps and osrc == "oracle":
            for si, (ax, _) in enumerate(sw):
                for side, cs in (("O", csO), ("I", csI)):
                    swrows.append(pd.DataFrame(dict(toy=lib["toy"], variant=variant, Q=Q, fidelity=fid, seed=seed, channel=ch, mask=mask, side=side,
                                                    sweep="-".join(str(a + 1) for a in ax), metric=ids, valid=cs["valid"][si], minset=cs["minset"][si],
                                                    perfect=cs["perfect"][si], marginal=cs["marginal"][si], strict_perfect=cs["strict_perfect"][si],
                                                    margin=cs["margin"][si], interior_tie=cs["interior_tie"][si], endpoint_diff=cs["endpoint_diff"][si], range=cs["range"][si])))
    return rows, corr, swrows


def collect(panel=None, batches_fid=None, want_sweeps=False, Qs=None):
    idx = lib_index(); rows, corr, sws = [], [], []
    for f, run, variant, Q, fid, seed, ch, mask in iter_loss_files():
        lib = idx[run]
        if panel and lib["panel"] != panel:
            continue
        if batches_fid and fid not in batches_fid:
            continue
        if Qs and Q not in Qs:
            continue
        r, c, s = analyse_file(f, lib, variant, Q, fid, seed, ch, mask, want_sweeps)
        rows += r; corr += c; sws += s
    return pd.DataFrame(rows), pd.DataFrame(corr), (pd.concat(sws) if sws else None)


def mask_policy(df, policy):
    """bank libraries contribute the chosen mask arm, the others their single FULL arm"""
    return df[(df["mask"] == "FULL") | (df["mask"] == policy)].assign(mask_policy=policy)


def chan_group(ch):
    return "P" if ch == "YP" else "L/C"


def class_counts(ps):
    """number of metrics passing on EVERY library of a scope, per class / channel group / config"""
    out = []
    ps = ps[ps.analytic_source == "oracle"].copy(); ps["chgrp"] = ps.channel.map(chan_group)
    for policy in ("MASK-ALL", "MASK-COL"):
        d = mask_policy(ps, policy)
        for keys, g in d.groupby(["cls", "chgrp", "Q", "fidelity", "seed"]):
            nlib = {"C": g[g.fam_group == "C"].groupby(["toy", "variant"]).ngroups, "CR": g[g.fam_group == "CR"].groupby(["toy", "variant"]).ngroups}
            for scope, gg in (("C-family", g[g.fam_group == "C"]), ("CR-family", g[g.fam_group == "CR"]), ("all", g)):
                n = gg.groupby(["toy", "variant"]).ngroups
                if n == 0:
                    continue
                agg = gg.groupby("metric")[[f"{s}_{fl}" for s in ("O", "I", "both") for fl in FLAGS] + ["both_perfect_legacy", "both_marginal_legacy"]].sum()
                row = dict(cls=keys[0], channel_group=keys[1], Q=keys[2], fidelity=keys[3], seed=keys[4], mask_policy=policy, scope=scope, n_libraries=n)
                for c in agg.columns:
                    row[f"n_metrics_all_{c}"] = int((agg[c] == n).sum())
                out.append(row)
    return pd.DataFrame(out)


def fourway(ps):
    ps = ps[(ps.analytic_source == "oracle")].copy(); out = []
    for keys, g in ps.groupby(["cls", "fam_group", "channel", "Q", "fidelity", "seed", "mask"]):
        O, I = g.O_marginal, g.I_marginal
        out.append(dict(zip(["cls", "fam_group", "channel", "Q", "fidelity", "seed", "mask"], keys), cells=len(g), O_pass_I_pass=int((O & I).sum()),
                        O_pass_I_fail=int((O & ~I).sum()), O_fail_I_pass=int((~O & I).sum()), O_fail_I_fail=int((~O & ~I).sum()),
                        O_fail_constant=int((g.O_fail == "constant").sum()), O_fail_wrong_min=int((g.O_fail == "wrong_minimum").sum()),
                        I_fail_constant=int((g.I_fail == "constant").sum()), I_fail_wrong_min=int((g.I_fail == "wrong_minimum").sum()),
                        O_fail_nonfinite=int((g.O_fail == "nonfinite").sum()), I_fail_nonfinite=int((g.I_fail == "nonfinite").sum())))
    return pd.DataFrame(out)


def freeze_priority():
    ps, _, _ = collect("primary", ("pilot",), Qs=("Q01",)); ps = ps[(ps.Q == "Q01") & (ps.analytic_source == "oracle") & (ps.channel != "YP")]
    rows = []
    for policy in ("MASK-ALL", "MASK-COL"):
        d = mask_policy(ps, policy); n = d.groupby(["toy", "variant"]).ngroups
        agg = d.groupby("metric").agg(O_perfect=("O_perfect", "sum"), O_marginal=("O_marginal", "sum"), both_perfect=("both_perfect", "sum"), libs=("O_perfect", "size"))
        agg["policy"] = policy; agg["n_libraries"] = n; rows.append(agg.reset_index())
    sel = pd.concat(rows); sel["selected"] = (sel.O_perfect == sel.n_libraries)
    sel.sort_values(["metric", "policy"]).to_csv(RES / "priority_selection.csv", index=False)
    chosen = sorted(set(sel[sel.selected].metric))
    roles = dict(FIXED_ROLE)
    for m in chosen:
        roles.setdefault(m, "W1-01_rule_analytic_perfect_all_24_logit_continuous_libraries")
    ids = list(dict.fromkeys(FIXED_PANEL + chosen))
    # one failing control per recipe family present in the panel (fewest analytic-perfect libraries; ties by name)
    allm = sel[sel.policy == "MASK-ALL"].set_index("metric")
    fam = lambda m: m.split("|")[0] + "|" + m.split("|")[1].split(".")[0]
    for fm in sorted({fam(m) for m in ids}):
        cand = allm[[fam(m) == fm and m not in ids for m in allm.index]]
        if len(cand):
            c = sorted(cand[cand.O_perfect == cand.O_perfect.min()].index)[0]
            ids.append(c); roles[c] = "failing_control_of_family"
    panel = dict(frozen=time.strftime("%Y-%m-%d %H:%M:%S"), source="batch A (Q01, pilot, seed 31), logit/continuous channel, oracle arm",
                 rule="W1-01 rule: analytic (oracle) harmonised all-sweep PERFECT on all 24 primary logit/continuous libraries (12 toys x S9/S9_2) under the "
                      "MASK-ALL or the MASK-COL policy; plus the fixed 01_first_wave_README section 4.2 list; plus one failing control per recipe family "
                      "(fewest analytic-perfect libraries under MASK-ALL, ties by name). Development selection on the current toys, not confirmation.",
                 ids=ids, roles={i: roles.get(i, "") for i in ids}, n=len(ids))
    (CONFIGS / "priority_panel.json").write_text(json.dumps(panel, indent=1))
    print(f"priority panel frozen: {len(ids)} metrics ({len(chosen)} by the W1-01 rule)")


def freeze_escalation():
    ps, _, _ = collect("primary", ("pilot", "C")); pan = json.loads((CONFIGS / "priority_panel.json").read_text())
    ps = ps[ps.metric.isin(pan["ids"]) & (ps.analytic_source == "oracle")]
    key = ["toy", "variant", "Q", "channel", "mask", "metric"]
    reasons = {}
    C = ps[ps.fidelity == "C"]
    for m, g in C.groupby("metric"):
        r = []
        st = g.groupby(key)[["O_marginal", "I_marginal", "O_perfect", "I_perfect"]].agg(lambda s: s.nunique() > 1).any(axis=1)
        if st.any(): r.append("pass status changes across seeds")
        pil = ps[(ps.fidelity == "pilot") & (ps.metric == m)].set_index(key)[["O_marginal", "I_marginal"]]
        c31 = g[g.seed == 31].set_index(key)[["O_marginal", "I_marginal"]]
        jn = pil.join(c31, lsuffix="_p", rsuffix="_c", how="inner")
        if len(jn) and ((jn.O_marginal_p != jn.O_marginal_c) | (jn.I_marginal_p != jn.I_marginal_c)).any(): r.append("pass status changes pilot -> C")
        mm = pd.concat([g.O_min_margin, g.I_min_margin]).abs()
        if (mm < 0.02).any(): r.append("near-tie margin |m| < 0.02")
        for (cls, fg, Q, seed, ch), gg in mask_policy(g, "MASK-ALL").groupby(["cls", "fam_group", "Q", "seed", "channel"]):
            if gg.both_marginal.all() and gg.groupby(["toy", "variant"]).ngroups == 6:      # 3 toys x 2 variants of a family group
                r.append("proposed all-family pass"); break
        if pan["roles"].get(m) == "failing_control_of_family": r.append("retained failing control")
        if r: reasons[m] = sorted(set(r))
    ids = sorted(reasons)
    (CONFIGS / "escalation_D.json").write_text(json.dumps(dict(frozen=time.strftime("%Y-%m-%d %H:%M:%S"), rule="escalate every priority candidate with a changed pass status across seeds or pilot->C, a near-tie margin |m|<0.02, or an all-family (6-library) both-marginal pass; keep the failing controls", ids=ids, reasons=reasons, n=len(ids)), indent=1))
    print(f"escalation frozen: {len(ids)} metrics")


def integration(ps):
    pan = json.loads((CONFIGS / "priority_panel.json").read_text()); ps = ps[ps.metric.isin(pan["ids"]) & (ps.analytic_source == "oracle")]
    key = ["toy", "cls", "variant", "Q", "channel", "mask", "metric"]; out = []
    for k, g in ps.groupby(key):
        r = dict(zip(key, k))
        for fid in ("pilot", "C", "D"):
            gg = g[g.fidelity == fid]
            if len(gg):
                for side in ("O", "I", "both"):
                    r[f"{fid}_{side}_marginal_seeds"] = int(gg[f"{side}_marginal"].sum()); r[f"{fid}_{side}_perfect_seeds"] = int(gg[f"{side}_perfect"].sum())
                r[f"{fid}_n_seeds"] = len(gg); r[f"{fid}_I_margin_min"] = float(gg.I_min_margin.min()); r[f"{fid}_I_margin_max"] = float(gg.I_min_margin.max())
                r[f"{fid}_O_margin_min"] = float(gg.O_min_margin.min()); r[f"{fid}_pearson_min"] = float(gg.pearson.min()); r[f"{fid}_pearson_max"] = float(gg.pearson.max())
        last = [f for f in ("C", "D") if f"{f}_n_seeds" in r]
        if len(last) == 2:
            r["stable_last_two_fidelities"] = all(r[f"{f}_{s}_marginal_seeds"] in (0, r[f"{f}_n_seeds"]) for f in last for s in ("O", "I")) and \
                all((r[f"C_{s}_marginal_seeds"] > 0) == (r[f"D_{s}_marginal_seeds"] > 0) for s in ("O", "I"))
        out.append(r)
    return pd.DataFrame(out)


def coverage(corr):
    return corr.groupby(["toy", "cls", "variant", "Q", "fidelity", "seed", "channel", "mask", "analytic_source"]).agg(
        valid_oracle=("valid_oracle", "all"), valid_nn=("valid_nn", "all"), coverage_min=("coverage_min", "min"), n_points_oracle=("n_points_oracle", "max"),
        n_points_nn=("n_points_nn", "max"), nonfinite_oracle=("nonfinite_oracle", "max"), nonfinite_nn=("nonfinite_nn", "max")).reset_index()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--panel", default="primary"); ap.add_argument("--freeze-priority", action="store_true")
    ap.add_argument("--freeze-escalation", action="store_true"); a = ap.parse_args()
    if a.freeze_priority:
        return freeze_priority()
    if a.freeze_escalation:
        return freeze_escalation()
    ANA.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ps, corr, sw = collect(None, None, want_sweeps=True)
    print(f"{len(ps)} rows, {ps.groupby(['toy','variant']).ngroups} libraries [{time.time()-t0:.0f}s]")
    ps.to_csv(RES / "pass_summary.csv.gz", index=False); corr.to_csv(RES / "correlations.csv.gz", index=False)
    if sw is not None:
        sw.to_csv(ANA / "sweep_minima.csv.gz", index=False)
    prim = ps[ps.panel == "primary"]
    class_counts(prim).to_csv(RES / "class_pass_counts.csv", index=False)
    fourway(prim).to_csv(RES / "fourway.csv", index=False)
    coverage(corr).to_csv(RES / "coverage.csv", index=False)
    if (CONFIGS / "priority_panel.json").is_file():
        integration(prim).to_csv(RES / "integration_diagnostics.csv.gz", index=False)
        sec = ps[ps.panel == "secondary"]
        if len(sec):
            class_counts(pd.concat([sec])).to_csv(RES / "class_pass_counts_secondary.csv", index=False)
    print(f"done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
