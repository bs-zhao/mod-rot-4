"""EVALUATION-ONLY (W2-02 sections 11.1-11.4): ideal and transferred T-U recovery of the replayed analytic solutions; fine-angle summary.

Ideal (11.1):    U_ideal = Q T_s on the toy's held-out TEST T (T_s standardised with the toy's S8 t_mean / t_std).
Transfer (11.2): U_pert = Q R* (V - c*) for each coordinate variant of each toy (a truth-assisted diagnostic, not a blind solution).
Matching / leakage: Hungarian on |corr|, identical to W2-01.  Ideal 'good' = MCC >= .95 and max leak < .20 (sensitivity flags .98 / .99 and
.10 / .15 also saved); transfer 'good' = Lab 05 ceiling-relative criterion (MCC >= supervised - .05, leak < .2 C / .3 CR).
Exact analytic preference comes from the Lab 07 W1-04 category at distance 1e-3 (replayed identically here).
"""
import json
import numpy as np, pandas as pd
from .common import OUT, RES, LAB07, leak_threshold, family_group
from .evaluate_recovery import EvalData, matched
from .blind_inputs import load_payload, to_v
from .optimizers import d_sp

OD = OUT / "oracle_diagnostics"


def exact_label(cat):
    if not isinstance(cat, str):
        return "inconclusive"
    if cat.startswith("empirical global support"):
        return "empirically supported"
    if cat == "global counterexample":
        return "displaced minimum"
    if cat == "nonunique minimum":
        return "flat / tied non-permutations"
    return "inconclusive"


def outcome(exact, ideal_good, transfer_good, lab07=""):
    if exact == "empirically supported":                # no reproducible lower point: the analytic minimum found is I itself
        return "theoretical priority (weak coverage: no start reached the identity basin)" if "no start ended" in str(lab07) else             "theoretical priority (subject to search coverage)"
    if exact == "inconclusive":
        return "inconclusive: improve search / support"
    if exact == "flat / tied non-permutations":
        return "objective cannot identify a unique axis system"
    if exact == "empirically supported":
        return "theoretical priority (subject to search coverage)" if (ideal_good and transfer_good) else "supported but solution recovery poor"
    if ideal_good and transfer_good:
        return "approximate simplicity may be sufficient in practice"
    if ideal_good and not transfer_good:
        return "learned geometry / ceiling interaction matters"
    return "important metric failure"


def fine_summary(fine, sol):
    out = []
    for (t, m), g in fine.groupby(["target", "metric"]):
        s = sol[(sol.target == t) & (sol.metric == m)]
        tau = float(s.tau.iloc[0]) if len(s) and "tau" in s and np.isfinite(s.tau.iloc[0]) else np.nan
        loc = g[g.path.str.startswith(("plane", "dir")) & (g.theta != 0)]
        lower = loc[(loc.dL < -tau) & (loc.dD73 < 0) & (loc.dD101 < 0)] if np.isfinite(tau) else loc.iloc[:0]
        flat = loc[np.abs(loc.dL) <= tau] if np.isfinite(tau) else loc.iloc[:0]
        curv = {}
        for h in (.5, 1, 2):
            pl = g[g.path.str.startswith("plane")]
            c = []
            for pth, gg in pl.groupby("path"):
                v = gg.set_index("theta").L
                if {h, -h, 0} <= set(v.index):
                    c.append((v[h] + v[-h] - 2 * v[0]) / np.deg2rad(h) ** 2)
            curv[h] = float(np.min(c)) if c else np.nan
        geo = g[g.path == "geodesic_to_susp"].sort_values("theta")
        out.append(dict(target=t, metric=m, n_local_points=len(loc), n_lower_than_I_reproduced=len(lower), min_dL_fine=float(loc.dL.min()) if len(loc) else np.nan,
                        best_lower_theta=float(lower.loc[lower.dL.idxmin(), "theta"]) if len(lower) else np.nan,
                        best_lower_path=lower.loc[lower.dL.idxmin(), "path"] if len(lower) else "",
                        n_flat=len(flat), min_plane_curvature_h0p5=curv[.5], min_plane_curvature_h1=curv[1], min_plane_curvature_h2=curv[2],
                        geodesic_monotone_decrease=bool(len(geo) > 2 and np.all(np.diff(geo.L.values) <= 1e-12 * (abs(geo.L.values).max() + 1e-12))),
                        identity_local_min_fine=bool(len(lower) == 0)))
    return pd.DataFrame(out)


def merge():
    gl = pd.read_csv(LAB07 / "results" / "global_oracle_summary.csv"); gl = gl[gl.channel == "L"]
    sols, reps, locs, fines, Qsave, rows = [], [], [], [], {}, []
    for f in sorted(OD.glob("*_solutions.csv")):
        tag = f.name[:-len("_solutions.csv")]; target = tag.replace("_L_", ":L:")
        sol = pd.read_csv(f); sols.append(sol); reps.append(pd.read_csv(OD / f"{tag}_replay.csv"))
        lf = OD / f"{tag}_local.csv"
        if lf.is_file() and lf.stat().st_size > 5:
            locs.append(pd.read_csv(lf))
        fines.append(pd.read_csv(OD / f"{tag}_fine.csv.gz"))
        fin = pd.read_csv(OD / f"{tag}_finalists.csv.gz"); z = np.load(OD / f"{tag}.npz")
        fam, _, mask = target.split(":")
        for toy in (fam, fam + "L"):
            pl = load_payload(toy); ed = EvalData(toy); te = ed.test; T = ed.T[te]
            ref9 = ed.refs["s9"]; Ts = (T - ref9["t_mean"]) / ref9["t_std"]
            Ustar = {v: (to_v(pl, v, pl.z[te]) - ed.refs[v]["z_center"]) @ ed.refs[v]["R_star"].T for v in ed.refs}
            sup = {v: matched(T, Ustar[v]) for v in Ustar}; thr = leak_threshold(toy)
            for _, s in sol.iterrows():
                m = s.metric
                if f"{m}|Q" not in z.files:
                    continue
                Q_all = z[f"{m}|Q"]; cat = gl[(gl.family == fam) & (gl["mask"] == mask) & (gl.metric == m)]
                exact = exact_label(cat.category.iloc[0] if len(cat) else None)
                for kind in ("best_any", "best_susp", "best_reeval"):
                    i = int(s[kind]) if np.isfinite(s[kind]) else -1
                    if i < 0:
                        continue
                    Q = Q_all[i]; Qsave[f"{target}|{m}|{kind}"] = Q
                    ide = matched(T, Ts @ Q.T)
                    r = dict(target=target, family=fam, mask=mask, toy=toy, cls="C" if toy.endswith("L") else "B", fam_group=family_group(toy), metric=m, solution=kind,
                             idx=i, d_sp=d_sp(Q), dL=float(z[f"{m}|f"][i] - s.L_identity), dD73=float(z[f"{m}|dD"][i, 0]), dD101=float(z[f"{m}|dD"][i, 1]),
                             exact=exact, lab07_category=cat.category.iloc[0] if len(cat) else "", priority=bool(s.priority),
                             ideal_MCC=ide["MCC"], ideal_min=ide["min"], ideal_leak_mean=ide["leak_mean"], ideal_leak_max=ide["leak_max"])
                    for a in (.95, .98, .99):
                        r[f"ideal_MCC>={a}"] = ide["MCC"] >= a
                    for b in (.10, .15, .20):
                        r[f"ideal_leak<{b}"] = ide["leak_max"] < b
                    r["ideal_good"] = bool(ide["MCC"] >= .95 and ide["leak_max"] < .20)
                    tg = []
                    for v in Ustar:
                        tr = matched(T, Ustar[v] @ Q.T); rs = matched(Ustar[v], Ustar[v] @ Q.T)
                        good = bool(tr["MCC"] >= sup[v]["MCC"] - .05 and tr["leak_max"] < thr); tg.append(good)
                        r.update({f"transfer_{v}_MCC": tr["MCC"], f"transfer_{v}_leak_max": tr["leak_max"], f"transfer_{v}_supMCC": sup[v]["MCC"],
                                  f"transfer_{v}_good": good, f"transfer_{v}_MCC_vs_Ustar": rs["MCC"]})
                    r["transfer_good_all_variants"] = bool(all(tg))
                    r["outcome"] = outcome(exact, r["ideal_good"], r["transfer_good_all_variants"], r["lab07_category"]) if kind == "best_any" else ""
                    if kind == "best_any" and exact == "empirically supported":     # practical recovery of the analytic minimum = identity
                        r["ideal_good_minimum"] = True; r["transfer_good_minimum"] = True
                    elif kind == "best_any":
                        r["ideal_good_minimum"] = r["ideal_good"]; r["transfer_good_minimum"] = r["transfer_good_all_variants"]
                    rows.append(r)
            # every finalist: distance vs ideal MCC (figure 3)
            for _, fr in fin.iterrows():
                Q = z[f"{fr.metric}|Q"][int(fr.idx)]; ide = matched(T, Ts @ Q.T)
                rows.append(dict(target=target, family=fam, mask=mask, toy=toy, cls="C" if toy.endswith("L") else "B", fam_group=family_group(toy), metric=fr.metric,
                                 solution="finalist", idx=int(fr.idx), d_sp=fr.d_sp, dL=fr.dL, dD73=fr.dD73, dD101=fr.dD101, ideal_MCC=ide["MCC"], ideal_leak_max=ide["leak_max"],
                                 ideal_good=bool(ide["MCC"] >= .95 and ide["leak_max"] < .20)))
        print("  merged", target, flush=True)
    rec = pd.DataFrame(rows); sol = pd.concat(sols); fine = pd.concat(fines)
    rec[rec.solution != "finalist"].to_csv(RES / "oracle_recovery.csv", index=False)
    rec[rec.solution == "finalist"].to_csv(RES / "oracle_finalist_recovery.csv.gz", index=False)
    sol.to_csv(RES / "oracle_solution_registry.csv", index=False); pd.concat(reps).to_csv(RES / "oracle_replay_audit.csv", index=False)
    if locs:
        pd.concat(locs).to_csv(RES / "oracle_local_escape.csv", index=False)
    fine.to_csv(RES / "oracle_fine_sweeps.csv.gz", index=False); fine_summary(fine, sol).to_csv(RES / "oracle_fine_summary.csv", index=False)
    np.savez_compressed(RES / "oracle_solutions.npz", **Qsave)
    print("oracle_recovery.csv:", int((rec.solution != "finalist").sum()), "rows")
