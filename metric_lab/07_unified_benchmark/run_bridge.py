"""W1-03: representation diagnostics, oracle bridges, bridge-decoder interpolation and the minimal T-free surrogate.

usage: python run_bridge.py --toys all|C1,... [--resume]
Points: Q01, pilot (256 TEST subjects x 32 own trials), seed 31 -- the same points as batch A.
Arms (all on ONE common point subset, fixed before rotation):
  O       true response, true T (labelled sweep G)                       [evaluation only]
  D       frozen decoder                                                 [blind]
  S       polynomial ridge (C) / logistic ridge (B), degree 2-3, TRAIN fit, VALIDATION selection   [blind]
  BO, BA  B-orth and B-aff bridges                                       [evaluation only]
  BO:a, BA:a  (1-a) bridge + a decoder, a in {0.25, 0.5, 0.75} (a=0 is the bridge, a=1 is D); Y-P mixes probabilities
Common subset: points whose B-orth AND B-aff predicted T lie inside the parameter support (CR families); never clipped.
A bridge with coverage < 0.5 is marked invalid (rule fixed before running).
Writes results/representation_diagnostics.csv, fit_diagnostics.csv, bridge_curves.csv.gz, bridge_energy_diagnostics.csv,
       bridge_transitions.csv, and outputs/.../W1/<run>/bridge/<variant>_<channel>_<mask>.npz (losses of every arm).
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2 import common
from wave4_2.common import RES, CONFIGS, OUT, libraries, library, channels
import numpy as np, pandas as pd
from wave4_2.inputs import load
from wave4_2.surfaces import DecoderSurface, fit_poly
from wave4_2.measures import blind_design, col_mask
from wave4_2.derivatives import ad_all, channel_hessian
from wave4_2.moments import Moments, contract, jac, mask_modes
from wave4_2 import metrics as MT
from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1
from wave4_2.truth import oracle_eta, in_support
from wave4_2.bridge import representation, bridge_eta, bridge_T, affine_fit

RUN = OUT / "W1"; ALPHAS = (0.25, 0.5, 0.75); MIN_COVERAGE = 0.5


def moments_of(H, cm, mode, k, p):
    m = Moments(k, p, mode); m.add(H, cm); return m.finish()


def arm_losses(mo, rots, oracle, ids):
    J = (lambda r: jac(r.t_std, r.G)) if oracle else (lambda r: jac(r.z_std, r.R))
    A0, B0 = contract(mo["K"], mo["J"], J(rots[0])); zero, _, _ = MT.zero_flags(A0, B0)
    st = MT.l03_null(mo["K"], mo["J"], rots[0].t_std if oracle else rots[0].z_std, contract, jac)
    L = np.array([MT.evaluate_ids(*contract(mo["K"], mo["J"], J(r)), st, zero, ids) for r in rots])
    A, B = contract(mo["K"], mo["J"], J(rots[0]))
    return L, A, B


def run_toy(toy, ids, rows):
    t0 = time.time(); b, ev, rots, src = load(toy); lib = library(toy); k, p = b.k, b.p
    d = blind_design(b, "Q01", "pilot", 31); xs = (d["x"] - b.x_mean) / b.x_std; cm_all = col_mask(d["x"], b.x_domain)
    Pz = np.column_stack((d["lat"], xs)); Pt = np.column_stack((ev.T[d["sid"]], xs))
    dec = DecoderSurface(b).eta(); fo = oracle_eta(ev.family, b.x_mean, b.x_std)
    poly, fitd = fit_poly(b)
    # derivative stability of the surrogate: refit on two disjoint TRAIN halves with the selected (degree, lambda)
    halves = np.array_split(np.random.default_rng(9).permutation(b.splits["train"]), 2); stab = []
    from dataclasses import replace
    for h in halves:
        bh = replace(b, splits=dict(b.splits, train=np.sort(h)))
        ph, _ = fit_poly(bh, degrees=(fitd["selected_degree"],), lambdas=(fitd["selected_lambda"],), n_train=512)
        e, g, H = ad_all(ph.eta(), Pz[:2048]); stab.append((H[:, :k, k:] ** 2).mean(0))
    fitd.update(toy=toy, half_refit_UX_rel_diff=float(np.abs(stab[0] - stab[1]).max() / (np.maximum(stab[0], stab[1]).max() + 1e-300)))
    ed, gd, Hd = ad_all(dec, Pz)
    es, gs, Hs = ad_all(poly.eta(), Pz); eo, go, Ho = ad_all(fo, Pt)
    fitd["S_vs_D_eta_corr_Q01"] = float(np.corrcoef(es, ed)[0, 1]); fitd["S_vs_O_eta_corr_Q01(eval-only)"] = float(np.corrcoef(es, eo)[0, 1])
    fitd["D_vs_O_eta_corr_Q01(eval-only)"] = float(np.corrcoef(ed, eo)[0, 1])
    rows["fit"].append(fitd)
    out_dir = RUN / lib["run"] / "bridge"; out_dir.mkdir(parents=True, exist_ok=True)
    for v in [v for v in common.VARIANTS if v in rots]:
        ref = ev.refs[v]; rep, (A, bb) = representation(b, ev, ev.refs, v)
        rows["rep"].append(dict(toy=toy, cls=b.cls, family=ev.family, variant=v, **{kk: (json.dumps(vv) if isinstance(vv, list) else vv) for kk, vv in rep.items()}))
        TB = {"BO": bridge_T(b, ev, ref, v, "B-orth", d["lat"]), "BA": bridge_T(b, ev, ref, v, "B-aff", d["lat"], A, bb)}
        okp = {kk: in_support(ev.family, T_) for kk, T_ in TB.items()}
        common_ok = okp["BO"][0] & okp["BA"][0]; cov = float(common_ok.mean())
        sup = dict(toy=toy, variant=v, n_points=len(common_ok), common_coverage=cov, bridge_valid=cov >= MIN_COVERAGE)
        for kk in TB:
            sup[f"{kk}_joint_violation_frac"] = float(1 - okp[kk][0].mean()); sup[f"{kk}_violation_by_param"] = json.dumps((1 - okp[kk][1].mean(0)).round(4).tolist())
        bf = {"BO": bridge_eta(b, ev, ref, v, "B-orth"), "BA": bridge_eta(b, ev, ref, v, "B-aff", A, bb)}
        sel = np.where(common_ok)[0]; Pzs = Pz[sel]; cm = cm_all[sel]
        base = {"O": (eo[sel], go[sel], Ho[sel]), "D": (ed[sel], gd[sel], Hd[sel]), "S": (es[sel], gs[sel], Hs[sel])}
        for kk, f in bf.items():
            base[kk] = ad_all(f, Pzs)
        man = [dict(name=r.name, candidate_type=r.candidate_type, axes=list(r.axes), angle=float(r.angle)) for r in rots[v]]; sw = sweeps(man)
        nexp = 2 ** k - k - 1
        for ch in channels(b.cls):
            H = {kk: channel_hessian(*vals, ch) for kk, vals in base.items()}
            for br in ("BO", "BA"):
                for a in ALPHAS:
                    H[f"{br}:{a}"] = (1 - a) * H[br] + a * H["D"]
                H[f"{br}+D"] = H[br] + H["D"]                         # for the signed cross moment E[H_b * H_d]
            for mode in mask_modes(b):
                mo = {kk: moments_of(h, cm, mode, k, p) for kk, h in H.items()}
                arr = dict(ids=np.array(ids), manifest=np.array(json.dumps(man))); Ls = {}
                for kk in [x for x in H if not x.endswith("+D")]:
                    L, A0, B0 = arm_losses(mo[kk], rots[v], kk == "O", ids); Ls[kk] = L; arr[f"{kk}|L"] = L; arr[f"{kk}|A"] = A0; arr[f"{kk}|B"] = B0
                np.savez_compressed(out_dir / f"{v}_{ch}_{mode}.npz", **arr)
                cs = {kk: library_flags(curve_stats(L, sw, "harmonised"), nexp) for kk, L in Ls.items()}
                for kk, L in Ls.items():
                    e1 = eval1(Ls["O"], L, man, sw) if kk != "O" else None
                    for j, mid in enumerate(ids):
                        rows["curves"].append(dict(toy=toy, cls=b.cls, family=ev.family, variant=v, channel=ch, mask=mode, arm=kk, metric=mid,
                                                   bridge_valid=sup["bridge_valid"], coverage=cov, arm_perfect=bool(cs[kk]["perfect"][j]), arm_marginal=bool(cs[kk]["marginal"][j]),
                                                   arm_strict_perfect=bool(cs[kk]["strict_perfect"][j]), arm_min_margin=float(cs[kk]["min_margin"][j]),
                                                   arm_constant=bool(cs[kk]["any_constant"][j]), O_perfect=bool(cs["O"]["perfect"][j]), O_marginal=bool(cs["O"]["marginal"][j]),
                                                   both_perfect=bool(cs[kk]["perfect"][j] and cs["O"]["perfect"][j]), both_marginal=bool(cs[kk]["marginal"][j] and cs["O"]["marginal"][j]),
                                                   pearson_vs_O=float(e1["pearson"][j]) if e1 else np.nan))
                # energy diagnostics at the baseline candidate (index 0 = unswept R*): weak-entry support and the cross term
                def norm(kk):
                    A_, B_ = contract(mo[kk]["K"], mo[kk]["J"], jac(rots[v][0].t_std, rots[v][0].G) if kk == "O" else jac(rots[v][0].z_std, rots[v][0].R))
                    E = A_.sum() + B_.sum(); return A_ / E, B_ / E, A_, B_
                nO = norm("O")
                for kk in ("D", "S", "BO", "BA"):
                    nK = norm(kk); r_ = dict(toy=toy, cls=b.cls, variant=v, channel=ch, mask=mode, arm=kk, coverage=cov, bridge_valid=sup["bridge_valid"])
                    r_["UX_rel_err"] = float(np.linalg.norm(nK[1] - nO[1]) / (np.linalg.norm(nO[1]) + 1e-300))
                    r_["UU_rel_err"] = float(np.linalg.norm(nK[0] - nO[0]) / (np.linalg.norm(nO[0]) + 1e-300)) if np.linalg.norm(nO[0]) > 0 else np.nan
                    r_["UX_weak_gained"] = int(((nO[1] < 1e-3) & (nK[1] >= 1e-3)).sum()); r_["UX_weak_lost"] = int(((nO[1] >= 1e-3) & (nK[1] < 1e-3)).sum())
                    r_["UU_weak_gained"] = int(((nO[0] < 1e-3) & (nK[0] >= 1e-3)).sum()); r_["UU_weak_lost"] = int(((nO[0] >= 1e-3) & (nK[0] < 1e-3)).sum())
                    if kk in ("BO", "BA"):
                        nS = norm(f"{kk}+D"); nD = norm("D")
                        cross_B = (nS[3] - nK[3] - nD[3]) / 2; cross_A = (nS[2] - nK[2] - nD[2]) / 2
                        r_["cross_UX_over_sum"] = float(cross_B.sum() / (nK[3].sum() + nD[3].sum() + 1e-300)); r_["cross_UU_over_sum"] = float(cross_A.sum() / (nK[2].sum() + nD[2].sum() + 1e-300))
                        r_["cross_UX_min_signed"] = float(cross_B.min())
                    rows["energy"].append(dict(r_, **{kk_: vv for kk_, vv in sup.items() if kk_ not in ("toy", "variant")}))
        print(f"  [{toy}] {v}: coverage {cov:.2f} [{time.time()-t0:.0f}s]", flush=True)


def transitions(curves):
    """first interpolation weight at which the arm's all-sweep (marginal) classification changes, per bridge"""
    out = []
    for keys, g in curves[curves.arm.str.match(r"^(BO|BA)")].groupby(["toy", "variant", "channel", "mask", "metric"]):
        for br in ("BO", "BA"):
            seq = [(0.0, br)] + [(a, f"{br}:{a}") for a in ALPHAS]
            st = []
            for a, arm in seq:
                gg = g[g.arm == arm]
                if len(gg): st.append((a, bool(gg.arm_marginal.iloc[0])))
            dd = curves[(curves.toy == keys[0]) & (curves.variant == keys[1]) & (curves.channel == keys[2]) & (curves["mask"] == keys[3]) & (curves.metric == keys[4]) & (curves.arm == "D")]
            if len(dd): st.append((1.0, bool(dd.arm_marginal.iloc[0])))
            first = next((a for (a0, s0), (a, s) in zip(st, st[1:]) if s != s0), np.nan)
            out.append(dict(zip(["toy", "variant", "channel", "mask", "metric"], keys), bridge=br, sequence=";".join(f"{a}:{int(s)}" for a, s in st), first_change_at=first,
                            n_changes=int(sum(s != s0 for (_, s0), (_, s) in zip(st, st[1:])))))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toys", default="all"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--resume", action="store_true", help="accepted for run_all.sh; bridge outputs are recomputed per toy"); a = ap.parse_args()
    if a.merge:                                  # concatenate per-toy result files written by parallel runs
        for stem, ext in (("representation_diagnostics", ".csv"), ("fit_diagnostics", ".csv"), ("bridge_curves", ".csv.gz"), ("bridge_energy_diagnostics", ".csv"), ("bridge_transitions", ".csv")):
            parts = [RES / f"{stem}_{l['toy']}{ext}" for l in libraries("primary")]
            missing = [x.name for x in parts if not x.is_file()]
            if missing:
                raise FileNotFoundError(f"missing per-toy files: {missing}")
            pd.concat([pd.read_csv(x) for x in parts]).to_csv(RES / f"{stem}{ext}", index=False)
            for x in parts: x.unlink()
        return
    ids = [i for i in json.loads((CONFIGS / "priority_panel.json").read_text())["ids"]]
    toys = [l["toy"] for l in libraries("primary")] if a.toys == "all" else a.toys.split(",")
    rows = dict(fit=[], rep=[], curves=[], energy=[])
    for t in toys:
        run_toy(t, ids, rows)
    suffix = "" if a.toys == "all" else "_" + a.toys.replace(",", "_")
    pd.DataFrame(rows["rep"]).to_csv(RES / f"representation_diagnostics{suffix}.csv", index=False)
    pd.DataFrame([{kk: (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in r.items()} for r in rows["fit"]]).to_csv(RES / f"fit_diagnostics{suffix}.csv", index=False)
    cur = pd.DataFrame(rows["curves"]); cur.to_csv(RES / f"bridge_curves{suffix}.csv.gz", index=False)
    pd.DataFrame(rows["energy"]).to_csv(RES / f"bridge_energy_diagnostics{suffix}.csv", index=False)
    transitions(cur).to_csv(RES / f"bridge_transitions{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
