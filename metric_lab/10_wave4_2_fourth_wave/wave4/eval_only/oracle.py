"""EVALUATION-ONLY W4-03 (04_fourth_wave sections 12-14): analytic O-true moments, exact / practical Gap 1 and Gap 2 Eval 1 / Eval 2
under labelled measures.

O-true (primary scientific oracle): latent = the TRUE T of the empirical TRAIN subjects drawn by the M-E design (permutation [seed,1]),
crossed with the SAME physical X panel as the blind Q05 / M-* designs of that seed (independent complete rows or the full bank) and the
same derivative masks; analytic logit (class B) / continuous mean (class C) of the generating formula, AD float64.  Rotation frame:
T standardised with the saved TRAIN T scaler (U = Q T_s, dT/dU = diag(t_std) Q^T).  Empirical T is supported by construction (no normal
approximation, no clipping), so no invalid fraction arises.  O-ref (normal approximation to T) and the Z->T bridge are NOT run (see README).
Gap 1 (8 analytic targets of Lab 07-09; B toy of each family): W2 S0 global search from identity + Haar starts on the C seed-31 O-true
cache, search seeds 31 / 73 / 101; independent paired checks L_D(Q) - L_D(I) on O-true D caches of seeds 211 / 613.  Categories and
practical criteria as Lab 09 (wave3/eval_only/oracle_global.py).
Gap 2: the saved S9 / S9_2 sweep rotations (R = G R*, Q = G), oracle O-true vs inferred blind measure at matched seed / fidelity:
Q05 C 31/73/101 vs O-true C 31/73/101, Q05 D 211/613 vs O-true D 211/613; Lab 07 flag code (harmonised) and Eval1 correlations.
"""
import numpy as np, pandas as pd
from ..common import OUT, CAND, E02, B01, spec_of, PRIMARY
from .. import moments as MO, designs as DS
from wave3.objectives import build
from wave2.optimizers import search, d_sp
from wave2.evaluate_recovery import EvalData, matched, recovery
from wave2.blind_inputs import load_payload
from wave4_2.inputs import load
from wave4_2.truth import oracle_eta
from wave4_2.evaluation import sweeps
import run_baseline as RB7
from wave3.eval_only import local as LO9

OC = OUT / "eval_only" / "oracle_cache"
OG = OUT / "eval_only" / "oracle_global"
TARGETS = ("C1:L:FULL", "C2:L:FULL", "C3:L:FULL", "CR3:L:MASK-ALL", "CR3:L:MASK-COL", "CR4:L:FULL", "CR5:L:MASK-ALL", "CR5:L:MASK-COL")
CIDS = (CAND, E02, B01)
_L = {}


def loaded(toy):
    if toy not in _L:
        _L[toy] = load(toy)
    return _L[toy]


def otrue(toy, fid, seed):
    """{(channel, mask): moments} of the O-true law (cached under eval_only/)"""
    from wave2.objectives import save_moments, load_moments
    f = OC / toy.replace("#", "_") / f"otrue_{fid}_s{seed}.npz"
    if f.is_file():
        return load_moments(f)[0]
    pl = load_payload(toy); b, ev, _, _ = loaded(toy); d = DS.design(pl, "M-E", fid, seed, "train")
    dO = dict(d, lat=ev.T[d["sid"]]); res, _ = MO.compute(pl, oracle_eta(ev.family, pl.x_mean, pl.x_std), dO, MO.masks_for(pl))
    save_moments(f, res, dict(design=dict(d["meta"], latent="true T of the M-E subjects (O-true)"), n_points=int(len(d["x"])))); return res


def _mo(toy, fid, seed, mask):
    pl = load_payload(toy); c = "YC" if pl.cls == "C" else "YL"; return otrue(toy, fid, seed)[(c, mask)]


def gap1(cid, target):
    out = OG / f"{cid.replace('@', '_at_')}__{target.replace(':', '_')}.csv"
    if out.is_file():
        return pd.read_csv(out)
    OG.mkdir(parents=True, exist_ok=True)
    fam, _, mask = target.split(":"); b, ev, _, _ = loaded(fam); k = b.k; ts = ev.refs["s9"]["t_std"]; I = np.eye(k); sp = spec_of(cid)
    Ls = {s: build(sp, _mo(fam, "C", s, mask), ts) for s in (31, 73, 101)}; LD = {s: build(sp, _mo(fam, "D", s, mask), ts) for s in (211, 613)}
    L = Ls[31]; l0 = L(I); row = dict(cid=cid, target=target, family=fam, mask=mask, measure="O-true", L_identity=l0)
    if not np.isfinite(l0):
        row["category"] = "invalid"; pd.DataFrame([row]).to_csv(out, index=False); return pd.DataFrame([row])
    tau = max(1e-9 * abs(l0), float(np.ptp([Ls[s](I) for s in Ls]))); fins = []
    for ss in (31, 73, 101):
        r = search(L, k, ss, "S0")
        for f in r["finalists"]:
            fins.append(dict(Q=f["Q"], f=f["f"], d=d_sp(f["Q"]), conv=f["converged"]))
    F = np.array([f["f"] for f in fins]); D = np.array([f["d"] for f in fins]); far = D > .05
    lower = far & (F < l0 - tau); tied = far & (np.abs(F - l0) <= tau)
    dD = np.array([[LD[s](f["Q"]) - LD[s](I) for s in LD] for f in fins]); conf = lower & (dD < 0).all(1)
    cat = "global counterexample" if conf.any() else ("inconclusive" if lower.any() else ("tied non-permutation" if tied.any() else "empirical support"))
    best = int(np.argmin(F)); Qb = fins[best]["Q"]
    ed = EvalData(fam); te = np.sort(ed.test); Tsd = (ed.T - ed.refs["s9"]["t_mean"]) / ts; ideal = matched(ed.T[te], Tsd[te] @ Qb.T)
    row.update(tau=tau, category=cat, n_finalists=len(fins), n_lower=int(lower.sum()), n_confirmed=int(conf.sum()), n_tied=int(tied.sum()), best_dL=float(F[best] - l0),
               best_dsp=float(D[best]), best_converged=bool(fins[best]["conv"]), ideal_MCC=ideal["MCC"], ideal_leak_max=ideal["leak_max"],
               ideal_good=bool(ideal["MCC"] >= .95 and ideal["leak_max"] < .2), dL_D211=float(dD[best, 0]), dL_D613=float(dD[best, 1]))
    tr = []
    for toy in (fam, fam + "L"):
        pl = load_payload(toy); edt = EvalData(toy)
        for v in edt.refs:
            rec, _ = recovery(pl, edt, v, Qb @ edt.refs[v]["R_star"]); tr.append(rec["pass_lab05"]); row[f"transfer_{toy}_{v}"] = bool(rec["pass_lab05"])
    row["transfer_all"] = bool(all(tr)); row["practical_adequate"] = bool(row["ideal_good"] and row["transfer_all"])
    np.savez_compressed(out.with_suffix(".npz"), Q=np.stack([f["Q"] for f in fins]), f=F, d=D, dD=dD)
    df = pd.DataFrame([row]); OG.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); return df


MATCH = [("Q05", "C", s, "O-true", "C", s) for s in (31, 73, 101)] + [("Q05", "D", s, "O-true", "D", s) for s in (211, 613)]


def gap2(toy):
    out = OUT / "eval_only" / "gap2" / f"{toy.replace('#', '_')}.csv.gz"
    if out.is_file():
        return pd.read_csv(out)
    pl = load_payload(toy); b, ev, rots, _ = loaded(toy); k = b.k; n_exp = 2 ** k - k - 1; c = "YC" if pl.cls == "C" else "YL"; rows = []
    for M, fid, s, OM, ofid, os_ in MATCH:
        for mask in MO.masks_for(pl):
            mi = MO.get(pl, M, fid, s, "train")[0][(c, mask)]; mo = otrue(toy, ofid, os_)[(c, mask)]
            for v in rots:
                man = RB7.manifest_rows(rots[v]); sw = sweeps(man)
                Lo = np.column_stack([[build(spec_of(cid), mo, r.t_std)(r.G) for r in rots[v]] for cid in CIDS])
                Li = np.column_stack([[build(spec_of(cid), mi, r.z_std)(r.R) for r in rots[v]] for cid in CIDS])
                for r in LO9._flags(Lo, Li, man, sw, n_exp, list(CIDS)):
                    rows.append(dict(toy=toy, variant=v, inferred=f"{M}@{fid}", seed=s, oracle=f"{OM}@{ofid}", oracle_seed=os_, channel=c, mask=mask, **r))
        print(f"  gap2 {toy} {M}@{fid} s{s}", flush=True)
    df = pd.DataFrame(rows); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); return df
