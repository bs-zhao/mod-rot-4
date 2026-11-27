"""EVALUATION-ONLY exact / practical Gap 1 for W3 candidates (03_third_wave section 20), on the 8 analytic logit / continuous targets of
Lab 07 / 08 (C1, C2, C3, CR4: FULL; CR3, CR5: MASK-ALL and MASK-COL).

Analytic surface: AD of the generating formula at the paired true T of the Q01 C seed-31 TEST design (Lab 07 oracle_moments); U = Q T_s,
d T / d U = diag(t_std) Q^T.  Search: the W2 S0 optimizer (Haar + identity starts, decreasing spans), search seeds 31 / 73 / 101.
Independent check: paired L_D(Q) - L_D(I) on the Lab 07 D-fidelity caches of seeds 73 and 101.
Exact categories (tau = max(1e-9 |L(I)|, spread of L(I) over the three C seeds)):
  empirical support          no finalist with d_SP > .05 below L(I) - tau (finite search: support only, never a proof)
  global counterexample      a finalist with d_SP > .05, L < L(I) - tau and lower than I on BOTH independent caches
  inconclusive               lower on the search cache but not confirmed on both independent caches
  tied non-permutation       a finalist with d_SP > .05 within tau of L(I) and none lower
  invalid                    non-finite / undefined at I
Fusion candidates (analytic analogue, section 20): both component metrics (E = B08.M03.c0.03, X = SL0.UXUU.e0.01) on the SAME analytic surface,
each z-scored with its own analytic Haar-512 calibration; the epsilon reference is the analytic S0 minimum of z_E (3 search seeds).  This is an
analogue of the structural preference only - it does not model the AD / RBF estimator pair.
Practical (W2 criteria): the lowest finalist Q*: ideal recovery U = Q* T_s on TEST (MCC >= .95 and leak max < .2) and transferred recovery
Q* R* V (Lab 05 criterion) for the B and C toys of the family and both variants.
"""
import numpy as np, pandas as pd
from ..common import OUT, RES
from ..objectives import build
from wave2.optimizers import search, d_sp
from wave2.evaluate_recovery import EvalData, matched, recovery
from wave2.blind_inputs import load_payload
from wave4_2.inputs import load
import run_global_oracle as G7

TARGETS = ("C1:L:FULL", "C2:L:FULL", "C3:L:FULL", "CR3:L:MASK-ALL", "CR3:L:MASK-COL", "CR4:L:FULL", "CR5:L:MASK-ALL", "CR5:L:MASK-COL")
OG = OUT / "oracle_global"
_MO = {}


def oracle_mo(fam, mask, fid, seed):
    key = (fam, mask, fid, seed)
    if key not in _MO:
        b, ev, _, _ = load(fam); _MO[key] = G7.oracle_moments(b, ev, fid, seed, "L", mask)[0]
    return _MO[key]


def objective(spec, mo, ts, k):
    if spec.get("type") != "fusion":
        return build(spec, mo, ts)
    from .. import fusion as FU
    E = build(dict(type="registry", metric="L06|B08.M03.c0.03"), mo, ts); X = build(dict(type="registry", metric="L03|SL0.UXUU.e0.01"), mo, ts)
    fz = spec["fusion"]; calE = FU.calibrate(E, k, fz.get("norm", "iqr")); calX = FU.calibrate(X, k, fz.get("norm", "iqr")); zE = FU.Z(E, calE); zmin = None
    if fz["kind"] == "eps":
        zmin = min(search(zE, k, ss, "S0")["f"] for ss in (31, 73, 101))
    return FU.Fused(zE, FU.Z(X, calX), fz["kind"], fz.get("w"), fz.get("delta"), zmin)


def run(cid, spec, target):
    out = OG / f"{cid.replace('@', '_at_')}__{target.replace(':', '_')}.csv"
    if out.is_file():
        return pd.read_csv(out)
    fam, _, mask = target.split(":"); b, ev, _, _ = load(fam); k = b.k; ts = ev.refs["s9"]["t_std"]; I = np.eye(k)
    Ls = {s: objective(spec, oracle_mo(fam, mask, "C", s), ts, k) for s in (31, 73, 101)}
    LD = {s: objective(spec, oracle_mo(fam, mask, "D", s), ts, k) for s in (73, 101)}
    L = Ls[31]; l0 = L(I)
    row = dict(cid=cid, target=target, family=fam, mask=mask, L_identity=l0)
    if not np.isfinite(l0):
        row["category"] = "invalid"; pd.DataFrame([row]).to_csv(out, index=False); return pd.DataFrame([row])
    tau = max(1e-9 * abs(l0), float(np.ptp([Ls[s](I) for s in Ls]))); fins = []
    for ss in (31, 73, 101):
        r = search(L, k, ss, "S0")
        for f in r["finalists"]:
            fins.append(dict(Q=f["Q"], f=f["f"], d=d_sp(f["Q"]), conv=f["converged"]))
    F = np.array([f["f"] for f in fins]); D = np.array([f["d"] for f in fins]); far = D > 0.05
    lower = far & (F < l0 - tau); tied = far & (np.abs(F - l0) <= tau)
    dD = np.array([[LD[s](f["Q"]) - LD[s](I) for s in LD] for f in fins])
    conf = lower & (dD < 0).all(1)
    cat = "global counterexample" if conf.any() else ("inconclusive" if lower.any() else ("tied non-permutation" if tied.any() else "empirical support"))
    best = int(np.argmin(F)); Qb = fins[best]["Q"]
    # practical: ideal (true T) and transferred (Q R* V) recovery of the lowest finalist
    ed = EvalData(fam); te = np.sort(ed.test); Ts = (ed.T - ed.refs["s9"]["t_mean"]) / ts; ideal = matched(ed.T[te], Ts[te] @ Qb.T)
    row.update(tau=tau, category=cat, n_finalists=len(fins), n_lower=int(lower.sum()), n_confirmed=int(conf.sum()), n_tied=int(tied.sum()),
               best_dL=float(F[best] - l0), best_dsp=float(D[best]), best_converged=bool(fins[best]["conv"]), ideal_MCC=ideal["MCC"], ideal_leak_max=ideal["leak_max"],
               ideal_good=bool(ideal["MCC"] >= .95 and ideal["leak_max"] < .2))
    tr = []
    for toy in (fam, fam + "L"):
        pl = load_payload(toy); edt = EvalData(toy)
        for v in edt.refs:
            rec, _ = recovery(pl, edt, v, Qb @ edt.refs[v]["R_star"]); tr.append(rec["pass_lab05"]); row[f"transfer_{toy}_{v}"] = bool(rec["pass_lab05"])
    row["transfer_all"] = bool(all(tr)); row["practical_adequate"] = bool(row["ideal_good"] and row["transfer_all"])
    df = pd.DataFrame([row]); OG.mkdir(parents=True, exist_ok=True); df.to_csv(out, index=False); return df
