"""W3-01 RW-A (EVALUATION-ONLY mechanism audit, 03_third_wave sections 5 and 7).

usage: python run_repair_diag.py            -> results/repair_decomposition.csv.gz, repair_geodesic.csv.gz, repair_old_vs_ref.csv.gz
For every primary toy x variant x W2 BS-B E02 cell (integration seeds 31/73/101 x search seeds 31/73/101), on the identical frozen TRAIN cache:
  * B(R) at the blind solution R_hat and at the supervised reference R*; rows matched by Hungarian on |R_hat R*^T| (evaluation-only labels:
    reference row i = parameter i);  exact symmetric decomposition  L(R_hat) - L(R*) = sum wbar ds + sum sbar dw  (+ residual, ~1e-16);
  * archived W2 loss reproduced at R_hat;
  * geodesic R* -> aligned R_hat (21 points): loss, weight term and score term along the path (fine pair-angle scan of the mechanism);
  * every registered repair recipe and control: L(R*) vs L(R_hat) on the same cache  ->  does the old wrong solution still beat R*?
Truth (R*, labels) enters only here, after the W2 solutions were locked; nothing feeds back into any blind configuration.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT08, RES, PRIMARY, VARIANTS, chan, prim_mask
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm, logm
from wave3 import rowmetrics as RM, jobs as JB
from wave3.objectives import build, ux_factory
from wave2.blind_inputs import load_payload, variant_frame
from wave2.objectives import load_moments
from wave2.evaluate_recovery import EvalData, recovery
from wave2.optimizers import rel_dist

E02 = dict(kind="POW", a=1.0, c=0.03)


def align(Rh, Rs):
    """signed permutation P with P R_hat row i matched to R* row i; det fixed to +1 by flipping the weakest match if needed"""
    C = Rh @ Rs.T; r, c = linear_sum_assignment(-np.abs(C)); perm = np.empty(len(r), int); perm[c] = r       # perm[i] = row of R_hat for ref row i
    sgn = np.sign(C[perm, np.arange(len(r))]); sgn[sgn == 0] = 1; P = np.zeros_like(C); P[np.arange(len(r)), perm] = sgn
    A = P @ Rh
    if np.linalg.det(A @ Rs.T) < 0:
        w = np.argmin(np.abs(C[perm, np.arange(len(r))])); A[w] *= -1
    return perm, A


def main():
    t0 = time.time(); reg = JB.registry(); cands = {c: e["spec"] for c, e in reg.items() if e["family"] in ("RW", "CTRL") and e["surface"] == "AD" and c != "X01"}
    dec_rows, geo_rows, ovr_rows = [], [], []
    for toy in PRIMARY:
        pl = load_payload(toy); ed = EvalData(toy); c = chan(pl); m = prim_mask(pl)
        z = np.load(OUT08 / "blind" / "BS-B" / toy / "E02.npz"); diag = json.loads(str(z["diag"])); mc = {}
        for key, d in diag.items():
            v = d["variant"]
            if v not in ed.refs:
                continue
            if d["cache"] not in mc:
                mc[d["cache"]] = load_moments(OUT08 / "moment_cache" / toy / f"{d['cache']}.npz")[0][(c, m)]
            mo = mc[d["cache"]]; sc = variant_frame(pl, v)[1]; Bf = ux_factory(mo["J"], sc)
            Rh = z[f"{key}|R"]; Rs = ed.refs[v]["R_star"]; perm, Ra = align(Rh, Rs)
            rec, _ = recovery(pl, ed, v, Rh)
            dec = RM.decompose(E02, Bf(Rs), Bf(Rh), perm); ta, tb = RM.terms(E02, Bf(Rs)), RM.terms(E02, Bf(Rh))
            base = dict(toy=toy, variant=v, iseed=d["iseed"], sseed=d["sseed"], pass_lab05=rec["pass_lab05"], MCC=rec["MCC"], leak_max=rec["leak_max"],
                        dist_to_Rstar=rec["dist_to_Rstar"])
            for i in range(pl.k):
                dec_rows.append(dict(base, axis=i, w_ref=dec["w_a"][i], w_blind=dec["w_b"][i], s_ref=dec["s_a"][i], s_blind=dec["s_b"][i],
                                     score_term=dec["score_term"][i], weight_term=dec["weight_term"][i], act_ref=bool(dec["act_a"][i]), act_blind=bool(dec["act_b"][i]),
                                     r_ref=ta["r"][i], r_blind=tb["r"][perm][i]))
            dec_rows.append(dict(base, axis="total", L_ref=ta["L"], L_blind=tb["L"], dL=dec["dL"], score_term=float(dec["score_term"].sum()),
                                 weight_term=float(dec["weight_term"].sum()), residual=dec["residual"], support_change=dec["support_change"],
                                 archived_L=d["f"], archived_rel_err=abs(tb["L"] - d["f"]) / abs(d["f"])))
            # geodesic R* -> aligned R_hat
            S = np.real(logm(Ra @ Rs.T))
            for t in np.linspace(0, 1, 21):
                Rt = expm(t * S) @ Rs; dd = RM.decompose(E02, Bf(Rs), Bf(Rt), np.arange(pl.k))
                geo_rows.append(dict(base, t=float(t), L=RM.value(E02, Bf(Rt)), dL=dd["dL"], score_term=float(dd["score_term"].sum()),
                                     weight_term=float(dd["weight_term"].sum()), dist_to_Rstar=rel_dist(Rt, Rs)))
            for cid, spec in cands.items():
                L = build(spec, mo, sc); ls, lh = L(Rs), L(Rh)
                ovr_rows.append(dict(base, cid=cid, L_Rstar=ls, L_Rhat_E02=lh, old_beats_ref=bool(lh < ls - 1e-12 * (abs(ls) + 1e-12))))
        print(f"  {toy} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(dec_rows).to_csv(RES / "repair_decomposition.csv.gz", index=False)
    pd.DataFrame(geo_rows).to_csv(RES / "repair_geodesic.csv.gz", index=False)
    pd.DataFrame(ovr_rows).to_csv(RES / "repair_old_vs_ref.csv.gz", index=False)


if __name__ == "__main__":
    main()
