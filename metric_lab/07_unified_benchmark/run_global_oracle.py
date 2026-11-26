"""W1-04: does the analytic loss prefer signed permutations of the true axes GLOBALLY on O(k)?  (analytic only; not Gap 3)

usage: python run_global_oracle.py --target C1:L:FULL | --list   [--metrics all-panel]
Target = family : channel group (L = eta, shared by the class B Y-L and class C Y-C toys when their points/scalers are identical;
P = sigmoid(eta), class B only) : mask.  Search measure: Q01, fidelity C (512 TEST subjects x 64 trials), seed 31 points.
Budget: Haar starts 400 (k = 3, 4) / 2000 (k = 6) per seed for seeds 31/73/101; best 8 / 30 distinct starts per seed refined.
Controls: L(I), 20 random signed permutations.  Suspicious minima (non-permutation, d_SP/sqrt(2k) > 1e-3, loss <= L(I) + tau) are
re-evaluated on independent Q01 points at fidelity D (seeds 73, 101).  tau = 1e-9 |L(I)| (numerical) and the integration
variability = spread of L(I) over the three fidelity-C seeds (substantive equivalence).
Writes outputs/.../W1/global_oracle/<target>.npz and <target>_starts / _summary tables; --merge -> results/global_oracle_summary.csv, global_oracle_search.csv.gz.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4_2 import common
from wave4_2.common import RES, CONFIGS, OUT, libraries
import numpy as np, pandas as pd
from wave4_2.inputs import load
from wave4_2.measures import blind_design, col_mask
from wave4_2.derivatives import ad_chunks, channel_hessian
from wave4_2.moments import Moments, contract, jac, mask_modes
from wave4_2 import metrics as MT
from wave4_2.truth import oracle_eta
from wave4_2.global_oracle import kron_contract_factory, metric_fn, haar, d_sp, nearest_signed_perm, refine, distinct_best

GO = OUT / "W1" / "global_oracle"; GRES = GO          # per-target intermediates stay with the large outputs; --merge writes results/
FAMILIES = ("C1", "C2", "C3", "CR3", "CR4", "CR5")
DIST_THRESHOLDS = (1e-3, 0.02, 0.05, 0.15)       # normalised d_SP; 0.15 ~ one 15-degree Givens step at k=3
EXCLUDE = {"L06|CTRL.EA", "L06|CTRL.EB"}          # rotation-invariant energies: constant by construction


def targets():
    out = []
    for fam in FAMILIES:
        masks = ("MASK-ALL", "MASK-COL") if fam in ("CR3", "CR5") else ("FULL",)
        for ch in ("L", "P"):
            for m in masks:
                out.append(f"{fam}:{ch}:{m}")
    return out


def oracle_moments(b, ev, fid, seed, ch, mask):
    d = blind_design(b, "Q01", fid, seed); xs = (d["x"] - b.x_mean) / b.x_std; cm = col_mask(d["x"], b.x_domain)
    P = np.column_stack((ev.T[d["sid"]], xs)); fo = oracle_eta(ev.family, b.x_mean, b.x_std); m = Moments(b.k, b.p, mask)
    for sl, e, g, H in ad_chunks(fo, P):
        m.add(channel_hessian(e, g, H, "YP" if ch == "P" else "YL"), cm[sl])
    return m.finish(), d


def shared_check(fam):
    """B and C toys of a family: identical analytic points / scalers?  (then the L target is shared and reported for both)"""
    bB, evB, _, _ = load(fam); bC, evC, _, _ = load(fam + "L")
    dB, dC = blind_design(bB, "Q01", "C", 31), blind_design(bC, "Q01", "C", 31)
    same = (np.array_equal(dB["x"], dC["x"]) and np.array_equal(evB.T[dB["sid"]], evC.T[dC["sid"]]) and np.allclose(bB.x_mean, bC.x_mean, atol=0)
            and np.allclose(bB.x_std, bC.x_std, atol=0) and np.allclose(evB.refs["s9"]["t_std"], evC.refs["s9"]["t_std"], atol=0))
    return bool(same)


def run_target(target, ids):
    fam, ch, mask = target.split(":"); t0 = time.time()
    b, ev, rots, _ = load(fam); k = b.k; ts = ev.refs["s9"]["t_std"]; shared = shared_check(fam) if ch == "L" else None
    mos = {s: oracle_moments(b, ev, "C", s, ch, mask)[0] for s in common.SEEDS}
    mosD = {s: oracle_moments(b, ev, "D", s, ch, mask)[0] for s in (73, 101)}
    AB = {s: kron_contract_factory(m["K"], m["J"], ts) for s, m in mos.items()}; ABD = {s: kron_contract_factory(m["K"], m["J"], ts) for s, m in mosD.items()}
    # sanity: fast contraction == reference contraction
    Qt = haar(k, 1, np.random.default_rng(1))[0]; A1, B1 = AB[31](Qt); A2, B2 = contract(mos[31]["K"], mos[31]["J"], jac(ts, Qt))
    assert np.allclose(A1, A2, rtol=1e-9, atol=1e-14) and np.allclose(B1, B2, rtol=1e-9, atol=1e-14)
    I = np.eye(k); nh, m_ref = (2000, 30) if k == 6 else (400, 8)
    sp = [np.eye(k)[np.random.default_rng(100 + i).permutation(k)] * np.random.default_rng(200 + i).choice([-1, 1], k)[:, None] for i in range(20)]
    starts, summ = [], []; saveQ = {}
    for mid in [i for i in ids if i not in EXCLUDE]:
        def mk(ABs, m_):
            A0, B0 = ABs(I); zero, _, _ = MT.zero_flags(A0, B0)
            st = MT.l03_null(m_["K"], m_["J"], ts, contract, jac); return metric_fn(mid, ABs, st, zero)
        Ls = {s: mk(AB[s], mos[s]) for s in mos}; LD = {s: mk(ABD[s], mosD[s]) for s in mosD}
        L0 = {s: Ls[s](I) for s in Ls}; L = Ls[31]; l0 = L0[31]
        row = dict(target=target, family=fam, channel=ch, mask=mask, metric=mid, k=k, L_identity=l0, shared_B_C=shared)
        if not np.isfinite(l0):
            summ.append(dict(row, category="inconclusive (invalid / undefined loss at I)")); continue
        perm_dev = max(abs(L(P) - l0) for P in sp) / (abs(l0) + 1e-12); row["signed_perm_rel_dev"] = perm_dev
        tau_num = 1e-9 * (abs(l0) + 1e-12); spread = float(np.ptp([L0[s] for s in L0])); tau = max(tau_num, spread)
        allQ, allf = [], []
        haar_vals = []
        for s in common.SEEDS:
            Hq = haar(k, nh, np.random.default_rng([s, 7, k])); hv = np.array([L(Q) for Q in Hq]); haar_vals.append(hv)
            for rank, i in enumerate(distinct_best(Hq, hv, m_ref)):
                Qf, ff, n, conv, nsw = refine(L, Hq[i])
                ds = d_sp(Qf); starts.append(dict(target=target, metric=mid, seed=s, start=int(i), rank=rank, L_initial=float(hv[i]), L_final=float(ff),
                                                  L_identity=l0, dL=float(ff - l0), d_sp_norm=ds, converged=conv, n_evals=n, sweeps=nsw))
                allQ.append(Qf); allf.append(ff)
        hv = np.concatenate(haar_vals); scale = float(np.nanpercentile(hv, 95) - np.nanmin(np.r_[hv, l0])) if np.isfinite(hv).any() else np.nan
        allf = np.array(allf); ds_all = np.array([d_sp(Q) for Q in allQ]); far = ds_all > 1e-3
        row.update(haar_scale=scale, best_L=float(np.nanmin(allf)), best_dL=float(np.nanmin(allf) - l0), best_dL_scaled=float((np.nanmin(allf) - l0) / scale) if scale > 0 else np.nan,
                   n_starts=len(allf), frac_starts_near_perm=float((~far).mean()), frac_converged=float(np.mean([r["converged"] for r in starts if r["metric"] == mid and r["target"] == target])),
                   tau_numerical=tau_num, tau_integration=spread, haar_constant=bool(np.nanmax(hv) - np.nanmin(hv) <= 1e-9 * (np.nanmax(np.abs(hv)) + 1e-12)))
        row["no_start_within_1e-3_of_permutation"] = bool(far.all())
        cats = {}
        for th in DIST_THRESHOLDS:                                # sensitivity of the verdict to the near-permutation distance
            susp = np.where((ds_all > th) & (allf <= l0 + tau))[0]
            if row["haar_constant"]:
                cat = "inconclusive (constant loss over Haar rotations)"
            elif len(susp) == 0:
                cat = "empirical global support" + (" (no start ended within this distance of a signed permutation; only other local minima)" if (ds_all > th).all() else "")
            else:
                j = susp[np.argmin(allf[susp])]; Qs = allQ[j]
                re = {s_: LD[s_](Qs) - LD[s_](I) for s_ in LD}
                tauD = max(1e-9 * (abs(LD[73](I)) + 1e-12), abs(LD[73](I) - LD[101](I)))
                if th == DIST_THRESHOLDS[0]:
                    saveQ[mid] = Qs; rp = {s_: LD[s_](nearest_signed_perm(Qs)) - LD[s_](I) for s_ in LD}
                    row.update(suspicious_d_sp=float(ds_all[j]), suspicious_dL=float(allf[j] - l0), reeval_dL_73=float(re[73]), reeval_dL_101=float(re[101]),
                               reeval_perm_dL_73=float(rp[73]), tau_D=tauD)
                if all(re[s_] < -tauD for s_ in re):
                    cat = "global counterexample"
                elif all(abs(re[s_]) <= tauD for s_ in re):
                    cat = "nonunique minimum"
                elif all(re[s_] > tauD for s_ in re):
                    cat = "empirical global support (search-measure minimum not reproduced)"
                else:
                    cat = "inconclusive (re-evaluation disagrees across seeds)"
            if row.get("frac_converged", 1) < 0.5 and cat.startswith("empirical global support"):
                cat = "inconclusive search (convergence)"
            cats[th] = cat; row[f"category_dsp>{th}"] = cat
        cat = cats[DIST_THRESHOLDS[0]]
        row["category"] = cat; summ.append(row)
        print(f"  [{target}] {mid}: {cat}  best dL={row.get('best_dL', np.nan):.3g}  [{time.time()-t0:.0f}s]", flush=True)
    GO.mkdir(parents=True, exist_ok=True); GRES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GO / f"{target.replace(':', '_')}.npz", **{m: Q for m, Q in saveQ.items()})
    pd.DataFrame(starts).to_csv(GRES / f"{target.replace(':', '_')}_starts.csv.gz", index=False)
    pd.DataFrame(summ).to_csv(GRES / f"{target.replace(':', '_')}_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target"); ap.add_argument("--list", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(targets())); return
    if a.merge:
        pd.concat([pd.read_csv(f) for f in sorted(GRES.glob("*_summary.csv"))]).to_csv(RES / "global_oracle_summary.csv", index=False)
        pd.concat([pd.read_csv(f) for f in sorted(GRES.glob("*_starts.csv.gz"))]).to_csv(RES / "global_oracle_search.csv.gz", index=False); return
    pan = json.loads((CONFIGS / "priority_panel.json").read_text())
    run_target(a.target, pan["ids"])


if __name__ == "__main__":
    main()
