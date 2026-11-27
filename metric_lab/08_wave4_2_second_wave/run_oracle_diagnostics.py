"""W2-02 (EVALUATION-ONLY): practical accuracy of the analytic global minima.  Never enters a blind configuration.

usage: python run_oracle_diagnostics.py --target CR3:L:MASK-ALL      (one of the 8 logit / continuous analytic targets)
       python run_oracle_diagnostics.py --merge                      (ideal + transfer recovery, categories -> results/)
Per target:
  1. deterministic REPLAY of the Lab 07 W1-04 search (identical code, moments, seeds and budgets: Lab 07 run_global_oracle functions),
     now saving EVERY refined finalist Q.  Identity check: replayed per-start final losses / distances against the persisted
     global_oracle_search.csv.gz, and the replayed suspicious Q against the persisted Lab 07 matrix.
  2. solution registry per metric: best unconstrained, best suspicious non-permutation (Lab 07 rule), best independently
     re-evaluated non-permutation finalist (paired L_D(Q) - L_D(I) on the 1024 x 128 seeds 73 / 101), identity loss, tied near-minima,
     non-converged finalists.
  3. fine-angle diagnostics around I: every pairwise plane and 8 random unit skew directions at
     theta in {0, +-.5, +-1, +-2, +-5, +-10, +-15} degrees, and the geodesic I -> best suspicious Q (11 points) plus its nearest signed
     permutation; on the search cache and as paired differences on the two independent high-fidelity caches.
  4. local refinements from I and from 8 small perturbations (2 each at 2, 5, 10, 20 degrees) for the prioritised metrics
     (A1, the 17 historical analytic-perfect metrics, E01/E02, and every metric with a C3 / CR counterexample).
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, RES, LAB07, OUT07, PRIMARY, VARIANTS, leak_threshold
import numpy as np, pandas as pd
from scipy.linalg import expm, logm
import run_global_oracle as G7                                   # Lab 07 W1-04 (truth-enabled, evaluation only)
from wave4_2 import common as C7
from wave4_2.inputs import load
from wave4_2.global_oracle import kron_contract_factory, metric_fn, haar, d_sp, nearest_signed_perm, refine, distinct_best, givens
from wave4_2.moments import contract, jac
from wave4_2 import metrics as MT

OD = OUT / "oracle_diagnostics"
L_TARGETS = [t for t in G7.targets() if t.split(":")[1] == "L"]
FINE = np.array([0, .5, -.5, 1, -1, 2, -2, 5, -5, 10, -10, 15, -15], float)
PERTURB = (2, 2, 5, 5, 10, 10, 20, 20)


def priority_ids():
    pan = json.loads((LAB07 / "configs" / "priority_panel.json").read_text())
    nw = pd.read_csv(LAB07 / "results" / "next_wave_candidates.csv"); gl = pd.read_csv(LAB07 / "results" / "global_oracle_summary.csv")
    a1 = set(nw[nw.lane == "A1"].id); h17 = {m for m, r in pan["roles"].items() if r == "hist17_analytic_perfect"}
    ce = set(gl[(gl.channel == "L") & (gl.category == "global counterexample") & gl.family.isin(["C3", "CR3", "CR4", "CR5"])].metric)
    return [i for i in pan["ids"] if i not in G7.EXCLUDE], (a1 | h17 | {"L06|B08.M03.c0.01", "L06|B08.M03.c0.03"} | ce)


def skew_dirs(k, n, seed=12):
    rng = np.random.default_rng([seed, k]); out = []
    for _ in range(n):
        A = rng.standard_normal((k, k)); S = (A - A.T) / 2; S /= np.linalg.norm(S) / np.sqrt(2)   # unit: a plane rotation of angle t
        out.append(S)
    return out


def run_target(target):
    out = OD / f"{target.replace(':', '_')}.npz"
    if out.is_file():
        print(f"  [{target}] done"); return
    ids, prio = priority_ids(); fam, ch, mask = target.split(":"); t0 = time.time()
    b, ev, rots, _ = load(fam); k = b.k; ts = ev.refs["s9"]["t_std"]
    mos = {s: G7.oracle_moments(b, ev, "C", s, ch, mask)[0] for s in C7.SEEDS}
    mosD = {s: G7.oracle_moments(b, ev, "D", s, ch, mask)[0] for s in (73, 101)}
    AB = {s: kron_contract_factory(m["K"], m["J"], ts) for s, m in mos.items()}; ABD = {s: kron_contract_factory(m["K"], m["J"], ts) for s, m in mosD.items()}
    I = np.eye(k); nh, m_ref = (2000, 30) if k == 6 else (400, 8)
    saved_starts = pd.read_csv(LAB07 / "results" / "global_oracle_search.csv.gz"); saved_starts = saved_starts[saved_starts.target == target]
    savedQ = np.load(OUT07 / "W1" / "global_oracle" / f"{target.replace(':', '_')}.npz")
    arr, fin_rows, fine_rows, loc_rows, sol_rows, rep_rows = {}, [], [], [], [], []
    dirs = skew_dirs(k, 8); planes = [(i, j) for i in range(k) for j in range(i + 1, k)]
    for mid in ids:
        def mk(ABs, m_):
            A0, B0 = ABs(I); zero, _, _ = MT.zero_flags(A0, B0)
            st = MT.l03_null(m_["K"], m_["J"], ts, contract, jac); return metric_fn(mid, ABs, st, zero)
        Ls = {s: mk(AB[s], mos[s]) for s in mos}; LD = {s: mk(ABD[s], mosD[s]) for s in mosD}
        L = Ls[31]; l0 = L(I); L0 = {s: Ls[s](I) for s in Ls}
        if not np.isfinite(l0):
            sol_rows.append(dict(target=target, metric=mid, status="undefined at I")); continue
        tau = max(1e-9 * (abs(l0) + 1e-12), float(np.ptp([L0[s] for s in L0])))
        allQ, allf, prov = [], [], []
        for s in C7.SEEDS:                                       # identical to Lab 07 run_target
            Hq = haar(k, nh, np.random.default_rng([s, 7, k])); hv = np.array([L(Q) for Q in Hq])
            for rank, i in enumerate(distinct_best(Hq, hv, m_ref)):
                Qf, ff, n, conv, nsw = refine(L, Hq[i]); allQ.append(Qf); allf.append(ff); prov.append(dict(seed=s, start=int(i), rank=rank, converged=conv, n_evals=n, sweeps=nsw))
        allf = np.array(allf); ds = np.array([d_sp(Q) for Q in allQ]); allQ = np.array(allQ)
        # replay identity vs the persisted Lab 07 search
        ss = saved_starts[saved_starts.metric == mid].sort_values(["seed", "rank"])
        if len(ss) == len(allf):
            rep_rows.append(dict(target=target, metric=mid, n=len(allf), max_abs_L_diff=float(np.max(np.abs(ss.L_final.values - allf))),
                                 max_abs_dsp_diff=float(np.max(np.abs(ss.d_sp_norm.values - ds))), savedQ=mid in savedQ.files))
        else:
            rep_rows.append(dict(target=target, metric=mid, n=len(allf), max_abs_L_diff=np.nan, max_abs_dsp_diff=np.nan, savedQ=mid in savedQ.files, note=f"saved rows {len(ss)}"))
        dD = np.array([[LD[s](Q) - LD[s](I) for s in LD] for Q in allQ])          # paired independent re-evaluation
        susp = np.where((ds > 1e-3) & (allf <= l0 + tau))[0]
        if mid in savedQ.files and len(susp):
            j = susp[np.argmin(allf[susp])]; rep_rows[-1]["savedQ_rel_diff"] = float(np.abs(savedQ[mid] - allQ[j]).max())
        arr[f"{mid}|Q"] = allQ; arr[f"{mid}|f"] = allf; arr[f"{mid}|dD"] = dD
        for i, (Q, f_, d_) in enumerate(zip(allQ, allf, ds)):
            fin_rows.append(dict(target=target, metric=mid, idx=i, L=f_, dL=f_ - l0, d_sp=d_, dD73=dD[i, 0], dD101=dD[i, 1], **prov[i]))
        far = ds > 1e-3; best_far_re = np.where(far & (dD < 0).all(1))[0]
        sol = dict(target=target, family=fam, mask=mask, metric=mid, L_identity=l0, tau=tau, n_finalists=len(allf), n_converged=int(sum(p["converged"] for p in prov)),
                   best_any=int(np.argmin(allf)), best_any_dL=float(allf.min() - l0), best_any_dsp=float(ds[np.argmin(allf)]),
                   best_susp=int(susp[np.argmin(allf[susp])]) if len(susp) else -1, n_susp=int(len(susp)),
                   best_reeval=int(best_far_re[np.argmin(dD[best_far_re].mean(1))]) if len(best_far_re) else -1, n_reeval_lower=int(len(best_far_re)),
                   n_tied=int(((np.abs(allf - allf.min()) <= tau) & far).sum()), any_start_reached_identity=bool((~far).any()), priority=mid in prio)
        sol_rows.append(sol)
        # fine-angle diagnostics around I (search cache + paired independent caches)
        def evalset(Q):
            return L(Q), [LD[s](Q) - LD[s](I) for s in LD]
        for lab, gen in [(f"plane{i}{j}", (lambda t, i=i, j=j: givens(k, i, j, np.deg2rad(t)))) for i, j in planes] + \
                        [(f"dir{d}", (lambda t, S=S: expm(np.deg2rad(t) * S))) for d, S in enumerate(dirs)]:
            for t in FINE:
                Q = gen(t); v, pdD = evalset(Q); fine_rows.append(dict(target=target, metric=mid, path=lab, theta=t, L=v, dL=v - l0, dD73=pdD[0], dD101=pdD[1]))
        if sol["best_susp"] >= 0:
            Qs = allQ[sol["best_susp"]]
            Qg = Qs if np.linalg.det(Qs) > 0 else np.diag([-1.0] + [1.0] * (k - 1)) @ Qs    # sign flip of one rotated axis: same loss, det +1
            Lg = np.real(logm(Qg))
            if np.all(np.isfinite(Lg)):
                for s_ in np.linspace(0, 1, 11):
                    Q = expm(s_ * (Lg - Lg.T) / 2); v, pdD = evalset(Q)
                    fine_rows.append(dict(target=target, metric=mid, path="geodesic_to_susp", theta=float(s_), L=v, dL=v - l0, dD73=pdD[0], dD101=pdD[1], d_sp=d_sp(Q)))
            P = nearest_signed_perm(Qs); v, pdD = evalset(P)
            fine_rows.append(dict(target=target, metric=mid, path="nearest_signed_perm_of_susp", theta=np.nan, L=v, dL=v - l0, dD73=pdD[0], dD101=pdD[1]))
        if mid in prio:                                           # local escape / basin coverage
            rng = np.random.default_rng([2026, k])
            for pi, deg in enumerate((0,) + PERTURB):
                S = skew_dirs(k, 1, seed=1000 + pi)[0]; Q0 = expm(np.deg2rad(deg) * S)
                Qf, ff, n, conv, nsw = refine(L, Q0)
                loc_rows.append(dict(target=target, metric=mid, start_deg=deg, start=pi, L_final=ff, dL=ff - l0, d_sp=d_sp(Qf), converged=conv,
                                     returned_to_identity=bool(d_sp(Qf) <= 1e-3), lower_than_identity=bool(ff < l0 - tau)))
                arr[f"{mid}|local{pi}"] = Qf
        print(f"  [{target}] {mid}: best dL={sol['best_any_dL']:.3g} susp={sol['n_susp']} [{time.time()-t0:.0f}s]", flush=True)
    OD.mkdir(parents=True, exist_ok=True); tag = target.replace(":", "_")
    pd.DataFrame(fin_rows).to_csv(OD / f"{tag}_finalists.csv.gz", index=False); pd.DataFrame(fine_rows).to_csv(OD / f"{tag}_fine.csv.gz", index=False)
    pd.DataFrame(loc_rows).to_csv(OD / f"{tag}_local.csv", index=False); pd.DataFrame(sol_rows).to_csv(OD / f"{tag}_solutions.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(OD / f"{tag}_replay.csv", index=False)
    np.savez_compressed(out, **arr)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target"); ap.add_argument("--list", action="store_true"); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(L_TARGETS)); return
    if a.merge:
        from wave2.oracle_merge import merge
        merge(); return
    run_target(a.target)


if __name__ == "__main__":
    main()
