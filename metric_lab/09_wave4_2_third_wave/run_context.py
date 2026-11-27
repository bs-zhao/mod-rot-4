"""W3-04 DC-N / DC-R disjoint-context diagnostic (03_third_wave sections 17-19).

usage
  python run_context.py --encode --toy CR3                 # blind, GPU: every nested / repeat representation (wave3.context)
  python run_context.py --search --toy CR3 --cid E02       # blind: S0 search on each DC-N representation (context seed 31), both frames
  python run_context.py --evaluate --toy CR3               # EVALUATION-ONLY: Stage A, same-context reference, recovery, local Gap 2
  python run_context.py --merge                            # -> results/context_diagnostics.csv, context_recovery.csv, context_local.csv
Representations searched: N|31|{25,50,100,200,all} (disjoint from the fixed query) and the official all-trial S5 Z ('full', query included).
Frames: s9 = raw Z;  s9_2 = TRAIN-fit marginal standardisation (strict; the historical all-subject S9_2 scaling is transductive and is NOT
used here).  Moments: Q01 C seed-31 TRAIN / VALIDATION designs with the representation's Z (same subjects / X rows as the main runs).
Evaluation: (i) same-context supervised reference = orthogonal Procrustes of TRAIN-standardised T on the TRAIN-centred frame (TRAIN subjects),
evaluated on TEST; (ii) official S5 reference as sensitivity; pass = Lab 05 criterion against the same-context ceiling.
"""
import sys, json, time, argparse, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT, RES, PRIMARY, chan, prim_mask, code_hash, leak_threshold
import numpy as np, pandas as pd
from wave3 import context as CX, jobs as JB
from wave3.objectives import build
from wave2.blind_inputs import load_payload
from wave2.designs import design
from wave2.objectives import compute_moments, save_moments, load_moments, masks_of
import run_blind as RB8

REPS = ("N|31|25", "N|31|50", "N|31|100", "N|31|200", "N|31|all", "full")


def rep_payload(pl, Z):
    return dataclasses.replace(pl, z=np.asarray(Z, float))


def frame(Z, train, variant):
    c = Z[train].mean(0)
    return (c, np.ones(Z.shape[1])) if variant == "s9" else (c, Z[train].std(0))


def rep_moments(pl, Z, key, pool):
    f = CX.CTX / pl.toy.replace("#", "_") / "moments" / f"{key.replace('|', '_')}_{pool}.npz"
    if f.is_file():
        return load_moments(f)[0]
    p2 = rep_payload(pl, Z); d = design(p2, "Q01", "C", 31, pool=pool)
    res = compute_moments(p2, RB8.eta_factory(pl)(), d, (chan(pl),), masks_of(pl)); save_moments(f, res, dict(design=d["meta"], rep=key)); return res


def load_reps(pl):
    return np.load(CX.CTX / pl.toy.replace("#", "_") / "reps.npz")


def search(toy, cid):
    out = CX.CTX / toy.replace("#", "_") / "blind" / f"{cid}.npz"
    if out.is_file():
        print(f"  [{toy}] DC-N {cid} done"); return
    pl = load_payload(toy); reps = load_reps(pl); m = prim_mask(pl); c = chan(pl); spec = JB.registry()[cid]["spec"]; recs = []; t0 = time.time()
    tr = np.sort(pl.splits["train"])
    for key in REPS:
        Z = reps[f"{key}|Z"]; mo = rep_moments(pl, Z, key, "train")[(c, m)]; mv = rep_moments(pl, Z, key, "validation")[(c, m)]
        for v in ("s9", "s9_2"):
            sc = frame(Z, tr, v)[1]; L = build(spec, mo, sc); Lv = build(spec, mv, sc)
            for ss in (31, 73, 101):
                r = RB8.run_cell(L, pl.k, ss, "S0", tie=Lv)
                recs.append((f"{key}|{v}|s{ss}", r, dict(rep=key, variant=v, Q="Q01", mask=m, channel=c, iseed=31, sseed=ss, budget="S0", cid=cid, cache=f"rep {key}",
                             val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])))
                print(f"  [{toy}] DC-N {cid} {key} {v} s{ss}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
    RB8.save_job(out, recs, dict(batch="DC-N", toy=toy, cid=cid, spec=spec, frames="s9 raw / s9_2 TRAIN-fit", code_hash=code_hash()))


# ---------------------------------------------------------------- evaluation-only
def evaluate(toy):
    from wave2.evaluate_recovery import EvalData, matched
    from wave2.optimizers import rel_dist
    from wave4_2.inputs import load
    from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1
    from wave4_2.surfaces import DecoderSurface
    from wave4_2.measures import blind_design
    from wave4_2.moments import jac
    import run_baseline as RB7
    from wave3.common import OUT07, toy_entry
    out = CX.CTX / toy.replace("#", "_") / "evaluation.npz"
    pl = load_payload(toy); ed = EvalData(toy); reps = load_reps(pl); meta = json.loads(str(reps["meta"])); T = ed.T; k = pl.k
    tr, te = np.sort(pl.splits["train"]), np.sort(pl.splits["test"]); dec = DecoderSurface(pl); thr = leak_threshold(toy)
    qc = reps["query_counts"]; qid = reps["query_ids"]; qoff = np.r_[0, np.cumsum(qc)]
    rows_diag, rows_rec, rows_loc = [], [], []
    Tm, Ts = T[tr].mean(0), T[tr].std(0)
    keys = [k_[:-2] for k_ in reps.files if k_.endswith("|Z")]
    b, ev, rots, _ = load(toy); n_exp = 2 ** k - k - 1
    t7 = np.load(OUT07 / "W1" / toy_entry(toy)["run"] / "tensors" / "Q01_pilot_s31.npz"); m = prim_mask(pl); ch = chan(pl)
    Jo = t7[f"oracle|{ch}|{m}|J"]
    d_pilot = blind_design(pl, "Q01", "pilot", 31)
    jobs = sorted((CX.CTX / toy.replace("#", "_") / "blind").glob("*.npz"))
    for key in keys:
        Z = reps[f"{key}|Z"]; LV = reps[f"{key}|LV"]
        ev_ = np.linalg.eigvalsh(np.cov(Z[tr], rowvar=False)); pr = ev_ / ev_.sum()
        row = dict(toy=toy, rep=key, ood_frac=meta["info"].get(key, {}).get("ood_frac", np.nan) if key.startswith("N|") else 0.0,
                   n_context_median=meta["info"].get(key, {}).get("n_context_median", np.nan), sd_mean=float(Z[tr].std(0).mean()), post_var_mean=float(np.exp(LV[tr]).mean()),
                   cov_cond=float(ev_.max() / max(ev_.min(), 1e-300)), eff_rank=float(np.exp(-(pr * np.log(pr + 1e-300)).sum())))
        # linear T ~ Z (TRAIN fit, TEST R^2)
        X1 = np.column_stack([np.ones(len(tr)), Z[tr]]); W = np.linalg.lstsq(X1, T[tr], rcond=None)[0]; P = np.column_stack([np.ones(len(te)), Z[te]]) @ W
        r2 = 1 - ((T[te] - P) ** 2).sum(0) / ((T[te] - T[te].mean(0)) ** 2).sum(0); row.update({f"R2_T{i}": float(r2[i]) for i in range(k)}, R2_mean=float(r2.mean()))
        # held-out query prediction on TEST subjects (contexts never contain these trials)
        if key.startswith("N|") or key == "full":
            idx = np.concatenate([pl.starts[s] + qid[qoff[s]:qoff[s + 1]] for s in te]); sid = pl.subject_index[idx]
            pred = dec.predict(Z[sid], (pl.x[idx] - pl.x_mean) / pl.x_std); y = pl.y[idx]
            if pl.cls == "B":
                pp = np.clip(1 / (1 + np.exp(-pred)), 1e-12, 1 - 1e-12); row.update(query_nll=float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))), query_acc=float(np.mean((pp > .5) == y)))
            else:
                row.update(query_mse=float(np.mean((pred - y) ** 2)), query_r2=float(1 - np.mean((pred - y) ** 2) / np.var(y)))
        # DC-R repeatability (pairs of disjoint blocks)
        if key.startswith("R|") and key.endswith("|0"):
            Zb = reps[f"{key[:-1]}1|Z"]; row["repeat_corr_mean"] = float(np.mean([np.corrcoef(Z[te, j], Zb[te, j])[0, 1] for j in range(Z.shape[1])]))
        # same-context supervised reference per frame
        for v in ("s9", "s9_2"):
            c_, s_ = frame(Z, tr, v); V = (Z - c_) / s_; Mx = ((T[tr] - Tm) / Ts).T @ V[tr]; U_, _, Wt = np.linalg.svd(Mx); Rs = U_ @ Wt
            sup = matched(T[te], V[te] @ Rs.T); row.update({f"sup_MCC_{v}": sup["MCC"], f"sup_leak_{v}": sup["leak_max"]})
            raw = matched(T[te], V[te]); row.update({f"raw_MCC_{v}": raw["MCC"]})
            # official-reference sensitivity: the saved S8 / S8_2 reference evaluated on this representation's frame
            if key == "full" and v in ed.refs:
                row[f"official_vs_emulated_dist_{v}"] = rel_dist(Rs, ed.refs[v]["R_star"])
            for jf in jobs:
                zj = np.load(jf); diag = json.loads(str(zj["diag"])); cid = jf.stem
                for kk, dd in diag.items():
                    if dd["rep"] != key or dd["variant"] != v:
                        continue
                    R = zj[f"{kk}|R"]; res = matched(T[te], V[te] @ R.T)
                    rows_rec.append(dict(toy=toy, rep=key, variant=v, cid=cid, sseed=dd["sseed"], L_hat=dd["f"], MCC=res["MCC"], leak_max=res["leak_max"],
                                         sup_MCC=sup["MCC"], sup_leak=sup["leak_max"], MCC_gap=sup["MCC"] - res["MCC"], dist_to_Rstar=rel_dist(R, Rs),
                                         pass_same_ctx=bool(res["MCC"] >= sup["MCC"] - .05 and res["leak_max"] < thr), ood_frac=row["ood_frac"]))
            # local Gap 2 on the same relative sweeps: decoder at this representation (TEST pilot points), oracle from the saved Lab 07 tensor
            if key in REPS and v in rots:
                p2 = rep_payload(pl, Z); dp = dict(d_pilot); dp["lat"] = Z[d_pilot["sid"]]
                mo = compute_moments(p2, RB8.eta_factory(pl)(), dp, (ch,), (m,))[(ch, m)]
                man = RB7.manifest_rows(rots[v]); sw = sweeps(man); cids = [jf.stem for jf in jobs]
                specs = [JB.registry()[c_]["spec"] for c_ in cids]
                Lo = np.zeros((len(rots[v]), len(cids))); Li = Lo.copy()
                for i, r in enumerate(rots[v]):
                    for j, sp in enumerate(specs):
                        Lo[i, j] = build(sp, dict(K=t7[f"oracle|{ch}|{m}|K"], J=Jo, valid=True), r.t_std)(r.G)
                        Li[i, j] = build(sp, mo, s_)(r.G @ Rs)
                fo, fi = library_flags(curve_stats(Lo, sw), n_exp), library_flags(curve_stats(Li, sw), n_exp); e1 = eval1(Lo, Li, man, sw)
                for j, cid in enumerate(cids):
                    rows_loc.append(dict(toy=toy, rep=key, variant=v, cid=cid, O_perfect=bool(fo["perfect"][j]), O_marginal=bool(fo["marginal"][j]),
                                         I_marginal=bool(fi["marginal"][j]), I_perfect=bool(fi["perfect"][j]), pearson=float(e1["pearson"][j]), spearman=float(e1["spearman"][j])))
        rows_diag.append(row)
    pd.DataFrame(rows_diag).to_csv(out.with_name("diagnostics.csv"), index=False); pd.DataFrame(rows_rec).to_csv(out.with_name("recovery.csv"), index=False)
    pd.DataFrame(rows_loc).to_csv(out.with_name("local.csv"), index=False)
    print(f"  [{toy}] evaluated", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--cid"); ap.add_argument("--encode", action="store_true")
    ap.add_argument("--search", action="store_true"); ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.encode:
        print(CX.representations(load_payload(a.toy), a.device))
    elif a.search:
        search(a.toy, a.cid)
    elif a.evaluate:
        evaluate(a.toy)
    elif a.merge:
        for name in ("diagnostics", "recovery", "local"):
            fs = sorted(CX.CTX.glob(f"*/{name}.csv")); pd.concat([pd.read_csv(f) for f in fs]).to_csv(RES / f"context_{name}.csv", index=False)


if __name__ == "__main__":
    main()
