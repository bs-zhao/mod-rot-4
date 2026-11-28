"""REP-4 (blind, 04_fourth_wave sections 8-9): every finalist of every REP-1 / REP-2 RW-GL Q05 cell re-scored on BOTH independent REP-3
D caches (Q05 D, integration seeds 211 / 613, TRAIN pool).  Lab 09 escalation flags: selected R not converged, separated near-minima
(> .05 within 1e-3), inconsistent selections across the cell's seeds (> .05), or a D cache preferring another finalist (> .05 away and
lower by more than 1e-9 relative).  Flagged (variant, integration seed) cells get ONE S1 (5x) search, search seed 31, on their own cache
(reported separately; the selected R is never replaced).  Output: OUT/blind/REP-4/<toy>/RW-GL...__S1.npz and <toy>.reeval.csv.
"""
import json, time
import numpy as np, pandas as pd
from .common import CAND, chan, prim_mask, code_hash
from .search import BLIND, job_path, objective
from wave2.blind_inputs import load_payload
from wave2.optimizers import rel_dist
import run_blind as RB8

D_SEEDS = (211, 613)


def reeval(toy):
    out = BLIND / "REP-4" / toy.replace("#", "_") / "reeval.csv"
    if out.is_file():
        print(f"  [{toy}] REP-4 done"); return
    pl = load_payload(toy); mask = prim_mask(pl); t0 = time.time(); rows, recs = [], []
    for batch in ("REP-1", "REP-2"):
        z = np.load(job_path(batch, toy, CAND, "Q05")); diag = json.loads(str(z["diag"]))
        for v in ("s9", "s9_2"):
            LD = {s: objective(pl, CAND, "Q05", "D", s, mask, v)[0] for s in D_SEEDS}
            keys = [k for k in diag if diag[k]["variant"] == v]; sel = {k: z[f"{k}|R"] for k in keys}
            spread = max([rel_dist(sel[a], sel[b]) for a in keys for b in keys] + [0.0]); flagged = set()
            for k_ in keys:
                F = z[f"{k_}|finalQ"]; best = diag[k_]["best_index"]; r = dict(toy=toy, batch=batch, key=k_, variant=v, iseed=diag[k_]["iseed"], sseed=diag[k_]["sseed"],
                                                                                L_C=diag[k_]["f"], seed_spread=spread)
                prefer = False
                for s in D_SEEDS:
                    fD = np.array([LD[s](Q) for Q in F]); j = int(np.nanargmin(fD))
                    alt = [i for i in range(len(F)) if rel_dist(F[i], F[best]) > .05 and fD[i] < fD[best] - 1e-9 * (abs(fD[best]) + 1e-12)]
                    r.update({f"L_D{s}_selected": float(fD[best]), f"L_D{s}_min": float(fD[j]), f"D{s}_best_is_selected": bool(j == best or rel_dist(F[j], F[best]) <= .05),
                              f"D{s}_rank_of_selected": int((fD < fD[best]).sum())})
                    prefer |= len(alt) > 0
                r.update(not_converged=not diag[k_]["best_converged"], separated_near_minima=diag[k_]["near_minima_max_sep"] > .05,
                         seed_inconsistent=spread > .05, D_prefers_other=prefer)
                r["escalate"] = bool(r["not_converged"] or r["separated_near_minima"] or r["seed_inconsistent"] or r["D_prefers_other"]); rows.append(r)
                if r["escalate"]:
                    flagged.add(diag[k_]["iseed"])
            for iseed in sorted(flagged):
                L, tag = objective(pl, CAND, "Q05", "C", iseed, mask, v); Lv, vtag = objective(pl, CAND, "Q05", "C", iseed, mask, v, "validation")
                r = RB8.run_cell(L, pl.k, 31, "S1", tie=Lv)
                recs.append((f"{v}|Q05@C|{mask}|i{iseed}|s31|S1", r, dict(batch="REP-4", cid=CAND, variant=v, Q="Q05", fidelity="C", mask=mask, channel=chan(pl), iseed=iseed,
                             sseed=31, budget="S1", cache=tag, val_cache=vtag, val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])))
                print(f"  [{toy}] REP-4 S1 {v} i{iseed}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
    if recs:
        RB8.save_job(job_path("REP-4", toy, CAND, "Q05", "_S1"), recs, dict(batch="REP-4", toy=toy, cid=CAND, code_hash=code_hash()))
    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); print(f"  [{toy}] REP-4 [{time.time()-t0:.0f}s]", flush=True)
