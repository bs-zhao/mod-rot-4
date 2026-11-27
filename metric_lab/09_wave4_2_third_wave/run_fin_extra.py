"""W3-05 finalist robustness (blind; 03_third_wave sections 22-23).

usage: python run_fin_extra.py --reeval --toy CR3 --cid <finalist cid>    independent high-fidelity re-evaluation + blind escalation
       python run_fin_extra.py --sens --toy CR3 --cid <finalist cid>      measure / mask sensitivity searches
--reeval  every finalist of every FIN cell is re-scored on an independent TRAIN cache whose subjects are disjoint from the three medium caches
          (AD: Lab 08 train_Q01_D_s211_indep, 1024 x 128; RBF: the saved W2 RBF state on the same D design; fusions: both components with the
          cell's TRAIN calibration).  Blind escalation (Lab 08 trigger): selected R not converged, separated near-minima (> .05) within 1e-3,
          inconsistent selections across the 3 x 3 seeds (> .05), or the D cache preferring another finalist -> an S1 (5x) search on the cell
          cache and an S0 search on the D cache.  The selected solution is NOT replaced (both are reported).
--sens    search seed 31 x 3 on the seed-31 TRAIN caches of Q02, Q05, Q04 (banks) and Q01 MASK-COL (banks), VALIDATION tie-break.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT, cfg, chan, prim_mask, code_hash
import numpy as np, pandas as pd
from wave3 import jobs as JB
from wave3.objectives import build
from wave2.blind_inputs import load_payload, variant_frame
from wave2.designs import design, used_subjects, available
from wave2.objectives import masks_of, load_moments, save_moments
from wave2.optimizers import rel_dist
import run_blind as RB8
import run_search as RS

D_SEED = 211


def fin_entry(cid):
    return {c["cid"]: c for c in cfg("finalists.json")["finalists"]}[cid]


def d_objective(pl, base, variant, mask):
    used = np.unique(np.concatenate([used_subjects(design(pl, "Q01", "C", s, pool="train")) for s in (31, 73, 101)]))
    sc = variant_frame(pl, variant)[1]; c = chan(pl)
    def he(surface):
        if surface == "AD":
            return JB.he_cache(pl, "train", "Q01", "D", D_SEED, exclude=used, tag_extra="_indep")[0][(c, mask)]
        tag = f"train_Q01_D_s{D_SEED}_indep_rbfD1"; f = JB.CACHE9 / pl.toy.replace("#", "_") / f"{tag}.npz"
        if f.is_file():
            return load_moments(f)[0][(c, mask)]
        from wave2 import surrogates as S
        import run_inference as RI
        z = np.load(JB.OUT08 / "inference" / "estimator_fits" / f"{pl.toy.replace('#', '_')}_RBF-block_D1_Smin.npz")
        est = S.RBFEta(z["m"], z["s"], z["C"], float(z["h"]), z["a"], float(z["b"])); d = design(pl, "Q01", "D", D_SEED, pool="train", exclude=used)
        res = {(c, m): v for m, v in RI.moments_from_H(lambda P: est.hessian(P), pl, d, masks_of(pl)).items()}; save_moments(f, res, dict(design=d["meta"])); return res[(c, mask)]
    if base in RS.FUSIONS:
        from wave3 import fusion as FU
        fz = RS.FUSIONS[base]; comp = RS.components(pl, variant, 31, mask); calE = FU.calibrate(comp["E"][0], pl.k, fz.get("norm", "iqr")); calX = FU.calibrate(comp["X"][0], pl.k, fz.get("norm", "iqr"))
        reg = JB.registry(); E = build(reg["MX-B08c03@AD"]["spec"], he("AD"), sc); X = build(reg["MX-SL0UXUU@RBF"]["spec"], he("RBF"), sc)
        zmin = RS.zE_minimum(pl, variant, 31, mask, calE)[0] if fz["kind"] == "eps" else None
        return FU.Fused(FU.Z(E, calE), FU.Z(X, calX), fz["kind"], fz.get("w"), fz.get("delta"), zmin)
    e = JB.registry()[base]
    if e["surface"] == "GRAD":
        from wave3 import structure as ST
        return build(e["spec"], ST.load_grad(ST.build_grad_cache(pl, "train", D_SEED, "C")), sc)       # TRAIN grid of a distinct seed
    return build(e["spec"], he(e["surface"]), sc)


def reeval(toy, cid):
    out = JB.job_path("FIN-D", toy, cid)
    if out.is_file():
        print(f"  [{toy}] FIN-D {cid} done"); return
    fin = fin_entry(cid); base = fin["base"]; pl = load_payload(toy); mask = prim_mask(pl); t0 = time.time()
    z = np.load(JB.job_path("FIN", toy, base)); diag = json.loads(str(z["diag"])); rows, recs = [], []
    for v in ("s9", "s9_2"):
        LD = d_objective(pl, base, v, mask); keys = [k for k in diag if diag[k]["variant"] == v]; sel = {k: z[f"{k}|R"] for k in keys}
        spread = max([rel_dist(sel[a], sel[b]) for a in keys for b in keys] + [0.0])
        for k_ in keys:
            F = z[f"{k_}|finalQ"]; fD = np.array([LD(Q) for Q in F]); best = diag[k_]["best_index"]
            alt = [i for i in range(len(F)) if rel_dist(F[i], F[best]) > 0.05 and fD[i] < fD[best] - 1e-9 * (abs(fD[best]) + 1e-12)]
            flag = dict(not_converged=not diag[k_]["best_converged"], separated_near_minima=diag[k_]["near_minima_max_sep"] > 0.05, seed_inconsistent=spread > 0.05,
                        D_prefers_other=len(alt) > 0)
            rows.append(dict(toy=toy, cid=cid, base=base, key=k_, variant=v, L_C=diag[k_]["f"], L_D_selected=float(fD[best]), L_D_min=float(np.nanmin(fD)),
                             D_best_is_selected=bool(np.nanargmin(fD) == best or rel_dist(F[int(np.nanargmin(fD))], F[best]) <= 0.05), seed_spread=spread, **flag,
                             escalate=any(flag.values())))
        if any(r["escalate"] for r in rows if r["variant"] == v):
            for iseed in sorted({int(r["key"].split("|")[3][1:]) for r in rows if r["variant"] == v and r["escalate"]}):
                fac = RS.fusion_factory(base) if base in RS.FUSIONS else None
                if fac:
                    L, Lv, tag = fac(pl, v, iseed, mask)
                else:
                    e = JB.registry()[base]; mo, mv, tag = JB.surface_caches(pl, e["surface"], iseed, mask); sc = variant_frame(pl, v)[1]; L, Lv = build(e["spec"], mo, sc), build(e["spec"], mv, sc)
                r = RB8.run_cell(L, pl.k, 31, "S1", tie=Lv)
                recs.append((f"{v}|Q01|{mask}|i{iseed}|s31|S1", r, dict(cid=base, variant=v, Q="Q01", mask=mask, iseed=iseed, sseed=31, budget="S1", cache=tag,
                             val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])))
                print(f"  [{toy}] FIN-D S1 {base} {v} i{iseed}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
            r = RB8.run_cell(LD, pl.k, 31, "S0")
            recs.append((f"{v}|Q01|{mask}|iD{D_SEED}|s31|S0", r, dict(cid=base, variant=v, Q="Q01", mask=mask, iseed=D_SEED, sseed=31, budget="S0", cache="independent D",
                         val_f=np.nan, val_finalists=[np.nan] * len(r["finalists"]))))
    RB8.save_job(out, recs, dict(batch="FIN-D", toy=toy, cid=cid, base=base, code_hash=code_hash()))
    pd.DataFrame(rows).to_csv(out.with_suffix(".reeval.csv"), index=False); print(f"  [{toy}] FIN-D {cid} [{time.time()-t0:.0f}s]", flush=True)


def sens(toy, cid):
    out = JB.job_path("FIN-SENS", toy, cid)
    if out.is_file():
        return
    fin = fin_entry(cid); base = fin["base"]; e = JB.registry().get(base)
    if base in RS.FUSIONS or e is None or e["surface"] != "AD":
        print(f"  [{toy}] FIN-SENS {cid}: measure sensitivity defined for AD HE candidates only (skipped)"); return
    pl = load_payload(toy); c = chan(pl); recs = []; t0 = time.time()
    cells = [("Q02", prim_mask(pl)), ("Q05", prim_mask(pl)), ("Q04", prim_mask(pl)), ("Q01", "MASK-COL")]
    for Q, m in cells:
        if not available(pl, Q) or m not in masks_of(pl):
            continue
        mo, tag = JB.he_cache(pl, "train", Q, "C", 31); mv, _ = JB.he_cache(pl, "validation", Q, "C", 31)
        for v in ("s9", "s9_2"):
            sc = variant_frame(pl, v)[1]; L, Lv = build(e["spec"], mo[(c, m)], sc), build(e["spec"], mv[(c, m)], sc)
            for ss in (31, 73, 101):
                r = RB8.run_cell(L, pl.k, ss, "S0", tie=Lv)
                recs.append((f"{v}|{Q}|{m}|i31|s{ss}|S0", r, dict(cid=base, variant=v, Q=Q, mask=m, channel=c, iseed=31, sseed=ss, budget="S0", cache=tag,
                             val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])))
        print(f"  [{toy}] FIN-SENS {base} {Q} {m} [{time.time()-t0:.0f}s]", flush=True)
    if recs:
        RB8.save_job(out, recs, dict(batch="FIN-SENS", toy=toy, cid=cid, base=base, code_hash=code_hash()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--cid"); ap.add_argument("--reeval", action="store_true"); ap.add_argument("--sens", action="store_true")
    ap.add_argument("--jobs", action="store_true"); a = ap.parse_args()
    if a.jobs:
        from wave3.common import PRIMARY
        for t in PRIMARY:
            for f in cfg("finalists.json")["finalists"]:
                print(f"{t} {f['cid']}")
    elif a.reeval:
        reeval(a.toy, a.cid)
    elif a.sens:
        sens(a.toy, a.cid)
