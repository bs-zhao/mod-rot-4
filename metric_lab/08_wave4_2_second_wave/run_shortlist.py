"""W2-04 -> W2-01 secondary pilot: blind search with the post-hoc shortlisted improved inferred objective (development only).

usage: python run_shortlist.py --toy CR3
X01  L03|SL0.UXUU.e0.01 on the frozen decoder (AD), split-safe BS-B caches (TRAIN objective, VALIDATION tie-breaker)
X02  the same metric on the E-RBF block-scaled D1 teacher surrogate, refitted deterministically with the IF-B seeds and selected by
     validation loss (S-min); its state (landmarks, coefficients, scaling, bandwidth) is saved; moments on the identical TRAIN / VALIDATION
     Q01 C seed-31 designs.  Search S0, seeds 31 / 73 / 101, both variants.  The choice of metric + surface used truth-referenced sweep flags
     (W2-04), so this is a development pilot, not confirmation.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, code_hash
import numpy as np, pandas as pd
from wave2.blind_inputs import load_payload, variant_frame
from wave2.designs import design
from wave2.objectives import compute_moments, save_moments, load_moments, CACHE, masks_of, Objective
from wave2 import surrogates as S
import run_blind as RB
import run_inference as RI

MID = "L03|SL0.UXUU.e0.01"


def rbf_moments(pl, est, pool, tag):
    path = CACHE / pl.toy.replace("#", "_") / f"{tag}.npz"
    if path.is_file():
        return load_moments(path)[0]
    d = design(pl, "Q01", "C", 31, pool=pool)
    res = {(RB.chan(pl), m): v for m, v in RI.moments_from_H(lambda P: est.hessian(P), pl, d, masks_of(pl)).items()}
    save_moments(path, res, dict(design=d["meta"], surface="E-RBF block D1 S-min")); return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy", required=True); a = ap.parse_args(); toy = a.toy
    pl = load_payload(toy); c = RB.chan(pl); mask = RB.prim_mask(pl); t0 = time.time()
    st = OUT / "inference" / "estimator_fits" / f"{toy}_RBF-block_D1_Smin.npz"
    if st.is_file():
        z = np.load(st); est = S.RBFEta(z["m"], z["s"], z["C"], float(z["h"]), z["a"], float(z["b"]))
    else:
        D = RI.data(pl); fits = RI.fit_rbf_family(pl, D, "block", ("D1",)); f = min(fits, key=lambda g: g["val_loss"]); est = f["est"]
        st.parent.mkdir(parents=True, exist_ok=True)
        np.savez(st, m=est.m, s=est.s, C=est.C, h=est.h, a=est.a, b=est.b, val_loss=f["val_loss"], factor=f["degree"], lam=f["lam"])
    print(f"  [{toy}] surrogate ready [{time.time()-t0:.0f}s]", flush=True)
    surf = {"X01": (RB.cache(pl, "train", "Q01", "C", 31)[0][(c, mask)], RB.cache(pl, "validation", "Q01", "C", 31)[0][(c, mask)], "train_Q01_C_s31"),
            "X02": (rbf_moments(pl, est, "train", "train_Q01_C_s31_rbfD1")[(c, mask)], rbf_moments(pl, est, "validation", "validation_Q01_C_s31_rbfD1")[(c, mask)],
                    "train_Q01_C_s31_rbfD1")}
    for cid, (mo, mv, tag) in surf.items():
        out = RB.job_path("BS-SL", toy, cid)
        if out.is_file():
            continue
        recs = []
        for v in ("s9", "s9_2"):
            sc = variant_frame(pl, v)[1]; L = Objective(mo, sc, MID); Lv = Objective(mv, sc, MID)
            for ss in (31, 73, 101):
                r = RB.run_cell(L, pl.k, ss, "S0", tie=Lv)
                recs.append((f"{v}|Q01|{mask}|i31|s{ss}|S0", r, dict(variant=v, Q="Q01", mask=mask, channel=c, iseed=31, sseed=ss, budget="S0", cache=tag,
                             val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f_["Q"])) for f_ in r["finalists"]])))
                print(f"  [{toy}] {cid} {v} s{ss}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
        RB.save_job(out, recs, dict(batch="BS-SL", toy=toy, cid=cid, metric=MID, code_hash=code_hash()))


if __name__ == "__main__":
    main()
