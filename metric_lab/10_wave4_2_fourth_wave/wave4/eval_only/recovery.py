"""EVALUATION-ONLY (04_fourth_wave sections 5, 9, 15): lock every Lab 10 blind job file, THEN T-U recovery of every selected rotation and
the post-lock L(R*) diagnosis on the IDENTICAL frozen TRAIN objective (same measure / fidelity / seed / mask / variant).
Categories (Lab 08/09): recovered / objective-estimand issue (L_hat < L* - tau) / optimizer (L_hat > L* + tau) / ambiguous (|.| <= tau),
tau = .01 x (p95 - min) of the start losses of the same search.  Extra fields: signed / absolute T-U matrices, min matched, absolute thresholds.
"""
import json
import numpy as np, pandas as pd
from ..common import OUT, RES, PRIMARY, SECONDARY, family_group, sha256, leak_threshold
from ..search import BLIND, objective
from wave2.blind_inputs import load_payload
from wave2.evaluate_recovery import EvalData, recovery, supervised


def job_files(batches=None):
    fs = [f for f in sorted(BLIND.glob("*/*/*.npz")) if ".tmp" not in f.name]
    return [f for f in fs if batches is None or f.parent.parent.name in batches]


def lock(files, name):
    df = pd.DataFrame([dict(file=str(f.relative_to(OUT)), sha256=sha256(f), bytes=f.stat().st_size) for f in files])
    RES.mkdir(exist_ok=True); df.to_csv(RES / name, index=False); return df


def category(passed, f, Ls, tau):
    if passed:
        return "recovered"
    if not (np.isfinite(f) and np.isfinite(Ls) and np.isfinite(tau)):
        return "undefined"
    if f < Ls - tau:
        return "objective / estimand issue"
    if f > Ls + tau:
        return "optimizer / search budget"
    return "ambiguous landscape (near-tied)"


def evaluate(batches, out_name, panel=PRIMARY + SECONDARY, lock_name=None):
    files = job_files(batches); lk = lock(files, lock_name or out_name.replace("recovery", "locks")); print(f"locked {len(lk)} job files", flush=True)
    rows, Rs, mats = [], {}, {}
    for toy in panel:
        fl = [f for f in files if f.parent.name == toy.replace("#", "_")]
        if not fl:
            continue
        pl = load_payload(toy); ed = EvalData(toy); sups = {v: supervised(pl, ed, v) for v in ed.refs}; Lc = {}
        base = dict(toy=toy, cls=pl.cls, fam_group=family_group(toy), panel="primary" if toy in PRIMARY else "secondary")
        for f in fl:
            z = np.load(f, allow_pickle=False); diag = json.loads(str(z["diag"])); meta = json.loads(str(z["meta"]))
            for key, d in diag.items():
                v = d["variant"]
                if v not in sups:
                    rows.append(dict(base, batch=d["batch"], cid=d["cid"], key=key, variant=v, status="no supervised reference", pass_lab05=np.nan)); continue
                R = z[f"{key}|R"]
                if R.shape != (pl.k, pl.k) or not np.all(np.isfinite(R)):
                    rows.append(dict(base, batch=d["batch"], cid=d["cid"], key=key, variant=v, Q=d["Q"], status=d.get("status"), pass_lab05=False,
                                     category="invalid objective (no valid start)")); continue
                rec, raw = recovery(pl, ed, v, R, sups[v])
                ck = (d["cid"], d["Q"], d["fidelity"], d["iseed"], d["mask"], v)
                if ck not in Lc:
                    Lc[ck] = objective(pl, d["cid"], d["Q"], d["fidelity"], d["iseed"], d["mask"], v)[0]
                Ls = float(Lc[ck](ed.refs[v]["R_star"])); tau = .01 * (d["start_p95"] - d["start_min"])
                rid = f"{d['batch']}|{toy}|{d['cid']}|{key}"
                row = dict(base, row_id=rid, batch=d["batch"], cid=d["cid"], key=key, variant=v, Q=d["Q"], fidelity=d["fidelity"], mask=d["mask"], iseed=d["iseed"],
                           sseed=d["sseed"], budget=d["budget"], cache=d["cache"], L_hat=d["f"], L_val=d.get("val_f"), L_Rstar=Ls, tau=tau, dL_vs_Rstar=d["f"] - Ls,
                           converged=d.get("best_converged"), n_tied=d.get("n_tied"), tie_used=d.get("tie_breaker_used"), near_minima_max_sep=d.get("near_minima_max_sep"),
                           n_evals=d.get("n_evals"), seconds=d.get("seconds"), **rec)
                row["abs_good"] = bool(rec["MCC"] >= .9 and rec["leak_max"] < leak_threshold(toy))
                row["category"] = category(rec["pass_lab05"], d["f"], Ls, tau); rows.append(row); Rs[rid] = R; mats[rid] = raw["signed"]
        print(f"  evaluated {toy}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / out_name, index=False)
    np.savez_compressed(RES / out_name.replace("recovery.csv", "found_rotations.npz"), **Rs)
    np.savez_compressed(RES / out_name.replace("recovery.csv", "recovery_correlations.npz"), **mats)
    print(f"{out_name}: {len(df)} rows"); return df
