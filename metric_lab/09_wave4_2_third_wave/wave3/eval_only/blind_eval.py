"""EVALUATION-ONLY (W3-05 section 22): lock every W3 blind job file, then T-U recovery of every selected rotation and the post-lock
L(R*) diagnosis on the identical frozen objective.  Order: (1) all job files hashed into results/blind_locks.csv; (2) only then T / R*.
Categories (as Lab 08): recovered / objective-estimand issue (L_hat < L* - tau) / optimizer (L_hat > L* + tau) / ambiguous (|.| <= tau),
tau = 0.01 x (p95 - min) of the Haar start losses of the same search.  Selector rows (MX-SEL) have no single objective: category NA.
"""
import json
import numpy as np, pandas as pd
from ..common import OUT, RES, PRIMARY, SECONDARY, family_group, sha256
from ..jobs import BLIND
from wave2.blind_inputs import load_payload
from wave2.evaluate_recovery import EvalData, recovery, supervised


def job_files(batches=None):
    fs = sorted(BLIND.glob("*/*/*.npz"))
    fs = [f for f in fs if not f.name.endswith(".tmp.npz") and not f.name.endswith(".reeval.npz")]
    return [f for f in fs if batches is None or f.parent.parent.name in batches]


def lock(files, name="blind_locks.csv"):
    rows = [dict(file=str(f.relative_to(OUT)), sha256=sha256(f), bytes=f.stat().st_size) for f in files]
    df = pd.DataFrame(rows); RES.mkdir(exist_ok=True); df.to_csv(RES / name, index=False); return df


def category(rec, f, Lstar, tau):
    if rec["pass_lab05"]:
        return "recovered"
    if not (np.isfinite(f) and np.isfinite(Lstar) and np.isfinite(tau)):
        return "undefined"
    if f < Lstar - tau:
        return "objective / estimand issue"
    if f > Lstar + tau:
        return "optimizer / search budget"
    return "ambiguous landscape (near-tied)"


def objective_for(batch, cid, pl, variant, iseed, mask, Q="Q01"):
    """rebuild the exact frozen TRAIN objective of a job (blind code path)"""
    import run_search as RS
    from .. import jobs as JB
    from ..objectives import build
    from wave2.blind_inputs import variant_frame
    if batch == "MX-SEL" or cid == "MX-SEL":
        return None
    if batch == "FIN-D" and iseed == 211:
        import run_fin_extra as FX
        return FX.d_objective(pl, cid, variant, mask)
    if batch == "FIN-SENS":
        from ..common import chan
        e = JB.registry()[cid]; mo, _ = JB.he_cache(pl, "train", Q, "C", iseed)
        return build(e["spec"], mo[(chan(pl), mask)], variant_frame(pl, variant)[1])
    if batch == "ST-E":
        import run_reduced as RR
        return RR.blind_loss(pl, variant)
    if cid in RS.FUSIONS:
        return RS.fusion_factory(cid)(pl, variant, iseed, mask)[0]
    e = JB.registry()[cid]; mo, _, _ = JB.surface_caches(pl, e["surface"], iseed, mask)
    return build(e["spec"], mo, variant_frame(pl, variant)[1])


def evaluate(batches=None, out_name="blind_recovery.csv", trace=True, panel=PRIMARY + SECONDARY):
    files = job_files(batches); locks = lock(files, out_name.replace("recovery", "locks")); print(f"locked {len(locks)} job files", flush=True)
    rows, tr_rows, Rs = [], [], {}
    for toy in panel:
        fl = [f for f in files if f.parent.name == toy.replace("#", "_")]
        if not fl:
            continue
        pl = load_payload(toy); ed = EvalData(toy); sups = {v: supervised(pl, ed, v) for v in ed.refs}
        base = dict(toy=toy, cls=pl.cls, fam_group=family_group(toy), panel="primary" if toy in PRIMARY else "secondary")
        Lcache = {}
        for f in fl:
            batch = f.parent.parent.name; z = np.load(f, allow_pickle=False); diag = json.loads(str(z["diag"])); meta = json.loads(str(z["meta"]))
            trf = f.with_suffix(".trace.csv.gz")
            try:
                tr = pd.read_csv(trf) if (trace and trf.is_file()) else pd.DataFrame()
            except pd.errors.EmptyDataError:
                tr = pd.DataFrame()
            for key, d in diag.items():
                v = d["variant"]; cid = d.get("cid", meta.get("cid"))
                if v not in sups:
                    continue
                R = z[f"{key}|R"]
                if R.shape != (pl.k, pl.k) or not np.all(np.isfinite(R)):
                    rows.append(dict(base, batch=batch, cid=cid, key=key, variant=v, Q=d.get("Q", "Q01"), mask=d.get("mask"), iseed=d.get("iseed"), sseed=d.get("sseed"),
                                     status=d.get("status"), pass_lab05=False, category="invalid objective (no valid start)")); continue
                rec, raw = recovery(pl, ed, v, R, sups[v])
                ck = (batch, cid, v, d["iseed"], d["mask"], d.get("Q", "Q01"))
                if ck not in Lcache:
                    try:
                        Lcache[ck] = objective_for(batch, cid, pl, v, d["iseed"], d["mask"], d.get("Q", "Q01"))
                    except Exception as ex:                                     # recorded, never silently dropped
                        print(f"  L(R*) unavailable {ck}: {ex}"); Lcache[ck] = None
                L = Lcache[ck]; Lstar = float(L(ed.refs[v]["R_star"])) if L is not None else np.nan
                tau = 0.01 * (d["start_p95"] - d["start_min"]) if np.isfinite(d.get("start_p95", np.nan)) else np.nan
                rid = f"{batch}|{toy}|{cid}|{key}"
                row = dict(base, row_id=rid, batch=batch, cid=cid, key=key, variant=v, Q=d.get("Q", "Q01"), mask=d["mask"], iseed=d["iseed"], sseed=d["sseed"], budget=d["budget"],
                           cache=d.get("cache"), L_hat=d["f"], L_val=d.get("val_f", np.nan), L_Rstar=Lstar, tau=tau, dL_vs_Rstar=d["f"] - Lstar,
                           converged=d.get("best_converged"), n_tied=d.get("n_tied"), near_minima_max_sep=d.get("near_minima_max_sep"), n_evals=d.get("n_evals"),
                           seconds=d.get("seconds"), selected=d.get("selected"), disagreement=d.get("disagreement"), abstain_flag=d.get("abstain_flag"), **rec)
                row["category"] = category(rec, d["f"], Lstar, tau) if L is not None else ("recovered" if rec["pass_lab05"] else "NA (selector)")
                rows.append(row); Rs[rid] = R
                if trace and len(tr) and f"{key}|finalQ" in z.files:
                    F = z[f"{key}|finalQ"]; t = tr[tr.key == key]
                    for (_, r_), Q in zip(t.iterrows(), F):
                        rr, _ = recovery(pl, ed, v, Q, sups[v])
                        tr_rows.append(dict(batch=batch, toy=toy, cid=cid, key=key, rank=r_["rank"], L_final=r_["L_final"], val_L=r_["val_L"], converged=r_["converged"],
                                            selected=r_["selected"], MCC=rr["MCC"], leak_max=rr["leak_max"], pass_lab05=rr["pass_lab05"], dist_to_Rstar=rr["dist_to_Rstar"],
                                            L_Rstar=Lstar))
        print(f"  evaluated {toy}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / out_name, index=False)
    if trace:
        pd.DataFrame(tr_rows).to_csv(RES / out_name.replace("recovery.csv", "search_trace.csv.gz"), index=False)
    np.savez_compressed(RES / out_name.replace("recovery.csv", "found_rotations.npz"), **Rs)
    print(f"{out_name}: {len(df)} rows")
    return df
