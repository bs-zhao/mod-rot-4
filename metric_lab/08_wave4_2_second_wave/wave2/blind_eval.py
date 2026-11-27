"""EVALUATION-ONLY (W2-01 section 8): lock check, T-U recovery of every saved blind solution, post-lock L(R*) diagnosis.

Order: (1) every blind job file is hashed and the hashes are written to results/blind_locks.csv; (2) only then are T and R* opened.
Rows: one per blind search (selected R) plus the no-search baselines N01 (identity) and N02 (fixed Haar), once per toy x variant.
Categories (L* = blind loss at R* on the identical frozen cache, tau = 0.01 x landscape scale of the Haar start losses):
  recovered (Lab 05 criterion)                         -> "recovered"
  not recovered, L_hat < L* - tau                      -> "objective / estimand issue"
  not recovered, L_hat > L* + tau                      -> "optimizer / search budget"
  not recovered, |L_hat - L*| <= tau                   -> "ambiguous landscape (near-tied)"
"""
import json
import numpy as np, pandas as pd
from .common import OUT, RES, PRIMARY, SECONDARY, VARIANTS, sha256, family_group, cfg
from .blind_inputs import load_payload, variant_frame
from .objectives import load_moments, Objective, CACHE
from .optimizers import haar
from .evaluate_recovery import EvalData, recovery, supervised

BLIND = OUT / "blind"
CANDS = {c["id"]: c for c in cfg("blind_candidates.json")["candidates"]}
HAAR_SEED = 20261126


def job_files():
    return sorted(BLIND.glob("*/*/*.npz"))


def lock():
    rows = [dict(file=str(f.relative_to(OUT)), sha256=sha256(f), bytes=f.stat().st_size) for f in job_files() if not f.name.endswith(".tmp.npz")]
    df = pd.DataFrame(rows); RES.mkdir(exist_ok=True); df.to_csv(RES / "blind_locks.csv", index=False); return df


def category(rec, f, Lstar, tau):
    if rec["pass_lab05"]:
        return "recovered"
    if not (np.isfinite(f) and np.isfinite(Lstar)):
        return "undefined"
    if f < Lstar - tau:
        return "objective / estimand issue"
    if f > Lstar + tau:
        return "optimizer / search budget"
    return "ambiguous landscape (near-tied)"


def evaluate_all():
    locks = lock(); print(f"locked {len(locks)} blind job files")
    rows, trace, Rs, Cs = [], [], {}, {}
    toys = list(PRIMARY) + list(SECONDARY); ctx = {}
    for toy in toys:
        files = [f for f in job_files() if f.parent.name == toy.replace("#", "_")]
        if not files:
            continue
        pl = load_payload(toy); ed = EvalData(toy); ctx[toy] = (pl, ed)
        sups = {}
        for v in VARIANTS:
            vref = v if v in ed.refs else "s9"
            if v not in ed.refs:            # secondary runs without S8_2: no matching supervised ceiling for the standardised variant
                continue
            sups[v] = supervised(pl, ed, vref)
        cls = pl.cls; base = dict(toy=toy, cls=cls, fam_group=family_group(toy), panel="primary" if toy in PRIMARY else "secondary")
        for v in sups:                      # no-search baselines
            for cid, R in (("N01", np.eye(pl.k)), ("N02", haar(pl.k, 1, np.random.default_rng([HAAR_SEED, pl.k]))[0])):
                rec, raw = recovery(pl, ed, v, R, sups[v]); rid = f"baseline|{toy}|{v}|{cid}"
                rows.append(dict(base, row_id=rid, batch="baseline", variant=v, cid=cid, metric=CANDS[cid]["metric"], **rec)); Rs[rid] = R; Cs[rid] = raw["signed"]
        for f in files:
            batch = f.parent.parent.name; cid_file = f.stem; z = np.load(f, allow_pickle=False); diag = json.loads(str(z["diag"]))
            trf = f.with_suffix(".trace.csv.gz")
            try:
                tr = pd.read_csv(trf) if trf.is_file() else pd.DataFrame()
            except pd.errors.EmptyDataError:
                tr = pd.DataFrame()
            mcache = {}
            for key, d in diag.items():
                cid = d.get("cid", cid_file); v = d["variant"]
                if v not in sups:
                    continue
                R = z[f"{key}|R"]
                if R.shape != (pl.k, pl.k) or not np.all(np.isfinite(R)):
                    rows.append(dict(base, batch=batch, cid=cid, metric=CANDS[cid]["metric"], variant=v, key=key, status=d.get("status"))); continue
                rec, raw = recovery(pl, ed, v, R, sups[v])
                if d["cache"] not in mcache:
                    mcache[d["cache"]] = load_moments(CACHE / toy.replace("#", "_") / f"{d['cache']}.npz")[0]
                mo = mcache[d["cache"]][(d["channel"] if "channel" in d else ("YC" if cls == "C" else "YL"), d["mask"])]
                L = Objective(mo, variant_frame(pl, v)[1], CANDS[cid]["metric"]); Lstar = L(ed.refs[v]["R_star"])
                tau = 0.01 * (d["start_p95"] - d["start_min"]) if np.isfinite(d.get("start_p95", np.nan)) else np.nan
                rid = f"{batch}|{toy}|{cid}|{key}"
                row = dict(base, row_id=rid, batch=batch, cid=cid, metric=CANDS[cid]["metric"], key=key, variant=v, Q=d["Q"], mask=d["mask"],
                           iseed=d["iseed"], sseed=d["sseed"], budget=d["budget"], cache=d["cache"], L_hat=d["f"], L_val=d.get("val_f", np.nan),
                           L_Rstar=Lstar, tau=tau, dL_vs_Rstar=d["f"] - Lstar, converged=d["best_converged"], n_tied=d["n_tied"],
                           tie_breaker_used=d["tie_breaker_used"], near_minima_max_sep=d["near_minima_max_sep"], n_evals=d["n_evals"], seconds=d["seconds"], **rec)
                row["category"] = category(rec, d["f"], Lstar, tau) if cid != "N03" else "invariant control (R arbitrary)"
                rows.append(row); Rs[rid] = R; Cs[rid] = raw["signed"]
                if len(tr):                       # finalist recovery (loss vs MCC scatter; never used for selection)
                    F = z[f"{key}|finalQ"]; t = tr[tr.key == key]
                    for (_, r_), Q in zip(t.iterrows(), F):
                        rr, _ = recovery(pl, ed, v, Q, sups[v])
                        trace.append(dict(batch=batch, toy=toy, cls=cls, cid=cid, key=key, rank=r_["rank"], start=r_["start"], identity_start=r_["identity_start"],
                                          L_initial=r_["L_initial"], L_final=r_["L_final"], val_L=r_["val_L"], converged=r_["converged"], sweeps=r_["sweeps"],
                                          n_evals=r_["n_evals"], selected=r_["selected"], MCC=rr["MCC"], leak_max=rr["leak_max"], pass_lab05=rr["pass_lab05"],
                                          dist_to_Rstar=rr["dist_to_Rstar"]))
        print(f"  evaluated {toy}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "blind_recovery.csv", index=False)
    pd.DataFrame(trace).to_csv(RES / "blind_search_trace.csv.gz", index=False)
    ref = df[df.batch != "baseline"][["row_id", "batch", "toy", "cid", "variant", "Q", "mask", "iseed", "sseed", "budget", "L_hat", "L_Rstar", "tau", "dL_vs_Rstar", "pass_lab05", "category"]]
    ref.to_csv(RES / "blind_reference_loss.csv", index=False)
    np.savez_compressed(RES / "found_rotations.npz", **{k: v for k, v in Rs.items()})
    np.savez_compressed(RES / "recovery_correlations.npz", **{k: v for k, v in Cs.items()})
    print(f"blind_recovery.csv: {len(df)} rows")
    return df
