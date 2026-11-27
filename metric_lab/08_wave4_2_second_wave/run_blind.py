"""W2-01: R*-free blind rotation search with frozen inferred objectives, then (separately) post-lock T-U evaluation.

usage
  python run_blind.py --batch BS-A|BS-B|BS-C|BS-E --toy C1 --caches            # blind moment caches of one toy (run first)
  python run_blind.py --batch BS-A|BS-B|BS-C|BS-E --toy C1 --metric E01         # blind searches of one toy x candidate
  python run_blind.py --batch BS-D --toy C1                                      # independent high-fidelity re-evaluation + S1 escalation
  python run_blind.py --evaluate                                                 # lock check, T-U recovery, L(R*) diagnosis -> results/
Batches (02_second_wave section 7):
  BS-A  historical replay: TEST pool, Q01, fidelity C, integration seed 31, 3 search seeds, E01/E02, C01-C04, N03
  BS-B  split-safe: TRAIN-pool objective, VALIDATION-pool tie-breaker / stability, TEST evaluation; Q01, C;
        E01/E02: integration seeds 31/73/101 x 3 search seeds; controls: seed 31 x 3
  BS-C  split-safe measure / mask sensitivity: Q02, Q05, Q04 (banks), Q01 MASK-COL (banks); integration seed 31, 3 search seeds
  BS-D  E01/E02 cells of BS-B: every finalist re-evaluated on an independent 1024 x 128 TRAIN cache (subjects disjoint from the medium
        caches); blind-diagnostic escalation: S1 (5x budget) on the cell's cache and an S0 search on the D cache
  BS-E  secondary NP runs: E01/E02, C01/C02; split-safe Q01 C, primary mask and MASK-COL (banks); seed 31, 3 search seeds
The blind part imports no truth module and opens only the blind payload, the frozen checkpoint and its own caches.
"""
import sys, json, time, argparse, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, RES, PRIMARY, SECONDARY, VARIANTS, cfg, code_hash, git_commit
import numpy as np, pandas as pd
from wave2.blind_inputs import load_payload, variant_frame
from wave2.designs import design, available, used_subjects
from wave2.objectives import cached_moments, masks_of, Objective
from wave2.optimizers import search, haar, rel_dist

BLIND = OUT / "blind"
CANDS = {c["id"]: c for c in cfg("blind_candidates.json")["candidates"]}
SEARCHABLE = [c for c in CANDS if CANDS[c]["search"]]
ALL_SEARCH_SEEDS = (31, 73, 101)
BATCH = {
    "BS-A": dict(mode="replay", pool="test", cells=[("Q01", "primary")], fid="C", iE=(31,), iC=(31,), metrics=SEARCHABLE, panel=PRIMARY),
    "BS-B": dict(mode="split", pool="train", cells=[("Q01", "primary")], fid="C", iE=(31, 73, 101), iC=(31,), metrics=SEARCHABLE, panel=PRIMARY),
    "BS-C": dict(mode="split", pool="train", cells=[("Q02", "primary"), ("Q05", "primary"), ("Q04", "primary"), ("Q01", "MASK-COL")], fid="C",
                 iE=(31,), iC=(31,), metrics=SEARCHABLE, panel=PRIMARY),
    "BS-E": dict(mode="split", pool="train", cells=[("Q01", "primary"), ("Q01", "MASK-COL")], fid="C", iE=(31,), iC=(31,),
                 metrics=["E01", "E02", "C01", "C02"], panel=SECONDARY),
}
D_SEED = 211


def chan(pl):
    return "YC" if pl.cls == "C" else "YL"


def prim_mask(pl):
    return "MASK-ALL" if masks_of(pl) == ("MASK-ALL", "MASK-COL") else "FULL"


def cells_of(pl, spec):
    out = []
    for Q, m in spec["cells"]:
        mask = prim_mask(pl) if m == "primary" else m
        if not available(pl, Q) or mask not in masks_of(pl):
            continue
        out.append((Q, mask))
    return out


def eta_factory(pl):
    from wave4_2.surfaces import DecoderSurface                       # Lab 07 frozen decoder adapter (truth-free)
    return lambda: DecoderSurface(pl).eta()


def cache(pl, pool, Q, fid, seed, exclude=None, tag_extra=""):
    tag = f"{pool}_{Q}_{fid}_s{seed}{tag_extra}"
    return cached_moments(pl, eta_factory(pl), lambda: design(pl, Q, fid, seed, pool=pool, exclude=exclude), tag, (chan(pl),), masks_of(pl))


def job_path(batch, toy, cid):
    return BLIND / batch / toy.replace("#", "_") / f"{cid}.npz"


def run_cell(L, k, sseed, budget, tie=None):
    t0 = time.time(); r = search(L, k, sseed, budget, identity_start=True, tie_breaker=tie); r["seconds"] = time.time() - t0; return r


def pack(records):
    """records: list of (key, result, extra) -> npz arrays + a finalist trace DataFrame"""
    arr, rows, diag = {}, [], {}
    for key, r, extra in records:
        arr[f"{key}|R"] = r["R"] if r["R"] is not None else np.full((1, 1), np.nan)
        arr[f"{key}|finalQ"] = np.stack([f["Q"] for f in r["finalists"]]) if r["finalists"] else np.zeros((0,))
        arr[f"{key}|start_vals"] = r["start_vals"]
        d = {kk: r.get(kk) for kk in ("f", "status", "best_index", "n_tied", "tie_breaker_used", "near_minima_max_sep", "n_near_minima", "best_converged", "n_evals", "seconds")}
        sv = r["start_vals"][np.isfinite(r["start_vals"])]
        d.update(start_min=float(sv.min()) if sv.size else np.nan, start_p95=float(np.percentile(sv, 95)) if sv.size else np.nan, n_starts=int(len(r["start_vals"])), **extra)
        diag[key] = d
        for i, f in enumerate(r["finalists"]):
            rows.append(dict(key=key, rank=f["rank"], start=f["start"], identity_start=f["identity_start"], L_initial=f["L_initial"], L_final=f["f"],
                             converged=f["converged"], sweeps=f["sweeps"], n_evals=f["n_evals"], final_span=f["final_span"],
                             val_L=extra.get("val_finalists", [np.nan] * len(r["finalists"]))[i], selected=bool(i == r.get("best_index"))))
    for d in diag.values():
        d.pop("val_finalists", None)
    arr["diag"] = np.array(json.dumps(diag, default=float))
    return arr, pd.DataFrame(rows)


def save_job(path, records, meta):
    arr, tr = pack(records); arr["meta"] = np.array(json.dumps(meta, default=float))
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **arr); tmp.replace(path)
    tr.to_csv(path.with_suffix(".trace.csv.gz"), index=False)


def run_job(batch, toy, cid):
    spec = BATCH[batch]; out = job_path(batch, toy, cid)
    if out.is_file():
        print(f"  [{toy}] {batch} {cid}: done"); return
    pl = load_payload(toy); mid = CANDS[cid]["metric"]; c = chan(pl); t0 = time.time(); recs = []
    iseeds = spec["iE"] if cid in ("E01", "E02") else spec["iC"]
    for Q, mask in cells_of(pl, spec):
        for iseed in iseeds:
            mo = cache(pl, spec["pool"], Q, spec["fid"], iseed)[0][(c, mask)]
            mv = cache(pl, "validation", Q, spec["fid"], iseed)[0][(c, mask)] if spec["mode"] == "split" else None
            for variant in VARIANTS:
                sc = variant_frame(pl, variant)[1]; L = Objective(mo, sc, mid); Lv = Objective(mv, sc, mid) if mv is not None else None
                for ss in ALL_SEARCH_SEEDS:
                    r = run_cell(L, pl.k, ss, "S0", tie=Lv)
                    extra = dict(variant=variant, Q=Q, mask=mask, channel=c, iseed=iseed, sseed=ss, budget="S0", cache=f"{spec['pool']}_{Q}_{spec['fid']}_s{iseed}",
                                 val_f=float(Lv(r["R"])) if (Lv is not None and r["R"] is not None) else np.nan,
                                 val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]] if Lv is not None else [np.nan] * len(r["finalists"]))
                    recs.append((f"{variant}|{Q}|{mask}|i{iseed}|s{ss}|S0", r, extra))
                    print(f"  [{toy}] {batch} {cid} {variant} {Q} {mask} i{iseed} s{ss}: L={r['f']:.6g} conv={r.get('best_converged')} "
                          f"evals={r.get('n_evals')} [{time.time()-t0:.0f}s]", flush=True)
    save_job(out, recs, dict(batch=batch, toy=toy, cid=cid, metric=mid, mode=spec["mode"], pool=spec["pool"], code_hash=code_hash(), git=git_commit(),
                             payload_hashes=pl.hashes))


def caches_only(batch, toy):
    spec = BATCH[batch]; pl = load_payload(toy); t0 = time.time()
    for Q, mask in cells_of(pl, spec):
        for iseed in sorted(set(spec["iE"]) | set(spec["iC"])):
            cache(pl, spec["pool"], Q, spec["fid"], iseed)
            if spec["mode"] == "split":
                cache(pl, "validation", Q, spec["fid"], iseed)
    print(f"  [{toy}] {batch} caches [{time.time()-t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- BS-D: independent high-fidelity re-evaluation + escalation
def run_bsd(toy):
    out = job_path("BS-D", toy, "E01E02")
    if out.is_file():
        print(f"  [{toy}] BS-D done"); return
    pl = load_payload(toy); c = chan(pl); mask = prim_mask(pl); t0 = time.time()
    used = np.unique(np.concatenate([used_subjects(design(pl, "Q01", "C", s, pool="train")) for s in (31, 73, 101)]))
    moD = cache(pl, "train", "Q01", "D", D_SEED, exclude=used, tag_extra="_indep")[0][(c, mask)]
    recs, reval = [], []
    for cid in ("E01", "E02"):
        z = np.load(job_path("BS-B", toy, cid), allow_pickle=False); diag = json.loads(str(z["diag"])); mid = CANDS[cid]["metric"]
        for variant in VARIANTS:
            sc = variant_frame(pl, variant)[1]; LD = Objective(moD, sc, mid)
            keys = [k_ for k_ in diag if k_.startswith(variant + "|")]
            sel = {k_: z[f"{k_}|R"] for k_ in keys}
            spread = max([rel_dist(sel[a], sel[b]) for a in keys for b in keys] + [0.0])
            for k_ in keys:
                F = z[f"{k_}|finalQ"]; fD = np.array([LD(Q) for Q in F]); best = diag[k_]["best_index"]
                alt = [i for i in range(len(F)) if rel_dist(F[i], F[best]) > 0.05 and fD[i] < fD[best] - 1e-9 * (abs(fD[best]) + 1e-12)]
                flag = dict(not_converged=not diag[k_]["best_converged"], separated_near_minima=diag[k_]["near_minima_max_sep"] > 0.05,
                            seed_inconsistent=spread > 0.05, D_prefers_other=len(alt) > 0)
                reval.append(dict(toy=toy, cid=cid, metric=mid, key=k_, variant=variant, L_C=diag[k_]["f"], L_D_selected=float(fD[best]), L_D_min_finalists=float(np.nanmin(fD)),
                                  D_best_is_selected=bool(np.nanargmin(fD) == best or rel_dist(F[int(np.nanargmin(fD))], F[best]) <= 0.05),
                                  seed_spread=spread, **flag, escalate=any(flag.values())))
            esc = [r for r in reval if r["cid"] == cid and r["variant"] == variant and r["escalate"]]
            if esc:                                                      # blind-diagnostic escalation (S1 on the cell caches, S0 on D)
                for iseed in sorted({int(r["key"].split("|")[3][1:]) for r in esc}):
                    mo = cache(pl, "train", "Q01", "C", iseed)[0][(c, mask)]; mv = cache(pl, "validation", "Q01", "C", iseed)[0][(c, mask)]
                    L = Objective(mo, sc, mid); Lv = Objective(mv, sc, mid)
                    r = run_cell(L, pl.k, 31, "S1", tie=Lv)
                    recs.append((f"{cid}|{variant}|Q01|{mask}|i{iseed}|s31|S1", r, dict(cid=cid, variant=variant, Q="Q01", mask=mask, iseed=iseed, sseed=31, budget="S1",
                                 cache=f"train_Q01_C_s{iseed}", val_f=float(Lv(r["R"])), val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])))
                    print(f"  [{toy}] BS-D S1 {cid} {variant} i{iseed}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
                r = run_cell(LD, pl.k, 31, "S0")
                recs.append((f"{cid}|{variant}|Q01|{mask}|iD{D_SEED}|s31|S0", r, dict(cid=cid, variant=variant, Q="Q01", mask=mask, iseed=D_SEED, sseed=31, budget="S0",
                             cache=f"train_Q01_D_s{D_SEED}_indep", val_f=np.nan)))
                print(f"  [{toy}] BS-D S0-on-D {cid} {variant}: L={r['f']:.6g} [{time.time()-t0:.0f}s]", flush=True)
    save_job(out, recs, dict(batch="BS-D", toy=toy, D_subjects_excluded=int(len(used)), code_hash=code_hash()))
    pd.DataFrame(reval).to_csv(out.with_suffix(".reeval.csv"), index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--batch"); ap.add_argument("--toy"); ap.add_argument("--metric"); ap.add_argument("--caches", action="store_true")
    ap.add_argument("--evaluate", action="store_true"); ap.add_argument("--jobs", action="store_true")
    a = ap.parse_args(argv)
    if a.evaluate:
        from wave2.blind_eval import evaluate_all
        evaluate_all(); return
    if a.jobs:                                                          # print the job list of a batch (scheduler only)
        spec = BATCH[a.batch]
        for t in spec["panel"]:
            for m in spec["metrics"]:
                print(f"{t} {m}")
        return
    if a.batch == "BS-D":
        run_bsd(a.toy); return
    if a.caches:
        caches_only(a.batch, a.toy); return
    run_job(a.batch, a.toy, a.metric)


if __name__ == "__main__":
    main()
