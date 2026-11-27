"""Blind job plumbing shared by every W3 search batch (truth-free).

Candidate ids map to (objective spec, surface).  Surfaces:
  AD    frozen decoder autograd Hessian moments: Lab 08 split-safe caches train/validation_Q01_C_s<seed> (read-only reuse)
  RBF   E-RBF block-scaled D1 teacher surrogate (S-min), Lab 08 caches train/validation_Q01_C_s31_rbfD1 (seed 31 only)
  GRAD  wave3.structure gradient / contrast caches (TRAIN product design, VALIDATION tie-break)
Missing HE caches are computed into the Lab 09 output tree (Lab 08 outputs are never written).
Job files: OUT/blind/<batch>/<toy>/<cid>.npz (+ .trace.csv.gz), written with Lab 08 run_blind.save_job (same layout).
"""
import json, time
import numpy as np
from .common import OUT, OUT08, cfg, chan, prim_mask, code_hash, git_commit
from . import rowmetrics as RM
from .objectives import build
from wave2.blind_inputs import load_payload, variant_frame
from wave2.designs import design
from wave2.objectives import compute_moments, save_moments, load_moments, masks_of
from wave2.optimizers import search
import run_blind as RB8                                    # Lab 08 runner (pack / save_job / eta_factory), imported read-only

BLIND = OUT / "blind"
CACHE9 = OUT / "moment_cache"
CACHE8 = OUT08 / "moment_cache"


def he_cache(pl, pool, Q, fid, seed, exclude=None, tag_extra=""):
    tag = f"{pool}_{Q}_{fid}_s{seed}{tag_extra}"; name = pl.toy.replace("#", "_")
    for root in (CACHE8, CACHE9):
        f = root / name / f"{tag}.npz"
        if f.is_file():
            return load_moments(f)[0], tag
    d = design(pl, Q, fid, seed, pool=pool, exclude=exclude)
    res = compute_moments(pl, RB8.eta_factory(pl)(), d, (chan(pl),), masks_of(pl))
    save_moments(CACHE9 / name / f"{tag}.npz", res, dict(design=d["meta"], n_points=int(len(d["x"]))))
    return res, tag


def rbf_cache(pl, pool, seed=31):
    """X02 surface: the saved E-RBF block D1 state (fitted once in W2, TRAIN subjects, S-min) differentiated on the Q01 C design of `seed`"""
    tag = f"{pool}_Q01_C_s{seed}_rbfD1"; name = pl.toy.replace("#", "_")
    for root in (CACHE8, CACHE9):
        f = root / name / f"{tag}.npz"
        if f.is_file():
            return load_moments(f)[0], tag
    from wave2 import surrogates as S
    import run_inference as RI
    st = OUT08 / "inference" / "estimator_fits" / f"{name}_RBF-block_D1_Smin.npz"; z = np.load(st)
    est = S.RBFEta(z["m"], z["s"], z["C"], float(z["h"]), z["a"], float(z["b"])); d = design(pl, "Q01", "C", seed, pool=pool)
    res = {(chan(pl), m): v for m, v in RI.moments_from_H(lambda P: est.hessian(P), pl, d, masks_of(pl)).items()}
    save_moments(CACHE9 / name / f"{tag}.npz", res, dict(design=d["meta"], surface="E-RBF block D1 S-min (W2 state)")); return res, tag


_REG = None


def registry():
    """cid -> dict(spec, surface).  Built from the frozen configs (never from results)."""
    global _REG
    if _REG is not None:
        return _REG
    reg = {}
    rc = cfg("repair_candidates.json")
    for cid, spec in RM.registry(rc).items():
        reg[cid] = dict(spec=dict(spec, type="row"), surface="AD", family="RW")
    for c in rc["controls"]:
        reg[c["id"]] = dict(spec=dict(type="registry", metric=c["metric"]), surface="AD", family="CTRL")
    mx = cfg("fusion_candidates.json")
    for f in mx["factorial"]:
        for surf in mx["surfaces"]:
            cid = f"MX-{f['id']}@{surf}"
            reg[cid] = dict(spec=dict(f["spec"]), surface=surf, family="MX-A")
    reg["X02"] = dict(spec=dict(type="registry", metric="L03|SL0.UXUU.e0.01"), surface="RBF", family="CTRL")
    reg["X01"] = dict(spec=dict(type="registry", metric="L03|SL0.UXUU.e0.01"), surface="AD", family="CTRL")
    try:
        st = cfg("structure_candidates.json")
        for c in st["candidates"]:
            reg[c["id"]] = dict(spec=dict(c["spec"], type="struct"), surface=c.get("surface", "GRAD"), family=c["family"])
    except FileNotFoundError:
        pass
    _REG = reg
    return reg


def surface_caches(pl, surface, iseed, mask, fid="C"):
    """-> (train moment dict, validation moment dict, cache tag) for one surface / integration seed / mask"""
    c = chan(pl)
    if surface == "AD":
        mo, tag = he_cache(pl, "train", "Q01", fid, iseed); mv, _ = he_cache(pl, "validation", "Q01", fid, iseed)
        return mo[(c, mask)], mv[(c, mask)], tag
    if surface == "RBF":
        mo, tag = rbf_cache(pl, "train", iseed); mv, _ = rbf_cache(pl, "validation", iseed)
        return mo[(c, mask)], mv[(c, mask)], tag
    if surface == "GRAD":
        from .structure import grad_cache
        return grad_cache(pl, "train", iseed, mask), grad_cache(pl, "validation", iseed, mask), f"grad_train_s{iseed}"
    raise ValueError(surface)


def job_path(batch, toy, cid):
    return BLIND / batch / toy.replace("#", "_") / f"{cid.replace('|', '_').replace('@', '_at_')}.npz"


def run_search_job(batch, toy, cid, iseeds=(31,), sseeds=(31, 73, 101), variants=("s9", "s9_2"), budget="S0", mask=None, objective_factory=None, meta_extra=None):
    """generic blind search job.  objective_factory(pl, variant, iseed, mask) -> (L, Lv, cache_tag) overrides the registry lookup."""
    out = job_path(batch, toy, cid)
    if out.is_file():
        print(f"  [{toy}] {batch} {cid}: done", flush=True); return out
    pl = load_payload(toy); mask = mask or prim_mask(pl); t0 = time.time(); recs = []
    entry = registry().get(cid)
    for iseed in iseeds:
        if objective_factory is None:
            mo, mv, tag = surface_caches(pl, entry["surface"], iseed, mask)
        for variant in variants:
            if objective_factory is None:
                sc = variant_frame(pl, variant)[1]; L = build(entry["spec"], mo, sc); Lv = build(entry["spec"], mv, sc)
            else:
                L, Lv, tag = objective_factory(pl, variant, iseed, mask)
            for ss in sseeds:
                r = RB8.run_cell(L, pl.k, ss, budget, tie=Lv)
                extra = dict(variant=variant, Q="Q01", mask=mask, channel=chan(pl), iseed=iseed, sseed=ss, budget=budget, cache=tag, cid=cid,
                             val_f=float(Lv(r["R"])) if (Lv is not None and r["R"] is not None) else np.nan,
                             val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]] if Lv is not None else [np.nan] * len(r["finalists"]))
                recs.append((f"{variant}|Q01|{mask}|i{iseed}|s{ss}|{budget}", r, extra))
                print(f"  [{toy}] {batch} {cid} {variant} i{iseed} s{ss}: L={r['f']:.6g} conv={r.get('best_converged')} [{time.time()-t0:.0f}s]", flush=True)
    meta = dict(batch=batch, toy=toy, cid=cid, spec=None if entry is None else entry, code_hash=code_hash(), git=git_commit(), payload_hashes=pl.hashes)
    meta.update(meta_extra or {})
    RB8.save_job(out, recs, meta)
    return out
