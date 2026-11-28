"""Truth-free W4 search jobs.  One job file per (batch, toy, candidate, measure); keys  <variant>|<measure>@<fid>|<mask>|i<iseed>|s<sseed>|<budget>.
Objective = the Lab 09 frozen builder (wave3.objectives.build) on a Lab 10 moment cache; optimizer = Lab 08 run_cell (S0 / S1) with the
VALIDATION cache of the same measure / fidelity / seed as numerical tie-break.  Files written with Lab 08 run_blind.save_job (same layout).
"""
import time
import numpy as np
from .common import OUT, cfg, chan, prim_mask, code_hash, git_commit, spec_of, SSEEDS
from . import moments as MO, designs as DS
from wave3.objectives import build
from wave2.blind_inputs import load_payload, variant_frame
import run_blind as RB8

BLIND = OUT / "blind"
BANK_TOYS = ("CR3", "CR5", "CR3L", "CR5L")


def job_path(batch, toy, cid, M, tag=""):
    return BLIND / batch / toy.replace("#", "_") / f"{cid.replace('@', '_at_')}__{M}{tag}.npz"


def mo_for(pl, M, fid, iseed, pool, mask):
    """moment dict of one (measure, fidelity, seed, pool, mask); MIX-a is built from the M-E and M-D caches"""
    c = chan(pl)
    if M.startswith("MIX-"):
        a = float(M[4:]); E = MO.get(pl, "M-E", fid, iseed, pool)[0][(c, mask)]; D = MO.get(pl, "M-D", fid, iseed, pool)[0][(c, mask)]
        return MO.mixture(E, D, a), MO.tag_of(pool, M, fid, iseed)
    Mx = M if pool == "train" else DS.val_recipe(M)
    return MO.get(pl, Mx, fid, iseed, pool)[0][(c, mask)], MO.tag_of(pool, Mx, fid, iseed)


def objective(pl, cid, M, fid, iseed, mask, variant, pool="train"):
    mo, tag = mo_for(pl, M, fid, iseed, pool, mask)
    return build(spec_of(cid), mo, variant_frame(pl, variant)[1]), tag


def run_job(batch, toy, cid, M, fid, iseeds, variants, masks=None, budget="S0", sseeds=SSEEDS, pl=None):
    out = job_path(batch, toy, cid, M, "" if masks is None else "_" + "+".join(masks))
    if out.is_file():
        print(f"  [{toy}] {batch} {cid} {M}: done", flush=True); return out
    pl = pl or load_payload(toy); masks = masks or [prim_mask(pl)]; t0 = time.time(); recs = []
    for mask in masks:
        for iseed in iseeds:
            for variant in variants:
                L, tag = objective(pl, cid, M, fid, iseed, mask, variant); Lv, vtag = objective(pl, cid, M, fid, iseed, mask, variant, "validation")
                for ss in sseeds:
                    r = RB8.run_cell(L, pl.k, ss, budget, tie=Lv)
                    extra = dict(batch=batch, cid=cid, variant=variant, Q=M, fidelity=fid, mask=mask, channel=chan(pl), iseed=iseed, sseed=ss, budget=budget,
                                 cache=tag, val_cache=vtag, val_f=float(Lv(r["R"])) if r["R"] is not None else np.nan,
                                 val_finalists=[float(Lv(f["Q"])) for f in r["finalists"]])
                    recs.append((f"{variant}|{M}@{fid}|{mask}|i{iseed}|s{ss}|{budget}", r, extra))
                    print(f"  [{toy}] {batch} {cid} {M}@{fid} {mask} {variant} i{iseed} s{ss}: L={r['f']:.6g} conv={r.get('best_converged')} [{time.time()-t0:.0f}s]", flush=True)
    RB8.save_job(out, recs, dict(batch=batch, toy=toy, cid=cid, measure=M, fidelity=fid, spec=spec_of(cid), code_hash=code_hash(), git=git_commit(), payload_hashes=pl.hashes))
    return out


def batch_jobs(batch):
    """-> list of (toy, cid, M, fid, iseeds, variants, masks)"""
    from .common import PRIMARY, SECONDARY
    b = cfg("replication_protocol.json")["batches"][batch]
    panel = dict(primary=PRIMARY, secondary=SECONDARY, bank=BANK_TOYS)[b["panel"]]; out = []
    pairs = [(c, M) for c in b["cids"] for M in b["measures"]] + [(e["cid"], e["measure"]) for e in b.get("extra", [])]
    for toy in panel:
        for c, M in pairs:
            if M == "Q04" and toy not in BANK_TOYS + ("CR3#b5e-5",):
                continue                                         # Q04 = joint Gaussian x full bank: fixed-bank designs only
            out.append((toy, c, M, b["fidelity"], tuple(b["iseeds"]), tuple(b["variants"]), tuple(b["masks"]) if "masks" in b else None))
    return out
