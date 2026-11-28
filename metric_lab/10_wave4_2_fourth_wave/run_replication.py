"""W4-01 / W4-02 / W4-04 blind searches (truth-free).  Batches and cells: configs/replication_protocol.json.

usage
  python run_replication.py --caches --toy CR4                  every moment cache the registered batches need (train + validation), once
  python run_replication.py --jobs REP-1                         "batch toy cid measure" lines (k=6 toys first) for the scheduler
  python run_replication.py --batch REP-1 --toy CR4 --cid RW-GL-l0.75-a1.0 --Q Q05
  python run_replication.py --reeval --toy CR4                   REP-4: REP-1 / REP-2 finalists on the REP-3 D caches + blind S1 escalation
"""
import sys, argparse, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave4.common import cfg, chan, prim_mask, PRIMARY, SECONDARY
from wave4 import search as SE, moments as MO, designs as DS
import numpy as np

ORDER = ("REP-1", "REP-2", "REP-2C", "REP-3", "MG", "MS", "MIX", "RB-Q", "RB-MASK", "RB-SEC")


def needed_caches(toy):
    out = []
    for b in ORDER:
        for (t, cid, M, fid, iseeds, variants, masks) in SE.batch_jobs(b):
            if t != toy:
                continue
            for s in iseeds:
                Ms = ["M-E", "M-D"] if M.startswith("MIX-") else [M]
                for m in Ms:
                    for pool, mm in (("train", m), ("validation", DS.val_recipe(m))):
                        if (mm, fid, s, pool) not in out:
                            out.append((mm, fid, s, pool))
    return out


def caches(toy):
    from wave2.blind_inputs import load_payload
    pl = load_payload(toy); t0 = time.time(); need = needed_caches(toy); need.sort(key=lambda x: x[1] == "D")
    for M, fid, s, pool in need:
        f = MO.path_of(pl, MO.tag_of(pool, M, fid, s))
        if f.is_file():
            continue
        if not DS.available(pl, M):
            continue
        MO.get(pl, M, fid, s, pool); print(f"  [{toy}] cache {pool} {M} {fid} s{s} [{time.time()-t0:.0f}s]", flush=True)
    print(f"  [{toy}] caches done ({len(need)}) [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--caches", action="store_true"); ap.add_argument("--jobs"); ap.add_argument("--batch"); ap.add_argument("--toy")
    ap.add_argument("--cid"); ap.add_argument("--Q"); ap.add_argument("--reeval", action="store_true"); a = ap.parse_args()
    if a.caches:
        caches(a.toy)
    elif a.jobs:
        js = [(a.jobs,) + j for j in SE.batch_jobs(a.jobs)]
        js.sort(key=lambda j: 0 if j[1].startswith("C3") else 1)
        for j in js:
            print(j[0], j[1], j[2], j[3])
    elif a.reeval:
        import wave4.reeval as RV
        RV.reeval(a.toy)
    else:
        j = next(j for j in SE.batch_jobs(a.batch) if j[0] == a.toy and j[1] == a.cid and j[2] == a.Q)
        SE.run_job(a.batch, *j)
