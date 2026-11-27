"""W3-03 ST-E reduced-axis pilot (03_third_wave section 15-16).

usage: python run_reduced.py --search --toy CR3     blind capped search (budget RA), both variants, search seed 31; GPU float32
       python run_reduced.py --local --toy CR3      EVALUATION-ONLY local sweeps: decoder arm (all toys) and analytic arm (C1/C2/C3 only:
                                                     the isotropic reference leaves the bounded CPT parameter support -> analytic arm NA for CR)
       python run_reduced.py --merge                -> results/reduced_axis_summary.csv.gz
Equivalence: the first --search call per toy records |L_cuda32 - L_cpu64| / |L| at I and at a Haar rotation (reduced_axis_equivalence.csv).
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT, RES, PRIMARY, chan, prim_mask, code_hash
import numpy as np, pandas as pd, torch
from wave3 import reduced_axis as RA, jobs as JB
from wave2.blind_inputs import load_payload, variant_frame
from wave2.optimizers import haar
import run_blind as RB8

RAD = OUT / "reduced_axis"


def blind_loss(pl, variant, device="cuda", dtype=torch.float32):
    c, s = variant_frame(pl, variant); Xs = RA.xrows(pl, "train", 31)
    return RA.DeletionLoss(RA.decoder_gfun(pl, c, s, device, dtype), pl.k, Xs, 31, device, dtype)


def search(toy):
    out = JB.job_path("ST-E", toy, "ST-E-RA")
    if out.is_file():
        print(f"  [{toy}] ST-E done"); return
    pl = load_payload(toy); recs = []; t0 = time.time(); eq = []
    for v in ("s9", "s9_2"):
        L = blind_loss(pl, v)
        L64 = blind_loss(pl, v, "cpu", torch.float64); Q = haar(pl.k, 1, np.random.default_rng(4))[0]
        for name, R in (("I", np.eye(pl.k)), ("haar", Q)):
            a, b = L(R), L64(R); eq.append(dict(toy=toy, variant=v, R=name, L_cuda32=a, L_cpu64=b, rel=abs(a - b) / abs(b)))
        r = RA.capped_search(L, pl.k, 31); r["seconds"] = time.time() - t0
        recs.append((f"{v}|RA|{prim_mask(pl)}|i31|s31|RA", r, dict(variant=v, Q="ISO", mask=prim_mask(pl), channel=chan(pl), iseed=31, sseed=31, budget="RA", cid="ST-E-RA",
                     cache="isotropic reference 64x16x64 (seed 31)", val_f=np.nan, val_finalists=[np.nan] * len(r["finalists"]))))
        print(f"  [{toy}] ST-E {v}: L={r['f']:.6g} evals={r['n_evals']} [{time.time()-t0:.0f}s]", flush=True)
    RB8.save_job(out, recs, dict(batch="ST-E", toy=toy, cid="ST-E-RA", budget=RA.RA_BUDGET, draws=[RA.NB, RA.NR, RA.NX], code_hash=code_hash()))
    RAD.mkdir(parents=True, exist_ok=True); pd.DataFrame(eq).to_csv(RAD / f"{toy}_equivalence.csv", index=False)


def local(toy):
    from wave4_2.inputs import load
    from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1
    from wave4_2.truth import oracle_eta
    import run_baseline as RB7
    out = RAD / f"{toy}_local.csv"
    if out.is_file():
        return
    pl = load_payload(toy); b, ev, rots, _ = load(toy); k = pl.k; n_exp = 2 ** k - k - 1; rows = []; Xs = RA.xrows(pl, "test", 31)
    for v in rots:
        man = RB7.manifest_rows(rots[v]); sw = sweeps(man); r0 = rots[v][0]
        Li = RA.DeletionLoss(RA.decoder_gfun(pl, r0.z_mean, r0.z_std), k, Xs, 31)
        li = np.array([[Li(r.R)] for r in rots[v]])
        if ev.family in ("C1", "C2", "C3"):
            f = oracle_eta(ev.family, pl.x_mean, pl.x_std); fv = torch.vmap(f)
            tm = torch.as_tensor(ev.refs[v]["t_mean"], dtype=torch.float64); ts = torch.as_tensor(r0.t_std, dtype=torch.float64)
            g = lambda V, X: fv(torch.cat((tm + ts * V, X), 1))                     # analytic arm: CPU float64 (the formula code is CPU-only)
            Lo = RA.DeletionLoss(g, k, Xs, 31, device="cpu", dtype=torch.float64); lo = np.array([[Lo(r.G)] for r in rots[v]])
        else:
            lo = np.full_like(li, np.nan)
        fo, fi = library_flags(curve_stats(lo, sw), n_exp), library_flags(curve_stats(li, sw), n_exp); e1 = eval1(lo, li, man, sw)
        rows.append(dict(toy=toy, variant=v, cid="ST-E-RA", analytic_arm="NA (bounded prior support)" if ev.family.startswith("CR") else "ok",
                         O_perfect=bool(fo["perfect"][0]), O_marginal=bool(fo["marginal"][0]), I_perfect=bool(fi["perfect"][0]), I_marginal=bool(fi["marginal"][0]),
                         pearson=float(e1["pearson"][0]), I_min_margin=float(fi["min_margin"][0])))
    pd.DataFrame(rows).to_csv(out, index=False); print(f"  [{toy}] ST-E local", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--search", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.search:
        search(a.toy)
    if a.local:
        local(a.toy)
    if a.merge:
        loc = pd.concat([pd.read_csv(f) for f in sorted(RAD.glob("*_local.csv"))]); eq = pd.concat([pd.read_csv(f) for f in sorted(RAD.glob("*_equivalence.csv"))])
        loc.to_csv(RES / "reduced_axis_summary.csv.gz", index=False); eq.to_csv(RES / "reduced_axis_equivalence.csv", index=False)
