"""W3 blind searches (truth-free).  Every batch: TRAIN-pool objective, VALIDATION tie-break, S0 search (Lab 08 optimizer), both variants.

usage
  python run_search.py --batch RW-C --jobs                 # print "toy cid" job lines (scheduler)
  python run_search.py --batch RW-C --toy CR3 --cid RW-POW-a0.5-c0.03
  python run_search.py --batch MX-SEL --toy CR3            # blind cross-objective selector (after MX-A / MX-B jobs of the toy)
Batches
  RW-C   W3-01: 39 registered repair recipes + controls E02 / E01 / B07c03 / B01c03; AD surface; integration seed 31; search seeds 31/73/101
  MX-A   W3-02: factorial {B08c03, SL0UXUU, SL0UX, B01c03} x {AD, RBF}; seed 31; search seeds 31/73/101
  MX-B   W3-02: same-frame fusions of E = B08c03@AD and X = SL0UXUU@RBF (Haar-512 median/IQR calibration on TRAIN)
  MX-SEL W3-02 section 9.2 selector over the E, X and F_w finalists (VALIDATION z-scores, TRAIN calibration)
  ST-B   W3-03: 52 gradient / contrast / control recipes on the TRAIN product-grid gradient cache; seed 31; search seeds 31/73/101
  FIN    W3-05: advanced candidates, integration seeds 31/73/101 x search seeds 31/73/101 (configs/finalists.json, frozen after pilots)
The blind part imports no truth module.
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave3.common import OUT, PRIMARY, SECONDARY, VARIANTS, cfg, chan, prim_mask, code_hash, git_commit
import numpy as np
from wave3 import jobs as JB, fusion as FU
from wave3.objectives import build
from wave2.blind_inputs import load_payload, variant_frame
import run_blind as RB8

FC = cfg("fusion_candidates.json")
FUSIONS = {f"MX-F-w{w}": dict(kind="w", w=w) for w in FC["fusion"]["weights"]}
FUSIONS["MX-F-max"] = dict(kind="max")
FUSIONS.update({f"MX-F-eps{d}": dict(kind="eps", delta=d) for d in FC["fusion"]["eps_delta"]})
FUSIONS.update({f"MX-F-w0.5-{n}": dict(kind="w", w=0.5, norm=n) for n in FC["fusion"]["calibration"]["sensitivity"]})


def batch_cids(batch):
    reg = JB.registry()
    if batch == "RW-C":
        return [c for c, e in reg.items() if e["family"] in ("RW", "CTRL") and c not in ("X01", "X02")]
    if batch == "MX-A":
        return [c for c, e in reg.items() if e["family"] == "MX-A"]
    if batch == "MX-B":
        return list(FUSIONS)
    if batch == "ST-B":
        return [c for c, e in reg.items() if c.startswith("ST-")]
    if batch == "FIN":
        return [c["cid"] for c in cfg("finalists.json")["finalists"]]
    raise ValueError(batch)


# ---------------------------------------------------------------- fusion objectives
def components(pl, variant, iseed, mask, E="MX-B08c03@AD", X="MX-SL0UXUU@RBF"):
    reg = JB.registry(); sc = variant_frame(pl, variant)[1]; out = {}
    for name, cid in (("E", E), ("X", X)):
        e = reg[cid]; mo, mv, tag = JB.surface_caches(pl, e["surface"], iseed, mask)
        out[name] = (build(e["spec"], mo, sc), build(e["spec"], mv, sc), tag)
    return out


def zE_minimum(pl, variant, iseed, mask, cal, E="MX-B08c03@AD"):
    f = JB.job_path("MX-A", pl.toy, E) if iseed == 31 else JB.job_path("FIN", pl.toy, E)
    z = np.load(f, allow_pickle=False); diag = json.loads(str(z["diag"]))
    fs = [d["f"] for k, d in diag.items() if d["variant"] == variant and d["iseed"] == iseed and d["mask"] == mask]
    return (min(fs) - cal["center"]) / cal["scale"], len(fs)


def fusion_factory(cid):
    fz = FUSIONS[cid]

    def make(pl, variant, iseed, mask):
        comp = components(pl, variant, iseed, mask); norm = fz.get("norm", "iqr")
        calE = FU.calibrate(comp["E"][0], pl.k, norm); calX = FU.calibrate(comp["X"][0], pl.k, norm)
        zmin = None
        if fz["kind"] == "eps":
            zmin, _ = zE_minimum(pl, variant, iseed, mask, calE)
        L = FU.Fused(FU.Z(comp["E"][0], calE), FU.Z(comp["X"][0], calX), fz["kind"], fz.get("w"), fz.get("delta"), zmin)
        Lv = FU.Fused(FU.Z(comp["E"][1], calE), FU.Z(comp["X"][1], calX), fz["kind"], fz.get("w"), fz.get("delta"), zmin)
        L.cal = dict(E=calE, X=calX, zE_min=zmin); Lv.cal = L.cal
        return L, Lv, comp["E"][2] + "+" + comp["X"][2]
    return make


def run_selector(toy, out_batch="MX-SEL", src_batches=("MX-A", "MX-B"), iseeds=(31,)):
    """blind section 9.2 selector: finalists of E, X and F_w searches of the same variant / seed; VALIDATION z-scores (TRAIN calibration)"""
    out = JB.job_path(out_batch, toy, "MX-SEL")
    if out.is_file():
        print(f"  [{toy}] {out_batch} MX-SEL done"); return
    pl = load_payload(toy); mask = prim_mask(pl); recs = []; rows = []
    srcs = [(src_batches[0], "MX-B08c03@AD"), (src_batches[0], "MX-SL0UXUU@RBF")] + [(src_batches[1], f"MX-F-w{w}") for w in FC["fusion"]["weights"]]
    for variant, iseed in [(v, i) for v in VARIANTS for i in iseeds]:
        comp = components(pl, variant, iseed, mask); calE = FU.calibrate(comp["E"][0], pl.k); calX = FU.calibrate(comp["X"][0], pl.k)
        zEv, zXv = FU.Z(comp["E"][1], calE), FU.Z(comp["X"][1], calX); zEt, zXt = FU.Z(comp["E"][0], calE), FU.Z(comp["X"][0], calX)
        for ss in (31, 73, 101):
            cands = []
            for b, cid in srcs:
                z = np.load(JB.job_path(b, toy, cid), allow_pickle=False); key = f"{variant}|Q01|{mask}|i{iseed}|s{ss}|S0"
                for i, Q in enumerate(z[f"{key}|finalQ"]):
                    cands.append(dict(key=f"{cid}#{i:03d}", Q=Q, zE_val=zEv(Q), zX_val=zXv(Q), zE_tr=zEt(Q), zX_tr=zXt(Q)))
            sel = FU.select(cands); best = next(c for c in cands if c["key"] == sel)
            near = [c for c in cands if np.isfinite(c["zE_val"]) and max(c["zE_val"], c["zX_val"]) <= max(best["zE_val"], best["zX_val"]) + 1e-3]
            # disagreement flag (reported only): distance between the E and X component minimisers of this seed
            zE = np.load(JB.job_path(src_batches[0], toy, "MX-B08c03@AD"))[f"{variant}|Q01|{mask}|i{iseed}|s{ss}|S0|R"]
            zX = np.load(JB.job_path(src_batches[0], toy, "MX-SL0UXUU@RBF"))[f"{variant}|Q01|{mask}|i{iseed}|s{ss}|S0|R"]
            from wave2.optimizers import rel_dist
            r = dict(R=best["Q"], f=max(best["zE_val"], best["zX_val"]), finalists=[dict(Q=c["Q"], f=max(c["zE_val"], c["zX_val"]), rank=i, start=-1, identity_start=False,
                     L_initial=np.nan, converged=True, sweeps=0, n_evals=0, final_span=np.nan) for i, c in enumerate(cands)], start_vals=np.array([max(c["zE_tr"], c["zX_tr"]) for c in cands]),
                     status="ok", best_index=[c["key"] for c in cands].index(sel), n_tied=1, tie_breaker_used=False, near_minima_max_sep=0.0, n_near_minima=len(near),
                     best_converged=True, n_evals=len(cands) * 2, seconds=0.0)
            recs.append((f"{variant}|Q01|{mask}|i{iseed}|s{ss}|SEL", r, dict(variant=variant, Q="Q01", mask=mask, channel=chan(pl), iseed=iseed, sseed=ss, budget="SEL",
                         cache="validation selector", cid="MX-SEL", selected=sel, val_f=r["f"], val_finalists=[max(c["zE_val"], c["zX_val"]) for c in cands],
                         disagreement=float(rel_dist(zE, zX)), abstain_flag=bool(rel_dist(zE, zX) > 0.10))))
            print(f"  [{toy}] MX-SEL {variant} s{ss}: {sel} max-z={r['f']:.4g} EX-dist={rel_dist(zE, zX):.3f}", flush=True)
    RB8.save_job(out, recs, dict(batch=out_batch, toy=toy, cid="MX-SEL", rule=FC["selector"], code_hash=code_hash()))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--batch", required=True); ap.add_argument("--toy"); ap.add_argument("--cid"); ap.add_argument("--jobs", action="store_true")
    ap.add_argument("--panel", default="primary")
    a = ap.parse_args(argv)
    panel = PRIMARY if a.panel == "primary" else SECONDARY
    if a.jobs:
        if a.batch == "MX-SEL":
            for t in panel:
                print(f"{t} MX-SEL")
            return
        for t in panel:
            for c in batch_cids(a.batch):
                print(f"{t} {c}")
        return
    if a.batch == "MX-SEL":
        run_selector(a.toy); return
    if a.batch == "MX-B":
        JB.run_search_job("MX-B", a.toy, a.cid, objective_factory=fusion_factory(a.cid), meta_extra=dict(fusion=FUSIONS[a.cid], components=FC["fusion"])); return
    if a.batch == "FIN":
        fin = {c["cid"]: c for c in cfg("finalists.json")["finalists"]}[a.cid]
        fac = fusion_factory(fin["base"]) if fin["base"] in FUSIONS else None
        if fin["base"] == "MX-SEL":
            run_selector(a.toy, out_batch="FIN", src_batches=("FIN", "FIN"), iseeds=(31, 73, 101)); return
        JB.run_search_job("FIN", a.toy, fin["base"], iseeds=tuple(fin.get("iseeds", (31, 73, 101))), sseeds=tuple(fin.get("sseeds", (31, 73, 101))),
                          objective_factory=fac, meta_extra=dict(finalist=fin)); return
    JB.run_search_job(a.batch, a.toy, a.cid)


if __name__ == "__main__":
    main()
