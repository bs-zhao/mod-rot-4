"""W2-03 conditional Hessian energy: CH-A (analytic arm), CH-B (frozen decoder, all 24 variants), CH-C (P2 / P5 / N0 / N2 / K / weights).

usage: python run_conditional.py --toy CR3 [--fid pilot|medium] [--seed 31]      one toy, both arms, every configuration, both variants
       python run_conditional.py --analyze                                         -> results/conditional_pass_summary.csv.gz, tables
Product design (02_second_wave 14.2): TRAIN subjects; random designs 128 x 128 (pilot) / 256 x 256 (medium) sampled complete X rows x
latents; fixed banks the full 500-row bank x 32 / 128 latents.  The blind arm uses the subjects' Z (payload only), the analytic arm their
paired true T (evaluation only).  X grouping uses observed X rows only.  Evaluated on the saved S9 / S9_2 rotation libraries with the
04.2 sweep flags (perfect / marginal / strict, margins) and evaluation 1 correlations; P0 is computed on the identical points.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, RES, PRIMARY, VARIANTS, cfg, code_hash, family_group
import numpy as np, pandas as pd
from scipy.stats import special_ortho_group
from wave2.blind_inputs import load_payload
from wave2.conditional import (product_design, product_moments, kmeans_groups, group_moments, contract_batch, Scorer, soft_fixed_scale, SOFT, NORM_ALT)
from wave2.objectives import masks_of
from wave4_2.moments import jac
from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1
from wave4_2 import metrics as MT

COND = OUT / "conditional"
PANEL = ["L06|B01.M03.c0.01", "L06|B01.M03.c0.03", "L06|B01.M04.c0.01", "L06|B08.M03.c0.01", "L06|B08.M03.c0.03", "L06|B08.M05",
         "L06|B09.w75.M03.c0.01", "L06|B09.w50.M03.c0.03", "L06|B10.2blk.M05", "L03|UX.lq.1", "L03|UXcol.a0.lq.25", "L03|POOL.lq.1",
         "L03|SL0.UXUU.e0.01", "L03|rowsp.a0.5.q0.1", "L03|Fz.A+B+C", "L06|CTRL.EB", "L06|CTRL.EA"]
SIZES = {"pilot": ((128, 128), 32), "medium": ((256, 256), 128)}
KS = (4, 8, 16)


def configs():
    out = ["P0", "P1", "P2"] + [f"P4K{K}{w}" for K in KS for w in ("", "eq")]
    return out


def chan(pl):
    return "YC" if pl.cls == "C" else "YL"


class Landscape:
    """all aggregated losses of one arm / mask / scale for one rotation library"""
    def __init__(self, pm, mode, groups, mids):
        self.pm, self.mode, self.mids = pm, mode, mids
        K_r, J_r, cm = pm["K_r"], pm["J_r"], pm["cm"]; nR, p = J_r.shape[0], J_r.shape[1]
        if mode == "MASK-ALL":
            self.rows = np.where(cm.all(1))[0]; self.rcols = None
        elif mode == "MASK-COL":
            self.rows = np.where(cm.any(1))[0]; self.rcols = cm
        else:
            self.rows = np.arange(nR); self.rcols = None
        self.omitted_rows = int(nR - len(self.rows))
        whole = group_moments(pm, np.zeros(nR, int), mode)[0]
        self.pool = (whole[0][None], whole[1][None])
        self.groups, self.gcols = {}, {}
        for K, lab in groups.items():
            gm = group_moments(pm, lab, mode)
            self.groups[K] = (np.stack([g[0] for g in gm]), np.stack([g[1] for g in gm]), np.array([g[2] for g in gm], float))
            self.gcols[K] = np.stack([g[3] > 0 for g in gm]) if mode == "MASK-COL" else None
        Kl, Jl, ruu, rux = pm["per_lat"][mode]; self.lat = (Kl, Jl)

    def rc(self, cb):
        """column validity per block for the MASK-COL valid-subspace estimand (None = every column valid)"""
        if self.mode != "MASK-COL":
            return None
        if cb == "P1":
            return self.rcols[self.rows]
        if cb.startswith("P4K"):
            return self.gcols[int(cb[3:])]
        return None

    def blocks(self, Jt):
        Kr, Jr = self.pm["K_r"][self.rows], self.pm["J_r"][self.rows]
        out = dict(P0=contract_batch(*self.pool, Jt), P1=contract_batch(Kr, Jr, Jt), P2=contract_batch(*self.lat, Jt))
        for K, (Kg, Jg, w) in self.groups.items():
            out[f"P4K{K}"] = contract_batch(Kg, np.nan_to_num(Jg), Jt)
        return out


def zero_of(A, B):
    return [MT.zero_flags(a, b)[0] for a, b in zip(A, B)]


def row_values(sc, A, B, zero, rcols=None):
    vals = []
    for i in range(len(A)):
        Bi = B[i] if rcols is None else B[i][:, rcols[i]]
        vals.append(sc.value(A[i], Bi, zero[i]))
    return vals


def aggregate(vals, w):
    v = np.array(vals, float); w = np.asarray(w, float)
    if v.ndim == 2:                                      # Fz components: weighted mean of each component
        ok = np.isfinite(v).all(1)
        return (v[ok] * w[ok, None]).sum(0) / w[ok].sum() if ok.any() else np.full(v.shape[1], np.nan), float(w[~ok].sum() / w.sum())
    ok = np.isfinite(v)
    return (float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan), float(w[~ok].sum() / w.sum())


def evaluate_library(land, rots, scale_of, mids, Jt_of):
    """-> L (n_rot, n_cfg_total), names, omitted mass.  Zero flags per group decided once at the first rotation (invariant energies)."""
    names, L, omit = None, [], []
    base = land.blocks(Jt_of(rots[0])); zeros = {c: zero_of(*base[c]) for c in base}
    wts = {"P0": np.ones(1), "P1": np.ones(len(land.rows)),
           "P2": np.ones(len(land.lat[0]))}
    for K, (_, _, w) in land.groups.items():
        wts[f"P4K{K}"] = w
    scorers = {m: Scorer(m) for m in mids}
    # Haar null of the aggregated Fz components (same configuration, no reference): 200 rotations, random_state 4
    null = {}
    fz = [m for m in mids if scorers[m].fz]
    if fz:
        hq = special_ortho_group.rvs(land.pm["K_r"].shape[1], size=200, random_state=4)
        acc = {(m, c): [] for m in fz for c in configs()}
        for Q in hq:
            bl = land.blocks((scale_of[:, None] * Q.T).T)
            for c in configs():
                cb = c[:-2] if c.endswith("eq") else c
                A, B = bl[cb]; w = wts[cb] if not c.endswith("eq") else np.ones(len(wts[cb]))
                for m in fz:
                    acc[(m, c)].append(aggregate(row_values(scorers[m], A, B, zeros[cb], land.rc(cb)), w)[0])
        null = {k_: (np.median(np.array(v), 0), np.std(np.array(v), 0) + 1e-12) for k_, v in acc.items()}
    E0 = {}
    for r_i, r in enumerate(rots):
        bl = land.blocks(Jt_of(r)); vals, nm, om = [], [], []
        for m in mids:
            sc = scorers[m]; per = {}
            for c in configs():
                cb = c[:-2] if c.endswith("eq") else c
                A, B = bl[cb]; w = wts[cb] if not c.endswith("eq") else np.ones(len(wts[cb]))
                rc = land.rc(cb)
                v, o = aggregate(row_values(sc, A, B, zeros[cb], rc), w)
                if sc.fz:
                    med, sd = null[(m, c)]; v = float(((v - med) / sd).sum())
                per[c] = v; vals.append(v); nm.append(f"{m}|{c}"); om.append(o)
            if m in SOFT:
                for lam in (.25, .5, .75):
                    vals.append(lam * per["P0"] + (1 - lam) * per["P1"]); nm.append(f"{m}|P5l{lam}"); om.append(np.nan)
            if m in NORM_ALT:
                c_ = float(m.rsplit(".c", 1)[1]); kind = "M03" if ".M03." in m else "M04"
                for cb in ("P1", "P4K8"):
                    A, B = bl[cb]; w = wts[cb]; rc = land.rc(cb)
                    Bs = [B[i] if rc is None else B[i][:, rc[i]] for i in range(len(B))]
                    if r_i == 0:
                        E0[cb] = float(bl["P0"][1][0].sum())               # pooled invariant UX scale, frozen
                    v0 = [soft_fixed_scale(b, E0[cb], c_, kind) for b in Bs]
                    vals.append(float(np.average(v0, weights=w))); nm.append(f"{m}|{cb}N0"); om.append(np.nan)
                    eB = np.array([b.sum() for b in Bs]); loc = np.array(row_values(sc, A, B, zeros[cb], rc), float); ok = np.isfinite(loc) & (eB > 0)
                    vals.append(float((loc[ok] * eB[ok]).sum() / eB[ok].sum()) if ok.any() else np.nan); nm.append(f"{m}|{cb}N2"); om.append(np.nan)
        L.append(vals); names = nm; omit.append(om)
    return np.array(L, float), names, np.nanmax(np.array(omit, float), 0)


def run_toy(toy, fid="pilot", seed=31):
    out = COND / f"{fid}_s{seed}" / f"{toy}.npz"
    if out.is_file():
        print(f"  [{toy}] done"); return
    from wave4_2.inputs import load                                   # evaluation harness: T for the analytic arm, saved libraries
    from wave4_2.truth import oracle_eta
    from wave4_2.surfaces import DecoderSurface
    t0 = time.time(); pl = load_payload(toy); b, ev, rots, _ = load(toy)
    sizes, nbl = SIZES[fid]; d = product_design(pl, (sizes[0], sizes[1] if pl.fixed_bank is None else nbl), seed)
    X = d["X"]; Xs = (X - pl.x_mean) / pl.x_std
    groups = {K: kmeans_groups(Xs, K) for K in KS}
    pm = {"nn": product_moments(DecoderSurface(pl).eta(), pl.z[d["sub"]], X, pl.x_mean, pl.x_std, pl.x_domain),
          "oracle": product_moments(oracle_eta(ev.family, pl.x_mean, pl.x_std), ev.T[d["sub"]], X, pl.x_mean, pl.x_std, pl.x_domain)}
    print(f"  [{toy}] moments [{time.time()-t0:.0f}s]", flush=True)
    # correctness: per-row moments reconstruct the pooled moment of the same points (FULL)
    chk = []; mode0 = masks_of(pl)[0]
    for arm in pm:
        whole = group_moments(pm[arm], np.zeros(len(X), int), mode0)[0]; ruu = pm[arm]["per_lat"][mode0][2]
        Kr, Jr = pm[arm]["K_r"][ruu].mean(0), pm[arm]["J_r"][ruu].mean(0)
        chk.append(dict(arm=arm, mode=mode0, K_pool_vs_rows=float(np.abs(whole[0] - Kr).max() / max(np.abs(Kr).max(), 1e-300)),
                        K_lat_vs_rows=float(np.abs(pm[arm]["per_lat"][mode0][0].mean(0) - Kr).max() / max(np.abs(Kr).max(), 1e-300)),
                        J_lat_vs_rows=float(np.abs(pm[arm]["per_lat"][mode0][1].mean(0) - Jr).max() / max(np.abs(Jr).max(), 1e-300))))
        assert max(chk[-1]["K_pool_vs_rows"], chk[-1]["K_lat_vs_rows"], chk[-1]["J_lat_vs_rows"]) < 1e-10, chk[-1]
    arr = dict(meta=np.array(json.dumps(dict(toy=toy, fid=fid, seed=seed, design=d["meta"], groups={K: int(len(np.unique(g))) for K, g in groups.items()},
                                             finite_rows={a: int(pm[a]["finite_rows"].sum()) for a in pm}, checks=chk, code_hash=code_hash()), default=float)))
    for v in rots:
        man = [dict(name=r.name, candidate_type=r.candidate_type, axes=list(r.axes), angle=float(r.angle)) for r in rots[v]]
        arr[f"{v}|manifest"] = np.array(json.dumps(man))
        for mode in masks_of(pl):
            for arm in pm:
                land = Landscape(pm[arm], mode, groups, PANEL)
                if arm == "oracle":
                    sc = rots[v][0].t_std; Jt_of = lambda r: jac(r.t_std, r.G).T
                else:
                    sc = rots[v][0].z_std; Jt_of = lambda r: jac(r.z_std, r.R).T
                L, names, om = evaluate_library(land, rots[v], np.asarray(sc), PANEL, Jt_of)
                arr[f"{v}|{mode}|{arm}|L"] = L; arr[f"{v}|{mode}|{arm}|omitted"] = om; arr["names"] = np.array(names)
                print(f"  [{toy}] {v} {mode} {arm}: {L.shape} [{time.time()-t0:.0f}s]", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **arr)


def analyze():
    rows = []
    for f in sorted(COND.glob("*/*.npz")):
        z = np.load(f); meta = json.loads(str(z["meta"])); toy = meta["toy"]; names = list(z["names"]); k = None
        for key in z.files:
            if not key.endswith("|oracle|L"):
                continue
            v, mode, _, _ = key.split("|"); man = json.loads(str(z[f"{v}|manifest"])); sw = sweeps(man)
            kk = max(max(r["axes"]) for r in man if r["axes"]) + 1; n_exp = 2 ** kk - kk - 1
            Lo, Li = z[f"{v}|{mode}|oracle|L"], z[f"{v}|{mode}|nn|L"]
            fo, fi = library_flags(curve_stats(Lo, sw), n_exp), library_flags(curve_stats(Li, sw), n_exp)
            e1 = eval1(Lo, Li, man, sw)
            for j, nm in enumerate(names):
                m, c = nm.rsplit("|", 1)
                rows.append(dict(toy=toy, cls="C" if toy.endswith("L") else "B", fam_group=family_group(toy), fid=meta["fid"], seed=meta["seed"], variant=v, mask=mode,
                                 metric=m, config=c, O_perfect=bool(fo["perfect"][j]), O_marginal=bool(fo["marginal"][j]), O_strict_perfect=bool(fo["strict_perfect"][j]),
                                 I_perfect=bool(fi["perfect"][j]), I_marginal=bool(fi["marginal"][j]), I_strict_marginal=bool(fi["strict_marginal"][j]),
                                 both_perfect=bool(fo["perfect"][j] and fi["perfect"][j]), both_marginal=bool(fo["marginal"][j] and fi["marginal"][j]),
                                 O_min_margin=float(fo["min_margin"][j]), I_min_margin=float(fi["min_margin"][j]), O_constant=bool(fo["any_constant"][j]), I_constant=bool(fi["any_constant"][j]),
                                 pearson=float(e1["pearson"][j]) if "pearson" in e1 else np.nan, spearman=float(e1["spearman"][j]) if "spearman" in e1 else np.nan,
                                 omitted_O=float(z[f"{v}|{mode}|oracle|omitted"][j]), omitted_I=float(z[f"{v}|{mode}|nn|omitted"][j])))
    ps = pd.DataFrame(rows); ps.to_csv(RES / "conditional_pass_summary.csv.gz", index=False); print(len(ps), "rows")
    return ps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--toy"); ap.add_argument("--fid", default="pilot"); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--analyze", action="store_true"); a = ap.parse_args()
    if a.analyze:
        analyze(); return
    run_toy(a.toy, a.fid, a.seed)


if __name__ == "__main__":
    main()
