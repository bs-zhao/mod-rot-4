"""W2-04 improved inference of FIXED metrics: estimator fitting (blind) and evaluation against the analytic losses (after fitting).

usage: python run_inference.py --batch IF-A|IF-B|IF-C --toy CR3       fit + evaluate one toy
       python run_inference.py --analyze                               -> results/inference_pass_summary.csv.gz, fit / derivative tables
IF-A  AD (frozen decoder), W1 polynomial (Lab 07 fit_poly, unchanged), E-SMOOTH h_V in {.025, .05, .1, .2} (16 antithetic draws)
IF-B  E-POLY mono / herm, degree 2-5, ridge 1e-7..1 (8); E-RBF iso / block, 1024 landmarks, factor .5 / 1 / 2, ridge 1e-6..1 (4);
      D0 observed data and D1 frozen-decoder teacher; TRAIN 1024 subjects x 64 trials, VALIDATION 256 x 64.  Logistic D0 (class B):
      degree <= 4 (degree 5 logistic skipped for cost, declared before fitting).
      Selectors (blind, predeclared): S-min = lowest validation loss; S-1pct = most regularised configuration within 1% relative of the
      minimum (polynomial: lowest degree, then largest ridge; RBF: largest bandwidth, then largest ridge); every per-degree best is also
      evaluated (diagnostic, never selected by truth).
IF-C  cross-fitting for the priority observed-data surrogate (the D0 family x basis with the lowest validation loss under S-min, blind):
      3 repeated disjoint TRAIN subject half-splits (512 + 512 subjects x 64 trials); energies: single fit, mean of the two fits' energies,
      symmetrised cross-product, cross-product with PSD-projected base blocks; Q01 medium (512 x 64) TEST points, seeds 31 / 73 / 101.
Evaluation points: Q01 TEST (IF-A / IF-B pilot 256 x 32 seed 31 = Lab 07 batch A; IF-C medium).  Same masks, coordinates and metric code for
every surface; the analytic arm is read from the Lab 07 losses at the identical points.
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave2.common import OUT, RES, OUT07, LAB07, PRIMARY, VARIANTS, cfg, code_hash, toy_entry, family_group
import numpy as np, pandas as pd, torch
from wave2.blind_inputs import load_payload
from wave2.designs import design
from wave2.objectives import masks_of
from wave2 import surrogates as S
from wave4_2.moments import Moments, contract, jac
from wave4_2.measures import col_mask
from wave4_2.derivatives import ad_chunks

INF = OUT / "inference"
BUD = cfg("budgets.json")["inference"]
PANEL = json.loads((LAB07 / "configs" / "priority_panel.json").read_text())["ids"]


def chan(pl):
    return "YC" if pl.cls == "C" else "YL"


# ---------------------------------------------------------------- Hessians of each surface at design points
def batch_hessian(fn, P, chunk=1024):
    """Hessian of a row-separable torch function fn(W (n, D)) -> (n,) by D + 1 backward passes"""
    out = []
    for i in range(0, len(P), chunk):
        W = torch.as_tensor(P[i:i + chunk], dtype=torch.float64).requires_grad_(True); y = fn(W)
        g = torch.autograd.grad(y.sum(), W, create_graph=True)[0]
        H = torch.stack([torch.autograd.grad(g[:, j].sum(), W, retain_graph=True)[0] for j in range(W.shape[1])], 1)
        out.append(H.detach().numpy())
    return np.concatenate(out)


def poly_hessian(pe, P):
    m, s, E, basis, w, b = pe.m, pe.s, pe.E, pe.basis, pe.w, pe.b
    chunk = max(32, int(2e6 / (len(E) * E.shape[1])))
    return batch_hessian(lambda W: b + S.phi_torch((W - m) / s, E, basis) @ w, P, chunk)


def moments_from_H(Hfun, pl, d, masks):
    xs = (d["x"] - pl.x_mean) / pl.x_std; cm = col_mask(d["x"], pl.x_domain); P = np.column_stack((d["lat"], xs))
    acc = {m: Moments(pl.k, pl.p, m) for m in masks}
    for i in range(0, len(P), 2048):
        H = Hfun(P[i:i + 2048])
        for m in masks:
            acc[m].add(H, cm[i:i + 2048])
    return {m: a.finish() for m, a in acc.items()}


def ad_H(eta):
    return lambda P: np.concatenate([h for _, _, _, h in ad_chunks(eta, P, 1024)])


# ---------------------------------------------------------------- evaluation against the saved Lab 07 analytic losses
def evaluate(pl, mos, Q, fid, seed, ids, rots):
    import run_baseline as RB7                                      # Lab 07 loss function (identical nn convention)
    from wave4_2.evaluation import sweeps, curve_stats, library_flags, eval1, matrix_errors
    run = OUT07 / "W1" / toy_entry(pl.toy)["run"]; rows = []; k = pl.k; n_exp = 2 ** k - k - 1
    for v in VARIANTS:
        for m in mos:
            lf = run / "losses" / v / f"{Q}_{fid}_s{seed}" / f"{chan(pl)}_{m}.npz"
            if not lf.is_file():
                continue
            z = np.load(lf); zid = list(z["ids"]); use = [i for i in ids if i in zid]; jj = [zid.index(i) for i in use]
            Lo = z["oracle|L"][:, jj]; man = json.loads(str(z["manifest"])); sw = sweeps(man)
            L, As, Bs, info = RB7.losses(mos[m], rots[v], "nn", use)
            fo, fi = library_flags(curve_stats(Lo, sw), n_exp), library_flags(curve_stats(L, sw), n_exp); e1 = eval1(Lo, L, man, sw)
            me = np.array([matrix_errors(z["oracle|A"][r], z["oracle|B"][r], As[r], Bs[r]) for r in range(len(As))])
            for j, mid in enumerate(use):
                rows.append(dict(variant=v, mask=m, metric=mid, O_perfect=bool(fo["perfect"][j]), O_marginal=bool(fo["marginal"][j]), I_perfect=bool(fi["perfect"][j]),
                                 I_marginal=bool(fi["marginal"][j]), I_strict_marginal=bool(fi["strict_marginal"][j]), I_min_margin=float(fi["min_margin"][j]),
                                 I_constant=bool(fi["any_constant"][j]), both_perfect=bool(fo["perfect"][j] and fi["perfect"][j]),
                                 both_marginal=bool(fo["marginal"][j] and fi["marginal"][j]), pearson=float(e1["pearson"][j]), spearman=float(e1["spearman"][j]),
                                 UU_rel_err=float(np.nanmean(me[:, 0])), UX_rel_err=float(np.nanmean(me[:, 1])), oracle_zero_false_energy=float(np.nanmean(me[:, 2])),
                                 valid=bool(mos[m]["valid"])))
    return rows


# ---------------------------------------------------------------- fitting
def data(pl, seed=7):
    tr, trs = S.rows_of(pl, "train", *BUD["train_cap"], seed); va, vas = S.rows_of(pl, "validation", *BUD["validation_cap"], seed)
    Wtr, Wva = S.inputs(pl, tr), S.inputs(pl, va)
    return dict(tr=tr, va=va, Wtr=Wtr, Wva=Wva, y0tr=pl.y[tr], y0va=pl.y[va], y1tr=S.teacher(pl, Wtr), y1va=S.teacher(pl, Wva), tr_subjects=trs)


def fit_poly_family(pl, D, basis, degrees, sources, log):
    """-> list of fitted configurations with validation losses (one Gram per degree shared by the MSE sources)"""
    m = D["Wtr"].mean(0); s = D["Wtr"].std(0); s[s <= 0] = 1; Utr, Uva = (D["Wtr"] - m) / s, (D["Wva"] - m) / s; out = []
    for deg in degrees:
        E = S.exponents(Utr.shape[1], deg); t0 = time.time()
        Fva = S.phi_np(Uva, E, basis)
        mse_src = [src for src in sources if not (src == "D0" and pl.cls == "B")]
        if mse_src:
            def F_iter(Utr=Utr, E=E):
                for i in range(0, len(Utr), 4096):
                    yield S.phi_np(Utr[i:i + 4096], E, basis)
            for src in mse_src:
                y = D["y1tr"] if src == "D1" else D["y0tr"]; yv = D["y1va"] if src == "D1" else D["y0va"]
                path, info = S.ridge_path(F_iter, y, BUD["poly_ridge"], len(Utr), len(E))
                for lam, w, b in path:
                    out.append(dict(family="POLY", basis=basis, degree=deg, lam=lam, source=src, val_loss=S.nll_or_mse(b + Fva @ w, yv, False),
                                    n_terms=len(E), cond=info["cond"], rank=info["rank"], est=S.PolyEta(m, s, E, basis, w, b)))
        if "D0" in sources and pl.cls == "B":
            if deg >= 5:
                log.append(dict(family="POLY", basis=basis, degree=deg, source="D0", status="skipped (logistic degree 5: declared cost cap)")); continue
            Ftr = S.phi_np(Utr, E, basis)
            for lam, w, b, ok, nit in S.logistic_path(Ftr, D["y0tr"], BUD["poly_ridge"]):
                out.append(dict(family="POLY", basis=basis, degree=deg, lam=lam, source="D0", val_loss=S.nll_or_mse(b + Fva @ w, D["y0va"], True), n_terms=len(E),
                                converged=ok, iters=nit, est=S.PolyEta(m, s, E, basis, w, b)))
            del Ftr
        print(f"    poly {basis} deg {deg}: {len(E)} terms [{time.time()-t0:.0f}s]", flush=True)
    return out


def fit_rbf_family(pl, D, scaling, sources, seed=11):
    W = D["Wtr"]; m = W.mean(0); s = W.std(0); s[s <= 0] = 1; U = (W - m) / s
    if scaling == "block":                                          # latent and X blocks each scaled to unit median distance
        k = pl.k; s = s.copy(); s[:k] *= S.median_dist(U[:, :k]); s[k:] *= S.median_dist(U[:, k:]); U = (W - m) / s
    rng = np.random.default_rng([seed, 61]); C = U[rng.choice(len(U), BUD["rbf_landmarks"], replace=False)]; md = S.median_dist(U); out = []
    for fac in BUD["rbf_bandwidth_factors"]:
        h = fac * md; est0 = S.RBFEta(m, s, C, h, np.zeros(len(C)), 0.0); Fva = est0.features(D["Wva"])
        mse_src = [src for src in sources if not (src == "D0" and pl.cls == "B")]
        if mse_src:
            def F_iter():
                for i in range(0, len(W), 8192):
                    yield est0.features(W[i:i + 8192])
            for src in mse_src:
                y = D["y1tr"] if src == "D1" else D["y0tr"]; yv = D["y1va"] if src == "D1" else D["y0va"]
                path, info = S.ridge_path(F_iter, y, BUD["rbf_ridge"], len(W), len(C))
                for lam, a, b in path:
                    out.append(dict(family="RBF", basis=scaling, degree=fac, lam=lam, source=src, val_loss=S.nll_or_mse(b + Fva @ a, yv, False), n_terms=len(C),
                                    cond=info["cond"], est=S.RBFEta(m, s, C, h, a, b)))
        if "D0" in sources and pl.cls == "B":
            Ftr = est0.features(W)
            for lam, a, b, ok, nit in S.logistic_path(Ftr, D["y0tr"], BUD["rbf_ridge"]):
                out.append(dict(family="RBF", basis=scaling, degree=fac, lam=lam, source="D0", val_loss=S.nll_or_mse(b + Fva @ a, D["y0va"], True), n_terms=len(C),
                                converged=ok, iters=nit, est=S.RBFEta(m, s, C, h, a, b)))
            del Ftr
    return out


def select(fits):
    """S-min and S-1pct per (family, basis, source); per-degree best (diagnostic)"""
    df = pd.DataFrame([{k: v for k, v in f.items() if k != "est"} for f in fits]); df["i"] = range(len(df)); chosen = {}
    for (fam, basis, src), g in df.groupby(["family", "basis", "source"]):
        best = g.loc[g.val_loss.idxmin()]; chosen[(fam, basis, src, "S-min")] = int(best.i)
        tol = g[g.val_loss <= best.val_loss * (1 + BUD["admissible_prediction_tolerance"])]
        tol = tol.sort_values(["degree", "lam"], ascending=[fam == "POLY", False]) if fam == "POLY" else tol.sort_values(["degree", "lam"], ascending=[False, False])
        chosen[(fam, basis, src, "S-1pct")] = int(tol.iloc[0].i)
        for deg, gd in g.groupby("degree"):
            chosen[(fam, basis, src, f"deg{deg}")] = int(gd.loc[gd.val_loss.idxmin()].i)
    return df, chosen


def surface_H(f):
    est = f["est"]
    return (lambda P: poly_hessian(est, P)) if f["family"] == "POLY" else (lambda P: est.hessian(P))


# ---------------------------------------------------------------- batches
def run_ifa(toy):
    out = INF / "IF-A" / f"{toy.replace('#', '_')}.csv"
    if out.is_file():
        print(f"  [{toy}] IF-A done"); return
    from wave4_2.inputs import load
    from wave4_2.surfaces import DecoderSurface, fit_poly
    t0 = time.time(); pl = load_payload(toy); _, _, rots, _ = load(toy); masks = masks_of(pl)
    d = design(pl, "Q01", "pilot", 31, pool="test"); dec = DecoderSurface(pl); eta = dec.eta(); rows, fit = [], []
    Dva = data(pl)                                                   # validation rows for the smoothing selection
    est = [("AD", "h0", lambda: moments_from_H(ad_H(eta), pl, d, masks))]
    poly, diag = fit_poly(pl); fit.append(dict(estimator="POLY-W1", **{k: v for k, v in diag.items() if k != "grid"}))
    est.append(("POLY-W1", f"deg{diag['selected_degree']}_lam{diag['selected_lambda']}", lambda: moments_from_H(ad_H(poly.eta()), pl, d, masks)))
    zsd = pl.z.std(0)
    for h in BUD["smooth_bandwidths"]:
        est.append(("SMOOTH", f"h{h}", lambda h=h: moments_from_H(lambda P: S.smooth_hessians(eta, P, zsd, pl.k, h, 16, seed=31), pl, d, masks)))
    # blind selection inputs: validation predictive loss of g_h against observed Y (class B: Bernoulli NLL, class C: MSE in Y units)
    rng = np.random.default_rng([31, 51]); half = rng.standard_normal((8, pl.k)); xi = np.concatenate([half, -half])
    for h in [0.0] + list(BUD["smooth_bandwidths"]):
        W = Dva["Wva"]; pred = np.mean([dec.predict(W[:, :pl.k] + h * zsd * e, W[:, pl.k:]) for e in xi], 0)
        fit.append(dict(estimator="SMOOTH", setting=f"h{h}", val_loss=S.nll_or_mse(pred, Dva["y0va"], pl.cls == "B"),
                        val_teacher_mse=float(np.mean((pred - Dva["y1va"]) ** 2))))
    for name, setting, mk in est:
        mos = mk()
        for r in evaluate(pl, mos, "Q01", "pilot", 31, PANEL, rots):
            rows.append(dict(toy=toy, batch="IF-A", estimator=name, setting=setting, source="teacher" if name == "SMOOTH" else ("decoder" if name == "AD" else "D0"), **r))
        print(f"  [{toy}] IF-A {name} {setting} [{time.time()-t0:.0f}s]", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); pd.DataFrame(fit).to_csv(out.with_suffix(".fit.csv"), index=False)


def run_ifb(toy):
    out = INF / "IF-B" / f"{toy.replace('#', '_')}.csv"
    if out.is_file():
        print(f"  [{toy}] IF-B done"); return
    from wave4_2.inputs import load
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); pl = load_payload(toy); _, _, rots, _ = load(toy); masks = masks_of(pl); D = data(pl); log = []
    fits = []
    for basis in ("mono", "herm"):
        fits += fit_poly_family(pl, D, basis, BUD["poly_degrees"], ("D0", "D1"), log)
    for sc in ("iso", "block"):
        fits += fit_rbf_family(pl, D, sc, ("D0", "D1")); print(f"    rbf {sc} [{time.time()-t0:.0f}s]", flush=True)
    df, chosen = select(fits); df.drop(columns=[c for c in ("est",) if c in df]).assign(toy=toy).to_csv(out.with_suffix(".fit.csv"), index=False)
    pd.DataFrame(log).to_csv(out.with_suffix(".skipped.csv"), index=False)
    d = design(pl, "Q01", "pilot", 31, pool="test"); rows = []; done = {}
    for (fam, basis, src, sel), i in chosen.items():
        if i not in done:
            done[i] = evaluate(pl, moments_from_H(surface_H(fits[i]), pl, d, masks), "Q01", "pilot", 31, PANEL, rots)
        for r in done[i]:
            f = fits[i]; rows.append(dict(toy=toy, batch="IF-B", estimator=f"{fam}-{basis}", source=src, selector=sel, setting=f"deg{f['degree']}_lam{f['lam']}",
                                          val_loss=f["val_loss"], **r))
        print(f"  [{toy}] IF-B {fam} {basis} {src} {sel} [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    sel = {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": int(v) for k, v in chosen.items()}; (out.with_suffix(".chosen.json")).write_text(json.dumps(sel))


def psd(M):
    ev, V = np.linalg.eigh((M + M.T) / 2); return (V * np.maximum(ev, 0)) @ V.T


def run_ifc(toy):
    out = INF / "IF-C" / f"{toy.replace('#', '_')}.csv"
    if out.is_file():
        print(f"  [{toy}] IF-C done"); return
    from wave4_2.inputs import load
    t0 = time.time(); pl = load_payload(toy); _, _, rots, _ = load(toy); masks = masks_of(pl); k, p = pl.k, pl.p
    fb = pd.read_csv(INF / "IF-B" / f"{toy.replace('#', '_')}.fit.csv"); ch = json.loads((INF / "IF-B" / f"{toy.replace('#', '_')}.chosen.json").read_text())
    cand = [(key, i) for key, i in ch.items() if key.endswith("|D0|S-min")]
    key, i = min(cand, key=lambda t: fb.loc[t[1], "val_loss"]); fam, basis, _, _ = key.split("|"); cfg_row = fb.loc[i]
    rows = []; pool = np.sort(pl.splits["train"])
    for rep in range(3):
        perm = np.random.default_rng([rep, 71]).permutation(pool); halves = (perm[:len(perm) // 2], perm[len(perm) // 2:])
        ests = []
        for hidx, hs in enumerate(halves):
            tr, _ = S.rows_of(pl, hs, 512, 64, 100 + rep); va, _ = S.rows_of(pl, "validation", *BUD["validation_cap"], 7)
            D = dict(Wtr=S.inputs(pl, tr), Wva=S.inputs(pl, va), y0tr=pl.y[tr], y0va=pl.y[va], y1tr=None, y1va=None)
            if fam == "POLY":
                f = [g for g in fit_poly_family(pl, D, basis, [int(cfg_row.degree)], ("D0",), []) if g["lam"] == cfg_row.lam][0]
            else:
                f = [g for g in fit_rbf_family(pl, D, basis, ("D0",)) if g["degree"] == cfg_row.degree and g["lam"] == cfg_row.lam][0]
            ests.append(f)
        for seed in (31, 73, 101):
            d = design(pl, "Q01", "C", seed, pool="test"); xs = (d["x"] - pl.x_mean) / pl.x_std; cm = col_mask(d["x"], pl.x_domain); P = np.column_stack((d["lat"], xs))
            HA, HB = surface_H(ests[0])(P), surface_H(ests[1])(P)
            for m in masks:
                keep = cm.all(1) if m == "MASK-ALL" else np.ones(len(P), bool); colv = cm if m == "MASK-COL" else np.ones(cm.shape, bool)
                def mom(H1, H2):
                    h1, h2 = H1[keep], H2[keep]; K = np.einsum("nab,ncd->abcd", h1[:, :k, :k], h2[:, :k, :k]) / len(h1)
                    J = np.stack([(h1[colv[keep][:, j], :k, k + j].T @ h2[colv[keep][:, j], :k, k + j]) / max(colv[keep][:, j].sum(), 1) for j in range(p)])
                    K = (K + K.transpose(2, 3, 0, 1)) / 2; J = (J + J.transpose(0, 2, 1)) / 2
                    return dict(K=K, J=J, valid=True, nK=len(h1), nJ=colv[keep].sum(0), n_points=len(P), nonfinite=0, coverage=colv[keep].sum(0) / len(P))
                single = mom(HA, HA); avg = {kk: (mom(HA, HA)[kk] + mom(HB, HB)[kk]) / 2 if kk in ("K", "J") else single[kk] for kk in single}
                cross = mom(HA, HB); K2 = cross["K"].reshape(k * k, k * k)
                crossp = dict(cross, K=psd(K2).reshape(k, k, k, k), J=np.stack([psd(J) for J in cross["J"]]))
                for name, mo in (("single", single), ("mean_energy", avg), ("cross", cross), ("cross_psd", crossp)):
                    neg = float(np.mean(np.concatenate([np.linalg.eigvalsh((J + J.T) / 2) for J in mo["J"]]) < 0))
                    for r in evaluate(pl, {m: mo}, "Q01", "C", seed, PANEL, rots):
                        rows.append(dict(toy=toy, batch="IF-C", estimator=f"{fam}-{basis}", source="D0", selector="S-min", setting=f"deg{cfg_row.degree}_lam{cfg_row.lam}",
                                         energy=name, rep=rep, seed=seed, frac_negative_J_eigen=neg, **r))
            print(f"  [{toy}] IF-C rep {rep} seed {seed} [{time.time()-t0:.0f}s]", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False)


def analyze():
    for b in ("IF-A", "IF-B", "IF-C"):
        fs = sorted((INF / b).glob("*.csv")); fs = [f for f in fs if f.name.count(".") == 1]
        if fs:
            pd.concat([pd.read_csv(f) for f in fs]).to_csv(RES / f"inference_{b}_summary.csv.gz", index=False)
        ff = sorted((INF / b).glob("*.fit.csv"))
        if ff:
            pd.concat([pd.read_csv(f).assign(toy=f.name.split(".")[0]) for f in ff]).to_csv(RES / f"inference_{b}_fits.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--batch"); ap.add_argument("--toy"); ap.add_argument("--analyze", action="store_true"); a = ap.parse_args()
    if a.analyze:
        analyze(); return
    {"IF-A": run_ifa, "IF-B": run_ifb, "IF-C": run_ifc}[a.batch](a.toy)


if __name__ == "__main__":
    main()
