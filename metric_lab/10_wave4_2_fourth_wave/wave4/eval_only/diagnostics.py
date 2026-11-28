"""W4-02 measure diagnostics (04_fourth_wave sections 10-11).

Blind parts (no T): support() distribution diagnostics of every integration law against empirical TRAIN / VALIDATION Z; tail_energy()
share of the base-frame UX energy carried by latent draws with max_i |(z_i - mu_i) / sd_i| > 2, > 3; predictive() decoder predictions on
ACTUAL VALIDATION rows stratified by the same tail measure (synthetic off-support points have no observed Y and are never "validated").
Evaluation-only parts (after the measure stage lock): decomposition() exact GL loss differences between archived bad Q01 / Q02 rotations,
the Q05-found rotation and the supervised R* on every controlled measure (row-score / row-weight / global terms, evaluator-only row
alignment); geodesics() continuous orthogonal paths between those rotations, fine pair-angle sweeps and random skew directions around the
Q05 rotation; block_bootstrap() latent-draw-block bootstrap of L(R_hat) - L(R*) (rows sharing one latent draw are resampled together).
"""
import json
import numpy as np, pandas as pd
from scipy.linalg import expm, logm
from scipy.optimize import linear_sum_assignment
from scipy import stats
from ..common import OUT, OUT09, CAND, chan, prim_mask, spec_of
from .. import moments as MO, designs as DS
from ..search import job_path, objective
from wave3 import rowmetrics as RM
from wave3.objectives import ux_factory
from wave2.blind_inputs import load_payload, variant_frame

LAWS = ("Q01", "Q02", "Q03", "M-E", "M-J", "M-D", "M-I", "RHO-0.25", "RHO-0.5", "RHO-0.75", "KAP-0.75", "KAP-1.25", "TRN-2", "TRN-3")
DEC_LAWS = ("Q01", "Q02", "M-E", "M-J", "M-D", "M-I", "RHO-0.5", "KAP-0.75", "KAP-1.25", "TRN-2", "TRN-3")


def _std(pl, Z):
    Zt = pl.z[np.sort(pl.splits["train"])]; return (Z - Zt.mean(0)) / Zt.std(0)


def support(toy, seed=31):
    pl = load_payload(toy); Zt = pl.z[np.sort(pl.splits["train"])]; Zv = pl.z[np.sort(pl.splits["validation"])]; mu, S = Zt.mean(0), np.cov(Zt, rowvar=False, ddof=0)
    Si = np.linalg.inv(S); maha = lambda Z: np.einsum("ni,ij,nj->n", Z - mu, Si, Z - mu)
    thr_m = np.quantile(maha(Zv), .99); rng = np.random.default_rng(0); ref = Zt[rng.choice(len(Zt), min(3000, len(Zt)), replace=False)]
    nn = lambda Z: np.sqrt(((Z[:, None, :] - ref[None]) ** 2).sum(-1)).min(1)
    thr_nn = np.quantile(nn(Zv[:2000]), .99); q = {a: np.quantile(Zt, [a, 1 - a], axis=0) for a in (.01, .005)}; rows = []
    sets = [("empirical TRAIN Z", Zt), ("empirical VALIDATION Z", Zv)]
    for M in LAWS:
        if not DS.available(pl, M):
            continue
        res = MO.get(pl, M, "C", seed, "train", blocks=True); sets.append((M, res[2]["lat_blocks"]))
    for name, Z in sets:
        zs = _std(pl, Z); C = np.corrcoef(Z.T); off = np.abs(C[~np.eye(pl.k, dtype=bool)])
        r = dict(toy=toy, law=name, n=len(Z), mean_abs_std_mean=float(np.abs(zs.mean(0)).mean()), sd_ratio_mean=float(zs.std(0).mean()),
                 skew_max=float(np.abs(stats.skew(Z, 0)).max()), exkurt_max=float(np.abs(stats.kurtosis(Z, 0)).max()), exkurt_mean=float(stats.kurtosis(Z, 0).mean()),
                 mean_abs_corr=float(off.mean()), max_abs_corr=float(off.max()), eig_ratio=float(np.linalg.eigvalsh(np.cov(Z.T))[::-1][0] / np.linalg.eigvalsh(np.cov(Z.T)).min()),
                 frac_out_1_99=float(((Z < q[.01][0]) | (Z > q[.01][1])).any(1).mean()), frac_out_05_995=float(((Z < q[.005][0]) | (Z > q[.005][1])).any(1).mean()),
                 frac_maha_gt_val99=float((maha(Z) > thr_m).mean()), frac_nn_gt_val99=float((nn(Z[:2000]) > thr_nn).mean()),
                 frac_tail2=float((np.abs(zs).max(1) > 2).mean()), frac_tail3=float((np.abs(zs).max(1) > 3).mean()))
        rows.append(r)
    return pd.DataFrame(rows)


def tail_energy(toy, seed=31):
    pl = load_payload(toy); rows = []
    for M in LAWS:
        if not DS.available(pl, M):
            continue
        res, meta, bl = MO.get(pl, M, "C", seed, "train", blocks=True); t = np.abs(_std(pl, bl["lat_blocks"])).max(1)
        for m in MO.masks_for(pl):
            SJ = bl[f"{m}|SJ"]; e = np.einsum("bjaa->b", SJ); E = e.sum()
            rows.append(dict(toy=toy, law=M, mask=m, n_blocks=len(e), energy_tail2=float(e[t > 2].sum() / E), energy_tail3=float(e[t > 3].sum() / E),
                             blocks_tail2=float((t > 2).mean()), blocks_tail3=float((t > 3).mean()), top1pct_block_energy=float(np.sort(e)[::-1][:max(1, len(e) // 100)].sum() / E),
                             valid_pairs=float(bl[f"{m}|NJ"].sum() / pl.p)))
    return pd.DataFrame(rows)


def predictive(toy):
    """decoder vs observed Y on VALIDATION rows, by latent tail stratum (blind: no T)"""
    from wave4_2.surfaces import DecoderSurface
    pl = load_payload(toy); ds = DecoderSurface(pl); V = np.sort(pl.splits["validation"]); rows_ = np.concatenate([pl.starts[s] + np.arange(pl.counts[s]) for s in V])
    sid = pl.subject_index[rows_]; xs = (pl.x[rows_] - pl.x_mean) / pl.x_std; eta = np.concatenate([ds.predict(pl.z[sid[i:i + 50000]], xs[i:i + 50000]) for i in range(0, len(sid), 50000)])
    y = pl.y[rows_]; t = np.abs(_std(pl, pl.z[sid])).max(1); out = []
    for lo, hi in ((0, 2), (2, 3), (3, np.inf)):
        m = (t > lo) & (t <= hi)
        if m.sum() == 0:
            continue
        if pl.cls == "C":
            r2 = 1 - np.mean((y[m] - eta[m]) ** 2) / np.var(y[m]); out.append(dict(toy=toy, stratum=f"({lo},{hi}]", n_rows=int(m.sum()), frac_subjects=float(np.isin(V, np.unique(sid[m])).mean()), R2=float(r2)))
        else:
            p = 1 / (1 + np.exp(-eta[m])); ll = -np.mean(y[m] * np.log(p + 1e-12) + (1 - y[m]) * np.log(1 - p + 1e-12))
            out.append(dict(toy=toy, stratum=f"({lo},{hi}]", n_rows=int(m.sum()), frac_subjects=float(np.isin(V, np.unique(sid[m])).mean()), logloss=float(ll), acc=float(((p > .5) == (y[m] > .5)).mean())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- evaluation-only: fixed rotations
def fixed_rotations(toy, variant="s9"):
    """{name: R}: supervised R*, the Lab 09 FIN Q01 selection, the Lab 09 FIN-SENS Q02 selection and the REP-1 Q05 selection (iseed 31, sseed 31)"""
    from wave2.evaluate_recovery import EvalData
    pl = load_payload(toy); ed = EvalData(toy); mask = prim_mask(pl); t = toy.replace("#", "_"); out = dict(R_star=ed.refs[variant]["R_star"])
    z = np.load(OUT09 / "blind" / "FIN" / t / f"{CAND}.npz"); out["R_Q01"] = z[f"{variant}|Q01|{mask}|i31|s31|S0|R"]
    z = np.load(OUT09 / "blind" / "FIN-SENS" / t / f"{CAND}-FIN.npz"); out["R_Q02"] = z[f"{variant}|Q02|{mask}|i31|s31|S0|R"]
    z = np.load(job_path("REP-1", toy, CAND, "Q05")); out["R_Q05"] = z[f"{variant}|Q05@C|{mask}|i31|s31|S0|R"]
    return out, pl, mask


def align(Ra, Rb):
    """signed permutation P (evaluator only) maximising tr(P Ra Rb^T) -> P Ra ; perm[i] = row of Ra matched to row i of Rb"""
    C = Rb @ Ra.T; r, c = linear_sum_assignment(-np.abs(C)); P = np.zeros_like(C); P[r, c] = np.sign(C[r, c]); return P @ Ra


def decomposition(toy, variant="s9", seed=31):
    rots, pl, mask = fixed_rotations(toy, variant); sc = variant_frame(pl, variant)[1]; sp = spec_of(CAND); c = chan(pl); rows, per = [], []
    for M in DEC_LAWS:
        if not DS.available(pl, M):
            continue
        mo = MO.get(pl, M, "C", seed, "train")[0][(c, mask)]; Bf = ux_factory(mo["J"], sc)
        Ls = {n: RM.value(sp, Bf(R)) for n, R in rots.items()}
        for a, b in (("R_star", "R_Q01"), ("R_star", "R_Q05"), ("R_star", "R_Q02"), ("R_Q05", "R_Q01")):
            Ra, Rb = rots[a], align(rots[b], rots[a]); d = RM.decompose(sp, Bf(Ra), Bf(Rb))
            rows.append(dict(toy=toy, variant=variant, law=M, a=a, b=b, L_a=Ls[a], L_b=Ls[b], dL=d["dL"], dL_row=d["dL_row"], score_term=float(d["score_term"].sum()),
                             weight_term=float(d["weight_term"].sum()), d_global=d["d_extra"], residual=d["residual"], b_preferred=bool(d["dL"] < 0)))
            for i in range(pl.k):
                per.append(dict(toy=toy, variant=variant, law=M, a=a, b=b, row=i, w_a=d["w_a"][i], w_b=d["w_b"][i], s_a=d["s_a"][i], s_b=d["s_b"][i],
                                score_term=d["score_term"][i], weight_term=d["weight_term"][i]))
        for n, R in rots.items():
            B = Bf(R)
            for i in range(pl.k):
                per.append(dict(toy=toy, variant=variant, law=M, a=n, b="(B row)", row=i, **{f"B_{j}": float(B[i, j]) for j in range(pl.p)}))
    return pd.DataFrame(rows), pd.DataFrame(per)


def geodesics(toy, variant="s9", seed=31, n_skew=20):
    rots, pl, mask = fixed_rotations(toy, variant); sc = variant_frame(pl, variant)[1]; sp = spec_of(CAND); c = chan(pl); k = pl.k; rows = []
    Ls = {M: (lambda Bf: (lambda R: RM.value(sp, Bf(R))))(ux_factory(MO.get(pl, M, "C", seed, "train")[0][(c, mask)]["J"], sc)) for M in DEC_LAWS if DS.available(pl, M)}
    ts = np.linspace(-.25, 1.25, 61)
    for a, b in (("R_Q05", "R_Q01"), ("R_star", "R_Q01"), ("R_star", "R_Q05")):
        Ra = rots[a]; Rb = align(rots[b], Ra)
        if np.linalg.det(Rb) * np.linalg.det(Ra) < 0:
            Rb = Rb.copy(); Rb[-1] *= -1                        # same component of O(k) (a signed permutation is allowed)
        Lg = np.real(logm(Rb @ Ra.T)); Lg = (Lg - Lg.T) / 2
        for t in ts:
            R = expm(t * Lg) @ Ra
            rows.append(dict(toy=toy, variant=variant, path=f"{a}->{b}", t=float(t), **{f"L_{M}": f(R) for M, f in Ls.items()}))
    R0 = rots["R_Q05"]; rng = np.random.default_rng(5)
    for i in range(k):
        for j in range(i + 1, k):
            for deg in (-15, -10, -5, -2, -1, -.5, 0, .5, 1, 2, 5, 10, 15):
                G = np.eye(k); th = np.deg2rad(deg); G[[i, i, j, j], [i, j, i, j]] = [np.cos(th), -np.sin(th), np.sin(th), np.cos(th)]
                rows.append(dict(toy=toy, variant=variant, path=f"pair{i}{j}@R_Q05", t=float(deg), **{f"L_{M}": f(G @ R0) for M, f in Ls.items()}))
    for s in range(n_skew):
        A = rng.standard_normal((k, k)); A = (A - A.T) / 2; A /= np.linalg.norm(A) / np.sqrt(2)
        for deg in (0, 1, 2, 5, 10):
            rows.append(dict(toy=toy, variant=variant, path=f"skew{s}@R_Q05", t=float(deg), **{f"L_{M}": f(expm(np.deg2rad(deg) * A) @ R0) for M, f in Ls.items()}))
    return pd.DataFrame(rows)


def block_bootstrap(toy, variant="s9", n_boot=400, measures=(("Q05", "C", 31), ("Q05", "C", 73), ("Q05", "C", 101), ("Q05", "D", 211), ("Q05", "D", 613), ("Q01", "C", 31))):
    rots, pl, mask = fixed_rotations(toy, variant); sc = variant_frame(pl, variant)[1]; sp = spec_of(CAND); rows = []; rng = np.random.default_rng(11)
    for M, fid, s in measures:
        res, meta, bl = MO.get(pl, M, fid, s, "train", blocks=True); SJ, NJ = bl[f"{mask}|SJ"], bl[f"{mask}|NJ"]; nb = len(SJ)
        def L_of(idx_w):
            J = np.einsum("b,bjxy->jxy", idx_w, SJ) / np.maximum(idx_w @ NJ, 1)[:, None, None]; Bf = ux_factory(J, sc)
            return {n: RM.value(sp, Bf(R)) for n, R in rots.items()}
        full = L_of(np.ones(nb)); dd = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float); l = L_of(w); dd.append([l["R_Q05"] - l["R_star"], l["R_Q01"] - l["R_star"], l["R_Q05"] - l["R_Q01"]])
        dd = np.array(dd)
        rows.append(dict(toy=toy, variant=variant, measure=f"{M}@{fid}", seed=s, n_blocks=nb, dL_Q05_star=full["R_Q05"] - full["R_star"], dL_Q01_star=full["R_Q01"] - full["R_star"],
                         dL_Q05_Q01=full["R_Q05"] - full["R_Q01"], sd_Q05_star=float(dd[:, 0].std()), sd_Q01_star=float(dd[:, 1].std()), sd_Q05_Q01=float(dd[:, 2].std()),
                         p_Q05_below_star=float((dd[:, 0] < 0).mean()), p_Q01_below_star=float((dd[:, 1] < 0).mean()), p_Q05_below_Q01=float((dd[:, 2] < 0).mean())))
    return pd.DataFrame(rows)
